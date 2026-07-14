"""Matcher adapter for the SAMed paper's 8 biological-segmentation tasks.

Wraps the official Matcher (Liu et al., ICLR 2024) for N-shot evaluation on
our datasets. Training-free baseline using DINOv2 (vit_l14) for matching and
SAM (vit_b for consistency with MetaTune) as the mask generator.

Usage:
    python baselines/matcher_samed.py \
        --root_path $DATA_ROOT/blood-cell/train/Images \
        --test_path $DATA_ROOT/blood-cell/test/Images \
        --dataset blood --num_data 4 --seed 42 --gpu_id 0 \
        --output ./output_baselines/matcher
"""

import os
import sys
import random
import argparse
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from scipy.ndimage import zoom

warnings.filterwarnings("ignore")

# Vendored Matcher (must be on path)
MATCHER_DIR = os.path.join(os.path.dirname(__file__), "Matcher")
sys.path.insert(0, MATCHER_DIR)
from matcher.Matcher import build_matcher_oss  # noqa: E402

# Re-use SAMed's dice_score so numbers are directly comparable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cal_dice import dice_score  # noqa: E402


# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root_path", required=True)
    p.add_argument("--test_path", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--num_data", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--input_size", type=int, default=518, help="DINOv2 ViT-L14 expects 518x518")
    p.add_argument("--low_res", type=int, default=64, help="resolution at which Dice is computed (matches MetaTune)")
    # Encoder (DINOv2)
    p.add_argument("--dinov2_size", type=str, default="vit_large")
    p.add_argument("--dinov2_weights", type=str,
                   default=os.path.join(MATCHER_DIR, "models/dinov2_vitl14_pretrain.pth"))
    # SAM (use vit_b for consistency with MetaTune)
    p.add_argument("--sam_size", type=str, default="vit_b")
    p.add_argument("--sam_weights", type=str,
                   default="./checkpoints/sam_vit_b_01ec64.pth")
    # Matcher hyperparams — defaults from main_oss.py
    p.add_argument("--points_per_side", type=int, default=64)
    p.add_argument("--pred_iou_thresh", type=float, default=0.88)
    p.add_argument("--stability_score_thresh", type=float, default=0.95)
    p.add_argument("--sel_stability_score_thresh", type=float, default=0.0)
    p.add_argument("--iou_filter", type=float, default=0.0)
    p.add_argument("--box_nms_thresh", type=float, default=1.0)
    p.add_argument("--output_layer", type=int, default=3)
    p.add_argument("--dense_multimask_output", type=int, default=0)
    p.add_argument("--use_dense_mask", type=int, default=0)
    p.add_argument("--multimask_output", type=int, default=0)
    p.add_argument("--num_centers", type=int, default=8)
    p.add_argument("--use_box", action="store_true")
    p.add_argument("--use_points_or_centers", action="store_true", default=True)
    p.add_argument("--sample_range", type=str, default="(4,6)")
    p.add_argument("--max_sample_iterations", type=int, default=30)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--exp", type=float, default=0.0)
    p.add_argument("--emd_filter", type=float, default=0.0)
    p.add_argument("--purity_filter", type=float, default=0.0)
    p.add_argument("--coverage_filter", type=float, default=0.0)
    p.add_argument("--use_score_filter", action="store_true")
    p.add_argument("--deep_score_norm_filter", type=float, default=0.1)
    p.add_argument("--deep_score_filter", type=float, default=0.33)
    p.add_argument("--topk_scores_threshold", type=float, default=0.7)
    p.add_argument("--num_merging_mask", type=int, default=10)
    p.add_argument("--output", default="./output_baselines/matcher")
    return p.parse_args()


# ---------------------------------------------------------------------------
def load_image_and_mask(image_path, img_size):
    """Read RGB image + binary mask, resize to (img_size, img_size). Returns (img_uint8_HWC, mask_HW)."""
    img = np.array(Image.open(image_path).convert("RGB"))
    mask_path = image_path.replace("/Images", "/Masks")
    mask = np.uint8(np.array(Image.open(mask_path).convert("RGB")) > 0)[:, :, 0]
    x, y, _ = img.shape
    if x != img_size or y != img_size:
        img = zoom(img, (img_size / x, img_size / y, 1), order=3).clip(0, 255).astype(np.uint8)
        mask = (zoom(mask.astype(np.float32), (img_size / x, img_size / y), order=0) > 0.5).astype(np.uint8)
    return img, mask


def to_tensor_chw(img_uint8_hwc):
    """uint8 HWC -> float32 CHW in [0, 1]."""
    return torch.from_numpy(img_uint8_hwc).float().permute(2, 0, 1) / 255.0


# ---------------------------------------------------------------------------
def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.sample_range = eval(args.sample_range)  # tuple

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    exp_name = f"{args.dataset}{args.num_data}_matcher_img{args.input_size}_seed{args.seed}"
    out_dir = os.path.join(args.output, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[load] DINOv2 {args.dinov2_size} + SAM {args.sam_size}")
    matcher = build_matcher_oss(args)

    # ----- Sample N support images -----
    all_train = sorted(os.listdir(args.root_path))
    rng = random.Random(args.seed)
    support_files = rng.sample(all_train, args.num_data)
    print(f"[support] {args.num_data} files: {support_files}")

    support_imgs = []   # list of (3, H, W) float in [0,1]
    support_masks = []  # list of (H, W) {0,1}
    for fname in support_files:
        path = os.path.join(args.root_path, fname)
        img_np, msk_np = load_image_and_mask(path, args.input_size)
        if msk_np.sum() == 0:
            print(f"[skip support] {fname}: empty mask")
            continue
        support_imgs.append(to_tensor_chw(img_np))
        support_masks.append(torch.from_numpy(msk_np).float())

    if len(support_imgs) == 0:
        raise RuntimeError("No usable support images.")

    # Stack: imgs (1, ns, 3, H, W), masks (1, ns, H, W)
    support_imgs_t = torch.stack(support_imgs, dim=0).unsqueeze(0).to(args.device)  # (1, ns, 3, H, W)
    support_masks_t = torch.stack(support_masks, dim=0).unsqueeze(0).to(args.device)  # (1, ns, H, W)

    # ----- Test -----
    print(f"[test] Evaluating on {args.test_path}")
    test_files = sorted(os.listdir(args.test_path))
    dice_list = []
    for ti, tf in enumerate(test_files):
        path = os.path.join(args.test_path, tf)
        try:
            img_np, gt_np = load_image_and_mask(path, args.input_size)
        except Exception as e:
            print(f"[skip] {tf}: {e}")
            continue

        query_img = to_tensor_chw(img_np).unsqueeze(0).to(args.device)  # (1, 3, H, W)

        try:
            with torch.no_grad():
                matcher.set_reference(support_imgs_t, support_masks_t)
                matcher.set_target(query_img)
                pred_mask = matcher.predict()  # (1, H, W) on device
                matcher.clear()
        except Exception as e:
            # Some images may fail (e.g., empty SAM mask candidates); skip and continue
            print(f"[predict_fail] {tf}: {e}")
            continue

        pred_np = pred_mask.squeeze().cpu().numpy().astype(np.uint8)

        # Downsample pred + GT to low_res (matches MetaTune's evaluation)
        low_res = args.low_res
        ph, pw = pred_np.shape
        pred_lr = (zoom(pred_np.astype(np.float32), (low_res / ph, low_res / pw), order=0) > 0.5).astype(np.int64)
        gh, gw = gt_np.shape
        gt_lr = (zoom(gt_np.astype(np.float32), (low_res / gh, low_res / gw), order=0) > 0.5).astype(np.int64)

        # 2-class logits for dice_score
        logit_2c = np.stack([1 - pred_lr, pred_lr], axis=0)[None, ...]
        logit_2c_t = torch.tensor(logit_2c, dtype=torch.float32)
        gt_lr_t = torch.tensor(gt_lr, dtype=torch.long).unsqueeze(0)
        d = dice_score(logit_2c_t, gt_lr_t, bg=False).item()
        dice_list.append(d)
        if (ti + 1) % 50 == 0:
            print(f"  [test] {ti + 1}/{len(test_files)} dice running mean = {np.mean(dice_list):.4f}")

    mean_dice = float(np.mean(dice_list))
    print(f"[result] {args.dataset} seed={args.seed} num_data={args.num_data}: mean Dice = {mean_dice:.4f} over {len(dice_list)} test images")

    with open(os.path.join(out_dir, "result.txt"), "w") as f:
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"num_data: {args.num_data}\n")
        f.write(f"support_files: {support_files}\n")
        f.write(f"mean_dice: {mean_dice:.4f}\n")
        f.write(f"num_test_images: {len(dice_list)}\n")

    suffix = f"_{int(round(mean_dice * 10000)):04d}"
    new_dir = out_dir + suffix
    os.rename(out_dir, new_dir)
    print(f"[result] renamed dir -> {new_dir}")


if __name__ == "__main__":
    main()
