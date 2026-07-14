"""PerSAM-F adapter for the SAMed paper's 8 biological-segmentation tasks.

Implements the N-shot extension of PerSAM-F (Zhang et al., ICLR 2024) for fair
comparison against MetaTune:
  - SAM ViT-B (same backbone as MetaTune).
  - N support images, sampled with a per-task seed.
  - Target embedding is the average of N per-support target features.
  - Mask_Weights (2 learnable scalars) are trained jointly on all N supports.
  - Test Dice is computed at low-res (64x64) using the same dice_score
    function as inference.py, so the numbers are directly comparable to the
    paper's reported MetaTune values.

Usage:
    python baselines/persam_f_samed.py \
        --root_path $DATA_ROOT/blood-cell/train/Images \
        --test_path $DATA_ROOT/blood-cell/test/Images \
        --dataset blood --num_data 4 --seed 42 --gpu_id 0 \
        --output ./output_baselines/persam_f
"""

import os
import sys
import random
import argparse
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
from scipy.ndimage import zoom

warnings.filterwarnings("ignore")

# Vendored PerSAM SAM
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "PerSAM"))
from per_segment_anything import sam_model_registry, SamPredictor

# Re-use SAMed's dice_score for apples-to-apples evaluation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cal_dice import dice_score


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root_path", required=True, help="train Images dir")
    p.add_argument("--test_path", required=True, help="test Images dir")
    p.add_argument("--dataset", required=True)
    p.add_argument("--num_data", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--low_res", type=int, default=64)
    p.add_argument("--sam_type", default="vit_b")
    p.add_argument("--ckpt", default="./checkpoints/sam_vit_b_01ec64.pth")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train_epoch", type=int, default=500)  # ~4 supports x 500 = 2000 steps, similar budget to PerSAM-F default
    p.add_argument("--topk", type=int, default=16,
                   help="Number of high-similarity points sampled from the cosine map as positive prompts. "
                        "Original PerSAM-F uses 1 (single-instance personalization); we use >1 to handle multi-instance "
                        "binary cell segmentation where the GT covers many instances of the same class.")
    p.add_argument("--output", default="./output_baselines/persam_f")
    return p.parse_args()


class Mask_Weights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)


def point_selection(mask_sim, topk=1):
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    return topk_xy.cpu().numpy(), topk_label


def calculate_dice_loss(inputs, targets, num_masks=1):
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    return (1 - (numerator + 1) / (denominator + 1)).sum() / num_masks


def calculate_focal_loss(inputs, targets, num_masks=1, alpha=0.25, gamma=2):
    prob = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_masks


# ---------------------------------------------------------------------------
# Image loading — matches the SAMed dataset_*.py convention.
# ---------------------------------------------------------------------------
def load_image_and_mask(image_path, img_size):
    """Read RGB image + binary mask, resize to img_size x img_size."""
    img = np.array(Image.open(image_path).convert("RGB"))
    mask_path = image_path.replace("/Images", "/Masks")
    mask = np.uint8(np.array(Image.open(mask_path).convert("RGB")) > 0)
    x, y, _ = img.shape
    if x != img_size or y != img_size:
        img = zoom(img, (img_size / x, img_size / y, 1), order=3).clip(0, 255).astype(np.uint8)
        mask = zoom(mask, (img_size / x, img_size / y, 1), order=0)
    return img, mask[:, :, 0]  # mask: (H, W) {0,1}


def extract_target_feat(predictor, image_rgb, mask_uint8):
    """Run SAM image encoder on a reference image, extract avg+max masked features."""
    # set_image with mask arg expects a 3-channel mask (PerSAM's modified set_image returns the resized mask tensor)
    mask_for_set = np.stack([mask_uint8 * 255] * 3, axis=-1).astype(np.uint8)
    ref_mask_t = predictor.set_image(image_rgb, mask_for_set)
    ref_feat = predictor.features.squeeze().permute(1, 2, 0)  # (h, w, C)
    ref_mask_t = F.interpolate(ref_mask_t, size=ref_feat.shape[:2], mode="bilinear").squeeze()[0]
    target_feat = ref_feat[ref_mask_t > 0]
    if target_feat.numel() == 0:
        return None
    target_feat_mean = target_feat.mean(0)
    target_feat_max = target_feat.max(0)[0]
    return (target_feat_mean / 2 + target_feat_max / 2).detach()  # (C,)


def get_point_prior(predictor, target_feat_unit, image_rgb, topk=1):
    """Run SAM encoder on `image_rgb` and pick top-K points via cosine sim to target_feat_unit."""
    predictor.set_image(image_rgb)
    feat = predictor.features.squeeze()
    C, h, w = feat.shape
    feat = feat / feat.norm(dim=0, keepdim=True)
    feat = feat.reshape(C, h * w)
    sim = target_feat_unit @ feat  # (h*w,)
    sim = sim.reshape(1, 1, h, w)
    sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
    sim = predictor.model.postprocess_masks(
        sim, input_size=predictor.input_size, original_size=predictor.original_size
    ).squeeze()
    return point_selection(sim, topk=topk)


# ---------------------------------------------------------------------------
def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Output dir
    exp_name = f"{args.dataset}{args.num_data}_persam_f_img{args.img_size}_seed{args.seed}"
    out_dir = os.path.join(args.output, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    # Load SAM
    print(f"[load] SAM {args.sam_type} from {args.ckpt}")
    sam = sam_model_registry[args.sam_type](checkpoint=args.ckpt).cuda()
    for p in sam.parameters():
        p.requires_grad = False
    predictor = SamPredictor(sam)

    # Sample N support images
    all_train = sorted(os.listdir(args.root_path))
    rng = random.Random(args.seed)
    support_files = rng.sample(all_train, args.num_data)
    print(f"[support] {args.num_data} files: {support_files}")

    supports = []  # list of (image_rgb, mask_binary, target_feat_unit)
    target_feats = []
    for fname in support_files:
        path = os.path.join(args.root_path, fname)
        img, mask = load_image_and_mask(path, args.img_size)
        tf = extract_target_feat(predictor, img, mask)
        if tf is None:
            print(f"[skip] {fname}: empty mask")
            continue
        target_feats.append(tf)
        supports.append((img, mask))

    if len(supports) == 0:
        raise RuntimeError("No usable support images.")

    # Average target embeddings across N supports, then unit-normalize
    target_feat_mean = torch.stack(target_feats, dim=0).mean(0)
    target_feat_unit = (target_feat_mean / target_feat_mean.norm()).unsqueeze(0)  # (1, C)

    # Cache per-support: gt_mask (flat), top-1 point in support's own image
    support_cache = []
    for (img, mask) in supports:
        gt = torch.tensor(mask).float().unsqueeze(0).flatten(1).cuda()  # (1, H*W)
        topk_xy, topk_label = get_point_prior(predictor, target_feat_unit, img, topk=args.topk)
        # Image is already set by get_point_prior; we need a fresh predict using SAM's cached features.
        # Run an initial 3-mask predict to grab logits_high for training:
        masks, scores, logits, logits_high = predictor.predict(
            point_coords=topk_xy, point_labels=topk_label, multimask_output=True
        )
        support_cache.append({
            "gt": gt,
            "topk_xy": topk_xy,
            "topk_label": topk_label,
            "logits_high_init": logits_high.detach(),  # (3, H, W)
        })

    # Train Mask_Weights jointly on all N supports
    print(f"[train] Mask_Weights for {args.train_epoch} epochs over {len(supports)} supports")
    mask_weights = Mask_Weights().cuda()
    mask_weights.train()
    optimizer = torch.optim.AdamW(mask_weights.parameters(), lr=args.lr, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.train_epoch)

    for ep in range(args.train_epoch):
        loss_total = 0.0
        for c in support_cache:
            logits_high = c["logits_high_init"].flatten(1)  # (3, H*W)
            w = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
            lh = (logits_high * w).sum(0).unsqueeze(0)  # (1, H*W)
            loss = calculate_dice_loss(lh, c["gt"]) + calculate_focal_loss(lh, c["gt"])
            loss_total = loss_total + loss
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()
        if ep % 100 == 0:
            print(f"  ep {ep}: loss={loss_total.item():.4f}")

    mask_weights.eval()
    weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0).detach()
    weights_np = weights.cpu().numpy()
    print(f"[train] final mask_weights = {weights_np.flatten()}")

    # ----- Test -----
    print(f"[test] Evaluating on {args.test_path}")
    test_files = sorted(os.listdir(args.test_path))
    dice_list = []
    for tf in test_files:
        path = os.path.join(args.test_path, tf)
        try:
            img, gt = load_image_and_mask(path, args.img_size)
        except Exception as e:
            print(f"[skip] {tf}: {e}")
            continue

        topk_xy, topk_label = get_point_prior(predictor, target_feat_unit, img, topk=args.topk)
        masks, scores, logits, logits_high = predictor.predict(
            point_coords=topk_xy, point_labels=topk_label, multimask_output=True
        )
        # Weighted combination
        logits_high = logits_high * weights.unsqueeze(-1)  # (3, H, W) * (3, 1, 1)
        logit_high = logits_high.sum(0)
        mask = (logit_high > 0).detach().cpu().numpy().astype(np.uint8)

        # Cascaded refinement (1 stage)
        logits_w = logits * weights_np[..., None]
        logit = logits_w.sum(0)
        ys, xs = np.nonzero(mask)
        if len(xs) > 0:
            box = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
            masks2, scores2, logits2, _ = predictor.predict(
                point_coords=topk_xy,
                point_labels=topk_label,
                box=box[None, :],
                mask_input=logit[None, :, :],
                multimask_output=True,
            )
            best = int(np.argmax(scores2))
            mask_final = masks2[best].astype(np.uint8)
        else:
            mask_final = mask  # no foreground

        # Downsample prediction + GT to low_res, evaluate via SAMed's dice_score
        low_res = args.low_res
        mask_lr = zoom(mask_final.astype(np.float32), (low_res / args.img_size, low_res / args.img_size), order=0)
        mask_lr = (mask_lr > 0.5).astype(np.int64)
        gt_lr = zoom(gt.astype(np.float32), (low_res / args.img_size, low_res / args.img_size), order=0)
        gt_lr = (gt_lr > 0.5).astype(np.int64)

        # dice_score expects (B, C, H, W) logits and (B, H, W) target
        # Build 2-class logits from binary mask: channel 0 = 1-m, channel 1 = m
        logit_2c = np.stack([1 - mask_lr, mask_lr], axis=0)[None, ...]  # (1, 2, H, W)
        logit_2c_t = torch.tensor(logit_2c, dtype=torch.float32)
        gt_lr_t = torch.tensor(gt_lr, dtype=torch.long).unsqueeze(0)  # (1, H, W)
        d = dice_score(logit_2c_t, gt_lr_t, bg=False).item()
        dice_list.append(d)

    mean_dice = float(np.mean(dice_list))
    print(f"[result] {args.dataset} seed={args.seed} num_data={args.num_data}: mean Dice = {mean_dice:.4f} over {len(dice_list)} test images")

    # Persist result
    with open(os.path.join(out_dir, "result.txt"), "w") as f:
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"num_data: {args.num_data}\n")
        f.write(f"support_files: {support_files}\n")
        f.write(f"mean_dice: {mean_dice:.4f}\n")
        f.write(f"per_image_dice: {dice_list}\n")
        f.write(f"final_mask_weights: {weights_np.flatten().tolist()}\n")

    # Also rename the output dir to append _NNNN (Dice × 10000), matching the SAMed convention
    suffix = f"_{int(round(mean_dice * 10000)):04d}"
    new_dir = out_dir + suffix
    os.rename(out_dir, new_dir)
    print(f"[result] renamed dir -> {new_dir}")


if __name__ == "__main__":
    main()
