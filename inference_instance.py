"""Inference for BLO-SAM-instance.

Loads a trained BLO-SAM-instance checkpoint, predicts (dy, dx, prob) per test
image, runs cellpose.dynamics.compute_masks to get instance labels, and reports
AP@[0.5, 0.75, 0.9] + F1@0.5 (using cellpose.metrics.average_precision -- same
metric as the Cellpose / Cellpose-SAM / StarDist baselines).
"""
import os, sys, argparse, json, random
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom
from importlib import import_module

from segment_anything import sam_model_registry
from cellpose import dynamics, metrics as cp_metrics


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_imgs", required=True)
    p.add_argument("--test_masks", required=True, help="instance-ID masks dir")
    p.add_argument("--lora_ckpt", required=True)
    p.add_argument("--module", default="sam_lora_mask_decoder_instance")
    p.add_argument("--dataset", required=True)
    p.add_argument("--num_classes", type=int, default=1)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--vit_name", type=str, default="vit_b")
    p.add_argument("--ckpt", default="/data1/li/Auto_SAMed/checkpoints/sam_vit_b_01ec64.pth")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default=None, help="dir to write result.json; defaults to ckpt's parent")
    return p.parse_args()


def load_image(p, img_size):
    img = np.array(Image.open(p).convert("RGB"))
    H, W = img.shape[:2]
    if (H, W) != (img_size, img_size):
        img_r = zoom(img.astype(np.float32), (img_size / H, img_size / W, 1), order=3).clip(0, 255)
    else:
        img_r = img.astype(np.float32)
    return img_r / 255.0, (H, W)


def main():
    args = get_args()
    args.device = torch.device(f"cuda:{args.gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    sam, img_embedding_size = sam_model_registry[args.vit_name](
        image_size=args.img_size, num_classes=args.num_classes,
        checkpoint=args.ckpt, pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
    )
    low_res = img_embedding_size * 4
    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank)
    net.load_lora_parameters(args.lora_ckpt, args.device)
    net = net.to(args.device).eval()
    multimask = args.num_classes > 1

    test_files = sorted(os.listdir(args.test_imgs))
    masks_true, masks_pred = [], []
    for fn in test_files:
        msk_p = os.path.join(args.test_masks, fn)
        if not os.path.isfile(msk_p):
            continue
        img_p = os.path.join(args.test_imgs, fn)
        img, (Horig, Worig) = load_image(img_p, args.img_size)
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(args.device)
        with torch.no_grad():
            out = net(x, multimask, args.img_size)
        flow_logits = out["flow_logits"][0].cpu().numpy()   # (3, 256, 256)
        dy, dx = flow_logits[0], flow_logits[1]
        # Use SAM's semantic head as cellprob (well-trained via CE+Dice supervision).
        # low_res_logits is (2, 64, 64); upsample to 256x256 to match flow_logits resolution.
        sem_lr = out["low_res_logits"][0]  # tensor (2, 64, 64)
        sem_hr = torch.nn.functional.interpolate(sem_lr.unsqueeze(0),
                                                 size=flow_logits.shape[-2:],
                                                 mode="bilinear", align_corners=False)[0]
        sem_hr = sem_hr.cpu().numpy()
        cellprob = sem_hr[1] - sem_hr[0]  # log-odds at 256x256
        # cellpose flow→masks. Stack as (2, H, W) for compute_masks (dY, dX) plus cellprob.
        # The cellpose API: compute_masks(dP, cellprob, ...) where dP is (2, H, W).
        dP = np.stack([dy, dx], axis=0)
        try:
            # cellprob_threshold=-2 (permissive) and flow_threshold=0 (no flow-consistency filter)
            # work best for our small flow head; see smoke sweep in baselines/iterate notes.
            inst_lr = dynamics.compute_masks(dP, cellprob, niter=200,
                                             cellprob_threshold=-2.0, flow_threshold=0.0)
            if isinstance(inst_lr, tuple):
                inst_lr = inst_lr[0]
        except Exception as e:
            print(f"[warn] compute_masks failed on {fn}: {e}")
            inst_lr = np.zeros_like(cellprob, dtype=np.int32)
        # Upsample instance labels to original size (nearest preserves IDs)
        inst_full = zoom(inst_lr.astype(np.int32),
                         (Horig / inst_lr.shape[0], Worig / inst_lr.shape[1]),
                         order=0).astype(np.int32)
        gt = np.array(Image.open(msk_p)).astype(np.int32)
        masks_true.append(gt)
        masks_pred.append(inst_full)

    ap, tp, fp, fn_ = cp_metrics.average_precision(masks_true, masks_pred, threshold=[0.5, 0.75, 0.9])
    ap = np.nan_to_num(ap, nan=0.0)
    per_img_f1 = 2 * tp[:, 0] / np.maximum(2 * tp[:, 0] + fp[:, 0] + fn_[:, 0], 1)
    mean_ap = ap.mean(axis=0)
    mean_f1 = float(per_img_f1.mean())

    print(f"[result] {args.dataset} seed={args.seed} ckpt={args.lora_ckpt}")
    print(f"  AP@0.5={mean_ap[0]:.4f}  AP@0.75={mean_ap[1]:.4f}  AP@0.9={mean_ap[2]:.4f}  F1@0.5={mean_f1:.4f}")

    out_dir = args.output_dir or os.path.dirname(args.lora_ckpt)
    with open(os.path.join(out_dir, "instance_result.json"), "w") as f:
        json.dump({
            "dataset": args.dataset, "seed": args.seed, "lora_ckpt": args.lora_ckpt,
            "AP@0.5": float(mean_ap[0]), "AP@0.75": float(mean_ap[1]), "AP@0.9": float(mean_ap[2]),
            "F1@0.5": mean_f1, "n_test": len(masks_true),
        }, f, indent=2)


if __name__ == "__main__":
    main()
