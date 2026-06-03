"""Cellpose-SAM (cpsam) + BLO-SAM-style bilevel meta-finetuning hybrid.

Hypothesis: cpsam already has strong biology-domain pretraining. Adding
bilevel meta-finetuning (in the BLO-SAM spirit) might push it further by
treating the small readout layer as a meta-parameter learned on a held-out
support subset.

N support images are split 50/50 into D1 (non-meta) and D2 (meta).
  - main_opt (most params)   trained on D1 inner loop.
  - meta_opt (readout + diam_labels) trained on D2 outer loop.
Iterations alternate inner -> outer.

Compares against plain Cellpose-SAM (cellpose_samed.py with cpsam) which uses
the same N supports but trains everything jointly without bilevel split.
"""
import os, sys, argparse, random, json
import numpy as np
import torch
from PIL import Image

from cellpose import models, transforms, dynamics, metrics
from cellpose.train import _loss_fn_seg

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_imgs", required=True)
    p.add_argument("--train_masks", required=True)
    p.add_argument("--test_imgs", required=True)
    p.add_argument("--test_masks", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--num_data", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--pretrained_model", default="cpsam")
    p.add_argument("--n_epochs", type=int, default=50)
    p.add_argument("--lr_main", type=float, default=1e-5)
    p.add_argument("--lr_meta", type=float, default=1e-4)
    p.add_argument("--meta_set", default="out_diam",
                   choices=["out_diam", "out_diam_pos", "out_diam_neck", "pos_only", "out_neck_lastblk"],
                   help="Which params are the meta (upper-level) variables.")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--output", default="./output_baselines/cpsam_bilevel")
    return p.parse_args()


def load_pair(img_path, mask_path):
    img = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path)).astype(np.int32)
    return img, mask


def compute_flow_label(masks, device):
    """Use cellpose.dynamics.labels_to_flows to make GT flows for training (3 ch + binary)."""
    flows = dynamics.labels_to_flows([m[None] for m in masks], device=device)
    return flows


def forward_loss(net, imgs, labels, device, bsize=256):
    """Sum loss over imgs. Each img gets a random 256x256 CROP (matches cellpose's
    own train_seg) rather than a resize -- preserves native resolution on large
    images like FluoRed."""
    import torch.nn.functional as F
    total = 0.0
    for im, lab in zip(imgs, labels):
        im_n = transforms.normalize_img(im)
        H, W = im_n.shape[:2]
        # random crop coordinates (or resize if image smaller than bsize)
        if H >= bsize and W >= bsize:
            yy = np.random.randint(0, H - bsize + 1)
            xx = np.random.randint(0, W - bsize + 1)
            im_crop = im_n[yy:yy + bsize, xx:xx + bsize]
            lab_crop = lab[:, yy:yy + bsize, xx:xx + bsize]
        else:
            # Small image: resize to bsize
            im_t = torch.from_numpy(im_n).permute(2, 0, 1).float().unsqueeze(0)
            im_t = F.interpolate(im_t, size=(bsize, bsize), mode="bilinear", align_corners=False)
            im_crop = im_t.squeeze(0).permute(1, 2, 0).numpy()
            lab_t = torch.from_numpy(lab).float().unsqueeze(0)
            lab_t = F.interpolate(lab_t, size=(bsize, bsize), mode="nearest")
            lab_crop = lab_t.squeeze(0).numpy()

        x = torch.from_numpy(im_crop).permute(2, 0, 1).float().unsqueeze(0).to(device)
        net_dtype = next(net.parameters()).dtype
        x = x.to(net_dtype)
        out = net(x)
        y = out[0] if isinstance(out, tuple) else out
        y = y.float()
        lbl = torch.from_numpy(lab_crop).float().unsqueeze(0).to(device)
        total = total + _loss_fn_seg(lbl, y, device)
    return total / max(len(imgs), 1)


def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = f"{args.dataset}{args.num_data}_cpsam_bilevel_seed{args.seed}"
    out_dir = os.path.join(args.output, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    # ---- support set ----
    files = sorted(os.listdir(args.train_imgs))
    rng = random.Random(args.seed); rng.shuffle(files)
    sup = files[:args.num_data]
    print(f"[support] {sup}")

    imgs, masks = [], []
    for fn in sup:
        msk = os.path.join(args.train_masks, fn)
        if not os.path.isfile(msk): continue
        img, m = load_pair(os.path.join(args.train_imgs, fn), msk)
        imgs.append(img); masks.append(m)

    # 50/50 split into D1 (non-meta) and D2 (meta)
    n1 = max(1, len(imgs) // 2)
    d1_imgs, d1_masks = imgs[:n1], masks[:n1]
    d2_imgs, d2_masks = imgs[n1:] or imgs[:1], masks[n1:] or masks[:1]

    # Pre-compute GT flows once
    print(f"[gt-flows] D1={len(d1_imgs)} D2={len(d2_imgs)}")
    d1_lbl = compute_flow_label(d1_masks, device=device)
    d2_lbl = compute_flow_label(d2_masks, device=device)

    # ---- load cpsam ----
    print(f"[load] CellposeModel pretrained={args.pretrained_model}")
    # Float32 (use_bfloat16=False) for stable, reproducible training.
    model = models.CellposeModel(gpu=True, pretrained_model=args.pretrained_model, use_bfloat16=False)
    net = model.net
    # Identify meta params per --meta_set
    def is_meta(name, choice):
        if choice == "out_diam":
            return ("out" in name) or ("diam" in name)
        if choice == "out_diam_pos":
            return ("out" in name) or ("diam" in name) or ("pos_embed" in name)
        if choice == "out_diam_neck":
            return ("out" in name) or ("diam" in name) or ("neck" in name)
        if choice == "pos_only":
            return ("pos_embed" in name)
        if choice == "out_neck_lastblk":
            return ("out" in name) or ("neck" in name) or ("blocks.23" in name)  # last ViT-L block
        return False
    meta_names = [n for n, _ in net.named_parameters() if is_meta(n, args.meta_set)]
    meta_params = [p for n, p in net.named_parameters() if n in meta_names]
    main_params = [p for n, p in net.named_parameters() if n not in meta_names]
    print(f"[params] meta={sum(p.numel() for p in meta_params)/1e3:.1f}K, main={sum(p.numel() for p in main_params)/1e6:.1f}M")

    main_opt = torch.optim.AdamW(main_params, lr=args.lr_main, weight_decay=args.weight_decay)
    meta_opt = torch.optim.AdamW(meta_params, lr=args.lr_meta, weight_decay=args.weight_decay)

    # ---- bilevel training ----
    print(f"[train] {args.n_epochs} epochs, bilevel alternating")
    for ep in range(args.n_epochs):
        # inner: main params on D1
        loss_d1 = forward_loss(net, d1_imgs, d1_lbl, device)
        main_opt.zero_grad()
        loss_d1.backward()
        main_opt.step()

        # outer: meta params on D2
        loss_d2 = forward_loss(net, d2_imgs, d2_lbl, device)
        meta_opt.zero_grad()
        loss_d2.backward()
        meta_opt.step()

        if (ep + 1) % 20 == 0:
            print(f"  ep {ep+1}: D1 loss={loss_d1.item():.4f}  D2 loss={loss_d2.item():.4f}")

    # ---- evaluate ----
    print(f"[test] {args.test_imgs}")
    net.eval()
    test_files = sorted(os.listdir(args.test_imgs))
    masks_true, masks_pred = [], []
    for fn in test_files:
        msk = os.path.join(args.test_masks, fn)
        if not os.path.isfile(msk): continue
        img = np.array(Image.open(os.path.join(args.test_imgs, fn)).convert("RGB"))
        gt = np.array(Image.open(msk)).astype(np.int32)
        pred, _, _ = model.eval(img)
        masks_true.append(gt); masks_pred.append(pred.astype(np.int32))

    ap, tp, fp, fn_ = metrics.average_precision(masks_true, masks_pred, threshold=[0.5, 0.75, 0.9])
    ap = np.nan_to_num(ap, nan=0.0)
    per_img_f1 = 2 * tp[:, 0] / np.maximum(2 * tp[:, 0] + fp[:, 0] + fn_[:, 0], 1)
    mean_ap = ap.mean(axis=0)
    mean_f1 = float(per_img_f1.mean())
    print(f"[result] cpsam+bilevel {args.dataset} seed={args.seed}")
    print(f"  AP@0.5={mean_ap[0]:.4f}  AP@0.75={mean_ap[1]:.4f}  AP@0.9={mean_ap[2]:.4f}  F1@0.5={mean_f1:.4f}")
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({
            "method": "cpsam+bilevel", "dataset": args.dataset, "seed": args.seed,
            "support_files": sup,
            "AP@0.5": float(mean_ap[0]), "AP@0.75": float(mean_ap[1]), "AP@0.9": float(mean_ap[2]),
            "F1@0.5": mean_f1, "n_test": len(masks_true),
        }, f, indent=2)
    suffix = f"_{int(round(mean_ap[0]*10000)):04d}"
    new_dir = out_dir + suffix
    os.rename(out_dir, new_dir)
    print(f"[result] dir -> {new_dir}")


if __name__ == "__main__":
    main()
