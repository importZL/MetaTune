"""Cellpose / Cellpose-SAM N-shot instance-segmentation runner for the
biological-segmentation tasks.

Fine-tunes a pretrained Cellpose model on N support images per task and
evaluates instance AP/F1 on the test set. Outputs metrics per seed.

Uses Cellpose v4 (cellpose>=4.0). For Cellpose-SAM: --pretrained_model cpsam.
For classical Cellpose v3 cyto3: --pretrained_model cyto3.
"""
import os, sys, argparse, random, json
import numpy as np
from PIL import Image

from cellpose import models, train, metrics, io

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_imgs", required=True, help="train Images dir")
    p.add_argument("--train_masks", required=True, help="train instance-mask dir (Masks_instance)")
    p.add_argument("--test_imgs", required=True)
    p.add_argument("--test_masks", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--num_data", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--pretrained_model", default="cpsam",
                   help="cpsam = Cellpose-SAM (default); cyto3 = classical Cellpose v3")
    p.add_argument("--n_epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--output", default="./output_baselines/cellpose")
    return p.parse_args()

def load_pair(img_path, mask_path):
    img = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path))
    return img, mask.astype(np.int32)

def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    random.seed(args.seed); np.random.seed(args.seed)
    try:
        import torch; torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    tag = f"cellpose-{args.pretrained_model}"
    exp_name = f"{args.dataset}{args.num_data}_{tag}_seed{args.seed}"
    out_dir = os.path.join(args.output, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    # ----- support set: N images sampled with seed -----
    train_files = sorted(os.listdir(args.train_imgs))
    rng = random.Random(args.seed)
    sup_files = rng.sample(train_files, args.num_data)
    print(f"[support] {sup_files}")

    train_data, train_labels = [], []
    for fn in sup_files:
        img_p = os.path.join(args.train_imgs, fn)
        msk_p = os.path.join(args.train_masks, fn)
        if not os.path.isfile(msk_p):
            print(f"[skip support] {fn}: no instance mask"); continue
        img, msk = load_pair(img_p, msk_p)
        train_data.append(img); train_labels.append(msk)

    if len(train_data) == 0:
        raise RuntimeError("No usable support pairs.")

    # ----- model + fine-tune -----
    print(f"[load] CellposeModel pretrained={args.pretrained_model}")
    model = models.CellposeModel(gpu=True, pretrained_model=args.pretrained_model)

    print(f"[train] {args.n_epochs} epochs on {len(train_data)} support images")
    # Note: min_train_masks default is 5; FluoRed images can have as few as ~5 instances which is borderline.
    train.train_seg(
        model.net,
        train_data=train_data,
        train_labels=train_labels,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=1,
        min_train_masks=1,  # Allow images with few instances
        save_path=out_dir,
        model_name=f"finetuned_{exp_name}",
    )

    # ----- evaluate on test set -----
    print(f"[test] evaluating on {args.test_imgs}")
    test_files = sorted(os.listdir(args.test_imgs))
    masks_true, masks_pred = [], []
    for fn in test_files:
        img_p = os.path.join(args.test_imgs, fn)
        msk_p = os.path.join(args.test_masks, fn)
        if not os.path.isfile(msk_p):
            continue
        img = np.array(Image.open(img_p).convert("RGB"))
        gt = np.array(Image.open(msk_p)).astype(np.int32)
        pred, _, _ = model.eval(img)
        masks_true.append(gt)
        masks_pred.append(pred.astype(np.int32))

    # AP at IoU thresholds 0.5, 0.75, 0.9 (Cellpose default)
    ap, tp, fp, fn_ = metrics.average_precision(masks_true, masks_pred,
                                                threshold=[0.5, 0.75, 0.9])
    # AP can be NaN for images with empty GT and empty prediction (0/0); treat as 0
    ap = np.nan_to_num(ap, nan=0.0)
    # F1 at IoU 0.5
    per_img_f1_05 = 2*tp[:,0] / np.maximum(2*tp[:,0] + fp[:,0] + fn_[:,0], 1)

    mean_ap = ap.mean(axis=0)  # [3,] for [0.5, 0.75, 0.9]
    mean_f1_05 = float(per_img_f1_05.mean())
    print(f"[result] {args.dataset} seed={args.seed} num={args.num_data} pretrained={args.pretrained_model}")
    print(f"  AP@0.5={mean_ap[0]:.4f}  AP@0.75={mean_ap[1]:.4f}  AP@0.9={mean_ap[2]:.4f}  F1@0.5={mean_f1_05:.4f}")

    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({
            "dataset": args.dataset, "seed": args.seed, "num_data": args.num_data,
            "pretrained_model": args.pretrained_model,
            "support_files": sup_files,
            "AP@0.5": float(mean_ap[0]), "AP@0.75": float(mean_ap[1]), "AP@0.9": float(mean_ap[2]),
            "F1@0.5": mean_f1_05,
            "n_test": len(masks_true),
        }, f, indent=2)
    # Rename dir with AP@0.5 ×10000 for parity with other baselines' convention
    suffix = f"_{int(round(mean_ap[0]*10000)):04d}"
    new_dir = out_dir + suffix
    os.rename(out_dir, new_dir)
    print(f"[result] dir -> {new_dir}")

if __name__ == "__main__":
    main()
