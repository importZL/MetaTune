"""StarDist (2D) N-shot instance-segmentation runner.

Trains StarDist on N support images per task and evaluates instance AP/F1 on
the test set. We reuse cellpose.metrics.average_precision so the metric is
identical to the Cellpose runner (apples-to-apples).
"""
import os, sys, argparse, random, json
import numpy as np
from PIL import Image

# silence TF noise before imports
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from stardist.models import Config2D, StarDist2D
from stardist import gputools_available, fill_label_holes
from csbdeep.utils import normalize
from cellpose import metrics as cp_metrics  # reuse the same metric

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
    p.add_argument("--n_rays", type=int, default=32)
    p.add_argument("--n_epochs", type=int, default=100)
    p.add_argument("--steps_per_epoch", type=int, default=50)
    p.add_argument("--output", default="./output_baselines/stardist")
    return p.parse_args()

def load_img(p, gray=True):
    im = Image.open(p).convert("L" if gray else "RGB")
    return np.array(im).astype(np.float32)

def load_label(p):
    return np.array(Image.open(p)).astype(np.int32)

def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    random.seed(args.seed); np.random.seed(args.seed)

    exp_name = f"{args.dataset}{args.num_data}_stardist_seed{args.seed}"
    out_dir = os.path.join(args.output, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    # --- support set ---
    train_files = sorted(os.listdir(args.train_imgs))
    rng = random.Random(args.seed)
    sup_files = rng.sample(train_files, args.num_data)
    print(f"[support] {sup_files}")

    X_train, Y_train = [], []
    for fn in sup_files:
        msk_p = os.path.join(args.train_masks, fn)
        if not os.path.isfile(msk_p): continue
        img = load_img(os.path.join(args.train_imgs, fn))
        lab = load_label(msk_p)
        img = normalize(img, 1, 99.8)
        lab = fill_label_holes(lab)
        X_train.append(img); Y_train.append(lab)

    # Use a held-out support image as validation if N>=2; else duplicate
    if len(X_train) >= 2:
        X_val, Y_val = [X_train[-1]], [Y_train[-1]]
        X_train, Y_train = X_train[:-1], Y_train[:-1]
    else:
        X_val, Y_val = X_train, Y_train

    # --- model ---
    config = Config2D(
        n_rays=args.n_rays,
        grid=(2, 2),
        n_channel_in=1,
        use_gpu=gputools_available(),
        train_patch_size=(256, 256),
        train_batch_size=1,
        train_epochs=args.n_epochs,
        train_steps_per_epoch=args.steps_per_epoch,
    )
    print(f"[load] StarDist2D fresh model (n_rays={args.n_rays})")
    model = StarDist2D(config, name=exp_name, basedir=out_dir)

    print(f"[train] {args.n_epochs} epochs ({args.steps_per_epoch} steps/epoch) on {len(X_train)} support imgs")
    model.train(X_train, Y_train, validation_data=(X_val, Y_val))
    # Optimize NMS thresholds on validation
    try:
        model.optimize_thresholds(X_val, Y_val)
    except Exception as e:
        print(f"[warn] optimize_thresholds failed: {e}")

    # --- test ---
    print(f"[test] evaluating on {args.test_imgs}")
    test_files = sorted(os.listdir(args.test_imgs))
    masks_true, masks_pred = [], []
    for fn in test_files:
        msk_p = os.path.join(args.test_masks, fn)
        if not os.path.isfile(msk_p): continue
        img = load_img(os.path.join(args.test_imgs, fn))
        gt  = load_label(msk_p)
        img_n = normalize(img, 1, 99.8)
        labels, _ = model.predict_instances(img_n)
        masks_true.append(gt); masks_pred.append(labels.astype(np.int32))

    ap, tp, fp, fn_ = cp_metrics.average_precision(masks_true, masks_pred, threshold=[0.5, 0.75, 0.9])
    ap = np.nan_to_num(ap, nan=0.0)
    per_img_f1_05 = 2*tp[:,0] / np.maximum(2*tp[:,0] + fp[:,0] + fn_[:,0], 1)
    mean_ap = ap.mean(axis=0)
    mean_f1_05 = float(per_img_f1_05.mean())

    print(f"[result] {args.dataset} seed={args.seed} num={args.num_data} stardist")
    print(f"  AP@0.5={mean_ap[0]:.4f}  AP@0.75={mean_ap[1]:.4f}  AP@0.9={mean_ap[2]:.4f}  F1@0.5={mean_f1_05:.4f}")

    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({
            "dataset": args.dataset, "seed": args.seed, "num_data": args.num_data,
            "method": "stardist", "support_files": sup_files,
            "AP@0.5": float(mean_ap[0]), "AP@0.75": float(mean_ap[1]), "AP@0.9": float(mean_ap[2]),
            "F1@0.5": mean_f1_05, "n_test": len(masks_true),
        }, f, indent=2)
    suffix = f"_{int(round(mean_ap[0]*10000)):04d}"
    new_dir = out_dir + suffix
    os.rename(out_dir, new_dir)
    print(f"[result] dir -> {new_dir}")

if __name__ == "__main__":
    main()
