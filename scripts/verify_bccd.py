"""Check the BCCD (blood-cell) results against the values reported in the paper.

Stage 1 (default, ~2 min on one GPU): evaluate the released Zenodo checkpoints of the three
MetaTune runs averaged in Figure 2 and of the SAMed baseline (Figure 3) on the 159 test images.
Inference is deterministic, so each score must match the reported value.

Stage 2 (--retrain, ~5 min more): retrain MetaTune from scratch on the fixed split in
splits/bccd/ with the released code and compare the test Dice with the reported value of the
corresponding run.

Usage:
    python scripts/verify_bccd.py --data_root ../datasets --download_checkpoints
    python scripts/verify_bccd.py --data_root ../datasets --checkpoints_dir ./zenodo --retrain

Exits with status 1 if any check fails.
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import urllib.request
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZENODO_URL = "https://zenodo.org/api/records/20517421/files/metatune_v1.zip/content"

# (label, checkpoint folder in the Zenodo archive, reported test Dice)
RELEASED = [
    ("MetaTune run 1 (Fig. 2)", "semantic_main/blood4_auto_first_img256_20240826-055350_8649", 0.8649),
    ("MetaTune run 2 (Fig. 2)", "semantic_main/blood4_auto_first_img256_20241030-202935_8704", 0.8704),
    ("MetaTune run 3 (Fig. 2)", "semantic_main/blood4_auto_first_img256_20241030-203538", 0.8710),
    ("SAMed baseline (Fig. 3)", "ablations/vanilla_joint/blood4_vanilla_img256_20240826-053437_8549", 0.8549),
]
REPORTED_MEAN = 0.87          # Figure 2, MetaTune on BCCD (mean of the three runs)
EVAL_TOL = 0.002              # released checkpoints: only JPEG-decoding differences in 13 test images
RETRAIN = dict(label="MetaTune retrained (run 3)", split_dir="splits/bccd/run_1", seed=10,
               base_lr=5e-3, prompt_lr=5e-3, reported=0.8710)
RETRAIN_TOL = 0.005           # retraining: allows for GPU / library nondeterminism


def download_checkpoints(dest):
    zip_path = os.path.join(dest, "metatune_v1.zip")
    os.makedirs(dest, exist_ok=True)
    if not os.path.exists(zip_path):
        print(f"Downloading {ZENODO_URL} (~716 MB) ...")
        urllib.request.urlretrieve(ZENODO_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest, [n for n in zf.namelist() if "blood4_" in n or n.endswith("README_zenodo.md")])
    return os.path.join(dest, "metatune_v1", "metatune_zenodo_v1")


def run_inference(python, test_images, lora_ckpt, sam_ckpt, gpu):
    out = subprocess.run(
        [python, "-W", "ignore", "inference.py", "--volume_path", test_images, "--lora_ckpt", lora_ckpt,
         "--gpu_id", str(gpu), "--module", "sam_lora_mask_decoder", "--dataset", "blood",
         "--num_classes", "1", "--ckpt", sam_ckpt],
        cwd=REPO, capture_output=True, text=True)
    match = re.search(r"Test dice score: ([0-9.]+)", out.stdout)
    if match is None:
        sys.exit(f"inference.py failed for {lora_ckpt}:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")
    return float(match.group(1))


def retrain(python, train_images, sam_ckpt, gpu, output):
    env = dict(os.environ, PYTHON=python, DATASET="blood", TRAIN_IMAGES=train_images, NUM_DATA="4",
               BASE_LR=str(RETRAIN["base_lr"]), PROMPT_LR=str(RETRAIN["prompt_lr"]), SEED=str(RETRAIN["seed"]),
               SPLIT_DIR=RETRAIN["split_dir"], GPU=str(gpu), SAM_CKPT=sam_ckpt, OUTPUT=output)
    before = set(glob.glob(os.path.join(output, "blood4_auto_first_img256_*")))
    subprocess.run(["bash", "train.sh"], cwd=REPO, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    (run_dir,) = set(glob.glob(os.path.join(output, "blood4_auto_first_img256_*"))) - before
    return os.path.join(run_dir, "best.pth")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", required=True, help="Folder containing blood-cell/ (see prepare_bccd.py)")
    parser.add_argument("--checkpoints_dir", default="./zenodo", help="Where the Zenodo archive is (or will be) unpacked")
    parser.add_argument("--download_checkpoints", action="store_true", help="Download the Zenodo archive first")
    parser.add_argument("--sam_ckpt", default="./checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--retrain", action="store_true", help="Also retrain MetaTune on the fixed split")
    parser.add_argument("--output", default="./output_verify", help="Output folder for --retrain")
    args = parser.parse_args()

    sam_ckpt = os.path.abspath(args.sam_ckpt)
    test_images = os.path.abspath(os.path.join(args.data_root, "blood-cell", "test", "Images"))
    train_images = os.path.abspath(os.path.join(args.data_root, "blood-cell", "train", "Images"))
    if args.download_checkpoints:
        ckpt_root = download_checkpoints(os.path.abspath(args.checkpoints_dir))
    else:
        ckpt_root = os.path.abspath(args.checkpoints_dir)
        nested = os.path.join(ckpt_root, "metatune_v1", "metatune_zenodo_v1")
        ckpt_root = nested if os.path.isdir(nested) else ckpt_root

    rows, failed, metatune_scores = [], False, []
    for label, rel, reported in RELEASED:
        score = run_inference(args.python, test_images, os.path.join(ckpt_root, rel, "best.pth"), sam_ckpt, args.gpu)
        ok = abs(score - reported) <= EVAL_TOL
        failed |= not ok
        rows.append((label, reported, score, EVAL_TOL, ok))
        if label.startswith("MetaTune"):
            metatune_scores.append(score)

    mean = sum(metatune_scores) / len(metatune_scores)
    ok = round(mean, 2) == REPORTED_MEAN
    failed |= not ok
    rows.append(("MetaTune mean of 3 runs (Fig. 2)", REPORTED_MEAN, mean, "rounds to", ok))

    if args.retrain:
        ckpt = retrain(args.python, train_images, sam_ckpt, args.gpu, os.path.abspath(args.output))
        score = run_inference(args.python, test_images, ckpt, sam_ckpt, args.gpu)
        ok = abs(score - RETRAIN["reported"]) <= RETRAIN_TOL
        failed |= not ok
        rows.append((RETRAIN["label"], RETRAIN["reported"], score, RETRAIN_TOL, ok))

    print(f"\n{'Check':<36}{'Reported':>10}{'Obtained':>10}{'Tolerance':>11}  Result")
    for label, reported, score, tol, ok in rows:
        tol_s = f"±{tol}" if isinstance(tol, float) else tol
        print(f"{label:<36}{reported:>10.4f}{score:>10.4f}{tol_s:>11}  {'PASS' if ok else 'FAIL'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
