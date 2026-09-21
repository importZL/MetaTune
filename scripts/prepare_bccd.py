"""Download and organize the BCCD blood-cell dataset into the layout MetaTune expects.

    <data_root>/blood-cell/{train,test}/{Images,Masks}/<uuid>.png

Source: Kaggle dataset jeetblahiri/bccd-dataset-with-mask (see DATA.md). The archive stores
most images as PNG and 119 as JPEG (106 train, 13 test); these are converted to PNG with
Pillow, as in the original experiments. PNG files are copied unchanged.

Usage:
    python scripts/prepare_bccd.py --data_root ../datasets                 # download via the Kaggle API
    python scripts/prepare_bccd.py --data_root ../datasets --zip bccd.zip  # use an existing download
"""
import argparse
import os
import shutil
import subprocess
import zipfile

from PIL import Image

KAGGLE_ID = "jeetblahiri/bccd-dataset-with-mask"
ARCHIVE_ROOT = "BCCD Dataset with mask"
EXPECTED = {"train": 1169, "test": 159}
SPLITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "splits", "bccd")


def download(dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "bccd-dataset-with-mask.zip")
    if not os.path.exists(zip_path):
        # Requires `pip install kaggle` and an API token in ~/.kaggle/kaggle.json.
        subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_ID, "-p", dest_dir], check=True)
    return zip_path


def organize(zip_path, out_dir):
    extract_dir = os.path.join(os.path.dirname(zip_path), "bccd_raw")
    if not os.path.isdir(os.path.join(extract_dir, ARCHIVE_ROOT)):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    src_root = os.path.join(extract_dir, ARCHIVE_ROOT)

    for split in ("train", "test"):
        for src_sub, dst_sub in (("original", "Images"), ("mask", "Masks")):
            src, dst = os.path.join(src_root, split, src_sub), os.path.join(out_dir, split, dst_sub)
            os.makedirs(dst, exist_ok=True)
            for name in os.listdir(src):
                stem, ext = os.path.splitext(name)
                target = os.path.join(dst, stem + ".png")
                if os.path.exists(target):
                    continue
                if ext.lower() == ".png":
                    shutil.copy(os.path.join(src, name), target)
                else:
                    Image.open(os.path.join(src, name)).save(target)


def check(out_dir):
    for split, n in EXPECTED.items():
        images = set(os.listdir(os.path.join(out_dir, split, "Images")))
        masks = set(os.listdir(os.path.join(out_dir, split, "Masks")))
        assert len(images) == n, f"{split}/Images: expected {n} files, found {len(images)}"
        assert images == masks, f"{split}: image and mask filenames differ"
    with open(os.path.join(SPLITS_DIR, "test.txt")) as f:
        test = {line.strip() for line in f if line.strip()}
    assert test == set(os.listdir(os.path.join(out_dir, "test", "Images"))), "test set differs from splits/bccd/test.txt"
    for seed_dir in sorted(d for d in os.listdir(SPLITS_DIR) if d.startswith("seed_")):
        with open(os.path.join(SPLITS_DIR, seed_dir, "train.txt")) as f:
            for name in (line.strip() for line in f if line.strip()):
                assert os.path.exists(os.path.join(out_dir, "train", "Images", name)), f"missing {name} ({seed_dir})"
    print(f"OK: {out_dir} has {EXPECTED['train']} train / {EXPECTED['test']} test images "
          f"and contains every image listed in splits/bccd/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", required=True, help="Folder that will contain blood-cell/")
    parser.add_argument("--zip", default=None, help="Path to an already-downloaded Kaggle zip")
    args = parser.parse_args()

    zip_path = args.zip or download(os.path.join(args.data_root, "_downloads"))
    out_dir = os.path.join(args.data_root, "blood-cell")
    organize(zip_path, out_dir)
    check(out_dir)
