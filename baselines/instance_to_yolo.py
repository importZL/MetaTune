"""Convert instance-mask PNGs to YOLO polygon-segmentation labels.

For each image, reads /Masks_instance/{filename} (uint16, pixel=instance ID),
extracts a polygon contour per instance via cv2.findContours, and writes a YOLO
polygon label file (class_id x1 y1 x2 y2 ... normalized to [0,1]) alongside the
images.

Also writes train.txt / val.txt with absolute image paths.
"""
import os, sys, argparse
import numpy as np
import cv2
from PIL import Image


def mask_to_polygons(inst_mask: np.ndarray):
    """Yield (class_id=0, polygon_pts_normalized) for each instance in a HxW uint16 mask."""
    H, W = inst_mask.shape
    polys = []
    for inst_id in np.unique(inst_mask):
        if inst_id == 0:
            continue
        binary = (inst_mask == inst_id).astype(np.uint8) * 255
        # Use RETR_EXTERNAL to get one outer contour per instance
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        # Take largest contour if there are multiple disconnected pieces
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 3:
            continue
        # Simplify slightly to keep file sizes reasonable
        epsilon = 0.0015 * cv2.arcLength(c, True)
        c_approx = cv2.approxPolyDP(c, epsilon, True)
        if len(c_approx) < 3:
            continue
        # Normalize to [0,1]
        pts = c_approx.reshape(-1, 2).astype(np.float32)
        pts[:, 0] /= W
        pts[:, 1] /= H
        polys.append(pts)
    return polys


def convert_dataset(images_dir, masks_dir, labels_dir, listfile=None, name=""):
    os.makedirs(labels_dir, exist_ok=True)
    image_paths = []
    n_polys = 0
    files = sorted(os.listdir(images_dir))
    for fn in files:
        msk_p = os.path.join(masks_dir, fn)
        if not os.path.isfile(msk_p):
            # Try matching base name without extension
            base, _ = os.path.splitext(fn)
            cand = [m for m in os.listdir(masks_dir) if os.path.splitext(m)[0] == base]
            if not cand:
                continue
            msk_p = os.path.join(masks_dir, cand[0])
        inst = np.array(Image.open(msk_p)).astype(np.int32)
        polys = mask_to_polygons(inst)
        n_polys += len(polys)
        label_path = os.path.join(labels_dir, os.path.splitext(fn)[0] + ".txt")
        with open(label_path, "w") as f:
            for pts in polys:
                line = "0 " + " ".join(f"{v:.6f}" for v in pts.flatten())
                f.write(line + "\n")
        image_paths.append(os.path.join(images_dir, fn))
    if listfile:
        with open(listfile, "w") as f:
            for p in image_paths:
                f.write(p + "\n")
    print(f"[{name}] wrote {len(image_paths)} label files; total {n_polys} polygons.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", required=True, help="output root dir (will contain images/, labels/, train.txt, val.txt)")
    p.add_argument("--train_imgs", required=True)
    p.add_argument("--train_masks", required=True)
    p.add_argument("--test_imgs", required=True)
    p.add_argument("--test_masks", required=True)
    args = p.parse_args()

    # YOLO expects labels/train/*.txt next to images/train/*.png (or simpler flat layout).
    train_lbl_dir = os.path.join(args.out_root, "labels", "train")
    val_lbl_dir   = os.path.join(args.out_root, "labels", "val")
    convert_dataset(args.train_imgs, args.train_masks, train_lbl_dir,
                    listfile=os.path.join(args.out_root, "train.txt"), name="train")
    convert_dataset(args.test_imgs, args.test_masks, val_lbl_dir,
                    listfile=os.path.join(args.out_root, "val.txt"), name="val")


if __name__ == "__main__":
    main()
