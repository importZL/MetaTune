"""Regenerate per-pixel instance-ID PNG masks for FluoRed (Fluorescent Neuronal
Cells v2) from its trainval / test COCO annotations.

FluoRed's COCO format is non-standard: each "annotation" entry covers a single
image and its `segmentation` field is a LIST of polygons (one polygon per
instance). So one COCO annotation = one image's worth of instances, not one
instance.

Output: DATA_ROOT/{train,test}/Masks_instance/{fname}
"""

import argparse
import json
import os
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

def parse_args():
    parser = argparse.ArgumentParser(description="Convert FluoRed COCO annotations to instance-ID masks.")
    parser.add_argument("--train-json", required=True, help="Training/validation COCO annotation JSON.")
    parser.add_argument("--test-json", required=True, help="Test COCO annotation JSON.")
    parser.add_argument("--data-root", required=True, help="FluoRed root containing train and test directories.")
    return parser.parse_args()

def regen(train_json, test_json, data_root):
    inputs = [(train_json, os.path.join(data_root, "train")), (test_json, os.path.join(data_root, "test"))]
    for json_path, out_root in inputs:
        coco = json.load(open(json_path))
        # id -> (filename, H, W)
        info = {im["id"]: (im["file_name"], im["height"], im["width"]) for im in coco["images"]}
        # gather annotations by image_id
        anns_by_img = {}
        for ann in coco["annotations"]:
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        # only process images whose file is in this split's existing /Images dir (some images may have been re-split)
        existing = set(os.listdir(os.path.join(out_root, "Images"))) if os.path.isdir(os.path.join(out_root, "Images")) else set()
        out_dir = os.path.join(out_root, "Masks_instance")
        os.makedirs(out_dir, exist_ok=True)

        n_written = 0
        for img_id, (fn, H, W) in info.items():
            if existing and fn not in existing:
                continue
            label = np.zeros((H, W), dtype=np.uint16)
            inst_id = 0
            for ann in anns_by_img.get(img_id, []):
                polys = ann.get("segmentation") or []
                for poly in polys:
                    if not poly:
                        continue
                    inst_id += 1
                    if inst_id >= 65535:
                        print(f"warn: instance id overflow in {fn}, capping at 65534")
                        inst_id = 65534
                    # FluoRed polygons are nested [[x,y], [x,y], ...]; pycocotools wants flat [x1,y1,x2,y2,...]
                    flat = [c for pt in poly for c in pt]
                    if len(flat) < 6:
                        continue  # need >= 3 points for a polygon
                    rle = mask_utils.frPyObjects([flat], H, W)
                    m = mask_utils.decode(rle)[..., 0].astype(bool)
                    label[m] = inst_id
            Image.fromarray(label).save(os.path.join(out_dir, fn))
            n_written += 1
        print(f"{out_root}: wrote {n_written} instance masks")

if __name__ == "__main__":
    args = parse_args()
    regen(args.train_json, args.test_json, args.data_root)
