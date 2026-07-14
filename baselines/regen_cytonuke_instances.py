"""Regenerate per-pixel instance-ID PNG masks for CytoNuke from coco.json.

For each image in coco.json:
  - Iterate per-instance annotations.
  - Rasterize each annotation's polygon(s) to a binary mask.
  - Paint into a uint16 label image with a unique integer ID per instance.

Output: writes Masks_instance/{filename}.png next to the existing /Images
directories (one for train, one for test), with pixel value = instance ID
(0 = background, 1..N = instance IDs).

The existing binary /Masks dirs are not touched.
"""

import argparse
import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

def parse_args():
    parser = argparse.ArgumentParser(description="Convert CytoNuke COCO annotations to instance-ID masks.")
    parser.add_argument("--coco-json", required=True, help="Path to the CytoNuke COCO annotation JSON.")
    parser.add_argument("--data-root", required=True, help="Dataset root containing train/Images and test/Images.")
    return parser.parse_args()

def regen(coco_json, data_root):
    coco = COCO(coco_json)
    # Build filename -> split lookup
    splits = {}
    for split in ("train", "test"):
        d = os.path.join(data_root, split, "Images")
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            splits[f] = split
        os.makedirs(os.path.join(data_root, split, "Masks_instance"), exist_ok=True)

    for img_id in coco.getImgIds():
        info = coco.loadImgs([img_id])[0]
        fn = info["file_name"]
        H, W = info["height"], info["width"]
        split = splits.get(fn)
        if split is None:
            continue
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        label = np.zeros((H, W), dtype=np.uint16)
        for i, ann in enumerate(anns, start=1):
            m = coco.annToMask(ann).astype(bool)
            # Later instances overwrite earlier ones in overlapping regions (typical convention).
            label[m] = i
        out = os.path.join(data_root, split, "Masks_instance", fn)
        Image.fromarray(label).save(out)
    print(f"Done. Train+test images processed: {len([f for f in splits])}")

if __name__ == "__main__":
    args = parse_args()
    regen(args.coco_json, args.data_root)
