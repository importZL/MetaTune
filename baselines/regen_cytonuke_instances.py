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

import os, sys
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

COCO_JSON = "/data2/li/workspace/data/CytoNuke/coco.json"
IMG_ROOT  = "/data2/li/workspace/data/CytoNuke"  # has train/Images, test/Images

def regen():
    coco = COCO(COCO_JSON)
    # Build filename -> split lookup
    splits = {}
    for split in ("train", "test"):
        d = os.path.join(IMG_ROOT, split, "Images")
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            splits[f] = split
        os.makedirs(os.path.join(IMG_ROOT, split, "Masks_instance"), exist_ok=True)

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
        out = os.path.join(IMG_ROOT, split, "Masks_instance", fn)
        Image.fromarray(label).save(out)
    print(f"Done. Train+test images processed: {len([f for f in splits])}")

if __name__ == "__main__":
    regen()
