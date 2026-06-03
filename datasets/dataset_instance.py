"""Instance-segmentation dataset for BLO-SAM-instance training.

Returns per-sample:
  - image           (3, H, W) float [0,1]
  - label           (H, W) int64 -- binary (0/1) at img_size for the semantic head
  - low_res_label   (H_low, W_low) int64 -- binary downsampled to low_res
  - flow_gt         (3, H_low, W_low) float -- (dy, dx, prob) at low_res for flow head
  - path / case_name

Reads instance masks (integer IDs per pixel) from /Masks_instance/.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from scipy.ndimage import zoom
import random

# Cellpose flow generation (v4: masks_to_flows_gpu accepts torch.device)
import torch as _torch
from cellpose import dynamics

_FLOW_DEVICE = _torch.device("cpu")  # flow gen is fast on CPU and avoids GPU contention


def _renumber_instances(m: np.ndarray) -> np.ndarray:
    """Renumber instance IDs to be 0, 1, 2, ..., N-1 (no gaps).
    cellpose's masks_to_flows_gpu requires contiguous IDs."""
    out = np.zeros_like(m, dtype=np.int32)
    new_id = 0
    for old in np.unique(m):
        if old == 0:
            continue
        new_id += 1
        out[m == old] = new_id
    return out


def instance_to_flow(inst_mask: np.ndarray):
    """Convert instance mask (HxW int) to (dy, dx, prob_binary) at the same H, W."""
    if inst_mask.max() == 0:
        return np.zeros((3, inst_mask.shape[0], inst_mask.shape[1]), dtype=np.float32)
    inst_mask = _renumber_instances(inst_mask)
    if inst_mask.max() == 0:
        return np.zeros((3, inst_mask.shape[0], inst_mask.shape[1]), dtype=np.float32)
    try:
        flow = dynamics.masks_to_flows_gpu(inst_mask, device=_FLOW_DEVICE)
        # Returns (2, H, W) numpy array
        if isinstance(flow, tuple):
            flow = flow[0]
    except Exception:
        flow = np.zeros((2, inst_mask.shape[0], inst_mask.shape[1]), dtype=np.float32)
    if flow.shape[0] != 2:
        flow = np.transpose(flow, (2, 0, 1))
    prob = (inst_mask > 0).astype(np.float32)
    return np.concatenate([flow.astype(np.float32), prob[None]], axis=0)


class RandomGenerator(object):
    """Resize image + instance mask to output_size; compute flow at low_res."""
    def __init__(self, output_size, low_res, split=None):
        self.output_size = output_size  # (img_size, img_size)
        self.low_res = low_res          # (low_res, low_res)
        self.split = split

    def __call__(self, sample):
        image = sample["image"]            # (H, W, 3) float [0,1]
        inst  = sample["instance_label"]   # (H, W) int instance IDs
        image_path = sample["path"]

        H, W, _ = image.shape
        oh, ow = self.output_size
        if H != oh or W != ow:
            image = zoom(image, (oh / H, ow / W, 1), order=3)
            inst  = zoom(inst.astype(np.int32), (oh / H, ow / W), order=0).astype(np.int32)

        lh, lw = self.low_res
        # binary semantic labels
        label    = (inst > 0).astype(np.int64)
        low_res_label_arr = zoom((inst > 0).astype(np.float32), (lh / oh, lw / ow), order=0)
        low_res_label = (low_res_label_arr > 0.5).astype(np.int64)

        # Compute GT flow at img_size (oh x ow) from the FULL-res instance mask.
        # The flow head outputs at img_size (256x256), so GT must match.
        flow_gt = instance_to_flow(inst.astype(np.int32))  # (3, oh, ow)

        image_t = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        return {
            "image":          image_t,
            "label":          torch.from_numpy(label).long(),
            "low_res_label":  torch.from_numpy(low_res_label).long(),
            "flow_gt":        torch.from_numpy(flow_gt).float(),
            "path":           image_path,
        }


class Synapse_dataset(Dataset):
    """Drop-in replacement for SAMed dataset_* classes, with instance flows.

    The mask root is auto-derived: replaces "/Images" with "/Masks_instance".
    """
    def __init__(self, train_dir, num_data=0, transform=None, dataset=None, seed=None):
        self.transform = transform
        self.train_file = train_dir
        self.num_data = num_data
        self.dataset = dataset
        files = sorted(os.listdir(train_dir))
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(files)
        if self.num_data > 0:
            self.sample_list = files[: self.num_data]
        else:
            self.sample_list = files

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        fn = self.sample_list[idx]
        image_path = os.path.join(self.train_file, fn)
        image = Image.open(image_path).convert("RGB")
        inst_path = image_path.replace("/Images", "/Masks_instance")
        inst = Image.open(inst_path)
        image = np.array(image).astype(np.float32) / 255.0  # (H, W, 3)
        inst  = np.array(inst).astype(np.int32)             # (H, W) instance IDs
        sample = {"image": image, "instance_label": inst, "path": image_path}
        if self.transform:
            sample = self.transform(sample)
        sample["case_name"] = fn
        return sample
