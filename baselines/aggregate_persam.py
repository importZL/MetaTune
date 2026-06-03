"""Aggregate PerSAM-F / Matcher results into a numpy array ready to paste
into figures.ipynb.

Usage:
    python baselines/aggregate_persam.py [BASE_DIR] [METHOD]
    # default BASE_DIR=output_baselines/persam_f, METHOD=persam_f
    # for Matcher:  python baselines/aggregate_persam.py output_baselines/matcher matcher

Expects subdirs named like
  {dataset}{num_data}_{method}_img{size}_seed{seed}_{NNNN}
where _{NNNN} is Dice × 10000.
"""

import os
import re
import sys
from collections import defaultdict
from pprint import pprint

BASE = "output_baselines/persam_f"
METHOD = "persam_f"
TASK_ORDER = [
    ("blood", "BCCD"),
    ("osteosarcoma", "Osteo"),
    ("cellBT474", "BT474"),
    ("cellHuh7", "Huh7"),
    ("multimodal", "MultiModal"),
    ("cyto", "Cyto"),
    ("fluocellRed", "FluoRed"),
    ("sartorius", "Sartorius"),
]
SEED_ORDER = [42, 40, 22]

def main():
    base = sys.argv[1] if len(sys.argv) > 1 else BASE
    method = sys.argv[2] if len(sys.argv) > 2 else METHOD
    # Dir pattern: {dataset}{num_data}_{method}_img{size}_seed{seed}_{NNNN}
    # We match by prefix-stripping known dataset names since some end in digits.
    known_datasets = [ds for ds, _ in TASK_ORDER]
    found = defaultdict(dict)
    pat = re.compile(rf"^(.+?)_{re.escape(method)}_img\d+_seed(\d+)_(\d{{4}})$")
    for d in sorted(os.listdir(base)):
        m = pat.match(d)
        if not m:
            continue
        ds_with_num, seed, suffix = m.groups()
        ds = None
        for known in known_datasets:
            if ds_with_num.startswith(known):
                ds = known
                break
        if ds is None:
            continue
        found[ds][int(seed)] = int(suffix) / 10000.0

    rows = []
    print(f"{'Task':<14} {'Seed 42':<10} {'Seed 40':<10} {'Seed 22':<10} {'mean':<8} {'std':<8}")
    print("-" * 70)
    for ds, label in TASK_ORDER:
        row = [found.get(ds, {}).get(s, None) for s in SEED_ORDER]
        if all(v is not None for v in row):
            mean = sum(row) / 3
            std = (sum((v - mean) ** 2 for v in row) / 3) ** 0.5
            print(f"{label:<14} {row[0]:<10.4f} {row[1]:<10.4f} {row[2]:<10.4f} {mean:<8.4f} {std:<8.4f}")
        else:
            present = [f"{v:.4f}" if v is not None else "  --  " for v in row]
            print(f"{label:<14} {present[0]:<10} {present[1]:<10} {present[2]:<10}  (incomplete)")
        rows.append(row)

    print()
    print(f"# Numpy array for figures.ipynb (rows = tasks in TASK_ORDER above):")
    print(f"data_{method} = np.array([")
    for (ds, label), row in zip(TASK_ORDER, rows):
        if all(v is not None for v in row):
            print(f"    [{row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f}],  # {label}")
        else:
            print(f"    # [..., ..., ...],  # {label}  -- INCOMPLETE")
    print("])")


if __name__ == "__main__":
    main()
