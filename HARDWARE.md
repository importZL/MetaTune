# HARDWARE.md — compute environment used for the paper

All experiments reported in the paper, ablations, baselines, and reviewer-response experiments were run on a single workstation with the following configuration.

## GPU

| Item | Value |
|---|---|
| Vendor / model | NVIDIA A100 80 GB PCIe ×2 |
| Per-GPU memory | 80 GB HBM2e |
| Driver | 560.35.03 |
| CUDA runtime (system toolkit) | 12.6.68 |
| CUDA used by PyTorch builds | 12.1 (`torch==2.5.1+cu121`) |
| CUDA used by TensorFlow (StarDist env) | 12.9 (TF 2.21 default) |

Single-GPU training was used throughout (no DDP / multi-GPU). The shell scripts route to `--gpu_id 0` or `--gpu_id 1` to parallelize *independent* runs across the two GPUs; no run uses more than one GPU.

## CPU / RAM

| Item | Value |
|---|---|
| CPU model | (Intel Xeon, model not load-bearing for results) |
| RAM | ≥ 256 GB (peak resident usage of any single training run is ≤ 30 GB) |
| Disk | NVMe; per-dataset working sets are ≤ 5 GB each |

## Operating system / Python

| Item | Value |
|---|---|
| OS | Ubuntu 22.04 LTS |
| Kernel | 6.8.0-41-generic |
| Shell | bash |
| Conda | 4.5.12 (used to manage the `metatune` and `stardist` envs) |
| Python (main env) | 3.10 (PyTorch 2.5.1+cu121) |
| Python (stardist env) | 3.10 (TensorFlow 2.21) |

## Per-run runtime characterization

These wall-clock values are for *one seed* on *one task*; multiply by 3 for the seed-replicate sweeps reported in the paper.

| Experiment | Compute time | Peak GPU memory |
|---|---|---|
| MetaTune training (semantic, 100 epochs, N=4, img 256) | ~3-5 min | ~14 GB |
| MetaTune training (semantic, 100 epochs, N=10) | ~3-5 min | ~14 GB |
| MetaTune training with `--unrolled` (2nd-order DARTS, 100 epochs, N=4) | ~30-45 min | ~25 GB |
| MetaTune inference (semantic, 159 test imgs) | ~10-20 s | ~3 GB |
| `--swap_meta` ablation training (100 epochs, N=4) | ~3-5 min | ~14 GB |
| PerSAM-F adapter (SAM ViT-B, 1000-epoch mask-weight fit + 200 test imgs) | ~5 s + ~3 min eval | ~3 GB |
| Matcher adapter (DINOv2-L + SAM AMG, `points_per_side=32`, 200 test imgs) | ~30-60 min | ~10 GB |
| Cellpose / Cellpose-SAM fine-tune (cpsam ViT-L, 100 epochs, N=4) | ~3 min | ~10 GB |
| StarDist train + eval (100 epochs × 50 steps, N=4) | ~5 min | ~6 GB |
| BLO-SAM-instance training (300 epochs, N=4) | ~7-10 min | ~15 GB |
| YOLOv7+SAM-bilevel training (20 epochs, N=4) | ~5 min | ~12 GB |
| **cpsam + BLO-SAM-bilevel** (50 epochs alternating, N=4) | ~7 min | ~17 GB (float32) / ~10 GB (bf16) |
| Full instance-seg sweep (4 methods × 2 datasets × 3 seeds) | ~3-6 h total | up to 17 GB |

## Reproducibility expectations

- **Determinism**: random seeds (`{42, 40, 22}`) are set for `random`, `numpy`, and `torch`. CuDNN's nondeterministic ops are *not* disabled, so byte-identical reproduction is not guaranteed; expect Dice/AP within ±0.005 of the values reported in the paper across hardware.
- **Network / data dependencies**: model checkpoints and datasets are fetched once (SAM ViT-B, cpsam, DINOv2 ViT-L); training does not require internet.
- **Mixed precision**: the cpsam + BLO-SAM-bilevel experiments were run in **float32** for reproducibility (we observed bfloat16 introduces a ~0.07 AP regression on FluoRed). All other experiments use the default precision of their underlying frameworks.

## Approximate end-to-end time

Reproducing the *complete* set of experiments reported in the paper (all 8 tasks × all methods × 3 seeds, including ablations and instance-segmentation extensions) requires approximately:

| Step | Wall-clock |
|---|---|
| Setup (env creation, weight downloads, data downloads/preprocessing) | ~4-6 h |
| MetaTune semantic-seg sweeps (Figs 2-4, 7 methods × 8 tasks × 3 seeds) | ~12 h |
| Ablations (Figs 6-10) on 6 tasks × 3 seeds × ~5 conditions | ~6 h |
| OOD experiments (Fig. 5) | ~2 h |
| Swap-meta ablation (Reviewer Comment #1) | ~1 h |
| Instance-segmentation extensions + baselines (Reviewer Comment #6) | ~6-8 h |
| Figure rendering via `figures.ipynb` | ~5 min |
| **Total** | **~32-40 h on 2× A100 80GB** |
