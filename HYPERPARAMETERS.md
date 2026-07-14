# HYPERPARAMETERS.md — full hyperparameter tables

All hyperparameters used in the paper's experiments, ablations, and baselines are listed below. Every reported run also ships its own `config.txt` (saved automatically by `train.py` at the start of training) with the complete argparse namespace; those files are bundled with the released checkpoints on Zenodo: [10.5281/zenodo.20517421](https://doi.org/10.5281/zenodo.20517421).

Conventions: all experiments use **3 seeds** `{42, 40, 22}` for replicates. The seed controls (i) random sampling of N support images from the train pool, (ii) network initialization, and (iii) data-loader shuffling.

---

## MetaTune (semantic segmentation) — shared settings

| Hyperparameter | Value | Notes |
|---|---|---|
| Backbone | SAM ViT-B | `sam_vit_b_01ec64.pth` |
| LoRA rank `r` | 4 | inserted into all mask-decoder transformer blocks (`sam_lora_mask_decoder.py`) |
| Image size | 256 | matches MetaTune's prompt-encoder resolution |
| Batch size | 1 | per-sample bilevel updates |
| Epochs | 100 | sufficient for convergence on N ∈ {4, 10} supports |
| Optimizer | AdamW (`betas=(0.9, 0.999)`) | main optimizer (`trainer.py`) |
| Weight decay | 0.1 | both `weight_decay` and `prompt_weight_decay` |
| Warmup | off (`--warmup` disabled) | per `train.sh` |
| LR schedule | polynomial decay `lr0 · (1 - t/T)^0.9` | applied per iteration |
| Dice/CE weighting (`--dice_param`) | 0.8 | loss = 0.2·CE + 0.8·Dice |
| Train/D₂ split | 50/50 of N supports | `--train_split 0.5` |
| `--module` | `sam_lora_mask_decoder` | LoRA on decoder transformer (default) |

## MetaTune per-task LRs

These are the per-task **learning rates** used in the experiments. `base_lr` controls the *lower-level* (non-meta) optimizer; `prompt_base_lr` controls the *upper-level* (meta) optimizer (which trains the `no_mask_embed` prompt embedding).

Note: an earlier version of the paper's Table 4 listed Meta/Non-meta LRs with the row labels transposed; the correct mapping is given here.

| Task | dataset key | N (supports) | `base_lr` (non-meta) | `prompt_base_lr` (meta) |
|---|---|---|---|---|
| Blood (BCCD) | `blood` | 4 | 5e-3 | 1e-3 |
| Osteosarcoma | `osteosarcoma` | 4 | 1e-3 | 1e-3 |
| Breast (BT474) | `cellBT474` | 4 | 5e-3 | 1e-3 |
| Liver (Huh7) | `cellHuh7` | 4 | 5e-3 | 1e-3 |
| Multi-modality | `multimodal` | 4 | 1e-3 | 5e-3 |
| HNSCC (CytoNuke) | `cyto` | 4 | 1e-3 | 5e-3 |
| FluoRed | `fluocellRed` | 10 | 1e-3 | 1e-3 |
| Sartorius | `sartorius` | 10 | 5e-3 | 1e-3 |
| Yeast (bright/contrast) | `yeast-bright` / `yeast-contrast` | 4 (ID), 4 (OOD) | 5e-3 | 5e-3 |

## Reviewer-response ablations

### Comment #1 — swap-meta (Figs 6c proposed)
Same as MetaTune above but pass `--swap_meta` to invert the meta/non-meta assignment: prompt-embedding becomes lower-level (trained on D₁) and LoRA + decoder becomes upper-level (trained on D₂). **LR mirror convention**: keep each parameter set's original LR — for, e.g., BCCD, set `--base_lr 1e-3 --prompt_base_lr 5e-3` (swapped from the default). See `train_swap.sh` and `launch_swap_gpu{0,1}.sh`.

Restriction: `--swap_meta` is only supported with 1st-order (omit `--unrolled`).

### Comment #6 — instance segmentation, Route B (BLO-SAM-instance)
Same base MetaTune setup + Cellpose-style flow head. Pass `--module sam_lora_mask_decoder_instance`.

| Hyperparameter | Value |
|---|---|
| Epochs | 300 |
| Flow loss weight (vs semantic loss) | 20 |
| Cellprob BCE weight | 5 |
| Flow head channels | 128 (hidden) → 64 (last block) |
| Flow head architecture | 4-stage progressive upsample (16 → 32 → 64 → 128 → 256) |
| Output `flow_scale` (init) | [5, 5, 1] for (dy, dx, prob) |
| `compute_masks` thresholds at inference | `cellprob_threshold=-2`, `flow_threshold=0` |

GT flows computed online from instance masks via `cellpose.dynamics.masks_to_flows_gpu` (`baselines/regen_*` produces the instance masks first).

### Comment #6 — instance segmentation, "Route D" (BLO-SAM semantic + distance-transform flow)
No training (uses an existing BLO-SAM-instance checkpoint). At inference:
- Predict semantic mask; threshold at log-odds 0
- Run distance-transform on the binary mask; take gradient as `(dy, dx)` flow
- `cellpose.dynamics.compute_masks(dP, cellprob)` → instance labels
See `inference_route_d.py`.

## Comparison baselines not contained in this repository

The archived files do not record complete run configurations for DeepLab, UNet, vanilla SAM, MedSA, SAMed, uSAM, or HSNet. In particular, learning rate, epoch count, N-shot sampling, backbone, augmentation, prompt construction, checkpoint selection, and upstream commit are not all recoverable. These values must be supplied from the original experiment records before the reported numbers can be independently reproduced; they are intentionally not guessed here.

| Method | Implementation in this repository | Complete reported-run configuration |
|---|---:|---:|
| DeepLab | No | No |
| UNet | No | No |
| vanilla SAM | No | No |
| MedSA | No | No |
| SAMed | Scaffolding lineage only | No |
| uSAM | No | No |
| HSNet | No | No |

---

## Baseline hyperparameters

### PerSAM-F (Zhang et al., ICLR 2024 — `baselines/persam_f_samed.py`)

| Hyperparameter | Value |
|---|---|
| Backbone | SAM ViT-B (same as MetaTune) |
| Trainable params | 2 mask-combination weights (`Mask_Weights`) |
| Train epochs | 1000 |
| LR | 1e-3 |
| `--topk` (positive points per query) | 32 |
| Aggregation (multi-shot) | average target feature across N support images |

### Matcher (Liu et al., ICLR 2024 — `baselines/matcher_samed.py`)

| Hyperparameter | Value |
|---|---|
| Encoder | DINOv2 ViT-L (`dinov2_vitl14_pretrain.pth`) |
| Mask generator | SAM ViT-B + `SamAutomaticMaskGenerator` |
| `points_per_side` (AMG dense grid) | 32 (reduced from default 64 for compute; documented as a deliberate compute trade-off in Methods) |
| `pred_iou_thresh`, `stability_score_thresh` | 0.88 / 0.95 (defaults) |
| Training | none (training-free) |
| Image size | 518 (DINOv2 native) |

### Cellpose-SAM (`cpsam`) and Cellpose v3 (`cyto3`) — `baselines/cellpose_samed.py`

| Hyperparameter | Value |
|---|---|
| `pretrained_model` | `cpsam` (Cellpose-SAM, ViT-L 300M, 1.2 GB checkpoint) or `cyto3` (Cellpose v3 CNN) |
| Train epochs | 100 |
| LR | 1e-5 |
| Weight decay | 0.1 |
| `min_train_masks` | 1 (default 5 is too restrictive for some sparse images) |
| Batch size | 1 |
| `use_bfloat16` | False (cellpose default True, but float32 is more reproducible) |
| Inference | `model.eval(img)` — handles large images via internal tiling |

### StarDist (Schmidt et al., MICCAI 2018 — `baselines/stardist_samed.py`)

| Hyperparameter | Value |
|---|---|
| `n_rays` | 32 |
| Grid | (2, 2) |
| Train patch size | (256, 256) |
| Train epochs × steps/epoch | 100 × 50 |
| Optimizer | Adam (StarDist default) |
| LR | 3e-4 (StarDist default) |
| Thresholds | optimized post-training on a held-out support image via `model.optimize_thresholds` |
| Backbone | Trained from scratch (no biology pretraining) |

### YOLOv7+SAM-bilevel — `launch_yolosam_*.sh` (uses `yolov7-sam/segment/train.py`)

| Hyperparameter | Value |
|---|---|
| Detector backbone | YOLOv7-seg, initialized from cell_count-finetuned checkpoint |
| `--imgsz` | 256 |
| `--batch` | 1 |
| `--epochs` | 20 |
| Hyp config | `data/hyp.scratch.custom.yaml`: `lr0=0.001`, `lr_sam=0.001`, `lrf=0.1`, `momentum=0.937`, `box=0.3`, `cls=0.3`, `obj=0.7`, `iou_t=0.2`, fliplr=0.5 |
| SAM | ViT-B with `--sam_ckpt sam_vit_b_01ec64.pth` |
| `--freeze_yolo`, `--full_ft_sam` | False (bilevel default) |

### cpsam + BLO-SAM-bilevel (ours, best instance-seg) — `baselines/cellpose_cpsam_bilevel.py`

| Hyperparameter | Value |
|---|---|
| Backbone | Cellpose-SAM (`cpsam`, ViT-L 300M) |
| Precision | float32 (`--use_bfloat16=False`) |
| Train epochs | 50 |
| `lr_main` (non-meta, big group) | 1e-5 |
| `lr_meta` (meta, small group) | 1e-3 |
| Meta-parameter set | `out` (final readout conv) + `diam_labels` (≈ 49K params) |
| Non-meta parameter set | encoder + neck + positional embeddings (≈ 305M params) |
| Inner/outer ratio | 1 inner D₁ step + 1 outer D₂ step per epoch |
| Train-time augmentation | random 256×256 crop (not resize) to preserve native resolution on large images |
| Inference | `cellpose.CellposeModel.eval(img)` (internal tiling) |

Hyperparameter selection: chosen via a small grid on CytoNuke seed 42:
- `lr_main ∈ {1e-5, 3e-5, 5e-5, 1e-4}` × `lr_meta ∈ {1e-4, 3e-4, 5e-4, 1e-3}`
- meta-parameter set ∈ {`out_diam`, `out_diam_pos`, `out_diam_neck`, `pos_only`, `out_neck_lastblk`}

---

## Inference-time hyperparameters (instance segmentation)

For methods that produce flow + cellprob (BLO-SAM-instance, cpsam+BLO-SAM-bilevel, Cellpose family):

| Parameter | Value | Used by |
|---|---|---|
| `cellprob_threshold` | -2 (BLO-SAM-instance) / 0 (Cellpose family default) | `cellpose.dynamics.compute_masks` |
| `flow_threshold` | 0 (BLO-SAM-instance, looser due to small flow head) / 0.4 (Cellpose default) | `cellpose.dynamics.compute_masks` |
| `niter` | 200 | `cellpose.dynamics.compute_masks` |
| Evaluation IoU thresholds | {0.5, 0.75, 0.9} | `cellpose.metrics.average_precision` |
| Reported metrics | AP50 (IoU 0.5), AP75 (IoU 0.75), AP90 (IoU 0.9), F1@0.5, mAP = mean(AP50, AP75, AP90) |
