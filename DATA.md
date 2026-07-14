# DATA.md — datasets used in the paper

This document gives the source URL, license, sample counts, train/test split, and preprocessing for every dataset referenced in the paper. All datasets are publicly available; we do **not** redistribute them in this repository. After download, organize each dataset as:

```
<DATA_ROOT>/<dataset_name>/
├── train/
│   ├── Images/         <image files>.{png,jpg}
│   ├── Masks/          binary semantic masks (same filename, .png)
│   └── Masks_instance/ instance-ID masks for instance-seg only (uint16 .png; see baselines/regen_*)
└── test/
    ├── Images/
    ├── Masks/
    └── Masks_instance/
```

The training scripts (`train.sh`, `train_instance.sh`, etc.) point at `<DATA_ROOT>/<dataset_name>/train/Images`. See `datasets/dataset_*.py` for the matching loader.

---

## 1. BCCD — Blood Cell Count and Detection (semantic + instance)
- **Source**: [Roboflow BCCD dataset](https://public.roboflow.com/object-detection/bccd) (originally derived from the BCCD Dataset)
- **License**: MIT
- **Task in paper**: Blood-cell semantic segmentation (Fig. 2 / 3 / 4); instance segmentation extension (Comment #6)
- **Sample counts (paper Table 2)**: 4 train (sampled), 159 test
- **Cell type**: red blood cells, white blood cells, platelets (all labeled as foreground)
- **Image dimensions**: 1200×1600 (RGB)
- **Loader**: `datasets/dataset_blood.py`
- **Instance masks**: derived via connected components on binary mask (cells are non-overlapping).

## 2. Osteosarcoma — BBBC039v1 / CellPose-curated nuclei
- **Source**: [BBBC039v1](https://bbbc.broadinstitute.org/BBBC039) (Broad Bioimage Benchmark Collection); we use the variant re-distributed with CellPose
- **License**: CC0
- **Task in paper**: Osteosarcoma nuclei semantic segmentation; instance ablation tasks (Figs 6-10)
- **Sample counts**: 4 train (sampled), 89 test
- **Cell type**: U2OS osteosarcoma cells stained with Hoechst (BBBC039v1)
- **Image dimensions**: 383×512 (grayscale → 3-channel)
- **Loader**: `datasets/dataset_osteosarcoma.py`

## 3. BT474 (breast cancer) — LiveCell benchmark
- **Source**: [LiveCell](https://sartorius-research.github.io/LIVECell/) (Edlund et al., *Nat Methods* 2021)
- **License**: CC BY-NC-SA 4.0
- **Task in paper**: Breast-cancer cell segmentation; instance ablation tasks
- **Sample counts**: 4 train (sampled), 168 test
- **Cell type**: BT474 HER2-positive breast-cancer line; phase-contrast microscopy
- **Image dimensions**: 520×704
- **Loader**: `datasets/dataset_cellBT474.py`
- **Note**: original LiveCell polygon annotations carry instance information; we initially store as binary semantic masks under `train/Masks/`. For instance segmentation, see `baselines/regen_cytonuke_instances.py` for the COCO-style approach (we apply the analogous logic to LiveCell).

## 4. Huh7 (liver cancer) — LiveCell benchmark
- **Source**: [LiveCell](https://sartorius-research.github.io/LIVECell/)
- **License**: CC BY-NC-SA 4.0
- **Task in paper**: Liver-cancer cell segmentation
- **Sample counts**: 4 train (sampled), 200 test
- **Cell type**: Huh7 hepatocellular carcinoma; phase-contrast
- **Image dimensions**: 520×704
- **Loader**: `datasets/dataset_cellBT474.py` (shared)

## 5. Multimodal — Multi-modality Cell Segmentation Challenge
- **Source**: [Multi-modality Cell Segmentation Challenge, NeurIPS 2022](https://neurips22-cellseg.grand-challenge.org/)
- **License**: see challenge terms
- **Task in paper**: Multi-modality cell segmentation
- **Sample counts**: 4 train (sampled), 50 test
- **Modalities**: fluorescence, phase-contrast, and tissue sections
- **Image dimensions**: 480×640
- **Loader**: `datasets/dataset_osteosarcoma.py` (handles `multimodal` dataset key)

## 6. CytoNuke / HNSCC — CytoNuke for HNSCC histology
- **Source**: [CytoNuke](https://github.com/dasilvalab/CytoNuke) (Da Silva et al., 2023, derived from CPTAC)
- **License**: as posted on the CytoNuke repository
- **Task in paper**: Head and neck squamous cell carcinoma (HNSCC) cell segmentation; main instance-seg evaluation dataset (Comment #6)
- **Sample counts**: 4 train (sampled), 40 test
- **Image dimensions**: 256×256 (already cropped patches from larger WSIs)
- **Annotation format**: COCO polygons in `coco.json` (provided with the dataset); we convert to per-pixel instance masks via `baselines/regen_cytonuke_instances.py`
- **Loader**: `datasets/dataset_osteosarcoma.py` (handles `cyto` dataset key)

## 7. FluoRed — Fluorescent Neuronal Cells v2 (red channel)
- **Source**: [Fluorescent Neuronal Cells v2](https://amsacta.unibo.it/id/eprint/7347/) (Clissa et al., 2022)
- **License**: CC BY 4.0
- **Task in paper**: Neuronal-cell fluorescence segmentation (red channel)
- **Sample counts**: 10 train (sampled), 46 test
- **Image dimensions**: 1200×1600 / 1704×2272 (varies)
- **Annotation format**: COCO polygons in `trainval_ori/ground_truths/COCO/annotations_red_trainval.json`; we convert via `baselines/regen_fluored_instances.py`
- **Loader**: `datasets/dataset_cellBT474.py` (handles `fluocellRed` dataset key)

## 8. Sartorius — Kaggle Neuronal Cell Instance Segmentation
- **Source**: [Sartorius Cell Instance Segmentation (Kaggle)](https://www.kaggle.com/c/sartorius-cell-instance-segmentation)
- **License**: as posted by the competition (research use)
- **Task in paper**: Neuronal-cell instance segmentation (Sartorius SH-SY5Y line)
- **Sample counts**: 10 train (sampled), 500 test (we use a held-out split from the public training set, since competition test labels are withheld)
- **Image dimensions**: 520×704 (phase contrast)
- **Annotation format**: per-instance RLE in `train.csv` (multiple rows per image); the `sartorius/process.py` script in the dataset distribution decodes these to binary masks. For per-instance masks, augment that script to assign unique integer IDs (see `baselines/regen_fluored_instances.py` pattern for COCO; an analogous RLE decoder is straightforward).
- **Loader**: `datasets/dataset_cellBT474.py` (handles `sartorius` dataset key)

---

## Yeast (for OOD experiments, Fig. 5)

For the in-distribution / out-of-distribution evaluation in the paper's Fig. 5, we use a privately-collected yeast cell segmentation dataset (Hao Lab, UCSD). It consists of:

- **Bright-field modality**: 34 individual time-lapse experiments (xy01 through xy34), each containing several phase-contrast images of *Saccharomyces cerevisiae*.
- **Phase-contrast modality**: same 34 individuals, different optical channel.

The ID/OOD split is:
- **ID**: train on 4 images from individual `xy01`, test on the remaining held-out images from `xy01` (261 images).
- **OOD**: train on 4 images from `xy01`, test on individuals `xy02-xy34` (11,913 images).

Data is available upon request from `liz113[at]ucsd.edu`.

- **Loader**: `datasets/dataset_cellBT474.py` (handles `yeast-bright`, `yeast-contrast`)

---

## Preprocessing pipeline (summary)

1. **Original masks downloaded** from each source (varied formats: COCO JSON, RLE CSV, PNG masks, polygon TXT).
2. **Conversion to binary semantic masks** (`/Masks/<filename>.png`, foreground=255, background=0). See per-dataset `process.py` / `transfer.py` / `generate_mask.py` shipped with each original dataset, or recreate via simple `pycocotools.coco.COCO().annToMask()` calls.
3. **Conversion to instance-ID masks** (`/Masks_instance/<filename>.png`, uint16, pixel value = instance ID 1..N). Scripts in `baselines/regen_cytonuke_instances.py` and `baselines/regen_fluored_instances.py` handle CytoNuke and FluoRed respectively. For datasets whose cells don't touch (BCCD, Osteo), connected components on the binary mask is used.
4. **No data augmentation** beyond what `datasets/dataset_*.py` and the cellpose / yolov7-sam trainers do internally (random crops, flips for cellpose; YOLO-default hsv/scale/flip).

---

## A note on the YOLOv7+SAM-bilevel comparison

The YOLOv7+SAM-bilevel results (Section "Instance segmentation extension") require the data to be in YOLO-polygon-label format. We provide a converter:

```bash
python baselines/instance_to_yolo.py \
    --out_root "$DATA_ROOT/yolo_CytoNuke" \
    --train_imgs /data_root/CytoNuke/train/Images \
    --train_masks /data_root/CytoNuke/train/Masks_instance \
    --test_imgs   /data_root/CytoNuke/test/Images \
    --test_masks  /data_root/CytoNuke/test/Masks_instance
```

This builds:
```
yolo_<dataset>/
├── images/{train,val}/   symlinks to original .png files
├── labels/{train,val}/   YOLO polygon-format .txt files
├── train.txt             absolute image paths
└── val.txt
```

Use `baselines/sample_yolo_nshot.py` to sample N-shot YOLO splits with reproducible seeds.
