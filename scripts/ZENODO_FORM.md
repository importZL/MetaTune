# Zenodo upload form — fields ready to copy-paste

> **Status (2026-06-03):** the deposit has been published.
> Live DOI: [10.5281/zenodo.20517421](https://doi.org/10.5281/zenodo.20517421).
> This file is kept as a historical record of the upload-time metadata and as a
> template for future revisions / new deposits.

Use this file when filling in the Zenodo "New upload" form for the MetaTune
checkpoint deposit. Each section below corresponds to one of Zenodo's metadata
fields.

---

## Upload type

**Dataset**

(The deposit is supplementary materials for a paper — model checkpoints + run
configs. Choose "Dataset" rather than "Software" because the runnable code
lives on GitHub; this archive is purely the trained artifacts.)

---

## Basic information

**Title:**

> MetaTune: trained model checkpoints and run configurations for the paper "Meta-Finetuning Foundation Models for Generalizable Biological Image Segmentation in Ultra Low-Data Regimes"

**Authors** (in paper order, with ORCIDs if available):

- Li Zhang — UC San Diego — *(ORCID)*
- Youwei Liang — UC San Diego — *(ORCID)*
- Phuc Nguyen — UC San Diego — *(ORCID)*
- Fanny Chapelin — UC San Diego — *(ORCID)*
- Nan Hao — UC San Diego — *(ORCID)*
- Pengtao Xie (corresponding) — UC San Diego — *(ORCID)*

**Description** (copy verbatim into the description field; Markdown is supported on Zenodo):

> This deposit contains the trained model checkpoints and full per-run hyperparameter configurations for every semantic-segmentation experiment reported in:
>
> **Meta-Finetuning Foundation Models for Generalizable Biological Image Segmentation in Ultra Low-Data Regimes**
> Li Zhang, Youwei Liang, Phuc Nguyen, Fanny Chapelin, Nan Hao, Pengtao Xie
> *Cell Reports Methods* (under review), manuscript CR-METHODS-D-26-00020.
>
> ### What's in the archive
>
> - `semantic_main/` — 63 MetaTune (bilevel) checkpoints covering eight biological tasks (BCCD, Osteosarcoma, BT474, Huh7, Multimodal Cell, CytoNuke / HNSCC, FluoRed, Sartorius) plus yeast bright-field and contrast (in-distribution and out-of-distribution), three random seeds per task.
> - `ablations/vanilla_joint/` — 28 joint-optimization baselines (same architecture as MetaTune but without the bilevel split), used in Fig. 6 and as the "SAMed*" entries of Fig. 3 and Table 1.
> - `ablations/swap_meta/` — 18 swap-meta ablation checkpoints addressing Reviewer Comment #1: invert the meta/non-meta assignment so the prompt embedding becomes the lower-level variable and LoRA+decoder become the upper-level variable.
> - `README_zenodo.md` — inside the archive, gives the directory layout, the filename convention, a Python loading example, and the cross-reference to the figures of the paper.
>
> Each run directory carries the exact argparse namespace (`config.txt`) that produced the checkpoint — random seed, learning rates, image size, weight decay, etc. — so every result is fully reproducible together with the open-source code at https://github.com/importZL/MetaTune.
>
> ### Methodological lineage
>
> The bilevel-optimization framework underlying these checkpoints was introduced as **BLO-SAM** (Zhang et al., *ICML 2024*). The contribution of the present paper, and of this deposit, is the application of that framework to biological image segmentation in ultra-low-data regimes plus an eight-task benchmark and an out-of-distribution evaluation. Please cite both the present paper and BLO-SAM when using these checkpoints.
>
> ### Instance-segmentation results
>
> Instance-segmentation checkpoints from the reviewer-response exploration (BLO-SAM-instance, YOLOv7+SAM-bilevel, Cellpose-SAM fine-tunes, cpsam+BLO-SAM-bilevel, StarDist) are *not* included in this v1 deposit. Reproducing them does not require shared weights: the baselines load their own publicly-available pretrained models automatically, and the fine-tunes train from those public weights using the runners in `baselines/` and `scripts/` of the GitHub repository. A future v2 deposit may add instance-segmentation checkpoints if the manuscript expands that contribution.

---

## License

**Creative Commons Attribution 4.0 International (CC BY 4.0)**

---

## Related identifiers

Add three "Related identifier" entries:

| Relation type | Identifier | Resource type |
|---|---|---|
| `is supplement to` | DOI of the published paper (assign after acceptance) | Publication / Journal article |
| `is supplement to` | https://github.com/importZL/MetaTune | Software |
| `cites` | https://arxiv.org/abs/2402.16338 *(BLO-SAM paper)* | Publication / Conference paper |
| `is published in` | DOI of the bioRxiv preprint *(if you posted one)* | Publication / Preprint |

---

## Keywords

```
segmentation, biological imaging, foundation model, SAM, Segment Anything Model,
meta-learning, bilevel optimization, low-data, few-shot, LoRA, BLO-SAM,
cell segmentation
```

---

## Funding (from the paper's Acknowledgments)

- NSF IIS2405974
- NSF IIS2339216
- NIH R35GM157217
- NIH R21GM154171

(On Zenodo, search for each grant in the funding picker; the system auto-fills the funder DOIs.)

---

## Communities (optional)

- *Cell Press* (Zenodo community)
- *Image Analysis* (Zenodo community)

---

## DOI (published)

The deposit has been published with DOI **10.5281/zenodo.20517421**
(https://doi.org/10.5281/zenodo.20517421). For future revisions of this
deposit, Zenodo will mint a new version-DOI; the unversioned concept-DOI above
will always resolve to the latest version.

---

## File to upload

```
/data2/li/workspace/zenodo_metatune_v1.tar.gz
```

Size: ~1.3 GB. Zenodo's web uploader handles this in one go; if you have
intermittent network use the API (instructions on Zenodo) for resumable upload.

After upload, double-check:
- The tarball appears under "Files" with the correct size.
- Clicking the file name lets Zenodo unpack the README_zenodo.md preview.

Then click **"Publish"**.

---

## After publishing — wire the DOI into the manuscript (DONE for v1)

The live DOI is **10.5281/zenodo.20517421**. It has already been wired into:

- `README.md` (Reproducibility checklist → Weights).
- `REPRODUCE.md` (Released checkpoints section).
- `HYPERPARAMETERS.md` (intro paragraph).

For the manuscript / response letter, use:

```
Code: https://github.com/importZL/MetaTune
Trained model checkpoints (with run configs): https://doi.org/10.5281/zenodo.20517421
```

Both sentences belong in the "Data availability" and "Code availability"
statements that Cell Press requires. The response letter to Reviewer #7 should
quote the same URL/DOI.
