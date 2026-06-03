# Zenodo upload form — fields ready to copy-paste

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

## Reserve DOI (recommended)

Click **"Reserve DOI"** in the Identifiers panel before publishing. This gives
you a stable DOI to cite in the paper *before* the deposit is finalized. The
DOI is reserved against your account, so you can still revise the metadata or
files until you click "Publish".

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

Then click **"Publish"**. Once published, the deposit gets a permanent DOI of
the form `10.5281/zenodo.NNNNNNNN`. Update the placeholder `DOI: reserved at submission`
in the GitHub repo's `README.md` and `REPRODUCE.md` with this DOI.

---

## After publishing — wire the DOI into the manuscript

1. Add to the paper's "Code availability" / "Data availability" statement (Cell Press requires both):
   > Code: https://github.com/importZL/MetaTune
   > Trained model checkpoints (with run configs): https://doi.org/10.5281/zenodo.NNNNNNNN
2. Update `README.md`, `REPRODUCE.md`, and any reviewer-response text that currently says "DOI: reserved at submission".
3. In the response letter to Reviewer #7, point at the Zenodo URL/DOI explicitly.
