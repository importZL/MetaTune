# BASELINES.md — setting up the semantic-segmentation comparison methods

This guide explains how to rebuild each semantic-segmentation baseline in Figures 2–4 from its public implementation. For each method it gives the upstream repository and commit, the exact edits we made, the command we ran and the evaluation. The adapted code itself is not redistributed. Baselines already shipped in this repository (PerSAM-F, Matcher, Cellpose, StarDist) are documented in [REPRODUCE.md](REPRODUCE.md) and [HYPERPARAMETERS.md](HYPERPARAMETERS.md).

Throughout, `<DATA_ROOT>/<task>/{train,test}/{Images,Masks}` is the layout from [DATA.md](DATA.md). Masks share the image filename, and any nonzero mask pixel is foreground.

| Method | Figure | Upstream | Trains on | Section |
|---|---|---|---|---|
| DeepLabV3 | 2 | torchvision `deeplabv3_resnet50` (hub `pytorch/vision:v0.10.0`), trained with a Swin-Unet-derived trainer | first N/2 support images | [1](#1-deeplabv3-and-unet) |
| UNet | 2 | [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) model code, same trainer | first N/2 support images | [1](#1-deeplabv3-and-unet) |
| Vanilla SAM | 2 | [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything), ViT-B | no training | [2](#2-vanilla-sam) |
| MedSA | 3 | [WuJunde/Medical-SAM-Adapter](https://github.com/WuJunde/Medical-SAM-Adapter) @ `bbb8927` | first N/2 support images | [3](#3-medsa) |
| SAMed | 3 | this repository (`train_vanilla.py`) | first N/2 support images | [4](#4-samed) |
| uSAM | 3 | [computational-cell-analytics/micro-sam](https://github.com/computational-cell-analytics/micro-sam) @ `589e521` | no training | [5](#5-usam) |
| HSNet | 4 | [juhongm999/hsnet](https://github.com/juhongm999/hsnet) @ `6b1bc12` | N/2 support + N/2 query episodes | [6](#6-hsnet) |

**Support-image selection (all trainable baselines).** Each loader takes the first N files returned by `os.listdir(<DATA_ROOT>/<task>/train/Images)`. That order depends on the filesystem. To use the same images as MetaTune, replace that line with the file list from `splits/<task>/…/train.txt`, in the order listed. The first N/2 entries are the training images and the last N/2 the validation images, as in MetaTune's D₁/D₂.

---

## 1. DeepLabV3 and UNet

Both models are trained from scratch by the same trainer. The trainer was derived from the SAMed/Swin-Unet training script ([HuCaoFighting/Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)). Only its `train.py`/`trainer.py` scaffolding is used; the Swin model is not.

### 1.1 Environment
Python 3.7, PyTorch ≥ 1.9, torchvision, `scipy`, `wandb` (runs with `mode="disabled"`), `tqdm`, `Pillow`.

### 1.2 Model code
- **UNet.** Copy `unet/unet_model.py` and `unet/unet_parts.py` from Pytorch-UNet, then make these three edits to `unet_parts.py`:
  1. In `Up.__init__` (bilinear branch): `nn.Upsample(scale_factor=20, mode='bilinear', align_corners=True)` (upstream: `scale_factor=2`). The following `F.pad` against the skip connection brings the tensor back to the skip size.
  2. In `Up.forward`: `x = torch.cat([x1, x2], dim=1)` (upstream: `[x2, x1]`).
  3. `OutConv`: `nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))` (upstream: a single 1×1 conv).

  Instantiate with `UNet(n_channels=3, n_classes=2, bilinear=True)`.
- **DeepLabV3.** `deeplab/deeplabv3.py`:
  ```python
  import torch

  def replace_batchnorm(module):
      for name, child in module.named_children():
          if isinstance(child, torch.nn.BatchNorm2d):
              setattr(module, name, torch.nn.InstanceNorm2d(child.num_features))
          else:
              replace_batchnorm(child)
      return module

  class DeepLabV3(torch.nn.Module):
      def __init__(self, num_classes):
          super().__init__()
          self.main = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50',
                                     pretrained=False, num_classes=num_classes)
          if self.main.aux_classifier:
              self.main.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, 1)
          self.main.classifier[4] = torch.nn.Conv2d(256, num_classes, 1)
          self.main = replace_batchnorm(self.main)   # BatchNorm -> InstanceNorm (batch size 1)

      def forward(self, x):
          return self.main(x)['out']
  ```
  Instantiate with `DeepLabV3(num_classes=2)`. There are no ImageNet weights for either the backbone or the head.

### 1.3 Data loader
Same structure as `datasets/dataset_blood.py` in this repository:
- the image is read as RGB and scaled to [0, 1];
- the mask is `(RGB mask > 0)`, channel 0;
- both are resized to 224 × 224 with `scipy.ndimage.zoom` (order 3 for the image, order 0 for the mask);
- there is no augmentation and no mean/std normalization.

### 1.4 Training
| Setting | Value |
|---|---|
| Input size | 224 × 224 |
| Output | 2 channels (background, foreground) |
| Training images | first N/2 of the N support images; the other N/2 are used only for checkpoint selection |
| Loss | `CrossEntropyLoss + DiceLoss(2, softmax=True)` (weights 1 : 1) |
| Optimizer | `AdamW(model.parameters(), lr=base_lr)` (PyTorch default weight decay 0.01) |
| LR schedule | `lr = base_lr · (1 − iter / max_iter)^0.9`, updated every iteration |
| Epochs / batch size | 100 / 1 |
| Checkpoint | the epoch with the highest Dice on the N/2 validation images |
| Seed | `random`, `numpy`, `torch` seeded with `--seed` |

Command (the recorded runs used `base_lr=1e-3`):
```bash
python train.py --model {deeplab|unet} --dataset Synapse \
  --cfg configs/swin_tiny_patch4_window7_224_lite.yaml \   # required by the argument parser only
  --root_path <DATA_ROOT>/<task>/train/Images --num_data <N> \
  --img_size 224 --batch_size 1 --max_epochs 100 --base_lr 1e-3 \
  --seed <seed> --output_dir ./out/<task>_<model>_s<seed>
```
<!-- TODO(authors): confirm the base_lr used for each task in Fig. 2; per-run logs were overwritten. -->

### 1.5 Evaluation
After training, the best checkpoint is evaluated on `<DATA_ROOT>/<task>/test/Images` at 224 × 224. The per-image Dice comes from `cal_dice.dice_score` (the same file as in this repository: argmax over the two channels, foreground class only), averaged over test images.

---

## 2. Vanilla SAM

There is no training. SAM ViT-B (`sam_vit_b_01ec64.pth`) is built with this repository's copy of `segment_anything`: `sam_model_registry['vit_b'](image_size=256, num_classes=1, checkpoint=...)`, with the default SAM pixel mean/std. The image encoder therefore runs at 256 px, not SAM's native 1024. The model is used through `SamPredictor` with one prompt set per test image, derived from that image's ground-truth mask. This is the complete algorithm:

1. **Preprocessing.** Load the image and mask with the same loader as in §1.3, but resize to 256 × 256. Pass the image to `predictor.set_image(uint8(image × 255))` in HWC RGB order.
2. **Point prompts** (one positive and one negative per image). Let `M` be the 256 × 256 binary mask.
   - Positive point: one pixel drawn uniformly at random from `argwhere(M == 1)`.
   - Negative point: one pixel drawn uniformly at random from `argwhere(M == 0)`.
   - Both are given to SAM as `(x, y) = (column, row)`, with labels `[1, 0]`.
   - Our runs did not seed `numpy`. To reproduce with a fixed draw, call `np.random.seed(<seed>)` before the loop.
3. **Box prompt.** Run `cv2.findContours(M, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)` and take `cv2.boundingRect` → `(x, y, w, h)` for every contour. The box passed to SAM is
   `[min(x), max(y), min(w), max(h)]`, computed over all contours.
   This vector is passed directly as SAM's `box` argument, which SAM reads as `[x0, y0, x1, y1]`.
4. **Prediction.** `predictor.predict(point_coords=points, point_labels=[1, 0], box=box, multimask_output=False)`. If `M` has no foreground pixels, the call is made with no prompts.
5. **Metric.** Dice per image, `(2·|P∩G| + 1e-4) / (|P| + |G| + 1e-4)` at 256 × 256, then averaged over the test set.

---

## 3. MedSA

Upstream: `git clone https://github.com/WuJunde/Medical-SAM-Adapter && git checkout bbb8927`. The repository has since moved to `ImprintLab/Medical-SAM-Adapter`. Use the environment from its `environment.yml`, with `tensorboardX`, `seaborn` and `sklearn` imports removed; the edits below make them unused.

### 3.1 Edits
- `conf/global_settings.py`: `EPOCH = 100`.
- `cfg.py`: `-val_freq` default `1`, so validation runs every epoch.
- `function.py`, `train_sam`: besides the upstream freeze of all non-`Adapter` image-encoder parameters, also freeze **all prompt-encoder and mask-decoder parameters**. Only the adapters are trained. Loss is computed as `lossfunc(pred.squeeze(), masks.squeeze())`.
- `train.py`:
  - Set `random/np/torch` seeds to the run seed at import time (our runs: 22) and use `worker_init_fn = random.seed(42 + worker_id)`.
  - Add a dataset branch that builds `Synapse_dataset(train_dir=args.data_path, num_data=N)` (loader in §3.2). The first N/2 images form the training loader and the last N/2 the validation loader, both with batch size 1.
  - Build a test loader from `args.data_path.replace('/train', '/test')`.
  - After training, reload `checkpoint_best.pth` and report `validation_sam` on the test loader.

### 3.2 Loader (`dataset/dataset_teeth.py` in our copy)
- The image (RGB, [0, 1]) is resized to `-image_size` 1024 × 1024, and the mask to 256 × 256 (nearest neighbour).
- Click prompt: `pt = random_click(mask, 1, 1)`, one foreground pixel drawn uniformly at random with `np.random` (upstream `utils.random_click`), with `p_label = 1`. It is redrawn every time a sample is loaded, including at test time.

### 3.3 Training and evaluation
| Setting | Value |
|---|---|
| Trainable parameters | image-encoder adapters only |
| Prompt | one random positive click from the ground-truth mask (train and test) |
| Loss | `BCEWithLogitsLoss(pos_weight=2)` (upstream `criterion_G`) |
| Optimizer | `Adam(lr=1e-4, betas=(0.9, 0.999), weight_decay=0)`; the upstream `StepLR(10, 0.5)` is created but never stepped, so the LR stays constant |
| Epochs / batch | 100 / 1 |
| Checkpoint | lowest validation loss on the N/2 validation images |
| Test metric | upstream `eval_seg`: Dice of `(logits > t)` vs mask, averaged over `t ∈ {0.1, 0.3, 0.5, 0.7, 0.9}` (thresholds are applied to logits, not probabilities) |

```bash
python -W ignore train.py -net sam -mod sam_adpt -exp_name <task>-<N> \
  -sam_ckpt ./checkpoints/sam_vit_b_01ec64.pth -image_size 1024 -b 1 \
  -gpu_device 0 -dataset <branch> -data_path <DATA_ROOT>/<task>/train/Images
```

---

## 4. SAMed

Run with this repository's `train_vanilla.py`. The configuration differs from the original SAMed in two ways:
- LoRA (rank 4) is applied to the **mask decoder** (`sam_lora_mask_decoder`), not the image encoder.
- The learnable prompt embedding `no_mask_embed` is trained jointly with the LoRA parameters under one loss, with no bilevel split.

```bash
python train_vanilla.py --root_path <DATA_ROOT>/<task>/train/Images --dataset <task> \
  --module sam_lora_mask_decoder --num_data <N> --max_epochs 100 --batch_size 1 \
  --base_lr <task base_lr from HYPERPARAMETERS.md> --num_classes 1 --seed <seed> \
  --exp_type vanilla --ckpt ./checkpoints/sam_vit_b_01ec64.pth --output ./output --wandb_mode disabled
# Sartorius used --module sam_lora_all.
DATASET=<task> VOLUME_PATH=<DATA_ROOT>/<task>/test/Images LORA_CKPT=./output/<run>/best.pth bash inference.sh
```

Training settings:
- Training uses the first N/2 support images, and the checkpoint with the best Dice on the other N/2 is kept.
- Loss `0.2·CE + 0.8·Dice`, AdamW (weight decay 0.1) and the polynomial LR schedule, as in MetaTune.
- Evaluation uses `inference.sh`, identical to MetaTune.

---

## 5. uSAM

There is no training. Install micro-sam at commit `589e521` (`git clone https://github.com/computational-cell-analytics/micro-sam && git checkout 589e521 && pip install -e .`). Use the released light-microscopy model `vit_b_lm` in automatic instance segmentation (AIS) mode, following the upstream `notebooks/automatic_segmentation.ipynb`:

```python
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation
import numpy as np, matplotlib.pyplot as plt, os, random, torch

random.seed(21); np.random.seed(21); torch.manual_seed(21); torch.cuda.manual_seed(21)
predictor, segmenter = get_predictor_and_segmenter(model_type="vit_b_lm", amg=False, is_tiled=False)

def dice(gt, pred, eps=1e-6):
    gt, pred = gt.astype(bool), pred.astype(bool)
    return (2 * np.sum(gt & pred) + eps) / (gt.sum() + pred.sum() + eps)

scores = []
for name in os.listdir(img_dir):                       # <DATA_ROOT>/<task>/test/Images
    image = plt.imread(os.path.join(img_dir, name))
    gt = plt.imread(os.path.join(img_dir.replace("Images", "Masks"), name))
    inst = automatic_instance_segmentation(predictor=predictor, segmenter=segmenter,
                                           input_path=image, ndim=2)
    scores.append(dice(gt, (inst > 0).astype(np.uint8)))   # instances -> binary foreground
print(np.mean(scores))
```

Evaluation is at the native image resolution. Any nonzero ground-truth value counts as foreground.

---

## 6. HSNet

Upstream: `git clone https://github.com/juhongm999/hsnet && git checkout 6b1bc12`. The backbone is ResNet-50 with ImageNet weights, as in the upstream default.

### 6.1 Edits
- `data/my_fss.py`: a new dataset class, registered in `data/dataset.py` under a new benchmark key (ours was `'lung'`).
  - **Training episodes** (`split='trn'`): with `files = os.listdir(train/Images)`, the supports are `files[:N/2]` and the queries `files[N/2:N]`. Episode *i* pairs support *i* with query *i* (1-shot).
  - **Test** (`split='test'`): every image in `test/Images` is a query, and its single support is `files[i % N]` for query index *i*.
  - Images (RGB) and masks (converted to grayscale) are resized to 256 × 256 with PIL's default `Image.resize` resampling. Images are scaled to [0, 1] with no ImageNet normalization; the mask is thresholded at > 0 after resizing.
- `data/dataset.py`: `initialize(..., shot)` stores the shot count; `build_dataloader` no longer shuffles and no longer forces `nworker=0` at test.
- `common/logger.py`: `nclass = 1` for the new benchmark; the TensorBoard writer is removed.
- `train.py`:
  - Call `set_seed(<seed>)` at start-up (our runs: 2) and remove `utils.fix_randseed`.
  - Defaults `--bsz 1 --niter 100 --nworker 4`, plus a new `--shot` argument.
  - Remove the per-epoch validation and best-mIoU checkpointing. **The model after the last epoch is evaluated.**
  - Evaluate on the test split with `cal_dice.dice_score` (argmax, foreground class) averaged over test images.

### 6.2 Command
```bash
python train.py --backbone resnet50 --benchmark lung --lr 1e-3 --bsz 1 \
  --logpath ./log --datapath <DATA_ROOT>/<task>/train/Images --shot <N>
```
Optimizer: `Adam(lr=1e-3)`; loss: upstream HSNet cross-entropy on the query mask.
