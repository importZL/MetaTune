"""Trainer for BLO-SAM instance-segmentation extension.

Same bilevel structure as trainer.py (Prompt module reused), but:
  - Uses datasets.dataset_instance (returns flow_gt at low_res).
  - Loss = semantic CE + Dice + flow MSE + prob BCE (Cellpose-style for flow head).
  - --module sam_lora_mask_decoder_instance produces low_res_logits AND flow_logits.
"""
import logging, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb

from utils import DiceLoss
from prompt import Prompt
from cal_dice import dice_score


def calc_loss(outputs, batch, ce_loss, dice_loss, dice_weight=0.8, flow_weight=20.0, prob_weight=5.0):
    """Joint loss: semantic (CE+Dice on low_res_logits) + flow (MSE on dy,dx) + prob (BCE on cellprob)."""
    low_res_logits = outputs["low_res_logits"]      # (B, 2, H, W) - 2-class semantic
    flow_logits    = outputs["flow_logits"]         # (B, 3, H, W) - (dy, dx, prob_logit)
    low_res_label  = batch["low_res_label"].cuda()  # (B, H, W) int 0/1
    flow_gt        = batch["flow_gt"].cuda()        # (B, 3, H, W) - (dy_gt, dx_gt, prob_gt)

    # Semantic
    loss_ce   = ce_loss(low_res_logits, low_res_label.long())
    loss_dice = dice_loss(low_res_logits, low_res_label, softmax=True)
    loss_sem  = (1 - dice_weight) * loss_ce + dice_weight * loss_dice

    # Flow (MSE on dy, dx) — only where there are cells
    flow_pred = flow_logits[:, :2]                  # (B, 2, H, W)
    flow_tgt  = flow_gt[:, :2]                      # (B, 2, H, W)
    fg_mask   = flow_gt[:, 2:3]                     # (B, 1, H, W) binary
    # Cellpose convention: MSE weighted by foreground (only flow where cells exist)
    loss_flow = ((flow_pred - flow_tgt) ** 2 * fg_mask).mean()

    # Prob (BCE on cellprob logit vs binary GT)
    prob_pred = flow_logits[:, 2:3]                 # (B, 1, H, W) logits
    prob_tgt  = flow_gt[:, 2:3]                     # (B, 1, H, W) binary
    loss_prob = F.binary_cross_entropy_with_logits(prob_pred, prob_tgt)

    total = loss_sem + flow_weight * loss_flow + prob_weight * loss_prob
    return total, loss_sem, loss_flow, loss_prob


@torch.no_grad()
def validate(args, model, validloader, multimask_output):
    model.eval()
    scores = []
    for sb in validloader:
        image_batch = sb["image"].cuda()
        out = model(image_batch, multimask_output, args.img_size)
        d = dice_score(out["low_res_logits"], sb["low_res_label"].cuda())
        scores.append(d.cpu().numpy())
    model.train()
    return float(np.mean(scores)) if scores else 0.0


def trainer(args, model, snapshot_path, multimask_output, low_res):
    # Single dataset module for all instance-seg tasks (CytoNuke, FluoRed for now)
    from datasets.dataset_instance import Synapse_dataset, RandomGenerator

    exp_name = f"{args.dataset}-{args.num_data}-{args.exp_type}-instance"
    logger = wandb.init(project="MetaTune-instance", name=exp_name, resume="allow",
                        anonymous="must", mode=args.wandb_mode)
    logger.config.update(vars(args))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    base_lr = args.base_lr
    num_classes = args.num_classes

    db = Synapse_dataset(
        train_dir=args.root_path, num_data=args.num_data, dataset=args.dataset,
        seed=args.seed,
        transform=transforms.Compose([RandomGenerator(
            output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])
        ]),
    )
    num_train = max(1, int(len(db) * 0.5))
    selector = range(len(db))
    print(f"len(db)={len(db)}; D1={num_train}; D2={len(db)-num_train}")

    def worker_init_fn(wid): random.seed(args.seed + wid)
    trainloader = DataLoader(db, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn, sampler=selector[:num_train])
    validloader = DataLoader(db, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn, sampler=selector[num_train:])

    model.train()
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes + 1)

    # Optimizer: all non-no_mask_embed params (LoRA + flow_head + mask_decoder trainable)
    optimizer = optim.AdamW(
        [p for n, p in model.named_parameters() if p.requires_grad and "no_mask_embed" not in n],
        lr=base_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay,
    )
    max_iter = args.max_epochs * len(trainloader)
    prompt_module = Prompt(model=model, args=args, max_iterations=max_iter)

    best_perf = 0.0
    iter_num = 0
    for epoch in range(args.max_epochs):
        for batch in trainloader:
            image_batch = batch["image"].cuda()
            out = model(image_batch, multimask_output, args.img_size)
            loss, lsem, lflow, lprob = calc_loss(out, batch, ce_loss, dice_loss, args.dice_param)
            logger.log({"loss/total": float(loss), "loss/sem": float(lsem),
                        "loss/flow": float(lflow), "loss/prob": float(lprob)})
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            # Bilevel step (prompt embedding meta-update on D2)
            try:
                valid_batch = next(iter(validloader))
                eta = optimizer.param_groups[0]["lr"]
                prompt_module.step(batch, valid_batch, eta, optimizer,
                                   unrolled=args.unrolled, cur_iter=iter_num)
            except StopIteration:
                pass

            # LR schedule (poly)
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * (iter_num + 1) / args.warmup_period
            else:
                shift = iter_num - args.warmup_period if args.warmup else iter_num
                lr_ = base_lr * (1.0 - shift / max_iter) ** 0.9
            for g in optimizer.param_groups: g["lr"] = lr_
            iter_num += 1

        v = validate(args, model, validloader, multimask_output)
        logging.info(f"Epoch {epoch+1}: valid Dice = {v:.4f}")
        logger.log({"info/valid_score": v})
        if v > best_perf:
            best_perf = v
            try:    model.save_lora_parameters(os.path.join(snapshot_path, "best.pth"))
            except: model.module.save_lora_parameters(os.path.join(snapshot_path, "best.pth"))

    try:    model.save_lora_parameters(os.path.join(snapshot_path, "final.pth"))
    except: model.module.save_lora_parameters(os.path.join(snapshot_path, "final.pth"))
    return "training finished"
