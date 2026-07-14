import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import DiceLoss, Focal_loss
from torchvision import transforms
from icecream import ic
import wandb
from medpy import metric
from cal_dice import dice_score


def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice


@torch.no_grad()
def validate(args, model, dice_loss, validloader, multimask_output):
    score_dice = []
    model.eval()
    for _, sampled_batch in enumerate(validloader):
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
        low_res_label_batch = sampled_batch['low_res_label']
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
        low_res_label_batch = low_res_label_batch.cuda()

        outputs = model(image_batch, multimask_output, args.img_size)

        low_res_logits = outputs['low_res_logits']
        dice = dice_score(low_res_logits, low_res_label_batch)        
        score_dice.append(dice.cpu().numpy())
    model.train()
    return np.mean(score_dice)

def trainer(args, model, snapshot_path, multimask_output, low_res):
    # Dataset routing: this paper benchmarks 8 biological tasks + yeast OOD.
    if args.dataset == 'blood':
        from datasets.dataset_blood import Synapse_dataset, RandomGenerator
    elif args.dataset in ['osteosarcoma', 'multimodal', 'cyto']:
        from datasets.dataset_osteosarcoma import Synapse_dataset, RandomGenerator
    elif args.dataset in ['cellBT474', 'cellHuh7', 'sartorius', 'fluocellRed',
                          'yeast-bright', 'yeast-contrast']:
        from datasets.dataset_cellBT474 import Synapse_dataset, RandomGenerator
    else:
        print(f"##### Unimplemented dataset: {args.dataset} #####")
        print("Supported datasets for this paper: blood, osteosarcoma, multimodal, cyto, "
              "cellBT474, cellHuh7, sartorius, fluocellRed, yeast-bright, yeast-contrast")
        sys.exit()
    
    logger = wandb.init(project='MetaTune', name=f"{args.dataset}-{args.num_data}-{args.exp_type}-sam-lora", resume='allow', anonymous='must', mode=args.wandb_mode)
    logger.config.update(vars(args))
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    base_lr = args.base_lr
    num_classes = args.num_classes
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(train_dir=args.root_path,
                               num_data=args.num_data, 
                               dataset=args.dataset,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])]))
    if not 0 < args.train_split < 1:
        raise ValueError("--train_split must be between 0 and 1")
    num_train = int(len(db_train) * args.train_split)
    num_valid = len(db_train) - num_train
    selector = range(len(db_train))
    logging.info("The length of train set is: {}".format(num_train))
    logging.info("The length of train set is: {}".format(num_valid))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn, sampler=selector[:num_train])
    validloader = DataLoader(db_train, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn, sampler=selector[num_train:])
    
    db_test = Synapse_dataset(train_dir=args.root_path.replace('/train', '/test'), dataset=args.dataset,
                              transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res], split="test")]))
    testloader = DataLoader(db_test, batch_size=1, num_workers=4, pin_memory=True)
    
    model.train()
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes + 1)

    if args.freeze_main and args.freeze_prompt:
        raise ValueError("--freeze_main and --freeze_prompt cannot be used together")
    for name, parameter in model.named_parameters():
        if args.freeze_main and "no_mask_embed" not in name:
            parameter.requires_grad_(False)
        if args.freeze_prompt and "no_mask_embed" in name:
            parameter.requires_grad_(False)

    optimizer = None if args.freeze_main else optim.AdamW(
        [p for n, p in model.named_parameters() if p.requires_grad and "no_mask_embed" not in n],
        lr=base_lr, betas=(0.9, 0.999), weight_decay=0.1)
    optimizer_prompt = None if args.freeze_prompt else optim.AdamW(
        [p for n, p in model.named_parameters() if p.requires_grad and "no_mask_embed" in n],
        lr=base_lr, betas=(0.9, 0.999), weight_decay=0.1)
    
    
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    
    scheduler = None if optimizer is None else torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=base_lr, total_steps=max_iterations)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    
    train_score_list = []
    valid_score_list = []
    test_score_list = []
    
    for epoch_num in range(max_epoch):
        for i_batch, sampled_batch in enumerate(trainloader):
            
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
            low_res_label_batch = sampled_batch['low_res_label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()
            assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'
            outputs = model(image_batch, multimask_output, args.img_size)
            loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param)
            logger.log({'info/stage1_loss': loss})
            # if 'vanilla' in args.exp_type:
            # logger.log({'info/stage2_loss': loss})
            if optimizer is not None:
                optimizer.zero_grad()
            if optimizer_prompt is not None:
                optimizer_prompt.zero_grad()
            loss.backward()
            if optimizer is not None:
                optimizer.step()
            if optimizer_prompt is not None:
                optimizer_prompt.step()
            # else:
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()
            
            #     valid_batch = next(iter(validloader))
            #     valid_image_batch, valid_label_batch = valid_batch['image'], valid_batch['label']  # [b, c, h, w], [b, h, w]
            #     valid_low_res_label_batch = valid_batch['low_res_label']
            #     valid_image_batch, valid_label_batch = valid_image_batch.cuda(), valid_label_batch.cuda()
            #     valid_low_res_label_batch = valid_low_res_label_batch.cuda()
            #     assert valid_image_batch.max() <= 3, f'image_batch max: {valid_image_batch.max()}'
            #     valid_outputs = model(valid_image_batch, multimask_output, args.img_size)
            #     valid_loss, valid_loss_ce, valid_loss_dice = calc_loss(valid_outputs, valid_low_res_label_batch, ce_loss, dice_loss, args.dice_param)
            #     logger.log({'info/stage2_loss': valid_loss})
            #     optimizer_prompt.zero_grad()
            #     valid_loss_dice.backward()
            #     optimizer_prompt.step()
            
            ##### Adjust Learning Rate #####
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups if optimizer is not None else []:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups if optimizer is not None else []:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1

            if iter_num % 10 == 0:
                logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            
                # if 'vanilla' in args.exp_type:
                image = image_batch[0, :, :, :]
                output_masks = outputs['masks'] 
                labs = label_batch[0, ...].unsqueeze(0)
                # else:
                #     image = valid_image_batch[0, :, :, :]
                #     output_masks = valid_outputs['masks'] 
                #     labs = valid_label_batch[0, ...].unsqueeze(0)
                    
                ims = {}
                image = (image - image.min()) / (image.max() - image.min())
                image = image.mul(255).permute(1, 2, 0).to('cpu').numpy()
                ims['train/Image'] = wandb.Image(image)
                   
                output_masks = torch.argmax(torch.softmax(output_masks, dim=1), dim=1, keepdim=True)[0, ...]                
                output_masks = output_masks.mul(255).to('cpu').numpy()
                ims['train/Prediction'] = wandb.Image(output_masks)

                labs = labs.mul(255).to('cpu').numpy()
                ims['train/GroundTruth'] = wandb.Image(labs)

                logger.log(ims)

        # validate the model at every epoch ending
        valid_score = validate(args, model, dice_loss, validloader, multimask_output)
        logging.info('Epoch %d : valid score : %f' % (epoch_num + 1, valid_score))
        logger.log({'info/valid_score': valid_score})
        
        # test the model on training and test sets at every epoch ending
        # test_score = validate(args, model, dice_loss, testloader, multimask_output)
        # logging.info('Epoch %d : test score : %f' % (epoch_num + 1, test_score))
        # train_score = validate(args, model, dice_loss, trainloader, multimask_output)
        # logging.info('Epoch %d : train score : %f' % (epoch_num + 1, train_score))
        
        # train_score_list.append(train_score)
        # valid_score_list.append(valid_score)
        # test_score_list.append(test_score)
        
        if valid_score > best_performance:
            best_performance = valid_score
            save_mode_path = os.path.join(snapshot_path, 'best.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
        
        # test_score = validate(args, model, dice_loss, testloader, multimask_output)
        # logger.log({'info/test_score': test_score})
        
    # f = open(f"./output_other/result_samed1_{args.dataset}.txt", "w")
    # for item in train_score_list:
    #     f.write(str(item) + " ")
    # f.write("\n")
    # for item in valid_score_list:
    #     f.write(str(item) + " ")
    # f.write("\n")
    # for item in test_score_list:
    #     f.write(str(item) + " ")
    # f.write("\n")
    # f.close()
    
    save_mode_path = os.path.join(snapshot_path, 'final.pth')
    try:
        model.save_lora_parameters(save_mode_path)
    except:
        model.module.save_lora_parameters(save_mode_path)
    
    return "Training Finished!"
