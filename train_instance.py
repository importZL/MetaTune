"""Entry point for BLO-SAM-instance training."""
import argparse, os, random, sys, time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from importlib import import_module

from segment_anything import sam_model_registry
from trainer_instance import trainer

p = argparse.ArgumentParser()
p.add_argument("--root_path", type=str, required=True, help="train Images dir (with sibling Masks_instance)")
p.add_argument("--output", type=str, default="./output_instance")
p.add_argument("--dataset", type=str, default="cyto")
p.add_argument("--num_classes", type=int, default=1)
p.add_argument("--max_epochs", type=int, default=100)
p.add_argument("--batch_size", type=int, default=1)
p.add_argument("--gpu_id", type=str, default="0")
p.add_argument("--base_lr", type=float, default=5e-3)
p.add_argument("--prompt_base_lr", type=float, default=1e-3)
p.add_argument("--img_size", type=int, default=256)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--vit_name", type=str, default="vit_b")
p.add_argument("--ckpt", type=str, default="/data1/li/Auto_SAMed/checkpoints/sam_vit_b_01ec64.pth")
p.add_argument("--lora_ckpt", type=str, default=None)
p.add_argument("--rank", type=int, default=4)
p.add_argument("--warmup", action="store_true")
p.add_argument("--warmup_period", type=int, default=250)
p.add_argument("--module", type=str, default="sam_lora_mask_decoder_instance")
p.add_argument("--dice_param", type=float, default=0.8)
p.add_argument("--num_data", type=int, default=4)
p.add_argument("--exp_type", type=str, default="instance")
p.add_argument("--weight_decay", type=float, default=0.1)
p.add_argument("--prompt_weight_decay", type=float, default=0.1)
p.add_argument("--unrolled", action="store_true")
p.add_argument("--swap_meta", action="store_true")
p.add_argument("--wandb_mode", type=str, default="disabled")
args = p.parse_args()

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    args.exp = f"{args.dataset}{args.num_data}_{args.exp_type}_img{args.img_size}"
    snapshot_path = os.path.join(args.output, args.exp + "_" + time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(snapshot_path, exist_ok=True)

    sam, img_embedding_size = sam_model_registry[args.vit_name](
        image_size=args.img_size, num_classes=args.num_classes,
        checkpoint=args.ckpt, pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
    )
    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()
    if args.lora_ckpt is not None:
        net.load_lora_parameters(args.lora_ckpt, torch.device("cuda"))

    multimask_output = args.num_classes > 1
    low_res = img_embedding_size * 4

    with open(os.path.join(snapshot_path, "config.txt"), "w") as f:
        for k, v in args.__dict__.items():
            f.write(f"{k}: {v}\n")

    t0 = time.time()
    trainer(args, net, snapshot_path, multimask_output, low_res)
    print(f"duration: {(time.time()-t0)/3600:.3f} h")
