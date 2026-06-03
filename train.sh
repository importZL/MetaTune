# Example training script. Edit --root_path / --dataset / --base_lr / --prompt_base_lr per task.
# Per-task LRs are tabulated in HYPERPARAMETERS.md.
python train.py \
    --root_path /path/to/CytoNuke/train/Images \
    --output ./output \
    --module sam_lora_mask_decoder \
    --max_epoch 100 \
    --num_data 4 \
    --wandb_mode disabled \
    --batch_size 1 \
    --dataset cyto \
    --exp_type auto_first \
    --base_lr 1e-3 \
    --prompt_base_lr 5e-3 \
    --gpu_id 0 \
    --num_classes 1 \
    --dice_param 0.8 \
    --rank 4 \
    --seed 42 \
    --ckpt ./checkpoints/sam_vit_b_01ec64.pth
