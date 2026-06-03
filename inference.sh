# Example inference script. Edit --volume_path / --lora_ckpt / --dataset per run.
python -W ignore inference.py \
    --volume_path /path/to/CytoNuke/test/Images \
    --lora_ckpt ./output/cyto4_auto_first_img256_<timestamp>/best.pth \
    --gpu_id 0 \
    --module sam_lora_mask_decoder \
    --dataset cyto \
    --num_classes 1 \
    --ckpt ./checkpoints/sam_vit_b_01ec64.pth
