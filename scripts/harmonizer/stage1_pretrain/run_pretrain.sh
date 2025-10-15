#!/bin/bash

# Check parameters
if [ $# -ne 2 ]; then
    echo "Usage: $0 <model_size> <num_latent_tokens>"
    echo "Model size: base, small, large"
    echo "Example: $0 base 128"
    exit 1
fi

# Set model size and corresponding abbreviation
MODEL_SIZE=$1
NUM_LATENT_TOKENS=$2

case $MODEL_SIZE in
    "base")
        ms="b"
        model_size="base"
        ;;
    "small")
        ms="s" 
        model_size="small"
        ;;
    "large")
        ms="l"
        model_size="large"
        ;;
    *)
        echo "Error: Invalid model size '$MODEL_SIZE'"
        echo "Please choose: base, small, large"
        exit 1
        ;;
esac

echo "Model size: $MODEL_SIZE (${ms})"
echo "Number of latent tokens: $NUM_LATENT_TOKENS"

python modules/harmonizer/stage1_pretrain/main_pretrain.py \
        --output_dir experiments/stage1_pretrain/debug_harmonizer_vit${ms}_${NUM_LATENT_TOKENS} \
        --log_dir experiments/stage1_pretrain/debug_harmonizer_vit${ms}_${NUM_LATENT_TOKENS} \
        --batch_size 16 \
        --model onetokreg_vit_${model_size}_patch16 \
        --norm_pix_loss \
        --mask_ratio 0.75 \
        --epochs 300 \
        --warmup_epochs 40 \
        --blr 5e-4 --weight_decay 0.05 \
        --data_root_dir experiments/stage0_embed/pretrain_embed \
        --num_latent_tokens ${NUM_LATENT_TOKENS}