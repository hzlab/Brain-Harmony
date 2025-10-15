#!/bin/bash

# 检查参数
if [ $# -ne 4 ]; then
    echo "用法: $0 <模型大小> <num_latent_tokens> <数据集> <随机种子>"
    echo "模型大小: base, small, large"
    echo "数据集: 数据集名称"
    echo "随机种子: 用于数据分割的随机种子"
    echo "例如: $0 base 128 dataset1 42"
    exit 1
fi

# 获取参数
MODEL_SIZE=$1
NUM_LATENT_TOKENS=$2
DATASET_NAME=$3
SPLIT_SEED=$4

# 验证模型大小并设置对应的缩写
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
        echo "错误: 无效的模型大小 '$MODEL_SIZE'"
        echo "请选择: base, small, large"
        exit 1
        ;;
esac

echo "模型大小: $MODEL_SIZE (${ms})"
echo "潜在令牌数量: $NUM_LATENT_TOKENS"
echo "数据集: $DATASET_NAME"
echo "随机种子: $SPLIT_SEED"


# 启动微调训练
python modules/harmonizer/stage2_finetune/main_finetune.py \
    --batch_size 16 \
    --model vit_base_patch16 \
    --output_dir experiments/stage2_finetune/harmonizer_vit${ms}_${NUM_LATENT_TOKENS} \
    --log_dir experiments/stage2_finetune/harmonizer_vit${ms}_${NUM_LATENT_TOKENS} \
    --epochs 50 \
    --lr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 \
    --dist_eval \
    --nb_classes 2 \
    --dataset_name ${DATASET_NAME} \
    --split_seed ${SPLIT_SEED} \
    --finetune checkpoints/harmonizer/model.pth