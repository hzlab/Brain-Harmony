#!/bin/bash

# Get parameters
config_file=$1

if [[ ! -f "$config_file" ]]; then
    echo "❌ Error: Configuration file '$config_file' does not exist!"
    exit 1
fi

# Display configuration
echo "=== BrainHarmonix Stage0 Embedding Extraction ==="
echo "Configuration file: $config_file"
echo "=== Starting Training ==="

# Run
accelerate launch \
    --main_process_port 9999 \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    modules/harmonizer/stage0_embed/embedding_pretrain.py \
    --config=${config_file}