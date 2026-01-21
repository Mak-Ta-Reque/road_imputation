#!/bin/bash
# Unified experiment runner for all datasets
# Usage: ./run_all_experiments.sh <data_base_path>

DATA_BASE_PATH=${1:-"/mnt/abka03/road_data"}
SAVE_BASE_PATH=${2:-"./explanations"}
GPU_ID=${3:-0}

echo "=================================================="
echo "Running Explanation Generation for All Datasets"
echo "=================================================="
echo "Data path: $DATA_BASE_PATH"
echo "Save path: $SAVE_BASE_PATH"
echo "GPU ID: $GPU_ID"
echo ""

# Explanation methods to run
EXPL_METHODS=("ig" "gb" "ig_sg" "gb_sg")

# ==================== CIFAR-10 ====================
echo ""
echo "==================== CIFAR-10 ===================="
for method in "${EXPL_METHODS[@]}"; do
    echo "Running $method for CIFAR-10..."
    python ExplanationGeneration_unified.py \
        --dataset cifar10 \
        --data_path "$DATA_BASE_PATH/cifar10" \
        --save_path "$SAVE_BASE_PATH/cifar10" \
        --expl_method "$method" \
        --gpu True \
        --gpu_id $GPU_ID \
        --batch_size 64 \
        --epochs 20
done

# ==================== Food-101 ====================
echo ""
echo "==================== Food-101 ===================="
for method in "${EXPL_METHODS[@]}"; do
    echo "Running $method for Food-101..."
    python ExplanationGeneration_unified.py \
        --dataset food101 \
        --data_path "$DATA_BASE_PATH/food-101" \
        --save_path "$SAVE_BASE_PATH/food101" \
        --expl_method "$method" \
        --gpu True \
        --gpu_id $GPU_ID \
        --batch_size 32 \
        --epochs 30
done

# ==================== ImageNet ====================
echo ""
echo "==================== ImageNet ===================="
for method in "${EXPL_METHODS[@]}"; do
    echo "Running $method for ImageNet..."
    python ExplanationGeneration_unified.py \
        --dataset imagenet \
        --data_path "$DATA_BASE_PATH/imagenet" \
        --save_path "$SAVE_BASE_PATH/imagenet" \
        --expl_method "$method" \
        --gpu True \
        --gpu_id $GPU_ID \
        --batch_size 32 \
        --test True
done

echo ""
echo "=================================================="
echo "All experiments completed!"
echo "=================================================="
