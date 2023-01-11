#!/bin/bash
# Call this script from the experiments/cifar10 folder as cwd.
export PYTHONPATH=../../
python NoRetrainingMethod.py \
		--gpu=True \
		--data_path='/mnt/XMG4THD/abka03_data/food-101/images/food-101' \
		--expl_path='/mnt/XMG4THD/abka03_data/food-101/images/expl'\
		--params_file='noretrain_params_gpu.json' \
		--model_path='/mnt/XMG4THD/abka03_data/food-101/weight/resnet50model.pth'\
		--batch_size=10 \
		--dataset="food101"
