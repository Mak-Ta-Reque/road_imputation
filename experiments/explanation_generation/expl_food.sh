#!/bin/bash
# Call this script from the experiments/explanation_generation folder as cwd.
python ExplanationGeneration_food.py \
		--test=True \
		--gpu=False \
		--input_path='/mnt/sda/abka03-data/food-101/images/food-101' \
		--save_path='/mnt/sda/abka03-data/food-101/images/expl/' \
		--model_path='//mnt/sda/abka03-data/food-101/weight/resnet50model.pth' \
		--batch_size=20 \
		--expl_method='gb_sg'
