#!/bin/bash
# Call this script from the experiments/explanation_generation folder as cwd.
python ExplanationGeneration_food.py \
		--test=True \
		--gpu=False \
		--input_path='/workspaces/data/food101/image/food-101' \
		--save_path='/workspaces/data/food101/image/food-101/expl/' \
		--model_path='/workspaces/data/food101/weight/resnet50model..pth' \
		--batch_size=2 \
		--expl_method='ig'
