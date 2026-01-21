# Unified ROAD Benchmark Pipeline

A comprehensive pipeline for running ROAD (Remove and Debias) benchmark experiments across multiple datasets (CIFAR-10, Food-101, ImageNet) with ResNet50 models.

## Features

- **Multiple Datasets**: CIFAR-10, Food-101, ImageNet
- **Multiple Imputation Methods**: Linear, Telea, NS, Fixed, Zero, GAIN
- **Both Ranking Strategies**: Sort-based and Threshold-based
- **Multiple Explanation Methods**: IG, GB, IG-SG, GB-SG, IG-SQ, GB-SQ, IG-VAR, GB-VAR
- **Comprehensive Analysis**: Accuracy tables, AUC metrics, Spearman correlations, Time complexity
- **Publication-Ready Outputs**: CSV, LaTeX, Markdown tables + PNG/PDF figures
- **Imputation Visualization**: Visual examples of how each imputation method modifies images

## Installation

### Prerequisites

```bash
# Create conda environment
conda create -n road python=3.9 -y
conda activate road

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt
```

### Required Packages

```
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.7.0
opencv-python>=4.5.0
captum>=0.5.0
tqdm>=4.62.0
```

## Quick Start

### Single Dataset Run

```bash
cd /path/to/road_imputation

# Set PYTHONPATH
export PYTHONPATH=$(pwd)

# Run CIFAR-10 with 200 test samples
python -m experiments.unified.run_pipeline \
    --dataset cifar10 \
    --test_subset_size 200 \
    --imputations linear telea ns \
    --stages explain benchmark analyze \
    --output_dir ./output
```

### Full Pipeline with All Stages

```bash
# Run all stages: train → explain → benchmark → analyze
python -m experiments.unified.run_pipeline \
    --dataset cifar10 \
    --stages all \
    --output_dir ./output
```

### With Imputation Visualization

```bash
# Generate imputation example figures (both sort and threshold variants)
python -m experiments.unified.run_pipeline \
    --dataset cifar10 \
    --stages analyze \
    --visualize_imputations \
    --output_dir ./output
```

## Full Experiment Commands

### CIFAR-10

```bash
python -m experiments.unified.run_pipeline \
    --dataset cifar10 \
    --imputations linear telea ns \
    --rankings sort threshold \
    --orders morf lerf \
    --explanation_methods ig gb ig_sg gb_sg \
    --stages explain benchmark analyze \
    --visualize_imputations \
    --output_dir ./output
```

### Food-101

```bash
python -m experiments.unified.run_pipeline \
    --dataset food101 \
    --data_path /path/to/food101 \
    --imputations linear telea ns \
    --rankings sort threshold \
    --stages explain benchmark analyze \
    --output_dir ./output
```

### ImageNet

```bash
python -m experiments.unified.run_pipeline \
    --dataset imagenet \
    --data_path /path/to/imagenet \
    --imputations linear telea ns \
    --test_subset_size 1000 \
    --stages explain benchmark analyze \
    --output_dir ./output
```

### Run All Datasets

```bash
python -m experiments.unified.run_pipeline --all_datasets --output_dir ./output
```

Or use the provided script:

```bash
./experiments/unified/run_all_experiments.sh
```

## Pipeline Stages

| Stage | Description | Output |
|-------|-------------|--------|
| `train` | Fine-tune ResNet50 on target dataset | `models/resnet50_{dataset}.pth` |
| `explain` | Generate saliency maps (IG, GB, etc.) | `explanations/{method}/` |
| `benchmark` | Run ROAD benchmark with all configurations | `results/noretrain.json` |
| `analyze` | Generate tables and figures | `tables/`, `figures/` |

### Stage Dependencies

```
train → explain → benchmark → analyze
```

You can skip earlier stages if outputs already exist:
- Skip `train` if `models/resnet50_{dataset}.pth` exists
- Skip `explain` if explanation pickles exist in `explanations/`
- Skip `benchmark` if `results/noretrain.json` exists

## CLI Arguments

### Dataset and Paths

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `cifar10` | Dataset: cifar10, food101, imagenet |
| `--data_path` | `./data` | Path to dataset directory |
| `--output_dir` | `./output` | Output directory for all results |

### Benchmark Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--imputations` | `linear fixed telea ns` | Imputation methods to benchmark |
| `--rankings` | `sort` | Ranking approaches: sort, threshold |
| `--orders` | `morf lerf` | Removal orders: morf, lerf |
| `--explanation_methods` | `ig gb ig_sg gb_sg` | Explanation methods |
| `--percentages` | `0.0 0.1 ... 0.9` | Pixel removal percentages |

### Efficiency Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--test_subset_size` | `None` | Limit test samples (None = use all) |
| `--batch_size` | `32` | Batch size for inference |

### Training Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | `10` | Training epochs for fine-tuning |
| `--lr` | `0.001` | Learning rate |
| `--force_retrain` | `False` | Force model retraining |

### Pipeline Control

| Argument | Default | Description |
|----------|---------|-------------|
| `--stages` | `all` | Stages: all, train, explain, benchmark, analyze |
| `--visualize_imputations` | `False` | Generate imputation visualization figures |

### Output Formats

| Argument | Default | Description |
|----------|---------|-------------|
| `--figure_format` | `png` | Figure format: png, pdf, svg |

## Output Structure

```
output/
└── cifar10/
    ├── models/
    │   └── resnet50_cifar10.pth
    ├── explanations/
    │   ├── ig/
    │   │   ├── base_test.pkl
    │   │   └── sg_test.pkl
    │   └── gb/
    │       └── ...
    ├── results/
    │   ├── noretrain.json
    │   └── metrics_cifar10.json
    ├── tables/
    │   ├── accuracy_linear_morf.csv
    │   ├── accuracy_linear_morf.tex
    │   ├── accuracy_linear_morf.md
    │   ├── auc_comparison.csv
    │   ├── spearman_morf_lerf.csv
    │   ├── time_complexity_cifar10.csv
    │   └── ...
    └── figures/
        ├── accuracy_linear_morf.png
        ├── morf_lerf_linear_ig_base.png
        ├── spearman_heatmap.png
        ├── imputation_examples_linear_sort_ig.png
        ├── imputation_examples_telea_threshold_ig.png
        └── ...
```

## Generated Tables

| Table | Description |
|-------|-------------|
| `accuracy_{imp}_{order}` | Accuracy at each percentage for imputation/order |
| `comparison_{method}_{modifier}_{order}` | Cross-imputation comparison |
| `ranking_{imp}` | Method rankings for each imputation |
| `auc_comparison` | Area Under Curve for all methods |
| `spearman_morf_lerf` | Spearman correlation between MoRF and LeRF |
| `time_complexity_{dataset}` | Execution time with relative percentages |
| `summary` | Overall summary statistics |

## Generated Figures

| Figure | Description |
|--------|-------------|
| `accuracy_{imp}_{order}` | Accuracy curves for all methods |
| `morf_lerf_{imp}_{method}_{modifier}` | MoRF vs LeRF comparison |
| `imputation_comparison_{method}_{modifier}_{order}` | Cross-imputation comparison |
| `spearman_heatmap` | Correlation heatmap |
| `ranking_boxplot_{imp}` | Method ranking distribution |
| `grid_comparison_{order}` | Multi-panel overview |
| `imputation_examples_{imp}_{ranking}_{expl}` | Visual imputation examples |

## Imputation Visualization

The `--visualize_imputations` flag generates grid figures showing:
- **Rows**: Removal percentages (10% to 90%)
- **Columns**: Original + each imputation method
- **Variants**: Both sort-based and threshold-based for each method

Example output: `imputation_examples_linear_sort_ig.png`

## Example Results

After running the pipeline, you can find:

1. **Accuracy Tables** (CSV/LaTeX/Markdown):
   ```
   tables/accuracy_linear_morf.csv
   tables/accuracy_linear_morf.tex
   ```

2. **Spearman Correlation Table**:
   ```
   tables/spearman_morf_lerf.csv
   ```

3. **Time Complexity Table**:
   ```
   tables/time_complexity_cifar10.csv
   ```

4. **Publication-Ready Figures**:
   ```
   figures/accuracy_linear_morf.png
   figures/spearman_heatmap.png
   ```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python -m experiments.unified.run_pipeline --batch_size 16 ...
```

### Missing Explanations

Force regeneration:
```bash
python -m experiments.unified.run_pipeline --force_explain --stages explain ...
```

### Missing Model

Train from scratch:
```bash
python -m experiments.unified.run_pipeline --force_retrain --stages train ...
```

## Citation

If you use this pipeline, please cite:

```bibtex
@article{rong2022road,
  title={A Consistent and Efficient Evaluation Strategy for Attribution Methods},
  author={Rong, Yao and others},
  journal={ICML},
  year={2022}
}
```

## License

MIT License - see LICENSE file for details.
