"""
Unified ROAD Benchmark Pipeline

A complete pipeline for benchmarking imputation techniques on CIFAR-10, Food-101, and ImageNet
using ResNet50 models with automatic model loading, explanation generation, and result analysis.
"""

from .config import DatasetConfig, DATASET_CONFIGS, parse_args
from .data import get_dataset, get_transforms, get_dataloader
from .models import load_resnet50, finetune_model
from .explanations import generate_explanations, load_explanations, save_explanations
from .benchmark import run_road_benchmark
from .metrics import (
    averaging_accuracy,
    calculate_spearman_morf_lerf,
    calculate_spearman_rank,
    calculate_pearson_correlation,
    compute_auc,
    rankdata
)
from .results import load_all_results, merge_results, ResultsDatabase
from .tables import generate_all_tables
from .figures import generate_all_figures

__version__ = "1.0.0"
__all__ = [
    "DatasetConfig",
    "DATASET_CONFIGS", 
    "parse_args",
    "get_dataset",
    "get_transforms",
    "get_dataloader",
    "load_resnet50",
    "finetune_model",
    "generate_explanations",
    "load_explanations",
    "save_explanations",
    "run_road_benchmark",
    "averaging_accuracy",
    "calculate_spearman_morf_lerf",
    "calculate_spearman_rank",
    "calculate_pearson_correlation",
    "compute_auc",
    "rankdata",
    "load_all_results",
    "merge_results",
    "ResultsDatabase",
    "generate_all_tables",
    "generate_all_figures",
]
