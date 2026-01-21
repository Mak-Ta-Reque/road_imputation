"""
Configuration module for unified ROAD benchmark pipeline.
Defines dataset configurations, imputation methods, and CLI arguments.
"""

import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


@dataclass
class DatasetConfig:
    """Configuration for a specific dataset."""
    name: str
    num_classes: int
    image_size: Tuple[int, int]
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    input_channels: int = 3
    

# Predefined dataset configurations
DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    'cifar10': DatasetConfig(
        name='cifar10',
        num_classes=10,
        image_size=(32, 32),
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
    'cifar100': DatasetConfig(
        name='cifar100',
        num_classes=100,
        image_size=(32, 32),
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
    ),
    'food101': DatasetConfig(
        name='food101',
        num_classes=101,
        image_size=(224, 224),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    'imagenet': DatasetConfig(
        name='imagenet',
        num_classes=1000,
        image_size=(224, 224),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
}

# Available imputation methods
IMPUTATION_METHODS = ['linear', 'fixed', 'telea', 'ns', 'zero', 'gain']

# Ranking approaches
RANKING_METHODS = ['sort', 'threshold']

# Removal orders
REMOVAL_ORDERS = ['morf', 'lerf']

# Explanation methods
EXPLANATION_METHODS = ['ig', 'gb', 'ig_sg', 'gb_sg', 'ig_sq', 'gb_sq', 'ig_var', 'gb_var']

# Default percentages for benchmarking
DEFAULT_PERCENTAGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the unified pipeline."""
    parser = argparse.ArgumentParser(
        description='Unified ROAD Benchmark Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ============== Dataset and Paths ==============
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset to benchmark')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset root directory')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directory to save all outputs')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pretrained model weights (optional)')
    parser.add_argument('--expl_path', type=str, default=None,
                        help='Path to precomputed explanations (optional)')
    parser.add_argument('--cache_dir', type=str, default='./model_cache',
                        help='Directory for caching models')
    
    # ============== Benchmark Parameters ==============
    parser.add_argument('--imputations', type=str, nargs='+', 
                        default=['linear', 'fixed', 'telea', 'ns'],
                        choices=IMPUTATION_METHODS,
                        help='Imputation methods to benchmark')
    parser.add_argument('--rankings', type=str, nargs='+',
                        default=['sort'],
                        choices=RANKING_METHODS,
                        help='Ranking approaches (sort or threshold)')
    parser.add_argument('--orders', type=str, nargs='+',
                        default=['morf', 'lerf'],
                        choices=REMOVAL_ORDERS,
                        help='Removal orders to test')
    parser.add_argument('--expl_methods', type=str, nargs='+',
                        default=['ig', 'gb'],
                        choices=EXPLANATION_METHODS,
                        help='Explanation methods to use')
    parser.add_argument('--modifiers', type=str, nargs='+',
                        default=['base', 'sg', 'sq', 'var'],
                        help='Explanation modifiers')
    parser.add_argument('--percentages', type=float, nargs='+',
                        default=DEFAULT_PERCENTAGES,
                        help='Percentage values for pixel removal')
    parser.add_argument('--n_runs', type=int, default=5,
                        help='Number of runs for stochastic benchmarks (averaging)')
    
    # ============== Subset and Efficiency ==============
    parser.add_argument('--test_subset_size', type=int, default=None,
                        help='Limit test set size for faster benchmarking (None=use all)')
    parser.add_argument('--train_subset_size', type=int, default=None,
                        help='Limit train set size for explanation generation')
    
    # ============== Training/Fine-tuning ==============
    parser.add_argument('--finetune', action='store_true',
                        help='Force fine-tuning even if pretrained exists')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Epochs for fine-tuning')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training and inference')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Data loading workers')
    
    # ============== Hardware ==============
    parser.add_argument('--gpu', action='store_true', default=True,
                        help='Use GPU if available')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU device ID')
    
    # ============== Pipeline Stages ==============
    parser.add_argument('--stages', type=str, nargs='+',
                        default=['explain', 'benchmark', 'metrics', 'tables', 'figures'],
                        choices=['explain', 'benchmark', 'metrics', 'tables', 'figures', 'all'],
                        help='Pipeline stages to run')
    
    # ============== Output Formats ==============
    parser.add_argument('--table_formats', type=str, nargs='+',
                        default=['csv', 'latex', 'markdown'],
                        choices=['csv', 'latex', 'markdown'],
                        help='Table output formats')
    parser.add_argument('--figure_format', type=str, default='png',
                        choices=['png', 'pdf', 'svg'],
                        help='Figure output format')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Figure DPI')
    
    # ============== Visualization ==============
    parser.add_argument('--visualize_imputations', action='store_true',
                        help='Generate imputation example visualizations')
    parser.add_argument('--viz_samples', type=int, default=3,
                        help='Number of sample images for imputation visualization')
    
    # ============== Misc ==============
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_runs', type=int, default=5,
                        help='Number of runs per configuration')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--list_cached_models', action='store_true',
                        help='List all cached models and exit')
    parser.add_argument('--min_accuracy', type=float, default=50.0,
                        help='Minimum model accuracy required to run benchmark (%%)')
    parser.add_argument('--skip_accuracy_check', action='store_true',
                        help='Skip model accuracy validation (not recommended)')
    
    parser.add_argument('--config_file', type=str, default=None,
                        help='Path to YAML configuration file')
    
    args = parser.parse_args()
    
    # Load config from YAML if provided
    if args.config_file:
        import yaml
        try:
            with open(args.config_file, 'r') as f:
                config_args = yaml.safe_load(f)
                
            # Update args with config values (CLI args take precedence)
            for key, value in config_args.items():
                if not hasattr(args, key) or getattr(args, key) == parser.get_default(key):
                    setattr(args, key, value)
        except Exception as e:
            print(f"Error loading config file: {e}")
            import sys; sys.exit(1)
            
    return args


def get_config(dataset_name: str) -> DatasetConfig:
    """Get dataset configuration by name."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Available: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset_name]
