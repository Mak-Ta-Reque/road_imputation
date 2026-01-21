#!/usr/bin/env python3
"""
Unified ROAD Benchmark Pipeline

Main entry point for running ROAD benchmark experiments on CIFAR-10, Food-101, and ImageNet
with ResNet50 models from torch hub.

Usage:
    python run_pipeline.py --dataset cifar10 --test_subset_size 500
    python run_pipeline.py --dataset food101 --data_path /path/to/data --stages train explain benchmark
    python run_pipeline.py --dataset imagenet --imputations linear fixed telea --stages all
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional

import torch

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified.config import (
    DATASET_CONFIGS, 
    IMPUTATION_METHODS, 
    RANKING_METHODS, 
    REMOVAL_ORDERS,
    parse_args
)
from unified.data import get_dataset, get_dataloader
from unified.models import (
    load_resnet50, 
    finetune_model, 
    evaluate_model, 
    validate_model_accuracy,
    ModelAccuracyError,
    MIN_ACCURACY_THRESHOLD,
    DEFAULT_CACHE_DIR,
    is_model_cached,
    list_cached_models
)
from unified.explanations import generate_explanations, load_explanations
from unified.benchmark import run_road_benchmark
from unified.metrics import compute_all_metrics
from unified.results import ResultsDatabase, load_results, merge_results
from unified.tables import generate_all_tables
from unified.figures import generate_all_figures, generate_imputation_visualizations


def check_prerequisites(args) -> bool:
    """
    Check if model and dataset exist.
    If not, try to load from config file.
    
    Returns:
        True if all prerequisites are met
    """
    # Check dataset existence
    dataset_exists = False
    try:
        # Just check root path
        if os.path.exists(args.data_path):
            dataset_exists = True
        else:
            print(f"Data path not found: {args.data_path}")
    except Exception:
        pass
        
    # Check model existence (cache or path)
    model_exists = False
    if args.model_path and os.path.exists(args.model_path):
        model_exists = True
    else:
        # Check cache
        cache_dir = getattr(args, 'cache_dir', DEFAULT_CACHE_DIR)
        is_cached, _ = is_model_cached(args.dataset, cache_dir)
        if is_cached:
            model_exists = True
        elif args.dataset == 'imagenet':
             # ImageNet model is downloaded automatically/available via torchvision
             model_exists = True
             
    if dataset_exists and model_exists:
        return True
        
    # If missing, prompt for config file if interactive (simple check for TTY)
    if not args.config_file and sys.stdin.isatty():
        print("\nMissing dataset or model.")
        print("Please provide a path to a YAML config file:")
        config_path = input("Config file path: ").strip()
        if config_path:
            args.config_file = config_path
            # Reload args using parse_args logic (need to re-run parsing or simulate it)
            # Simulating update here for simplicity since we can't easily re-run parse_args
            import yaml
            try:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                for k, v in config_data.items():
                    setattr(args, k, v)
                return check_prerequisites(args) # Recursive check
            except Exception as e:
                print(f"Failed to load config: {e}")
                return False
    
    return False




def setup_directories(output_dir: str) -> Dict[str, str]:
    """Create output directories."""
    dirs = {
        'root': output_dir,
        'models': os.path.join(output_dir, 'models'),
        'explanations': os.path.join(output_dir, 'explanations'),
        'results': os.path.join(output_dir, 'results'),
        'tables': os.path.join(output_dir, 'tables'),
        'figures': os.path.join(output_dir, 'figures'),
    }
    
    for name, path in dirs.items():
        os.makedirs(path, exist_ok=True)
    
    return dirs


def run_training_stage(args, dirs: Dict[str, str], device: torch.device) -> str:
    """Run model training/finetuning stage."""
    print("\n" + "="*60)
    print("STAGE: Model Training/Loading")
    print("="*60)
    
    config = DATASET_CONFIGS[args.dataset]
    model_name = getattr(args, 'model_name', 'resnet50')
    output_model_path = os.path.join(dirs['models'], f'{model_name}_{args.dataset}.pth')
    cache_dir = getattr(args, 'cache_dir', DEFAULT_CACHE_DIR)

    # Model selection logic
    def load_model_by_name(model_name, config, device, model_path=None, cache_dir=None):
        if model_name == 'resnet56':
            import sys
            sys.path.append('./road/gisp/models')
            from resnet import ResNet
            def resnet56(num_classes):
                return ResNet(BasicBlock, [9, 9, 9, 9], num_classes)
            model = resnet56(num_classes=config.num_classes)
            model = model.to(device)
            if model_path and os.path.exists(model_path):
                from unified.models import load_custom_weights
                load_custom_weights(model, model_path, device)
            return model
        if model_name == 'resnet50' and config.name == 'imagenet':
            from torchvision.models import resnet50, ResNet50_Weights
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            model = model.to(device)
            return model
        # Default to resnet50
        return load_resnet50(config=config, device=device, model_path=model_path, cache_dir=cache_dir)

    model = None
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from model_path: {args.model_path}")
        model = load_model_by_name(model_name, config, device, model_path=args.model_path, cache_dir=cache_dir)
    elif os.path.exists(output_model_path) and not args.force_retrain:
        print(f"Loading existing model from output directory: {output_model_path}")
        model = load_model_by_name(model_name, config, device, model_path=output_model_path, cache_dir=cache_dir)
    else:
        print("No existing model found, will train/fine-tune if required.")
        model = load_model_by_name(model_name, config, device, cache_dir=cache_dir)

    # Train if needed
    if (not args.model_path or not os.path.exists(args.model_path)) and (not os.path.exists(output_model_path) or args.force_retrain):
        if args.dataset != 'imagenet':
            print(f"Fine-tuning model for {args.dataset}...")
            train_dataset = get_dataset(args.dataset, args.data_path, train=True)
            train_loader = get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_dataset = get_dataset(args.dataset, args.data_path, train=False)
            val_loader = get_dataloader(val_dataset, batch_size=args.batch_size)
            model = finetune_model(
                model,
                train_loader,
                val_loader,
                device=device,
                config=config,
                epochs=args.epochs,
                lr=args.lr
            )
            torch.save({
                'model_state_dict': model.state_dict(),
                'dataset': args.dataset,
                'num_classes': config.num_classes,
            }, output_model_path)
            print(f"Saved model: {output_model_path}")
        else:
            print("Using ImageNet pretrained model (no fine-tuning needed)")
    
    # Evaluate model
    print("\nEvaluating model...")
    test_dataset = get_dataset(args.dataset, args.data_path, train=False)
    test_loader = get_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        subset_size=args.test_subset_size
    )
    accuracy = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    return output_model_path


def run_explanation_stage(args, dirs: Dict[str, str], device: torch.device) -> str:
    """Run explanation generation stage."""
    print("\n" + "="*60)
    print("STAGE: Explanation Generation")
    print("="*60)
    
    config = DATASET_CONFIGS[args.dataset]
    model_name = getattr(args, 'model_name', 'resnet50')
    cache_dir = getattr(args, 'cache_dir', DEFAULT_CACHE_DIR)
    output_model_path = os.path.join(dirs['models'], f'{model_name}_{args.dataset}.pth')

    def load_model_by_name(model_name, config, device, model_path=None, cache_dir=None):
        if model_name == 'resnet56':
            import sys
            sys.path.append('./road/gisp/models')
            from resnet import ResNet
            def resnet56(num_classes):
                return ResNet(BasicBlock, [9, 9, 9, 9], num_classes)
            model = resnet56(num_classes=config.num_classes)
            model = model.to(device)
            if model_path and os.path.exists(model_path):
                from unified.models import load_custom_weights
                load_custom_weights(model, model_path, device)
            return model
        if model_name == 'resnet50' and config.name == 'imagenet':
            from torchvision.models import resnet50, ResNet50_Weights
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            model = model.to(device)
            return model
        return load_resnet50(config=config, device=device, model_path=model_path, cache_dir=cache_dir)

    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from model_path: {args.model_path}")
        model = load_model_by_name(model_name, config, device, model_path=args.model_path, cache_dir=cache_dir)
    elif os.path.exists(output_model_path):
        print(f"Loading model from output directory: {output_model_path}")
        model = load_model_by_name(model_name, config, device, model_path=output_model_path, cache_dir=cache_dir)
    else:
        print("No existing model found, loading default model.")
        model = load_model_by_name(model_name, config, device, cache_dir=cache_dir)
    
    # Prepare test loader for validation
    test_dataset_eval = get_dataset(args.dataset, args.data_path, train=False)
    test_loader = get_dataloader(
        test_dataset_eval,
        batch_size=args.batch_size,
        subset_size=min(args.test_subset_size or 1000, 1000)
    )
    
    # Validate model accuracy before generating explanations
    skip_check = getattr(args, 'skip_accuracy_check', False)
    min_acc = getattr(args, 'min_accuracy', MIN_ACCURACY_THRESHOLD)
    
    if skip_check:
        print("\nWARNING: Skipping model accuracy validation (--skip_accuracy_check)")
        accuracy = evaluate_model(model, test_loader, device)
        print(f"Model accuracy: {accuracy:.2f}% (not validated)")
    else:
        print("\nValidating model accuracy before explanation generation...")
        try:
            accuracy = validate_model_accuracy(
                model, 
                test_loader, 
                device,
                min_accuracy=min_acc,
                dataset_name=args.dataset
            )
            print(f"Model validation passed! Accuracy: {accuracy:.2f}%")
        except ModelAccuracyError as e:
            print(str(e))
            print(f"\nTo fix this, run: python -m experiments.unified.run_pipeline --dataset {args.dataset} --stages train")
            raise
    
    # Get test data
    test_dataset = get_dataset(args.dataset, args.data_path, train=False)
    
    expl_dir = dirs['explanations']
    
    # Generate or load explanations for each method
    for expl_method in args.explanation_methods:
        expl_save_dir = os.path.join(expl_dir, expl_method, 'explanation', 'test')
        
        if os.path.exists(expl_save_dir) and not args.force_explain:
            print(f"Explanations already exist for {expl_method}, skipping...")
        else:
            print(f"Generating {expl_method} explanations...")
            generate_explanations(
                model=model,
                dataset=test_dataset,
                device=device,
                expl_method=expl_method,
                save_dir=expl_dir,
                split='test',
                subset_size=args.test_subset_size
            )
            print(f"Saved {expl_method} explanations")
    
    return expl_dir


def run_benchmark_stage(args, dirs: Dict[str, str], device: torch.device) -> str:
    """Run ROAD benchmark stage."""
    print("\n" + "="*60)
    print("STAGE: ROAD Benchmark")
    print("="*60)
    
    config = DATASET_CONFIGS[args.dataset]
    model_name = getattr(args, 'model_name', 'resnet50')
    cache_dir = getattr(args, 'cache_dir', DEFAULT_CACHE_DIR)
    output_model_path = os.path.join(dirs['models'], f'{model_name}_{args.dataset}.pth')

    def load_model_by_name(model_name, config, device, model_path=None, cache_dir=None):
        if model_name == 'resnet56':
            import sys
            sys.path.append('./road/gisp/models')
            from resnet import ResNet
            def resnet56(num_classes):
                return ResNet(BasicBlock, [9, 9, 9, 9], num_classes)
            model = resnet56(num_classes=config.num_classes)
            model = model.to(device)
            if model_path and os.path.exists(model_path):
                from unified.models import load_custom_weights
                load_custom_weights(model, model_path, device)
            return model
        if model_name == 'resnet50' and config.name == 'imagenet':
            from torchvision.models import resnet50, ResNet50_Weights
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            model = model.to(device)
            return model
        return load_resnet50(config=config, device=device, model_path=model_path, cache_dir=cache_dir)

    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from model_path: {args.model_path}")
        model = load_model_by_name(model_name, config, device, model_path=args.model_path, cache_dir=cache_dir)
    elif os.path.exists(output_model_path):
        print(f"Loading model from output directory: {output_model_path}")
        model = load_model_by_name(model_name, config, device, model_path=output_model_path, cache_dir=cache_dir)
    else:
        print("No existing model found, loading default model.")
        model = load_model_by_name(model_name, config, device, cache_dir=cache_dir)
    
    # Prepare test loader for validation
    test_dataset_eval = get_dataset(args.dataset, args.data_path, train=False)
    test_loader = get_dataloader(
        test_dataset_eval,
        batch_size=args.batch_size,
        subset_size=min(args.test_subset_size or 1000, 1000)  # Use up to 1000 samples for validation
    )
    
    # CRITICAL: Validate model accuracy before running benchmark
    skip_check = getattr(args, 'skip_accuracy_check', False)
    min_acc = getattr(args, 'min_accuracy', MIN_ACCURACY_THRESHOLD)
    
    if skip_check:
        print("\nWARNING: Skipping model accuracy validation (--skip_accuracy_check)")
        accuracy = evaluate_model(model, test_loader, device)
        print(f"Model accuracy: {accuracy:.2f}% (not validated)")
    else:
        print("\nValidating model accuracy before benchmark...")
        try:
            accuracy = validate_model_accuracy(
                model, 
                test_loader, 
                device,
                min_accuracy=min_acc,
                dataset_name=args.dataset
            )
            print(f"Model validation passed! Accuracy: {accuracy:.2f}%")
        except ModelAccuracyError as e:
            print(str(e))
            print(f"\nTo fix this, run: python -m experiments.unified.run_pipeline --dataset {args.dataset} --stages train")
            raise
    
    # Get normalize transform for after imputation
    from unified.data import get_normalize_transform, get_subset_dataset
    normalize_transform = get_normalize_transform(config)
    
    # Load explanations for each method
    expl_dir = dirs['explanations']
    explanations = {}
    num_explanations = None
    for expl_method in args.explanation_methods:
        try:
            expl_list, _ = load_explanations(expl_dir, expl_method, split='test')
            explanations[expl_method] = expl_list
            num_explanations = len(expl_list)
            print(f"Loaded {len(expl_list)} explanations for {expl_method}")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
    
    if not explanations:
        raise RuntimeError("No explanations found. Run explanation stage first.")
    
    # Get test data with tensor-only transform (for imputation)
    # Important: Create subset matching explanations count
    test_dataset = get_dataset(args.dataset, args.data_path, train=False, tensor_only=True)
    if num_explanations is not None and num_explanations < len(test_dataset):
        print(f"Creating subset of {num_explanations} samples to match explanations")
        test_dataset = get_subset_dataset(test_dataset, num_explanations)
    
    # Run benchmark
    results = run_road_benchmark(
        model=model,
        dataset=test_dataset,
        explanations=explanations,
        normalize_transform=normalize_transform,
        config=config,
        output_dir=dirs['results'],
        percentages=args.percentages,
        imputations=args.imputations,
        rankings=args.rankings,
        orders=args.orders,
        batch_size=args.batch_size,
        device=device
    )
    
    results_path = os.path.join(dirs['results'], 'noretrain.json')
    print(f"Results saved: {results_path}")
    
    return results_path


def run_analysis_stage(args, dirs: Dict[str, str]) -> None:
    """Run analysis stage (tables and figures)."""
    print("\n" + "="*60)
    print("STAGE: Analysis & Visualization")
    print("="*60)
    
    # Load results
    results_path = os.path.join(dirs['results'], 'noretrain.json')
    
    if not os.path.exists(results_path):
        print(f"Results not found: {results_path}")
        print("Run benchmark stage first.")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_all_metrics(results)
    
    metrics_path = os.path.join(dirs['results'], f'metrics_{args.dataset}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")
    
    # Generate tables
    print("Generating tables...")
    table_files = generate_all_tables(results, dirs['tables'])
    for name, path in table_files.items():
        print(f"  - {name}: {path}")
    
    # Generate figures
    print("Generating figures...")
    figure_files = generate_all_figures(results, dirs['figures'], format=args.figure_format)
    print(f"Generated {len(figure_files)} figures")
    
    # Generate imputation visualizations if requested
    if getattr(args, 'visualize_imputations', False):
        print("Generating imputation visualizations...")
        _generate_imputation_viz(args, dirs, results)


def _generate_imputation_viz(args, dirs: Dict[str, str], results: Dict[str, Any]) -> None:
    """Generate imputation visualization figures."""
    import pickle
    import torchvision
    import torchvision.transforms as transforms
    from road.utils import load_expl
    
    imputations = results.get('imputations', args.imputations)
    base_methods = results.get('base_methods', ['ig', 'gb'])
    
    # Try to load dataset
    try:
        from unified.config import DATASET_CONFIGS
        config = DATASET_CONFIGS[args.dataset]
        
        transform_tensor = torchvision.transforms.Compose([transforms.ToTensor()])
        
        if args.dataset == 'cifar10':
            dataset = torchvision.datasets.CIFAR10(
                root=args.data_path, train=False, download=True, transform=transform_tensor
            )
        elif args.dataset == 'cifar100':
            dataset = torchvision.datasets.CIFAR100(
                root=args.data_path, train=False, download=True, transform=transform_tensor
            )
        elif args.dataset == 'food101':
            dataset = torchvision.datasets.Food101(
                root=args.data_path, split='test', download=True, transform=transform_tensor
            )
        elif args.dataset == 'imagenet':
            from torchvision.datasets import ImageFolder
            dataset = ImageFolder(
                root=args.data_path, transform=transform_tensor
            )
        else:
            print(f"Dataset {args.dataset} not supported for visualization yet.")
            return
        
        print(f"Loaded dataset: {len(dataset)} samples")
    except Exception as e:
        print(f"Could not load dataset for visualization: {e}")
        return
    
    # Try to load explanations
    for method in base_methods:
        expl_dir = os.path.join(dirs['explanations'], method)
        expl_files = [
            os.path.join(expl_dir, 'base_test.pkl'),
            os.path.join(expl_dir, 'var_test.pkl'),
            os.path.join(expl_dir, f'{method}_test.pkl'),
        ]
        
        explanation_mask = None
        expl_name = method
        
        for expl_file in expl_files:
            if os.path.exists(expl_file):
                try:
                    _, explanation_mask, _, _ = load_expl(None, expl_file)
                    expl_name = os.path.basename(expl_file).replace('_test.pkl', '')
                    print(f"Loaded explanations from: {expl_file}")
                    break
                except Exception as e:
                    print(f"Could not load {expl_file}: {e}")
        
        if explanation_mask is None:
            print(f"No explanation files found for {method}, skipping visualization.")
            continue
        
        # Determine image IDs
        viz_samples = getattr(args, 'viz_samples', 3)
        import numpy as np
        np.random.seed(42)
        max_idx = min(len(dataset), len(explanation_mask))
        image_ids = np.random.choice(max_idx, size=min(viz_samples, max_idx), replace=False).tolist()
        
        # Generate visualizations
        try:
            viz_files = generate_imputation_visualizations(
                base_dataset=dataset,
                explanation_mask=explanation_mask,
                imputation_names=imputations,
                explanation_name=expl_name,
                output_dir=dirs['figures'],
                image_ids=image_ids,
                format=args.figure_format
            )
            print(f"Generated {len(viz_files)} imputation visualization figures for {method}")
        except Exception as e:
            print(f"Error generating visualizations for {method}: {e}")


def run_all_datasets(args) -> None:
    """Run pipeline for all datasets."""
    datasets = ['cifar10', 'food101', 'imagenet']
    
    for dataset in datasets:
        print(f"\n{'#'*70}")
        print(f"# Processing: {dataset.upper()}")
        print(f"{'#'*70}")
        
        args.dataset = dataset
        run_pipeline(args)


def run_pipeline(args) -> None:
    """Run the complete pipeline."""
    print(f"\n{'='*70}")
    print(f"ROAD Benchmark Pipeline - {args.dataset.upper()}")
    print(f"{'='*70}")
    
    # Check prerequisites
    if not check_prerequisites(args):
        # One last check: if we are supposed to TRAIN, maybe we don't need the model yet.
        # But we definitely need data.
        if 'train' not in args.stages:
             print("Prerequisites not met (dataset or model missing) and training is not requested.")
             print("Aborting.")
             sys.exit(1)
             
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    output_dir = os.path.join(args.output_dir, args.dataset)
    dirs = setup_directories(output_dir)
    print(f"Output directory: {output_dir}")
    
    # Determine stages to run
    stages = args.stages
    if 'all' in stages:
        stages = ['train', 'explain', 'benchmark', 'analyze']
        
    # Pre-flight Accuracy Check
    # If we are NOT explicitly training in this run, we must check if the current model is good enough.
    if 'train' not in stages:
        print("\nPre-flight: Checking model accuracy...")
        try:
             # Load model specifically for checking
            config = DATASET_CONFIGS[args.dataset]
            cache_dir = getattr(args, 'cache_dir', DEFAULT_CACHE_DIR)
            
            # This logic basically duplicates what's in run_training_stage but we need to check BEFORE deciding flow
            model = load_resnet50(config, device, args.model_path, cache_dir)
            
            # Load checkpoint if exists
            model_path_local = os.path.join(dirs['models'], f'resnet50_{args.dataset}.pth')
            load_path = args.model_path or (model_path_local if os.path.exists(model_path_local) else None)
            
            if load_path:
                 try:
                    load_custom_weights(model, load_path, device)
                 except: 
                    pass # load_resnet50 handles basics but we double check local output dir
            
            # Get small test set for quick valid
            test_dataset = get_dataset(args.dataset, args.data_path, train=False)
            # Use same subset size or at least 100 for valid
            val_subset = args.test_subset_size or 500
            test_loader = get_dataloader(test_dataset, batch_size=args.batch_size, subset_size=val_subset)
            
            acc = evaluate_model(model, test_loader, device)
            print(f"Current Model Accuracy: {acc:.2f}%")
            
            min_acc = getattr(args, 'min_accuracy', 50.0)
            if acc < min_acc:
                print(f"Accuracy below threshold ({min_acc}%).")
                print("Initiating fine-tuning stage...")
                stages.insert(0, 'train') # Add training to start of list
                args.force_retrain = True # Ensure we actually train
            else:
                 print("Accuracy check passed.")
                 
        except Exception as e:
            print(f"Warning during pre-flight check: {e}")
            # If check fails badly, we might just want to proceed and let stages handle it, 
            # OR assume we need to train.
            # For now, let's err on side of safety and add 'train' if it seems model is broken/missing
            if 'train' not in stages:
                print("Model seems missing or broken, adding 'train' stage.")
                stages.insert(0, 'train')

    
    print(f"Stages to run: {stages}")
    print(f"Imputations: {args.imputations}")
    print(f"Test subset size: {args.test_subset_size}")
    
    # Run stages
    if 'train' in stages:
        run_training_stage(args, dirs, device)
    
    if 'explain' in stages:
        run_explanation_stage(args, dirs, device)
    
    if 'benchmark' in stages:
        run_benchmark_stage(args, dirs, device)
    
    if 'analyze' in stages:
        run_analysis_stage(args, dirs)
    
    print(f"\n{'='*70}")
    print("Pipeline Complete!")
    print(f"{'='*70}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Unified ROAD Benchmark Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline for CIFAR-10
    python run_pipeline.py --dataset cifar10 --test_subset_size 500
    
    # Run only benchmark and analysis for Food-101
    python run_pipeline.py --dataset food101 --stages benchmark analyze
    
    # Run with specific imputations
    python run_pipeline.py --dataset imagenet --imputations linear fixed telea
    
    # Run all datasets
    python run_pipeline.py --all_datasets
        """
    )
    
    # Dataset options
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'food101', 'imagenet'],
                        help='Dataset to process')
    parser.add_argument('--data_path', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--all_datasets', action='store_true',
                        help='Run pipeline for all datasets')
    
    # Model options
    parser.add_argument('--epochs', type=int, default=10,
                        help='Training epochs for fine-tuning')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--force_retrain', action='store_true',
                        help='Force model retraining')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pretrained model weights (optional)')
    parser.add_argument('--expl_path', type=str, default=None,
                        help='Path to precomputed explanations (optional)')
    
    # Explanation options
    parser.add_argument('--explanation_methods', type=str, nargs='+',
                        default=['ig', 'gb', 'ig_sg', 'gb_sg'],
                        help='Explanation methods to use')
    parser.add_argument('--force_explain', action='store_true',
                        help='Force explanation regeneration')
    
    # Benchmark options
    parser.add_argument('--test_subset_size', type=int, default=None,
                        help='Number of test samples (None for all)')
    parser.add_argument('--imputations', type=str, nargs='+',
                        default=['linear', 'fixed', 'telea', 'ns'],
                        help='Imputation methods')
    parser.add_argument('--rankings', type=str, nargs='+',
                        default=['sort'],
                        help='Ranking methods (sort, threshold)')
    parser.add_argument('--orders', type=str, nargs='+',
                        default=['morf', 'lerf'],
                        help='Removal orders')
    parser.add_argument('--percentages', type=float, nargs='+',
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        help='Percentages of pixels to remove')
    
    # Stage control
    parser.add_argument('--stages', type=str, nargs='+',
                        default=['all'],
                        choices=['all', 'train', 'explain', 'benchmark', 'analyze'],
                        help='Pipeline stages to run')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory')
    parser.add_argument('--cache_dir', type=str, default='./model_cache',
                        help='Directory for caching models')
    parser.add_argument('--figure_format', type=str, default='png',
                        choices=['png', 'pdf', 'svg'],
                        help='Figure output format')
    
    # Model validation options
    parser.add_argument('--min_accuracy', type=float, default=50.0,
                        help='Minimum model accuracy required to run benchmark (%%)')
    parser.add_argument('--skip_accuracy_check', action='store_true',
                        help='Skip model accuracy validation (not recommended)')
    parser.add_argument('--list_cached_models', action='store_true',
                        help='List all cached models and exit')
    
    # Visualization options
    parser.add_argument('--visualize_imputations', action='store_true',
                        help='Generate imputation example visualizations')
    parser.add_argument('--viz_samples', type=int, default=3,
                        help='Number of sample images for visualization')
    
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
    
    # Handle list_cached_models option
    if args.list_cached_models:
        list_cached_models(args.cache_dir)
        return
    
    if args.all_datasets:
        run_all_datasets(args)
    else:
        run_pipeline(args)


if __name__ == '__main__':
    main()
