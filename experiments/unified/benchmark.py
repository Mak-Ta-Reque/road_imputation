"""
Benchmark runner module for unified ROAD benchmark pipeline.
Runs ROAD benchmarks with various imputation methods and ranking approaches.
"""

import os
import sys
import json
import time
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

# Add parent path for road module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from road import run_road
from road.imputations import (
    NoisyLinearImputer, 
    ChannelMeanImputer, 
    ImpaintingImputation,
    ImpaintingImputationNS,
    ZeroImputer,
    GAINImputer
)

from .config import DatasetConfig


def get_imputer(imputation_name: str, device: torch.device = None, gain_model_path: str = None):
    """
    Get imputer object from name.
    
    Args:
        imputation_name: Name of imputation method
        device: Device for GAIN imputer
        gain_model_path: Path to GAIN model weights
    
    Returns:
        Imputer object
    """
    if imputation_name == 'linear':
        return NoisyLinearImputer(noise=0.01)
    elif imputation_name == 'fixed':
        return ChannelMeanImputer()
    elif imputation_name == 'telea':
        return ImpaintingImputation()
    elif imputation_name == 'ns':
        return ImpaintingImputationNS()
    elif imputation_name == 'zero':
        return ZeroImputer()
    elif imputation_name == 'gain':
        if gain_model_path is None:
            raise ValueError("GAIN imputer requires gain_model_path")
        device_str = str(device) if device else 'cpu'
        return GAINImputer(gain_model_path, device_str)
    else:
        raise ValueError(f"Unknown imputation method: {imputation_name}")


def run_single_benchmark(
    model: torch.nn.Module,
    dataset: Dataset,
    explanations: List[np.ndarray],
    normalize_transform,
    percentages: List[float],
    imputation_name: str,
    ranking: str = 'sort',
    morf: bool = True,
    batch_size: int = 32,
    device: torch.device = None,
    gain_model_path: str = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run a single ROAD benchmark configuration.
    
    Args:
        model: Pretrained model
        dataset: Test dataset
        explanations: List of explanation arrays
        normalize_transform: Normalization transform
        percentages: List of removal percentages
        imputation_name: Imputation method name
        ranking: 'sort' or 'threshold'
        morf: True for MoRF, False for LeRF
        batch_size: Batch size
        device: Computation device
        gain_model_path: Path to GAIN model (if using)
    
    Returns:
        Tuple of (accuracy tensor, probability tensor)
    """
    imputer = get_imputer(imputation_name, device, gain_model_path)
    
    order_str = "MoRF" if morf else "LeRF"
    print(f"\nRunning: {imputation_name} | {ranking} | {order_str}")
    print(f"Percentages: {percentages}")
    
    start_time = time.time()
    
    res_acc, prob_acc = run_road(
        model=model,
        dataset_test=dataset,
        explanations_test=explanations,
        transform_test=normalize_transform,
        percentages=percentages,
        morf=morf,
        batch_size=batch_size,
        imputation=imputer,
        ranking=ranking
    )
    
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.1f}s | Accuracies: {res_acc.numpy()}")
    
    return res_acc, prob_acc


def run_road_benchmark(
    model: torch.nn.Module,
    dataset: Dataset,
    explanations: Dict[str, List[np.ndarray]],
    normalize_transform,
    config: DatasetConfig,
    output_dir: str,
    percentages: List[float] = None,
    imputations: List[str] = None,
    rankings: List[str] = None,
    orders: List[str] = None,
    batch_size: int = 32,
    device: torch.device = None,
    num_runs: int = 5,
    seeds: List[int] = None,
    gain_model_path: str = None
) -> Dict[str, Any]:
    """
    Run complete ROAD benchmark suite.
    
    Args:
        model: Pretrained model
        dataset: Test dataset
        explanations: Dict mapping expl_method -> list of explanations
        normalize_transform: Normalization transform
        config: Dataset configuration
        output_dir: Directory to save results
        percentages: List of removal percentages
        imputations: List of imputation methods
        rankings: List of ranking approaches
        orders: List of removal orders
        batch_size: Batch size
        device: Computation device
        num_runs: Number of runs per configuration
        seeds: Random seeds for runs
        gain_model_path: Path to GAIN model
    
    Returns:
        Results dictionary
    """
    # Defaults
    if percentages is None:
        percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if imputations is None:
        imputations = ['linear', 'fixed', 'telea', 'ns']
    if rankings is None:
        rankings = ['sort']
    if orders is None:
        orders = ['morf', 'lerf']
    if seeds is None:
        seeds = [2005, 42, 1515, 3333, 420]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results structure
    results = {
        'percentages': percentages,
        'base_methods': list(set(m.split('_')[0] if '_' in m else m for m in explanations.keys())),
        'modifiers': list(set(m.split('_')[1] if '_' in m else 'base' for m in explanations.keys())),
        'dataset': config.name,
        'imputations': imputations,
        'orders': orders,
        'rankings': rankings,
        'timestamp': datetime.now().isoformat()
    }
    
    # Run benchmarks
    for imputation in imputations:
        if imputation not in results:
            results[imputation] = {}
        
        for ranking in rankings:
            if ranking not in results[imputation]:
                results[imputation][ranking] = {}
            
            for expl_method, expl_list in explanations.items():
                # Parse method
                if '_' in expl_method:
                    base, modifier = expl_method.split('_', 1)
                else:
                    base, modifier = expl_method, 'base'
                
                if base not in results[imputation][ranking]:
                    results[imputation][ranking][base] = {}
                if modifier not in results[imputation][ranking][base]:
                    results[imputation][ranking][base][modifier] = {}
                
                for order in orders:
                    morf = (order == 'morf')
                    order_key = 'morf' if morf else 'lerf'
                    
                    if order_key not in results[imputation][ranking][base][modifier]:
                        results[imputation][ranking][base][modifier][order_key] = {}
                    
                    print(f"\n{'='*60}")
                    print(f"Benchmark: {imputation} | {ranking} | {expl_method} | {order}")
                    print(f"{'='*60}")
                    
                    for perc in percentages:
                        perc_key = str(perc)
                        if perc_key not in results[imputation][ranking][base][modifier][order_key]:
                            results[imputation][ranking][base][modifier][order_key][perc_key] = []
                        
                        for run_id in range(num_runs):
                            torch.manual_seed(seeds[run_id % len(seeds)])
                            
                            try:
                                res_acc, _ = run_single_benchmark(
                                    model=model,
                                    dataset=dataset,
                                    explanations=expl_list,
                                    normalize_transform=normalize_transform,
                                    percentages=[perc],
                                    imputation_name=imputation,
                                    ranking=ranking,
                                    morf=morf,
                                    batch_size=batch_size,
                                    device=device,
                                    gain_model_path=gain_model_path
                                )
                                
                                acc_value = res_acc[0].item()
                                results[imputation][ranking][base][modifier][order_key][perc_key].append(acc_value)
                                
                            except Exception as e:
                                print(f"Error in run {run_id}: {e}")
                                results[imputation][ranking][base][modifier][order_key][perc_key].append(f"error: {str(e)}")
                    
                    # Save intermediate results
                    save_results(results, output_dir)
    
    return results


def save_results(results: Dict[str, Any], output_dir: str, filename: str = 'noretrain.json') -> None:
    """Save results to JSON file."""
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {filepath}")


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_missing_run_parameters(
    results: Dict[str, Any],
    imputation: str,
    morf: bool,
    base_method: str,
    modifiers: List[str],
    percentages: List[float],
    num_runs: int = 5
) -> Optional[Tuple[str, float, int]]:
    """
    Get next missing run parameters.
    
    Args:
        results: Current results dictionary
        imputation: Imputation method
        morf: MoRF order
        base_method: Base explanation method
        modifiers: List of modifiers
        percentages: List of percentages
        num_runs: Expected number of runs
    
    Returns:
        Tuple of (modifier, percentage, run_id) or None if all complete
    """
    order_key = 'morf' if morf else 'lerf'
    
    if imputation not in results:
        return (modifiers[0], percentages[0], 0)
    
    for modifier in modifiers:
        for perc in percentages:
            perc_key = str(perc)
            
            try:
                current_runs = results[imputation][base_method][modifier][order_key].get(perc_key, [])
                # Filter out error entries and pending entries
                valid_runs = [r for r in current_runs if isinstance(r, (int, float))]
                
                if len(valid_runs) < num_runs:
                    return (modifier, perc, len(valid_runs))
            except KeyError:
                return (modifier, perc, 0)
    
    return None
