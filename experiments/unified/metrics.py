"""
Metrics module for unified ROAD benchmark pipeline.
Computes various metrics for benchmark evaluation.
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr
from typing import Dict, List, Tuple, Any, Optional
import copy


def averaging_accuracy(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Average accuracy values across multiple runs.
    
    Handles both old structure (imputation -> method -> modifier -> order -> perc)
    and new structure (imputation -> ranking -> method -> modifier -> order -> perc).
    
    Args:
        results: Raw results dictionary with lists of values
    
    Returns:
        Results dictionary with averaged values
    """
    averaged = copy.deepcopy(results)
    rankings = results.get('rankings', ['sort'])
    
    for imputation in averaged.get('imputations', []):
        if imputation not in averaged:
            continue
        
        # Check if new structure (with ranking level)
        has_ranking_level = any(r in averaged[imputation] for r in rankings if isinstance(averaged[imputation].get(r), dict))
        
        if has_ranking_level:
            # New structure: imputation -> ranking -> method -> modifier -> order -> perc
            for ranking in rankings:
                if ranking not in averaged[imputation]:
                    continue
                if not isinstance(averaged[imputation][ranking], dict):
                    continue
                    
                for base_method, base_dict in averaged[imputation][ranking].items():
                    if not isinstance(base_dict, dict):
                        continue
                        
                    for modifier, mod_dict in base_dict.items():
                        if not isinstance(mod_dict, dict):
                            continue
                            
                        for order, order_dict in mod_dict.items():
                            if not isinstance(order_dict, dict):
                                continue
                                
                            for perc, values in order_dict.items():
                                if isinstance(values, list):
                                    valid_values = [v for v in values if isinstance(v, (int, float))]
                                    if valid_values:
                                        averaged[imputation][ranking][base_method][modifier][order][perc] = np.mean(valid_values)
                                    else:
                                        averaged[imputation][ranking][base_method][modifier][order][perc] = 0.0
        else:
            # Old structure: imputation -> method -> modifier -> order -> perc
            for base_method, base_dict in averaged[imputation].items():
                if not isinstance(base_dict, dict):
                    continue
                    
                for modifier, mod_dict in base_dict.items():
                    if not isinstance(mod_dict, dict):
                        continue
                        
                    for order, order_dict in mod_dict.items():
                        if not isinstance(order_dict, dict):
                            continue
                            
                        for perc, values in order_dict.items():
                            if isinstance(values, list):
                                valid_values = [v for v in values if isinstance(v, (int, float))]
                                if valid_values:
                                    averaged[imputation][base_method][modifier][order][perc] = np.mean(valid_values)
                                else:
                                    averaged[imputation][base_method][modifier][order][perc] = 0.0
    
    return averaged


def compute_std(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute standard deviation across runs.
    
    Args:
        results: Raw results dictionary with lists of values
    
    Returns:
        Results dictionary with std values
    """
    std_results = copy.deepcopy(results)
    
    for imputation in std_results.get('imputations', []):
        if imputation not in std_results:
            continue
            
        for base_method, base_dict in std_results[imputation].items():
            if not isinstance(base_dict, dict):
                continue
                
            for modifier, mod_dict in base_dict.items():
                if not isinstance(mod_dict, dict):
                    continue
                    
                for order, order_dict in mod_dict.items():
                    if not isinstance(order_dict, dict):
                        continue
                        
                    for perc, values in order_dict.items():
                        if isinstance(values, list):
                            valid_values = [v for v in values if isinstance(v, (int, float))]
                            if valid_values:
                                std_results[imputation][base_method][modifier][order][perc] = np.std(valid_values)
                            else:
                                std_results[imputation][base_method][modifier][order][perc] = 0.0
    
    return std_results


def rankdata(values: List[float], morf: bool = True) -> np.ndarray:
    """
    Rank values (lower accuracy = higher rank for MoRF, opposite for LeRF).
    
    For MoRF: We want methods that cause accuracy to drop more (lower acc = better explanation)
    For LeRF: We want methods that maintain accuracy (higher acc = better explanation)
    
    Args:
        values: List of accuracy values
        morf: If True, lower accuracy is better (MoRF), else higher is better (LeRF)
    
    Returns:
        Array of ranks (1 = best)
    """
    array = np.array(values)
    if not morf:
        array = 1 - array
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array)) + 1
    return ranks


def calculate_spearman_rank(
    averaged_results: Dict[str, Any],
    imputation: str,
    morf: bool = True,
    base_methods: List[str] = None
) -> Dict[str, float]:
    """
    Calculate Spearman rank correlation for each explanation method.
    
    Args:
        averaged_results: Averaged results dictionary
        imputation: Imputation method
        morf: MoRF or LeRF order
        base_methods: List of base methods to include
    
    Returns:
        Dictionary mapping method name to Spearman correlation
    """
    if base_methods is None:
        base_methods = ['ig', 'gb']
    
    order_key = 'morf' if morf else 'lerf'
    rankings = {}
    
    if imputation not in averaged_results:
        return rankings
    
    for base_method in base_methods:
        if base_method not in averaged_results[imputation]:
            continue
            
        for modifier, mod_dict in averaged_results[imputation][base_method].items():
            if not isinstance(mod_dict, dict):
                continue
            if order_key not in mod_dict:
                continue
                
            order_dict = mod_dict[order_key]
            
            # Extract accuracies and percentages
            percs = sorted([float(p) for p in order_dict.keys()])
            accs = [order_dict[str(p)] for p in percs]
            
            # Filter out non-numeric values
            valid_pairs = [(p, a) for p, a in zip(percs, accs) if isinstance(a, (int, float))]
            if len(valid_pairs) < 3:
                continue
            
            percs, accs = zip(*valid_pairs)
            percs = np.array(percs)
            accs = np.array(accs)
            
            if morf:
                # For MoRF: higher % removal should lead to lower accuracy
                # Good explanation: negative correlation between % and acc
                accs_transformed = 1 - accs
            else:
                # For LeRF: higher % removal should maintain accuracy
                accs_transformed = 1 - accs
            
            score = spearmanr(percs, accs_transformed).correlation
            rankings[f"{base_method}-{modifier}"] = score
    
    # Sort by score
    rankings = {k: v for k, v in sorted(rankings.items(), key=lambda item: item[1])}
    return rankings


def calculate_spearman_morf_lerf(
    averaged_results: Dict[str, Any],
    imputation: str,
    ranking: str = None,
    base_methods: List[str] = None
) -> List[float]:
    """
    Calculate Spearman correlation between MoRF and LeRF rankings.
    
    Handles both old structure (no ranking level) and new structure (with ranking level).
    
    Args:
        averaged_results: Averaged results dictionary
        imputation: Imputation method
        ranking: Ranking strategy ('sort' or 'threshold'). If None, auto-detect structure.
        base_methods: List of base methods to include
    
    Returns:
        List of correlation scores
    """
    if base_methods is None:
        base_methods = ['ig', 'gb']
    
    scores = []
    
    if imputation not in averaged_results:
        return scores
    
    rankings_list = averaged_results.get('rankings', ['sort'])
    
    # Check if new structure (with ranking level)
    has_ranking_level = any(r in averaged_results[imputation] for r in rankings_list 
                           if isinstance(averaged_results[imputation].get(r), dict))
    
    if has_ranking_level:
        # New structure: imputation -> ranking -> method -> modifier -> order -> perc
        if ranking is None:
            ranking = rankings_list[0] if rankings_list else 'sort'
        
        if ranking not in averaged_results[imputation]:
            return scores
            
        imp_data = averaged_results[imputation][ranking]
    else:
        # Old structure: imputation -> method -> modifier -> order -> perc
        imp_data = averaged_results[imputation]
    
    for base_method in base_methods:
        if base_method not in imp_data:
            continue
            
        for modifier, mod_dict in imp_data[base_method].items():
            if not isinstance(mod_dict, dict):
                continue
            if 'morf' not in mod_dict or 'lerf' not in mod_dict:
                continue
            
            morf_dict = mod_dict['morf']
            lerf_dict = mod_dict['lerf']
            
            # Get common percentages
            common_percs = set(morf_dict.keys()) & set(lerf_dict.keys())
            common_percs = sorted([float(p) for p in common_percs])
            
            morf_accs = [morf_dict[str(p)] for p in common_percs]
            lerf_accs = [lerf_dict[str(p)] for p in common_percs]
            
            # Filter valid pairs
            valid_pairs = [
                (m, l) for m, l in zip(morf_accs, lerf_accs) 
                if isinstance(m, (int, float)) and isinstance(l, (int, float))
            ]
            
            if len(valid_pairs) < 3:
                continue
            
            morf_accs, lerf_accs = zip(*valid_pairs)
            score = spearmanr(morf_accs, lerf_accs).correlation
            scores.append(score)
    
    return scores


def calculate_spearman_by_method(
    averaged_results: Dict[str, Any],
    imputation: str,
    ranking: str = None,
    base_methods: List[str] = None
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate Spearman correlation per explanation method.
    
    Args:
        averaged_results: Averaged results dictionary
        imputation: Imputation method
        ranking: Ranking strategy ('sort' or 'threshold')
        base_methods: List of base methods to include
    
    Returns:
        Dict mapping method name to (mean, std) tuple
    """
    if base_methods is None:
        base_methods = ['ig', 'gb']
    
    method_scores = {}
    
    if imputation not in averaged_results:
        return method_scores
    
    rankings_list = averaged_results.get('rankings', ['sort'])
    
    # Check if new structure (with ranking level)
    has_ranking_level = any(r in averaged_results[imputation] for r in rankings_list 
                           if isinstance(averaged_results[imputation].get(r), dict))
    
    if has_ranking_level:
        if ranking is None:
            ranking = rankings_list[0] if rankings_list else 'sort'
        
        if ranking not in averaged_results[imputation]:
            return method_scores
            
        imp_data = averaged_results[imputation][ranking]
    else:
        imp_data = averaged_results[imputation]
    
    for base_method in base_methods:
        if base_method not in imp_data:
            continue
            
        for modifier, mod_dict in imp_data[base_method].items():
            if not isinstance(mod_dict, dict):
                continue
            if 'morf' not in mod_dict or 'lerf' not in mod_dict:
                continue
            
            method_name = f"{base_method}_{modifier}" if modifier != 'base' else base_method
            
            morf_dict = mod_dict['morf']
            lerf_dict = mod_dict['lerf']
            
            common_percs = set(morf_dict.keys()) & set(lerf_dict.keys())
            common_percs = sorted([float(p) for p in common_percs])
            
            morf_accs = [morf_dict[str(p)] for p in common_percs]
            lerf_accs = [lerf_dict[str(p)] for p in common_percs]
            
            valid_pairs = [
                (m, l) for m, l in zip(morf_accs, lerf_accs) 
                if isinstance(m, (int, float)) and isinstance(l, (int, float))
            ]
            
            if len(valid_pairs) < 3:
                continue
            
            morf_accs, lerf_accs = zip(*valid_pairs)
            score = spearmanr(morf_accs, lerf_accs).correlation
            
            if method_name not in method_scores:
                method_scores[method_name] = []
            method_scores[method_name].append(score)
    
    # Convert to mean, std
    result = {}
    for method, scores in method_scores.items():
        if scores:
            result[method] = (np.mean(scores), np.std(scores))
    
    return result


def calculate_pearson_correlation(
    averaged_results: Dict[str, Any],
    imputation: str,
    base_methods: List[str] = None
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate Pearson correlation between MoRF and LeRF.
    
    Args:
        averaged_results: Averaged results dictionary
        imputation: Imputation method
        base_methods: List of base methods
    
    Returns:
        Dictionary mapping method to (correlation, p-value)
    """
    if base_methods is None:
        base_methods = ['ig', 'gb']
    
    correlations = {}
    
    if imputation not in averaged_results:
        return correlations
    
    for base_method in base_methods:
        if base_method not in averaged_results[imputation]:
            continue
            
        for modifier, mod_dict in averaged_results[imputation][base_method].items():
            if not isinstance(mod_dict, dict):
                continue
            if 'morf' not in mod_dict or 'lerf' not in mod_dict:
                continue
            
            morf_dict = mod_dict['morf']
            lerf_dict = mod_dict['lerf']
            
            common_percs = set(morf_dict.keys()) & set(lerf_dict.keys())
            common_percs = sorted([float(p) for p in common_percs])
            
            morf_accs = [morf_dict[str(p)] for p in common_percs]
            lerf_accs = [lerf_dict[str(p)] for p in common_percs]
            
            valid_pairs = [
                (m, l) for m, l in zip(morf_accs, lerf_accs) 
                if isinstance(m, (int, float)) and isinstance(l, (int, float))
            ]
            
            if len(valid_pairs) < 3:
                continue
            
            morf_accs, lerf_accs = zip(*valid_pairs)
            corr, pval = pearsonr(morf_accs, lerf_accs)
            correlations[modifier] = (corr, pval)
    
    return correlations


def compute_auc(
    averaged_results: Dict[str, Any],
    imputation: str,
    base_method: str,
    modifier: str,
    morf: bool = True,
    ranking: str = None
) -> float:
    """
    Compute Area Under Curve for accuracy vs. percentage.
    
    Args:
        averaged_results: Averaged results dictionary
        imputation: Imputation method
        base_method: Base explanation method
        modifier: Modifier (e.g., 'base', 'sg')
        morf: MoRF or LeRF order
        ranking: Ranking strategy (for new structure with ranking level)
    
    Returns:
        AUC value (normalized to 0-1)
    """
    order_key = 'morf' if morf else 'lerf'
    
    try:
        if ranking is not None:
            # New structure: imputation -> ranking -> method -> modifier -> order
            order_dict = averaged_results[imputation][ranking][base_method][modifier][order_key]
        else:
            # Old structure: imputation -> method -> modifier -> order
            order_dict = averaged_results[imputation][base_method][modifier][order_key]
    except KeyError:
        return 0.0
    
    percs = sorted([float(p) for p in order_dict.keys()])
    accs = [order_dict[str(p)] for p in percs]
    
    # Filter valid values
    valid_pairs = [(p, a) for p, a in zip(percs, accs) if isinstance(a, (int, float))]
    if len(valid_pairs) < 2:
        return 0.0
    
    percs, accs = zip(*valid_pairs)
    
    # Compute AUC using trapezoidal rule
    auc = np.trapz(accs, percs)
    
    # Normalize by max possible AUC (1.0 * range of percentages)
    max_auc = max(percs) - min(percs)
    if max_auc > 0:
        auc = auc / max_auc
    
    return auc


def ranker(
    averaged_results: Dict[str, Any],
    imputation: str,
    morf: bool = True,
    base_methods: List[str] = None,
    max_percentages: int = 7
) -> Dict[str, np.ndarray]:
    """
    Rank explanation methods at each percentage level.
    
    This follows the notebook's ranking algorithm:
    1. Extract accuracies for each method at each percentage
    2. For MoRF: lower accuracy = better rank (explanation removes important features)
    3. For LeRF: higher accuracy = better rank (explanation correctly identifies unimportant features)
    
    Args:
        averaged_results: Averaged results dictionary
        imputation: Imputation method (e.g., 'linear', 'telea', 'ns')
        morf: True for MoRF order, False for LeRF order
        base_methods: Methods to include (default: ['ig', 'gb'])
        max_percentages: Limit number of percentages (default: 7 to match notebook)
    
    Returns:
        Dictionary mapping percentage to ranks array
    """
    if base_methods is None:
        base_methods = ['ig', 'gb']
    
    order_key = 'morf' if morf else 'lerf'
    
    # Build data structure matching notebook format
    new_dict = {}
    
    if imputation not in averaged_results:
        return {}
    
    # Get percentages from results
    percentages = averaged_results.get('percentages', [])
    if max_percentages and len(percentages) > max_percentages:
        percentages = percentages[:max_percentages]
    
    # Collect accuracies for each method
    for base_method in base_methods:
        if base_method not in averaged_results[imputation]:
            continue
            
        for modifier, mod_dict in averaged_results[imputation][base_method].items():
            if not isinstance(mod_dict, dict):
                continue
            if order_key not in mod_dict:
                continue
            
            method_key = f"{base_method}-{modifier}"
            accs = mod_dict[order_key]
            
            # Get accuracy values as list
            acc_list = list(accs.values())
            if max_percentages and len(acc_list) > max_percentages:
                acc_list = acc_list[:max_percentages]
            
            new_dict[method_key] = acc_list
    
    if not new_dict:
        return {}
    
    # Use percentages from data or generate indices
    if percentages:
        x_axis = percentages[:max_percentages] if max_percentages else percentages
    else:
        # Infer from first method's data length
        first_method = list(new_dict.keys())[0]
        x_axis = list(range(len(new_dict[first_method])))
    
    # Compute ranks for each percentage
    ranked_dict = {}
    method_names = sorted(new_dict.keys())
    
    for count, perc in enumerate(x_axis):
        list_item = []
        for method in method_names:
            if count < len(new_dict[method]):
                list_item.append(new_dict[method][count])
            else:
                list_item.append(0)
        
        ranked_dict[perc] = rankdata(list_item, morf=morf)
    
    return ranked_dict


def calculate_ranking_consistency(
    morf_ranks: Dict[float, np.ndarray],
    lerf_ranks: Dict[float, np.ndarray]
) -> List[float]:
    """
    Calculate Spearman correlation between MoRF and LeRF rankings at each percentage.
    
    Args:
        morf_ranks: MoRF ranking dictionary
        lerf_ranks: LeRF ranking dictionary
    
    Returns:
        List of correlation scores per percentage
    """
    scores = []
    common_percs = set(morf_ranks.keys()) & set(lerf_ranks.keys())
    
    for perc in sorted(common_percs):
        score = spearmanr(morf_ranks[perc], lerf_ranks[perc]).correlation
        scores.append(score)
    
    return scores


def compute_all_metrics(
    results: Dict[str, Any],
    imputations: List[str] = None,
    base_methods: List[str] = None
) -> Dict[str, Any]:
    """
    Compute all metrics for benchmark results.
    
    Args:
        results: Raw results dictionary
        imputations: Imputation methods to analyze
        base_methods: Explanation methods to analyze
    
    Returns:
        Dictionary with all computed metrics
    """
    if imputations is None:
        imputations = results.get('imputations', ['linear', 'fixed', 'telea', 'ns'])
    if base_methods is None:
        base_methods = results.get('base_methods', ['ig', 'gb'])
    
    averaged = averaging_accuracy(results)
    std = compute_std(results)
    
    metrics = {
        'averaged_results': averaged,
        'std_results': std,
        'spearman_ranks': {},
        'spearman_morf_lerf': {},
        'pearson_correlations': {},
        'auc_scores': {},
        'ranking_consistency': {}
    }
    
    rankings_list = results.get('rankings', ['sort'])
    
    for imputation in imputations:
        # Spearman rank correlations
        metrics['spearman_ranks'][imputation] = {
            'morf': calculate_spearman_rank(averaged, imputation, morf=True, base_methods=base_methods),
            'lerf': calculate_spearman_rank(averaged, imputation, morf=False, base_methods=base_methods)
        }
        
        # MoRF-LeRF correlation (per ranking method)
        metrics['spearman_morf_lerf'][imputation] = {}
        for ranking in rankings_list:
            metrics['spearman_morf_lerf'][imputation][ranking] = calculate_spearman_morf_lerf(
                averaged, imputation, ranking=ranking, base_methods=base_methods
            )
        
        # Pearson correlations
        metrics['pearson_correlations'][imputation] = calculate_pearson_correlation(averaged, imputation, base_methods)
        
        # AUC scores (need to handle ranking level in structure)
        metrics['auc_scores'][imputation] = {}
        
        # Check if new structure
        has_ranking_level = any(r in averaged.get(imputation, {}) for r in rankings_list 
                               if isinstance(averaged.get(imputation, {}).get(r), dict))
        
        if has_ranking_level:
            for ranking in rankings_list:
                if ranking not in averaged.get(imputation, {}):
                    continue
                metrics['auc_scores'][imputation][ranking] = {}
                for base in base_methods:
                    if base not in averaged[imputation][ranking]:
                        continue
                    metrics['auc_scores'][imputation][ranking][base] = {}
                    for modifier in averaged[imputation][ranking][base].keys():
                        metrics['auc_scores'][imputation][ranking][base][modifier] = {
                            'morf': compute_auc(averaged, imputation, base, modifier, morf=True, ranking=ranking),
                            'lerf': compute_auc(averaged, imputation, base, modifier, morf=False, ranking=ranking)
                        }
        else:
            for base in base_methods:
                if base not in averaged.get(imputation, {}):
                    continue
                metrics['auc_scores'][imputation][base] = {}
                for modifier in averaged[imputation][base].keys():
                    metrics['auc_scores'][imputation][base][modifier] = {
                        'morf': compute_auc(averaged, imputation, base, modifier, morf=True),
                        'lerf': compute_auc(averaged, imputation, base, modifier, morf=False)
                    }
        
        # Ranking consistency
        morf_ranks = ranker(averaged, imputation, morf=True, base_methods=base_methods)
        lerf_ranks = ranker(averaged, imputation, morf=False, base_methods=base_methods)
        consistency = calculate_ranking_consistency(morf_ranks, lerf_ranks)
        metrics['ranking_consistency'][imputation] = {
            'scores': consistency,
            'mean': np.mean(consistency) if consistency else 0.0
        }
    
    return metrics
