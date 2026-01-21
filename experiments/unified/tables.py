"""
Table generation module for unified ROAD benchmark pipeline.
Generates tables in CSV, LaTeX, and Markdown formats.
"""

import os
import re
import glob
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from .metrics import averaging_accuracy, compute_std, compute_auc, calculate_spearman_rank, calculate_spearman_morf_lerf, calculate_spearman_by_method


def create_accuracy_table(
    averaged_results: Dict[str, Any],
    imputation: str,
    order: str = 'morf',
    base_methods: List[str] = None
) -> pd.DataFrame:
    """
    Create accuracy table for a specific imputation and order.
    
    Args:
        averaged_results: Averaged results dictionary
        imputation: Imputation method
        order: 'morf' or 'lerf'
        base_methods: Methods to include
    
    Returns:
        DataFrame with accuracies
    """
    if base_methods is None:
        base_methods = ['ig', 'gb']
    
    data = {}
    
    if imputation not in averaged_results:
        return pd.DataFrame()
    
    for base in base_methods:
        if base not in averaged_results[imputation]:
            continue
        
        for modifier, mod_dict in averaged_results[imputation][base].items():
            if not isinstance(mod_dict, dict):
                continue
            if order not in mod_dict:
                continue
            
            method_name = f"{base}-{modifier}"
            order_dict = mod_dict[order]
            
            # Sort percentages
            percs = sorted([float(p) for p in order_dict.keys()])
            accs = [order_dict[str(p)] for p in percs]
            
            data[method_name] = {str(p): a for p, a in zip(percs, accs)}
    
    df = pd.DataFrame(data).T
    df.columns = [f"{float(c)*100:.0f}%" for c in df.columns]
    return df


def create_comparison_table(
    averaged_results: Dict[str, Any],
    base_method: str,
    modifier: str,
    order: str = 'morf',
    imputations: List[str] = None
) -> pd.DataFrame:
    """
    Create comparison table across imputations.
    
    Args:
        averaged_results: Averaged results
        base_method: Explanation method
        modifier: Modifier
        order: 'morf' or 'lerf'
        imputations: Imputation methods to compare
    
    Returns:
        DataFrame comparing imputations
    """
    if imputations is None:
        imputations = averaged_results.get('imputations', [])
    
    data = {}
    
    for imp in imputations:
        try:
            order_dict = averaged_results[imp][base_method][modifier][order]
            percs = sorted([float(p) for p in order_dict.keys()])
            accs = [order_dict[str(p)] for p in percs]
            data[imp] = {str(p): a for p, a in zip(percs, accs)}
        except KeyError:
            continue
    
    df = pd.DataFrame(data).T
    df.columns = [f"{float(c)*100:.0f}%" for c in df.columns]
    return df


def create_ranking_table(
    averaged_results: Dict[str, Any],
    imputation: str,
    base_methods: List[str] = None
) -> pd.DataFrame:
    """
    Create ranking table (Spearman correlations).
    
    Args:
        averaged_results: Averaged results
        imputation: Imputation method
        base_methods: Methods to include
    
    Returns:
        DataFrame with rankings
    """
    morf_ranks = calculate_spearman_rank(averaged_results, imputation, morf=True, base_methods=base_methods)
    lerf_ranks = calculate_spearman_rank(averaged_results, imputation, morf=False, base_methods=base_methods)
    
    all_methods = set(morf_ranks.keys()) | set(lerf_ranks.keys())
    
    data = []
    for method in sorted(all_methods):
        data.append({
            'Method': method,
            'MoRF Spearman': morf_ranks.get(method, np.nan),
            'LeRF Spearman': lerf_ranks.get(method, np.nan)
        })
    
    return pd.DataFrame(data)


def create_auc_table(
    averaged_results: Dict[str, Any],
    imputations: List[str] = None,
    base_methods: List[str] = None
) -> pd.DataFrame:
    """
    Create AUC comparison table.
    
    Args:
        averaged_results: Averaged results
        imputations: Imputation methods
        base_methods: Explanation methods
    
    Returns:
        DataFrame with AUC scores
    """
    if imputations is None:
        imputations = averaged_results.get('imputations', [])
    if base_methods is None:
        base_methods = ['ig', 'gb']
    
    data = []
    
    for imp in imputations:
        if imp not in averaged_results:
            continue
        
        for base in base_methods:
            if base not in averaged_results[imp]:
                continue
            
            for modifier in averaged_results[imp][base].keys():
                morf_auc = compute_auc(averaged_results, imp, base, modifier, morf=True)
                lerf_auc = compute_auc(averaged_results, imp, base, modifier, morf=False)
                
                data.append({
                    'Imputation': imp,
                    'Method': f"{base}-{modifier}",
                    'MoRF AUC': morf_auc,
                    'LeRF AUC': lerf_auc,
                    'Diff': lerf_auc - morf_auc
                })
    
    return pd.DataFrame(data)


def create_summary_table(
    averaged_results: Dict[str, Any],
    std_results: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Create summary statistics table.
    
    Args:
        averaged_results: Averaged results
        std_results: Standard deviation results
    
    Returns:
        DataFrame with summary statistics
    """
    imputations = averaged_results.get('imputations', [])
    
    data = []
    
    for imp in imputations:
        if imp not in averaged_results:
            continue
        
        # Collect all accuracies for this imputation
        all_accs_morf = []
        all_accs_lerf = []
        
        for base in averaged_results[imp].keys():
            if not isinstance(averaged_results[imp][base], dict):
                continue
            for modifier in averaged_results[imp][base].keys():
                try:
                    morf_dict = averaged_results[imp][base][modifier]['morf']
                    lerf_dict = averaged_results[imp][base][modifier]['lerf']
                    
                    all_accs_morf.extend([v for v in morf_dict.values() if isinstance(v, (int, float))])
                    all_accs_lerf.extend([v for v in lerf_dict.values() if isinstance(v, (int, float))])
                except KeyError:
                    continue
        
        if all_accs_morf:
            data.append({
                'Imputation': imp,
                'MoRF Mean Acc': np.mean(all_accs_morf),
                'MoRF Std': np.std(all_accs_morf),
                'LeRF Mean Acc': np.mean(all_accs_lerf) if all_accs_lerf else np.nan,
                'LeRF Std': np.std(all_accs_lerf) if all_accs_lerf else np.nan
            })
    
    return pd.DataFrame(data)


def save_table(
    df: pd.DataFrame,
    output_dir: str,
    name: str,
    formats: List[str] = None
) -> Dict[str, str]:
    """
    Save table in multiple formats.
    
    Args:
        df: DataFrame to save
        output_dir: Output directory
        name: Base name for files
        formats: List of formats ('csv', 'latex', 'markdown')
    
    Returns:
        Dictionary mapping format to filepath
    """
    if formats is None:
        formats = ['csv', 'latex', 'markdown']
    
    os.makedirs(output_dir, exist_ok=True)
    saved = {}
    
    if 'csv' in formats:
        path = os.path.join(output_dir, f'{name}.csv')
        df.to_csv(path)
        saved['csv'] = path
    
    if 'latex' in formats:
        path = os.path.join(output_dir, f'{name}.tex')
        with open(path, 'w') as f:
            f.write(df.to_latex(float_format="%.4f", escape=False))
        saved['latex'] = path
    
    if 'markdown' in formats:
        path = os.path.join(output_dir, f'{name}.md')
        with open(path, 'w') as f:
            f.write(df.to_markdown())
        saved['markdown'] = path
    
    return saved


def create_spearman_morf_lerf_table(
    averaged_results: Dict[str, Any],
    imputations: List[str],
    base_methods: List[str] = None,
    dataset_name: str = None
) -> pd.DataFrame:
    """
    Create Spearman rank correlation table between MoRF and LeRF curves.
    
    Format matches Table 2 style:
    Dataset | Strategy | Linear | Telea | NS | ...
    
    Shows correlation for each ranking strategy (sort, threshold) and imputation.
    
    Args:
        averaged_results: Averaged results dictionary
        imputations: List of imputation methods
        base_methods: Methods to include
        dataset_name: Name of dataset for table
    
    Returns:
        DataFrame with Spearman correlations per strategy and imputation
    """
    if base_methods is None:
        base_methods = ['ig', 'gb']
    
    if dataset_name is None:
        dataset_name = averaged_results.get('dataset', 'Dataset')
    
    rankings = averaged_results.get('rankings', ['sort'])
    data = []
    
    for ranking in rankings:
        row = {
            'Dataset': dataset_name,
            'Strategy': f"{ranking.capitalize()} (MoRF vs LeRF)"
        }
        
        for imp in imputations:
            if imp not in averaged_results:
                row[imp.capitalize()] = "N/A"
                continue
            
            spearman_values = calculate_spearman_morf_lerf(
                averaged_results, imp, ranking=ranking, base_methods=base_methods
            )
            
            if spearman_values:
                mean_val = np.mean(spearman_values)
                std_val = np.std(spearman_values)
                row[imp.capitalize()] = f"{mean_val:.2f}±{std_val:.2f}"
            else:
                row[imp.capitalize()] = "N/A"
        
        data.append(row)
    
    if not data:
        return pd.DataFrame()
    
    return pd.DataFrame(data)


def create_spearman_detailed_table(
    averaged_results: Dict[str, Any],
    imputations: List[str],
    base_methods: List[str] = None,
    dataset_name: str = None
) -> pd.DataFrame:
    """
    Create detailed Spearman table with breakdown by explanation method.
    
    Args:
        averaged_results: Averaged results dictionary
        imputations: List of imputation methods
        base_methods: Methods to include
        dataset_name: Dataset name
    
    Returns:
        DataFrame with detailed Spearman correlations
    """
    if base_methods is None:
        base_methods = ['ig', 'gb']
    
    if dataset_name is None:
        dataset_name = averaged_results.get('dataset', 'Dataset')
    
    rankings = averaged_results.get('rankings', ['sort'])
    data = []
    
    for ranking in rankings:
        for imp in imputations:
            if imp not in averaged_results:
                continue
            
            method_scores = calculate_spearman_by_method(
                averaged_results, imp, ranking=ranking, base_methods=base_methods
            )
            
            for method, (mean_val, std_val) in method_scores.items():
                data.append({
                    'Dataset': dataset_name,
                    'Strategy': ranking.capitalize(),
                    'Imputation': imp.capitalize(),
                    'Method': method,
                    'Spearman': f"{mean_val:.4f}±{std_val:.4f}",
                    'Mean': mean_val,
                    'Std': std_val
                })
    
    if not data:
        return pd.DataFrame()
    
    return pd.DataFrame(data)


def create_spearman_table2_format(
    averaged_results: Dict[str, Any],
    imputations: List[str],
    base_methods: List[str] = None,
    dataset_name: str = None
) -> pd.DataFrame:
    """
    Create Spearman table in Table 2 format from paper.
    
    Format: Dataset | Strategy | Linear | Telea | NS
    Each cell shows mean Spearman correlation across all explanation methods.
    
    Args:
        averaged_results: Averaged results dictionary
        imputations: List of imputation methods
        base_methods: Methods to include
        dataset_name: Dataset name
    
    Returns:
        DataFrame in Table 2 format
    """
    if base_methods is None:
        base_methods = ['ig', 'gb']
    
    if dataset_name is None:
        dataset_name = averaged_results.get('dataset', 'Dataset')
    
    rankings = averaged_results.get('rankings', ['sort'])
    data = []
    
    for ranking in rankings:
        row = {
            'Dataset': dataset_name,
            'Strategy': ranking.capitalize()
        }
        
        for imp in imputations:
            if imp not in averaged_results:
                row[imp.capitalize()] = '-'
                continue
            
            method_scores = calculate_spearman_by_method(
                averaged_results, imp, ranking=ranking, base_methods=base_methods
            )
            
            if method_scores:
                # Average across all methods
                all_means = [v[0] for v in method_scores.values()]
                all_stds = [v[1] for v in method_scores.values()]
                mean_val = np.mean(all_means)
                std_val = np.mean(all_stds) if all_stds else 0
                row[imp.capitalize()] = f"{mean_val:.4f}±{std_val:.4f}"
            else:
                row[imp.capitalize()] = '-'
        
        data.append(row)
    
    if not data:
        return pd.DataFrame()
    
    return pd.DataFrame(data)


def parse_time_file(filepath: str) -> List[float]:
    """
    Parse a time file containing Unix `time` command output.
    
    Extracts the 'real' time in seconds.
    
    Args:
        filepath: Path to time file
    
    Returns:
        List of time values in seconds
    """
    times = []
    if not os.path.exists(filepath):
        return times
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Match patterns like "real    0m12.345s" or "real 1m5.2s"
    pattern = r'real\s+(\d+)m([\d.]+)s'
    matches = re.findall(pattern, content)
    
    for minutes, seconds in matches:
        total_seconds = int(minutes) * 60 + float(seconds)
        times.append(total_seconds)
    
    return times


def parse_all_time_files(
    workspace_root: str,
    dataset: str,
    imputations: List[str]
) -> Dict[str, Dict[str, List[float]]]:
    """
    Parse all time files for a dataset and its imputations.
    
    Looks in both sort_time and threshold_time directories.
    
    Args:
        workspace_root: Root directory of workspace
        dataset: Dataset name (cifar10, food101)
        imputations: List of imputation methods
    
    Returns:
        Dictionary: {imputation: {'sort': [times], 'threshold': [times]}}
    """
    time_data = {}
    
    # Map dataset names
    dataset_dir_map = {
        'cifar10': 'cifar10',
        'cifar-10': 'cifar10',
        'food101': 'food101',
        'food-101': 'food101',
        'imagenet': 'imagenet'
    }
    
    dataset_dir = dataset_dir_map.get(dataset.lower(), dataset.lower())
    base_path = os.path.join(workspace_root, 'experiments', dataset_dir)
    
    for imp in imputations:
        time_data[imp] = {'sort': [], 'threshold': []}
        
        # Sort time
        sort_path = os.path.join(base_path, 'sort_time', f'file_{imp}_noretrain.txt')
        time_data[imp]['sort'] = parse_time_file(sort_path)
        
        # Threshold time
        threshold_path = os.path.join(base_path, 'threshold_time', f'file_{imp}_noretrain.txt')
        time_data[imp]['threshold'] = parse_time_file(threshold_path)
    
    return time_data


def create_time_complexity_table(
    workspace_root: str,
    dataset: str,
    imputations: List[str],
    baseline: str = 'linear'
) -> pd.DataFrame:
    """
    Create time complexity table with relative percentages.
    
    Shows mean time, std, and relative percentage compared to baseline.
    
    Args:
        workspace_root: Root directory of workspace
        dataset: Dataset name
        imputations: List of imputation methods
        baseline: Baseline imputation for relative comparison
    
    Returns:
        DataFrame with time complexity data
    """
    time_data = parse_all_time_files(workspace_root, dataset, imputations)
    
    data = []
    baseline_time = None
    
    # First pass: calculate all times and find baseline
    time_stats = {}
    for imp in imputations:
        all_times = time_data[imp]['sort'] + time_data[imp]['threshold']
        if all_times:
            mean_time = np.mean(all_times)
            std_time = np.std(all_times)
            time_stats[imp] = {'mean': mean_time, 'std': std_time}
            
            if imp == baseline:
                baseline_time = mean_time
    
    # Second pass: create table with relative percentages
    for imp in imputations:
        if imp not in time_stats:
            continue
        
        mean_time = time_stats[imp]['mean']
        std_time = time_stats[imp]['std']
        
        row = {
            'Imputation': imp,
            'Time (s)': f"{mean_time:.2f} ± {std_time:.2f}",
            'Mean (s)': mean_time,
            'Std (s)': std_time
        }
        
        if baseline_time and baseline_time > 0:
            relative_pct = (mean_time / baseline_time) * 100
            row['Relative (%)'] = f"{relative_pct:.1f}%"
        else:
            row['Relative (%)'] = "N/A"
        
        data.append(row)
    
    if not data:
        return pd.DataFrame()
    
    return pd.DataFrame(data)


def generate_all_tables(
    results: Dict[str, Any],
    output_dir: str,
    formats: List[str] = None
) -> Dict[str, str]:
    """
    Generate all tables from results.
    
    Args:
        results: Raw results dictionary
        output_dir: Directory to save tables
        formats: Output formats
    
    Returns:
        Dictionary of saved filepaths
    """
    if formats is None:
        formats = ['csv', 'latex', 'markdown']
    
    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    
    averaged = averaging_accuracy(results)
    std = compute_std(results)
    imputations = results.get('imputations', [])
    
    saved_files = {}
    
    # 1. Accuracy tables for each imputation
    for imp in imputations:
        for order in ['morf', 'lerf']:
            df = create_accuracy_table(averaged, imp, order)
            if not df.empty:
                name = f'accuracy_{imp}_{order}'
                saved_files[name] = save_table(df, tables_dir, name, formats)
    
    # 2. Comparison tables
    base_methods = results.get('base_methods', ['ig', 'gb'])
    for base in base_methods:
        if base not in averaged.get(imputations[0], {}):
            continue
        for modifier in averaged[imputations[0]][base].keys():
            for order in ['morf', 'lerf']:
                df = create_comparison_table(averaged, base, modifier, order, imputations)
                if not df.empty:
                    name = f'comparison_{base}_{modifier}_{order}'
                    saved_files[name] = save_table(df, tables_dir, name, formats)
    
    # 3. Ranking tables
    for imp in imputations:
        df = create_ranking_table(averaged, imp, base_methods)
        if not df.empty:
            name = f'ranking_{imp}'
            saved_files[name] = save_table(df, tables_dir, name, formats)
    
    # 4. AUC table
    df = create_auc_table(averaged, imputations, base_methods)
    if not df.empty:
        name = 'auc_comparison'
        saved_files[name] = save_table(df, tables_dir, name, formats)
    
    # 5. Summary table
    df = create_summary_table(averaged, std)
    if not df.empty:
        name = 'summary'
        saved_files[name] = save_table(df, tables_dir, name, formats)
    
    # 6. Spearman MoRF-LeRF correlation table (main format with strategies)
    dataset_name = results.get('dataset', 'Dataset')
    df = create_spearman_morf_lerf_table(averaged, imputations, base_methods, dataset_name)
    if not df.empty:
        name = 'spearman_morf_lerf'
        saved_files[name] = save_table(df, tables_dir, name, formats)
    
    # 7. Detailed Spearman table (per method breakdown)
    df = create_spearman_detailed_table(averaged, imputations, base_methods, dataset_name)
    if not df.empty:
        name = 'spearman_detailed'
        saved_files[name] = save_table(df, tables_dir, name, formats)
    
    # 8. Table 2 format (Dataset | Strategy | Linear | Telea | NS)
    df = create_spearman_table2_format(averaged, imputations, base_methods, dataset_name)
    if not df.empty:
        name = 'spearman_table2'
        saved_files[name] = save_table(df, tables_dir, name, formats)
    
    print(f"Generated {len(saved_files)} tables in {tables_dir}")
    return saved_files


def generate_all_tables_with_time(
    results: Dict[str, Any],
    output_dir: str,
    workspace_root: str,
    dataset: str,
    formats: List[str] = None
) -> Dict[str, str]:
    """
    Generate all tables including time complexity table.
    
    This function extends generate_all_tables with time complexity analysis.
    
    Args:
        results: Raw results dictionary
        output_dir: Directory to save tables
        workspace_root: Root directory of workspace (for time files)
        dataset: Dataset name for time file lookup
        formats: Output formats
    
    Returns:
        Dictionary of saved filepaths
    """
    # Generate standard tables
    saved_files = generate_all_tables(results, output_dir, formats)
    
    if formats is None:
        formats = ['csv', 'latex', 'markdown']
    
    tables_dir = os.path.join(output_dir, 'tables')
    imputations = results.get('imputations', [])
    
    # Add time complexity table
    df = create_time_complexity_table(workspace_root, dataset, imputations)
    if not df.empty:
        name = f'time_complexity_{dataset}'
        saved_files[name] = save_table(df, tables_dir, name, formats)
    
    print(f"Total tables generated (including time): {len(saved_files)}")
    return saved_files
