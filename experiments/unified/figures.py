"""
Figure generation module for unified ROAD benchmark pipeline.
Creates publication-ready figures for benchmark results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

from .metrics import averaging_accuracy, ranker, calculate_spearman_morf_lerf


# Set default style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("tab10")

# Default figure settings
DEFAULT_DPI = 300
DEFAULT_FIGSIZE = (10, 6)
LINEWIDTH = 2.5


def plot_accuracy_curves(
    averaged_results: Dict[str, Any],
    imputation: str,
    order: str = 'morf',
    base_methods: List[str] = None,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    title: str = None,
    save_path: str = None,
    dpi: int = DEFAULT_DPI
) -> plt.Figure:
    """
    Plot accuracy vs. percentage curves for multiple methods.
    
    Args:
        averaged_results: Averaged results dictionary
        imputation: Imputation method
        order: 'morf' or 'lerf'
        base_methods: Methods to include
        figsize: Figure size
        title: Optional title
        save_path: Path to save figure
        dpi: Resolution
    
    Returns:
        Matplotlib figure
    """
    if base_methods is None:
        base_methods = ['ig', 'gb']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    order_label = 'MoRF' if order == 'morf' else 'LeRF'
    
    if imputation not in averaged_results:
        return fig
    
    for base in base_methods:
        if base not in averaged_results[imputation]:
            continue
        
        for modifier, mod_dict in averaged_results[imputation][base].items():
            if not isinstance(mod_dict, dict):
                continue
            if order not in mod_dict:
                continue
            
            order_dict = mod_dict[order]
            
            percs = sorted([float(p) for p in order_dict.keys()])
            accs = [order_dict[str(p)] for p in percs]
            
            # Filter valid values
            valid = [(p, a) for p, a in zip(percs, accs) if isinstance(a, (int, float))]
            if not valid:
                continue
            
            percs, accs = zip(*valid)
            label = f"{base}-{modifier}"
            ax.plot(percs, accs, label=label, linewidth=LINEWIDTH)
    
    ax.set_xlabel(f'Portion Removed ({order_label})')
    ax.set_ylabel('Accuracy')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='best')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'{imputation.upper()} Imputation - {order_label}')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_morf_lerf_comparison(
    averaged_results: Dict[str, Any],
    imputation: str,
    base_method: str,
    modifier: str,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    save_path: str = None,
    dpi: int = DEFAULT_DPI
) -> plt.Figure:
    """
    Plot MoRF and LeRF curves side by side.
    
    Args:
        averaged_results: Averaged results
        imputation: Imputation method
        base_method: Explanation method
        modifier: Modifier
        figsize: Figure size
        save_path: Save path
        dpi: Resolution
    
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0]*1.5, figsize[1]))
    
    try:
        morf_dict = averaged_results[imputation][base_method][modifier]['morf']
        lerf_dict = averaged_results[imputation][base_method][modifier]['lerf']
    except KeyError:
        return fig
    
    # MoRF plot
    percs = sorted([float(p) for p in morf_dict.keys()])
    morf_accs = [morf_dict[str(p)] for p in percs]
    valid = [(p, a) for p, a in zip(percs, morf_accs) if isinstance(a, (int, float))]
    if valid:
        p, a = zip(*valid)
        ax1.plot(p, a, linewidth=LINEWIDTH, color='tab:blue', label='MoRF')
    ax1.set_xlabel('Portion Removed')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('MoRF (Most Relevant First)')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # LeRF plot
    percs = sorted([float(p) for p in lerf_dict.keys()])
    lerf_accs = [lerf_dict[str(p)] for p in percs]
    valid = [(p, a) for p, a in zip(percs, lerf_accs) if isinstance(a, (int, float))]
    if valid:
        p, a = zip(*valid)
        ax2.plot(p, a, linewidth=LINEWIDTH, color='tab:orange', label='LeRF')
    ax2.set_xlabel('Portion Removed')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('LeRF (Least Relevant First)')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    fig.suptitle(f'{imputation.upper()} - {base_method}-{modifier}')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_imputation_comparison(
    averaged_results: Dict[str, Any],
    base_method: str,
    modifier: str,
    order: str = 'morf',
    imputations: List[str] = None,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    save_path: str = None,
    dpi: int = DEFAULT_DPI
) -> plt.Figure:
    """
    Compare different imputation methods.
    
    Args:
        averaged_results: Averaged results
        base_method: Explanation method
        modifier: Modifier
        order: 'morf' or 'lerf'
        imputations: Imputations to compare
        figsize: Figure size
        save_path: Save path
        dpi: Resolution
    
    Returns:
        Matplotlib figure
    """
    if imputations is None:
        imputations = averaged_results.get('imputations', [])
    
    fig, ax = plt.subplots(figsize=figsize)
    order_label = 'MoRF' if order == 'morf' else 'LeRF'
    
    for imp in imputations:
        try:
            order_dict = averaged_results[imp][base_method][modifier][order]
            percs = sorted([float(p) for p in order_dict.keys()])
            accs = [order_dict[str(p)] for p in percs]
            
            valid = [(p, a) for p, a in zip(percs, accs) if isinstance(a, (int, float))]
            if valid:
                p, a = zip(*valid)
                ax.plot(p, a, label=imp.upper(), linewidth=LINEWIDTH)
        except KeyError:
            continue
    
    ax.set_xlabel(f'Portion Removed ({order_label})')
    ax.set_ylabel('Accuracy')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='best')
    ax.set_title(f'{base_method}-{modifier} - Imputation Comparison ({order_label})')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_spearman_heatmap(
    averaged_results: Dict[str, Any],
    imputations: List[str] = None,
    base_methods: List[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: str = None,
    dpi: int = DEFAULT_DPI
) -> plt.Figure:
    """
    Create heatmap of Spearman correlations.
    
    Args:
        averaged_results: Averaged results
        imputations: Imputations to include
        base_methods: Methods to include
        figsize: Figure size
        save_path: Save path
        dpi: Resolution
    
    Returns:
        Matplotlib figure
    """
    if imputations is None:
        imputations = averaged_results.get('imputations', [])
    if base_methods is None:
        base_methods = ['ig', 'gb']
    
    # Compute correlations for each imputation
    rankings = averaged_results.get('rankings', ['sort'])
    data = []
    
    for imp in imputations:
        for ranking in rankings:
            scores = calculate_spearman_morf_lerf(averaged_results, imp, ranking=ranking, base_methods=base_methods)
            if scores:
                data.append({
                    'Imputation': imp.upper(),
                    'Ranking': ranking.capitalize(),
                    'Mean Correlation': np.mean(scores),
                    'Std': np.std(scores)
                })
    
    if not data:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        return fig
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(df['Imputation'], df['Mean Correlation'], yerr=df['Std'], capsize=5)
    
    ax.set_ylabel('MoRF-LeRF Spearman Correlation')
    ax.set_xlabel('Imputation Method')
    ax.set_title('Ranking Consistency: MoRF vs LeRF')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_ranking_boxplot(
    averaged_results: Dict[str, Any],
    imputation: str,
    base_methods: List[str] = None,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    save_path: str = None,
    dpi: int = DEFAULT_DPI
) -> plt.Figure:
    """
    Create box plot of method rankings.
    
    Args:
        averaged_results: Averaged results
        imputation: Imputation method
        base_methods: Methods to include
        figsize: Figure size
        save_path: Save path
        dpi: Resolution
    
    Returns:
        Matplotlib figure
    """
    if base_methods is None:
        base_methods = ['ig', 'gb']
    
    morf_ranks = ranker(averaged_results, imputation, morf=True, base_methods=base_methods)
    lerf_ranks = ranker(averaged_results, imputation, morf=False, base_methods=base_methods)
    
    if not morf_ranks:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        return fig
    
    # Collect ranks per method
    num_methods = len(list(morf_ranks.values())[0])
    method_names = []
    
    # Infer method names from results
    if imputation in averaged_results:
        for base in base_methods:
            if base in averaged_results[imputation]:
                for modifier in averaged_results[imputation][base].keys():
                    method_names.append(f"{base}-{modifier}")
    
    if not method_names:
        method_names = [f"Method {i+1}" for i in range(num_methods)]
    
    # Prepare data for boxplot
    morf_data = {name: [] for name in method_names}
    lerf_data = {name: [] for name in method_names}
    
    for perc, ranks in morf_ranks.items():
        for i, rank in enumerate(ranks):
            if i < len(method_names):
                morf_data[method_names[i]].append(rank)
    
    for perc, ranks in lerf_ranks.items():
        for i, rank in enumerate(ranks):
            if i < len(method_names):
                lerf_data[method_names[i]].append(rank)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0]*1.5, figsize[1]))
    
    # MoRF boxplot
    ax1.boxplot([morf_data[m] for m in method_names], labels=method_names)
    ax1.set_ylabel('Rank')
    ax1.set_title(f'MoRF Ranking - {imputation.upper()}')
    ax1.tick_params(axis='x', rotation=45)
    
    # LeRF boxplot
    ax2.boxplot([lerf_data[m] for m in method_names], labels=method_names)
    ax2.set_ylabel('Rank')
    ax2.set_title(f'LeRF Ranking - {imputation.upper()}')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_multi_panel_grid(
    averaged_results: Dict[str, Any],
    imputations: List[str] = None,
    order: str = 'morf',
    base_methods: List[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    save_path: str = None,
    dpi: int = DEFAULT_DPI
) -> plt.Figure:
    """
    Create multi-panel grid comparing all imputations.
    
    Args:
        averaged_results: Averaged results
        imputations: Imputations to include
        order: 'morf' or 'lerf'
        base_methods: Methods to include
        figsize: Figure size
        save_path: Save path
        dpi: Resolution
    
    Returns:
        Matplotlib figure
    """
    if imputations is None:
        imputations = averaged_results.get('imputations', [])
    if base_methods is None:
        base_methods = ['ig', 'gb']
    
    n_imps = len(imputations)
    if n_imps == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig
    
    cols = min(2, n_imps)
    rows = (n_imps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_imps == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    order_label = 'MoRF' if order == 'morf' else 'LeRF'
    
    for idx, imp in enumerate(imputations):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        if imp not in averaged_results:
            ax.text(0.5, 0.5, f'No data for {imp}', ha='center', va='center')
            continue
        
        for base in base_methods:
            if base not in averaged_results[imp]:
                continue
            
            for modifier, mod_dict in averaged_results[imp][base].items():
                if not isinstance(mod_dict, dict):
                    continue
                if order not in mod_dict:
                    continue
                
                order_dict = mod_dict[order]
                percs = sorted([float(p) for p in order_dict.keys()])
                accs = [order_dict[str(p)] for p in percs]
                
                valid = [(p, a) for p, a in zip(percs, accs) if isinstance(a, (int, float))]
                if valid:
                    p, a = zip(*valid)
                    ax.plot(p, a, label=f"{base}-{modifier}", linewidth=LINEWIDTH)
        
        ax.set_xlabel('Portion Removed')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{imp.upper()}')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc='best', fontsize='small')
    
    # Hide empty subplots
    for idx in range(n_imps, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].set_visible(False)
    
    fig.suptitle(f'Imputation Comparison - {order_label}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def get_imputer_by_name(name: str):
    """
    Get imputer instance by name.
    
    Args:
        name: Imputation method name (linear, telea, ns, fixed, zero)
    
    Returns:
        Imputer instance
    """
    from road.imputations import (
        NoisyLinearImputer, 
        ImpaintingImputation, 
        ImpaintingImputationNS,
        ZeroImputer,
        ChannelMeanImputer
    )
    
    imputer_map = {
        'linear': NoisyLinearImputer(noise=0.01),
        'telea': ImpaintingImputation(),
        'ns': ImpaintingImputationNS(),
        'fixed': ChannelMeanImputer(),
        'zero': ZeroImputer(),
    }
    
    if name.lower() not in imputer_map:
        raise ValueError(f"Unknown imputation method: {name}. "
                        f"Available: {list(imputer_map.keys())}")
    
    return imputer_map[name.lower()]


def plot_imputation_examples(
    base_dataset,
    explanation_mask,
    imputation_names: List[str],
    image_ids: List[int] = None,
    percentages: List[float] = None,
    use_threshold: bool = False,
    remove: bool = True,
    figsize: Tuple[int, int] = None,
    save_path: str = None,
    dpi: int = DEFAULT_DPI
) -> plt.Figure:
    """
    Visualize imputation examples at different removal percentages.
    
    Creates a grid showing original images and imputed versions.
    
    Args:
        base_dataset: Original dataset (returns (image, label) tuples)
        explanation_mask: Explanation/saliency maps (same length as dataset)
        imputation_names: List of imputation method names to visualize
        image_ids: List of image indices to visualize (default: 3 random)
        percentages: Removal percentages (default: 0.1 to 0.9)
        use_threshold: If True, use ThresholdDataset; else ImputedDataset
        remove: If True, remove important pixels (MoRF); else keep them (LeRF)
        figsize: Figure size (auto-calculated if None)
        save_path: Path to save figure
        dpi: Resolution
    
    Returns:
        Matplotlib figure
    """
    from road import ImputedDataset, ThresholdDataset
    
    # Defaults
    if percentages is None:
        percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    if image_ids is None:
        # Default to 3 random samples
        np.random.seed(42)
        max_idx = min(len(base_dataset), len(explanation_mask))
        image_ids = np.random.choice(max_idx, size=min(3, max_idx), replace=False).tolist()
    
    DatasetClass = ThresholdDataset if use_threshold else ImputedDataset
    ranking_name = 'threshold' if use_threshold else 'sort'
    order_name = 'MoRF' if remove else 'LeRF'
    
    n_rows = len(percentages)
    n_cols = 1 + len(imputation_names)  # Original + each imputation
    
    if figsize is None:
        figsize = (3 * n_cols, 3 * n_rows)
    
    # Create one figure per image
    for img_idx, image_id in enumerate(image_ids):
        fig, ax_arr = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        for row, p in enumerate(percentages):
            for col in range(n_cols):
                ax = ax_arr[row, col] if n_rows > 1 else ax_arr[col]
                
                if col == 0:
                    # Original image
                    img, label = base_dataset[image_id]
                    if hasattr(img, 'cpu'):
                        img_display = img.cpu().permute(1, 2, 0).numpy()
                    else:
                        img_display = np.transpose(img, (1, 2, 0))
                    img_display = np.clip(img_display, 0, 1)
                    ax.imshow(img_display)
                    ax.set_ylabel(f"p={p:.1f}")
                    if row == 0:
                        ax.set_title("Original")
                else:
                    # Imputed image
                    imp_name = imputation_names[col - 1]
                    try:
                        imputer = get_imputer_by_name(imp_name)
                        ds_imputed = DatasetClass(
                            base_dataset,
                            mask=explanation_mask,
                            th_p=p,
                            remove=remove,
                            imputation=imputer
                        )
                        result = ds_imputed[image_id]
                        # Handle different return formats
                        if isinstance(result, tuple):
                            img = result[0]
                        else:
                            img = result
                        
                        if hasattr(img, 'cpu'):
                            img_display = img.cpu().permute(1, 2, 0).numpy()
                        else:
                            img_display = np.transpose(img, (1, 2, 0))
                        img_display = np.clip(img_display, 0, 1)
                        ax.imshow(img_display)
                    except Exception as e:
                        ax.text(0.5, 0.5, f"Error:\n{str(e)[:20]}", 
                               ha='center', va='center', transform=ax.transAxes)
                    
                    if row == 0:
                        ax.set_title(f"{imp_name.capitalize()} ({ranking_name})")
                
                ax.set_xticks([])
                ax.set_yticks([])
        
        fig.suptitle(f'Imputation Examples - Image {image_id} ({order_name}, {ranking_name})', fontsize=14)
        plt.tight_layout()
        
        if save_path and img_idx == 0:
            # Save only the first image's figure
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        elif save_path:
            # Save additional images with index suffix
            base, ext = os.path.splitext(save_path)
            fig.savefig(f"{base}_{img_idx}{ext}", dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_explanation_overlay(
    base_dataset,
    explanation_mask,
    image_id: int,
    figsize: Tuple[int, int] = (10, 4),
    save_path: str = None,
    dpi: int = DEFAULT_DPI
) -> plt.Figure:
    """
    Show original image alongside its explanation/saliency map.
    
    Args:
        base_dataset: Original dataset
        explanation_mask: Explanation maps
        image_id: Image index to visualize
        figsize: Figure size
        save_path: Save path
        dpi: Resolution
    
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Original image
    img, label = base_dataset[image_id]
    if hasattr(img, 'cpu'):
        img_display = img.cpu().permute(1, 2, 0).numpy()
    else:
        img_display = np.transpose(img, (1, 2, 0))
    img_display = np.clip(img_display, 0, 1)
    ax1.imshow(img_display)
    ax1.set_title(f'Original (label={label})')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Explanation map
    expl = explanation_mask[image_id]
    if hasattr(expl, 'numpy'):
        expl = expl.numpy()
    
    # Compute norm across channels if multi-channel
    if expl.ndim == 3:
        expl_display = np.linalg.norm(expl, axis=-1) if expl.shape[-1] <= 4 else np.linalg.norm(expl, axis=0)
    else:
        expl_display = expl
    
    ax2.matshow(expl_display, cmap='hot')
    ax2.set_title('Saliency Map')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def generate_imputation_visualizations(
    base_dataset,
    explanation_mask,
    imputation_names: List[str],
    explanation_name: str,
    output_dir: str,
    image_ids: List[int] = None,
    format: str = 'png',
    dpi: int = DEFAULT_DPI
) -> Dict[str, str]:
    """
    Generate all imputation visualization figures.
    
    Creates both sort-based and threshold-based visualizations for each imputation.
    
    Args:
        base_dataset: Original dataset
        explanation_mask: Explanation/saliency maps
        imputation_names: List of imputation method names
        explanation_name: Name of explanation method (e.g., 'ig', 'gb')
        output_dir: Directory to save figures
        image_ids: List of image indices to visualize
        format: Image format
        dpi: Resolution
    
    Returns:
        Dictionary of saved filepaths
    """
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    saved_files = {}
    
    # Generate for both ranking strategies
    for use_threshold in [False, True]:
        ranking_name = 'threshold' if use_threshold else 'sort'
        
        save_path = os.path.join(
            plots_dir, 
            f'imputation_examples_{ranking_name}_{explanation_name}.{format}'
        )
        
        try:
            plot_imputation_examples(
                base_dataset=base_dataset,
                explanation_mask=explanation_mask,
                imputation_names=imputation_names,
                image_ids=image_ids,
                use_threshold=use_threshold,
                remove=True,  # MoRF order
                save_path=save_path,
                dpi=dpi
            )
            saved_files[f'imputation_examples_{ranking_name}_{explanation_name}'] = save_path
        except Exception as e:
            print(f"Warning: Could not generate {ranking_name} visualization: {e}")
    
    # Generate explanation overlay for sample images
    if image_ids is None:
        np.random.seed(42)
        max_idx = min(len(base_dataset), len(explanation_mask))
        image_ids = np.random.choice(max_idx, size=min(3, max_idx), replace=False).tolist()
    
    for idx, image_id in enumerate(image_ids[:3]):  # Limit to 3
        save_path = os.path.join(
            plots_dir,
            f'explanation_overlay_{explanation_name}_{idx}.{format}'
        )
        try:
            plot_explanation_overlay(
                base_dataset=base_dataset,
                explanation_mask=explanation_mask,
                image_id=image_id,
                save_path=save_path,
                dpi=dpi
            )
            saved_files[f'explanation_overlay_{explanation_name}_{idx}'] = save_path
        except Exception as e:
            print(f"Warning: Could not generate overlay for image {image_id}: {e}")
    
    return saved_files


def generate_all_figures(
    results: Dict[str, Any],
    output_dir: str,
    format: str = 'png',
    dpi: int = DEFAULT_DPI
) -> Dict[str, str]:
    """
    Generate all figures from results.
    
    Args:
        results: Raw results dictionary
        output_dir: Directory to save figures
        format: Image format
        dpi: Resolution
    
    Returns:
        Dictionary of saved filepaths
    """
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    averaged = averaging_accuracy(results)
    imputations = results.get('imputations', [])
    base_methods = results.get('base_methods', ['ig', 'gb'])
    
    saved_files = {}
    
    # 1. Accuracy curves for each imputation
    for imp in imputations:
        for order in ['morf', 'lerf']:
            save_path = os.path.join(plots_dir, f'accuracy_{imp}_{order}.{format}')
            plot_accuracy_curves(averaged, imp, order, base_methods, save_path=save_path, dpi=dpi)
            saved_files[f'accuracy_{imp}_{order}'] = save_path
    
    # 2. MoRF-LeRF comparison for each method
    for imp in imputations:
        if imp not in averaged:
            continue
        for base in base_methods:
            if base not in averaged[imp]:
                continue
            for modifier in averaged[imp][base].keys():
                save_path = os.path.join(plots_dir, f'morf_lerf_{imp}_{base}_{modifier}.{format}')
                plot_morf_lerf_comparison(averaged, imp, base, modifier, save_path=save_path, dpi=dpi)
                saved_files[f'morf_lerf_{imp}_{base}_{modifier}'] = save_path
    
    # 3. Imputation comparison
    if imputations and base_methods:
        for base in base_methods:
            if base not in averaged.get(imputations[0], {}):
                continue
            for modifier in averaged[imputations[0]][base].keys():
                for order in ['morf', 'lerf']:
                    save_path = os.path.join(plots_dir, f'imputation_comparison_{base}_{modifier}_{order}.{format}')
                    plot_imputation_comparison(averaged, base, modifier, order, imputations, save_path=save_path, dpi=dpi)
                    saved_files[f'imputation_comparison_{base}_{modifier}_{order}'] = save_path
    
    # 4. Spearman heatmap
    save_path = os.path.join(plots_dir, f'spearman_heatmap.{format}')
    plot_spearman_heatmap(averaged, imputations, base_methods, save_path=save_path, dpi=dpi)
    saved_files['spearman_heatmap'] = save_path
    
    # 5. Ranking boxplots
    for imp in imputations:
        save_path = os.path.join(plots_dir, f'ranking_boxplot_{imp}.{format}')
        plot_ranking_boxplot(averaged, imp, base_methods, save_path=save_path, dpi=dpi)
        saved_files[f'ranking_boxplot_{imp}'] = save_path
    
    # 6. Multi-panel grid
    for order in ['morf', 'lerf']:
        save_path = os.path.join(plots_dir, f'grid_comparison_{order}.{format}')
        plot_multi_panel_grid(averaged, imputations, order, base_methods, save_path=save_path, dpi=dpi)
        saved_files[f'grid_comparison_{order}'] = save_path
    
    plt.close('all')
    
    print(f"Generated {len(saved_files)} figures in {plots_dir}")
    return saved_files
