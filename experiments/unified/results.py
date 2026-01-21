"""
Results aggregation module for unified ROAD benchmark pipeline.
Loads, merges, and queries benchmark results.
"""

import os
import json
import glob
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import copy


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Save results to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def load_all_results(base_path: str, pattern: str = '**/noretrain.json') -> Dict[str, Dict[str, Any]]:
    """
    Auto-discover and load all result files from a directory.
    
    Args:
        base_path: Base directory to search
        pattern: Glob pattern for result files
    
    Returns:
        Dictionary mapping file path to results
    """
    results = {}
    
    for filepath in glob.glob(os.path.join(base_path, pattern), recursive=True):
        try:
            results[filepath] = load_results(filepath)
            print(f"Loaded: {filepath}")
        except Exception as e:
            print(f"Failed to load {filepath}: {e}")
    
    return results


def merge_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple result dictionaries into one.
    
    Args:
        results_list: List of result dictionaries
    
    Returns:
        Merged results dictionary
    """
    if not results_list:
        return {}
    
    merged = copy.deepcopy(results_list[0])
    
    for results in results_list[1:]:
        # Merge imputations
        for imp in results.get('imputations', []):
            if imp not in merged.get('imputations', []):
                if 'imputations' not in merged:
                    merged['imputations'] = []
                merged['imputations'].append(imp)
            
            if imp not in merged:
                merged[imp] = {}
            
            if imp in results:
                _deep_merge(merged[imp], results[imp])
    
    return merged


def _deep_merge(base: Dict, update: Dict) -> None:
    """Recursively merge update into base."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        elif key in base and isinstance(base[key], list) and isinstance(value, list):
            # For accuracy lists, extend with new values
            base[key].extend(value)
        else:
            base[key] = value


@dataclass
class ResultQuery:
    """Query specification for filtering results."""
    dataset: Optional[str] = None
    imputation: Optional[str] = None
    ranking: Optional[str] = None
    order: Optional[str] = None
    base_method: Optional[str] = None
    modifier: Optional[str] = None
    percentage: Optional[float] = None


class ResultsDatabase:
    """
    Database for querying and analyzing ROAD benchmark results.
    """
    
    def __init__(self, results: Dict[str, Any] = None):
        """
        Initialize the results database.
        
        Args:
            results: Initial results dictionary
        """
        self.results = results or {}
        self._cache = {}
    
    def load(self, filepath: str) -> None:
        """Load results from file."""
        self.results = load_results(filepath)
        self._cache = {}
    
    def load_multiple(self, filepaths: List[str]) -> None:
        """Load and merge results from multiple files."""
        all_results = [load_results(fp) for fp in filepaths]
        self.results = merge_results(all_results)
        self._cache = {}
    
    def save(self, filepath: str) -> None:
        """Save results to file."""
        save_results(self.results, filepath)
    
    def get_imputations(self) -> List[str]:
        """Get list of available imputation methods."""
        return self.results.get('imputations', [])
    
    def get_base_methods(self) -> List[str]:
        """Get list of available base explanation methods."""
        return self.results.get('base_methods', [])
    
    def get_modifiers(self) -> List[str]:
        """Get list of available modifiers."""
        return self.results.get('modifiers', [])
    
    def get_percentages(self) -> List[float]:
        """Get list of percentages."""
        return self.results.get('percentages', [])
    
    def get_dataset(self) -> str:
        """Get dataset name."""
        return self.results.get('dataset', 'unknown')
    
    def query(
        self,
        imputation: str,
        base_method: str,
        modifier: str,
        order: str,
        percentage: Optional[float] = None
    ) -> Any:
        """
        Query specific results.
        
        Args:
            imputation: Imputation method
            base_method: Base explanation method
            modifier: Modifier
            order: 'morf' or 'lerf'
            percentage: Optional specific percentage
        
        Returns:
            Accuracy value(s) or None
        """
        try:
            order_dict = self.results[imputation][base_method][modifier][order]
            
            if percentage is not None:
                return order_dict.get(str(percentage))
            return order_dict
        except KeyError:
            return None
    
    def get_accuracy_series(
        self,
        imputation: str,
        base_method: str,
        modifier: str,
        order: str,
        averaged: bool = True
    ) -> Tuple[List[float], List[float]]:
        """
        Get accuracy vs. percentage series.
        
        Args:
            imputation: Imputation method
            base_method: Base explanation method
            modifier: Modifier
            order: 'morf' or 'lerf'
            averaged: Whether to average multiple runs
        
        Returns:
            Tuple of (percentages, accuracies)
        """
        order_dict = self.query(imputation, base_method, modifier, order)
        if order_dict is None:
            return [], []
        
        percentages = sorted([float(p) for p in order_dict.keys()])
        accuracies = []
        
        for p in percentages:
            val = order_dict[str(p)]
            if isinstance(val, list):
                if averaged:
                    valid = [v for v in val if isinstance(v, (int, float))]
                    accuracies.append(sum(valid) / len(valid) if valid else 0)
                else:
                    accuracies.append(val)
            else:
                accuracies.append(val if isinstance(val, (int, float)) else 0)
        
        return percentages, accuracies
    
    def get_all_methods(self, imputation: str) -> List[Tuple[str, str]]:
        """
        Get all (base_method, modifier) pairs for an imputation.
        
        Args:
            imputation: Imputation method
        
        Returns:
            List of (base_method, modifier) tuples
        """
        methods = []
        if imputation not in self.results:
            return methods
        
        for base, base_dict in self.results[imputation].items():
            if not isinstance(base_dict, dict):
                continue
            for modifier in base_dict.keys():
                methods.append((base, modifier))
        
        return methods
    
    def compare_imputations(
        self,
        base_method: str,
        modifier: str,
        order: str,
        imputations: List[str] = None
    ) -> Dict[str, Tuple[List[float], List[float]]]:
        """
        Compare multiple imputation methods for same explanation.
        
        Args:
            base_method: Base explanation method
            modifier: Modifier
            order: 'morf' or 'lerf'
            imputations: List of imputations to compare
        
        Returns:
            Dictionary mapping imputation to (percentages, accuracies)
        """
        if imputations is None:
            imputations = self.get_imputations()
        
        comparison = {}
        for imp in imputations:
            percs, accs = self.get_accuracy_series(imp, base_method, modifier, order)
            if percs:
                comparison[imp] = (percs, accs)
        
        return comparison
    
    def compare_methods(
        self,
        imputation: str,
        order: str,
        methods: List[Tuple[str, str]] = None
    ) -> Dict[str, Tuple[List[float], List[float]]]:
        """
        Compare multiple explanation methods for same imputation.
        
        Args:
            imputation: Imputation method
            order: 'morf' or 'lerf'
            methods: List of (base_method, modifier) tuples
        
        Returns:
            Dictionary mapping method name to (percentages, accuracies)
        """
        if methods is None:
            methods = self.get_all_methods(imputation)
        
        comparison = {}
        for base, modifier in methods:
            method_name = f"{base}-{modifier}"
            percs, accs = self.get_accuracy_series(imputation, base, modifier, order)
            if percs:
                comparison[method_name] = (percs, accs)
        
        return comparison
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics of the results."""
        return {
            'dataset': self.get_dataset(),
            'imputations': self.get_imputations(),
            'base_methods': self.get_base_methods(),
            'modifiers': self.get_modifiers(),
            'percentages': self.get_percentages(),
            'num_methods': len(self.get_all_methods(self.get_imputations()[0]) if self.get_imputations() else [])
        }
