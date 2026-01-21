"""
Explanation generation module for unified ROAD benchmark pipeline.
Generates and caches feature attributions using Captum.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import List, Dict, Optional, Callable, Tuple
import pickle


from captum.attr import IntegratedGradients, GuidedBackprop, NoiseTunnel
from captum.attr import LayerGradCam, LayerAttribution
def GRADCAM(model, sample: torch.Tensor, target: int) -> np.ndarray:
    """
    Grad-CAM explanation using the last convolutional layer (layer4 for ResNet).
    Args:
        model: Model (e.g., ResNet)
        sample: Input image tensor (1, C, H, W) or (C, H, W)
        target: Target class index
    Returns:
        Attribution map (H, W, 3)
    """
    device = next(model.parameters()).device
    if sample.dim() == 3:
        sample = sample.unsqueeze(0)
    sample = sample.to(device)
    sample.requires_grad = True

    # Use last conv layer for ResNet
    target_layer = getattr(model, 'layer4', None)
    if target_layer is None:
        raise ValueError("Model does not have 'layer4' attribute for GradCAM. Specify the correct target layer.")

    layer_gc = LayerGradCam(model, target_layer)
    attr = layer_gc.attribute(sample, target=target)
    upsampled_attr = LayerAttribution.interpolate(attr, sample.shape[2:])  # (1, 1, H, W)
    attr_np = upsampled_attr.detach().squeeze().cpu().numpy()  # (H, W)

    # Expand to 3 channels for compatibility
    if attr_np.ndim == 2:
        attr_np = np.expand_dims(attr_np, axis=-1)
        attr_np = np.repeat(attr_np, 3, axis=-1)
    return attr_np


def attribute_image_features(model, algorithm, input_tensor, target, **kwargs):
    """Generic attribution function."""
    model.zero_grad()
    tensor_attributions = algorithm.attribute(input_tensor, target=target, **kwargs)
    return tensor_attributions


def IG(model, sample: torch.Tensor, target: int) -> np.ndarray:
    """Integrated Gradients attribution."""
    ig = IntegratedGradients(model)
    attr_ig, _ = attribute_image_features(
        model, ig, sample, target,
        baselines=sample * 0,
        return_convergence_delta=True
    )
    attribution = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
    return attribution


def IG_SG(model, sample: torch.Tensor, target: int, nt_type: str = 'smoothgrad') -> np.ndarray:
    """Integrated Gradients with SmoothGrad."""
    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    attr_ig_nt = attribute_image_features(
        model, nt, sample, target,
        baselines=sample * 0,
        nt_type=nt_type,
        nt_samples=10,
        stdevs=0.2,
        internal_batch_size=10
    )
    attribution = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    return attribution


def GB(model, sample: torch.Tensor, target: int) -> np.ndarray:
    """Guided Backpropagation attribution."""
    gb = GuidedBackprop(model)
    attr_gb = attribute_image_features(model, gb, sample, target)
    attribution = np.transpose(attr_gb.squeeze().cpu().detach().numpy(), (1, 2, 0))
    return attribution


def GB_SG(model, sample: torch.Tensor, target: int, nt_type: str = 'smoothgrad') -> np.ndarray:
    """Guided Backpropagation with SmoothGrad."""
    gb = GuidedBackprop(model)
    nt = NoiseTunnel(gb)
    attr_gb_nt = attribute_image_features(
        model, nt, sample, target,
        nt_samples=10,
        nt_type=nt_type
    )
    attribution = np.transpose(attr_gb_nt.squeeze().cpu().detach().numpy(), (1, 2, 0))
    return attribution


def get_explanation_method(expl_str: str) -> Callable:
    """
    Get explanation function from string identifier.
    
    Args:
        expl_str: Explanation method string (e.g., 'ig', 'ig_sg', 'gb_var')
    
    Returns:
        Explanation function
    """
    def compute_explanation(model, sample, target):
        if expl_str == "ig":
            return IG(model, sample, target)
        elif expl_str == "gb":
            return GB(model, sample, target)
        elif expl_str == "ig_sg":
            return IG_SG(model, sample, target, nt_type='smoothgrad')
        elif expl_str == "gb_sg":
            return GB_SG(model, sample, target, nt_type='smoothgrad')
        elif expl_str == "ig_sq":
            return IG_SG(model, sample, target, nt_type='smoothgrad_sq')
        elif expl_str == "gb_sq":
            return GB_SG(model, sample, target, nt_type='smoothgrad_sq')
        elif expl_str == "ig_var":
            return IG_SG(model, sample, target, nt_type='vargrad')
        elif expl_str == "gb_var":
            return GB_SG(model, sample, target, nt_type='vargrad')
        elif expl_str == "gradcam":
            return GRADCAM(model, sample, target)
        else:
            raise ValueError(f"Unknown explanation method: {expl_str}")
    return compute_explanation


def parse_expl_method(base_method: str, modifier: str) -> str:
    """
    Combine base method and modifier into explanation string.
    
    Args:
        base_method: 'ig' or 'gb'
        modifier: 'base', 'sg', 'sq', or 'var'
    
    Returns:
        Combined method string (e.g., 'ig_sg')
    """
    if modifier == 'base':
        return base_method
    return f"{base_method}_{modifier}"


def generate_explanations(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    expl_method: str,
    save_dir: str,
    split: str = 'test',
    subset_size: Optional[int] = None,
    verbose: bool = True
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Generate explanations for a dataset.
    
    Args:
        model: Model for generating attributions
        dataset: Dataset to generate explanations for
        device: Computation device
        expl_method: Explanation method string
        save_dir: Directory to save explanations
        split: 'train' or 'test'
        subset_size: Limit number of samples
        verbose: Show progress bar
    
    Returns:
        Tuple of (explanations list, predictions list)
    """
    model.eval()
    get_expl = get_explanation_method(expl_method)
    
    # Create save directories
    expl_save_dir = os.path.join(save_dir, expl_method, 'explanation', split)
    pred_save_dir = os.path.join(save_dir, expl_method, 'prediction', split)
    os.makedirs(expl_save_dir, exist_ok=True)
    os.makedirs(pred_save_dir, exist_ok=True)
    
    num_samples = len(dataset) if subset_size is None else min(subset_size, len(dataset))
    
    explanations = []
    predictions = []
    
    iterator = range(num_samples)
    if verbose:
        iterator = tqdm(iterator, desc=f'Generating {expl_method} explanations ({split})')
    
    for i in iterator:
        torch.cuda.empty_cache()
        
        sample, label = dataset[i]
        sample = sample.unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(sample)
            _, predicted = torch.max(outputs.data, 1)
        
        expl = get_expl(model, sample, label)
        pred = predicted.data[0].cpu().numpy()
        
        explanations.append(expl)
        predictions.append(pred)
        
        # Save individual files
        np.save(os.path.join(expl_save_dir, f'{i}.npy'), expl)
        np.save(os.path.join(pred_save_dir, f'{i}.npy'), pred)
    
    return explanations, predictions


def save_explanations(
    explanations: List[np.ndarray],
    predictions: List[int],
    save_path: str,
    expl_method: str,
    split: str = 'test'
) -> None:
    """
    Save explanations as a pickle file (legacy format).
    
    Args:
        explanations: List of explanation arrays
        predictions: List of predictions
        save_path: Directory to save
        expl_method: Explanation method name
        split: 'train' or 'test'
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Parse method
    if '_' in expl_method:
        base, modifier = expl_method.split('_', 1)
    else:
        base, modifier = expl_method, 'base'
    
    base_dir = os.path.join(save_path, base)
    os.makedirs(base_dir, exist_ok=True)
    
    # Create dictionary format matching existing code
    expl_dict = {}
    for i, (expl, pred) in enumerate(zip(explanations, predictions)):
        expl_dict[i] = {
            'expl': expl,
            'prediction': pred
        }
    
    filename = f'{modifier}_{split}.pkl'
    with open(os.path.join(base_dir, filename), 'wb') as f:
        pickle.dump(expl_dict, f)
    
    print(f"Saved explanations to: {os.path.join(base_dir, filename)}")


def load_explanations(
    expl_path: str,
    expl_method: str,
    split: str = 'test'
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load explanations from disk.
    
    Args:
        expl_path: Base path to explanations
        expl_method: Explanation method name
        split: 'train' or 'test'
    
    Returns:
        Tuple of (explanations list, predictions list)
    """
    # Parse method
    if '_' in expl_method:
        base, modifier = expl_method.split('_', 1)
    else:
        base, modifier = expl_method, 'base'
    
    # Try pickle format first
    pkl_path = os.path.join(expl_path, base, f'{modifier}_{split}.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            expl_dict = pickle.load(f)
        
        explanations = []
        predictions = []
        for i in sorted(expl_dict.keys()):
            explanations.append(expl_dict[i]['expl'])
            predictions.append(expl_dict[i]['prediction'])
        
        return explanations, predictions
    
    # Try numpy format
    expl_dir = os.path.join(expl_path, expl_method, 'explanation', split)
    pred_dir = os.path.join(expl_path, expl_method, 'prediction', split)
    
    if os.path.exists(expl_dir):
        explanations = []
        predictions = []
        
        files = sorted([f for f in os.listdir(expl_dir) if f.endswith('.npy')],
                      key=lambda x: int(x.split('.')[0]))
        
        for f in files:
            idx = f.split('.')[0]
            explanations.append(np.load(os.path.join(expl_dir, f)))
            pred_file = os.path.join(pred_dir, f)
            if os.path.exists(pred_file):
                predictions.append(np.load(pred_file))
            else:
                predictions.append(0)
        
        return explanations, predictions
    
    raise FileNotFoundError(f"No explanations found at {expl_path} for method {expl_method}")


def load_expl_legacy(train_file: Optional[str], test_file: str) -> Tuple:
    """
    Load explanations in legacy format (for compatibility).
    
    Args:
        train_file: Path to training explanations (can be None)
        test_file: Path to test explanations
    
    Returns:
        Tuple of (expl_train, expl_test, pred_train, pred_test)
    """
    expl_train = []
    pred_train = []
    
    if train_file and os.path.exists(train_file):
        with open(train_file, 'rb') as f:
            train_dict = pickle.load(f)
        for i in sorted(train_dict.keys()):
            expl_train.append(train_dict[i]['expl'])
            pred_train.append(train_dict[i]['prediction'])
    
    with open(test_file, 'rb') as f:
        test_dict = pickle.load(f)
    
    expl_test = []
    pred_test = []
    for i in sorted(test_dict.keys()):
        expl_test.append(test_dict[i]['expl'])
        pred_test.append(test_dict[i]['prediction'])
    
    return expl_train, expl_test, pred_train, pred_test
