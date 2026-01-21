"""Model loading module for unified ROAD benchmark pipeline.
Handles ResNet50 loading from torch hub with automatic fine-tuning.
Includes model caching to avoid redownloading/retraining.
"""

import os
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Tuple
import warnings
import json

from .config import DatasetConfig, DATASET_CONFIGS

# Default model cache directory (used globally)
DEFAULT_CACHE_DIR = './model_cache'

# Minimum required accuracy to proceed with benchmark
MIN_ACCURACY_THRESHOLD = 50.0


class ModelAccuracyError(Exception):
    """Raised when model accuracy is below the minimum threshold."""
    pass


def _set_relu_inplace_false(model: nn.Module) -> None:
    """Set inplace=False for all ReLU layers (needed for gradient-based explanations)."""
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False


def get_cache_info_path(cache_dir: str) -> str:
    """Get path to cache info JSON file."""
    return os.path.join(cache_dir, 'model_cache_info.json')


def load_cache_info(cache_dir: str) -> dict:
    """Load cache info from JSON file."""
    info_path = get_cache_info_path(cache_dir)
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            return json.load(f)
    return {}


def save_cache_info(cache_dir: str, info: dict) -> None:
    """Save cache info to JSON file."""
    info_path = get_cache_info_path(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)


def get_cached_model_path(dataset_name: str, cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    """Get the path where a cached model would be stored."""
    return os.path.join(cache_dir, f'{dataset_name}_resnet50_finetuned.pth')


def is_model_cached(dataset_name: str, cache_dir: str = DEFAULT_CACHE_DIR) -> Tuple[bool, Optional[float]]:
    """
    Check if a fine-tuned model is cached for the given dataset.
    
    Returns:
        Tuple of (is_cached, cached_accuracy)
    """
    cache_info = load_cache_info(cache_dir)
    model_path = get_cached_model_path(dataset_name, cache_dir)
    
    if os.path.exists(model_path) and dataset_name in cache_info:
        return True, cache_info[dataset_name].get('accuracy')
    return False, None


def load_resnet50(
    config: DatasetConfig,
    device: torch.device,
    model_path: Optional[str] = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    force_finetune: bool = False
) -> nn.Module:
    """
    Load ResNet50 model from torch hub or cache.
    
    Priority order:
    1. Custom model_path if provided
    2. Cached fine-tuned model (in cache_dir)
    3. ImageNet pretrained model from torch hub (modified for num_classes)
    
    Args:
        config: Dataset configuration
        device: Device to load model on
        model_path: Path to custom pretrained weights
        cache_dir: Directory for caching models
        force_finetune: Force fine-tuning even if cached model exists
    
    Returns:
        Loaded and configured model
    """
    os.makedirs(cache_dir, exist_ok=True)
    torch.hub.set_dir(cache_dir)
    
    print(f"Loading ResNet50 for {config.name}...")
    
    # Check for cached fine-tuned model first (highest priority for non-ImageNet)
    cached_path = get_cached_model_path(config.name, cache_dir)
    is_cached, cached_acc = is_model_cached(config.name, cache_dir)
    
    if is_cached and not force_finetune and config.name != 'imagenet':
        print(f"Found cached fine-tuned model: {cached_path}")
        if cached_acc is not None:
            print(f"Cached model accuracy: {cached_acc:.2f}%")
        
        # Load base model structure first
        model = _load_base_resnet50()
        
        # Modify for target classes
        if config.num_classes != 1000:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, config.num_classes)
        
        _set_relu_inplace_false(model)
        model = model.to(device)
        
        # Load cached weights
        if load_custom_weights(model, cached_path, device):
            print(f"Successfully loaded cached model for {config.name}")
            return model
        else:
            print("Failed to load cached model, falling back to pretrained...")
    
    # Try custom model_path
    if model_path and os.path.exists(model_path):
        print(f"Attempting to load from custom path: {model_path}")
        model = _load_base_resnet50()
        if config.num_classes != 1000:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, config.num_classes)
        _set_relu_inplace_false(model)
        model = model.to(device)
        if load_custom_weights(model, model_path, device):
            return model
    
    # Load from torch hub (ImageNet pretrained)
    model = _load_base_resnet50()
    
    # Modify classifier for target number of classes
    if config.num_classes != 1000:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, config.num_classes)
        print(f"Modified classifier for {config.num_classes} classes (requires fine-tuning!)")
    
    # Set ReLU inplace=False for gradient computations
    _set_relu_inplace_false(model)
    
    model = model.to(device)
    
    return model


def _load_base_resnet50() -> nn.Module:
    """Load base ResNet50 from torch hub or torchvision."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = torch.hub.load('pytorch/vision', 'resnet50', weights='IMAGENET1K_V2')
            except:
                model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
        print("Loaded ResNet50 from torch hub")
    except Exception as e:
        print(f"torch.hub.load failed: {e}, using torchvision.models")
        try:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        except:
            model = models.resnet50(pretrained=True)
    return model


def load_custom_weights(
    model: nn.Module, 
    model_path: str, 
    device: torch.device
) -> bool:
    """
    Load custom weights from a checkpoint file.
    
    Args:
        model: Model to load weights into
        model_path: Path to checkpoint
        device: Device for loading
    
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Loading weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        print("Weights loaded successfully")
        return True
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return False


def finetune_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: DatasetConfig,
    epochs: int = 30,
    lr: float = 0.001,
    cache_dir: str = DEFAULT_CACHE_DIR,
    min_accuracy: float = MIN_ACCURACY_THRESHOLD
) -> nn.Module:
    """
    Fine-tune the model on the target dataset.
    
    Args:
        model: Model to fine-tune
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device for training
        config: Dataset configuration
        epochs: Number of training epochs
        lr: Learning rate
        cache_dir: Directory to save fine-tuned model
        min_accuracy: Minimum accuracy required (raises error if not met)
    
    Returns:
        Fine-tuned model
        
    Raises:
        ModelAccuracyError: If best accuracy is below min_accuracy threshold
    """
    print(f"\n{'='*60}")
    print(f"Fine-tuning ResNet50 on {config.name}")
    print(f"{'='*60}")
    
    os.makedirs(cache_dir, exist_ok=True)
    save_path = get_cached_model_path(config.name, cache_dir)
    
    # Freeze early layers for faster training
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False
    
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        scheduler.step()
        
        # Validation
        val_acc = evaluate_model(model, val_loader, device)
        print(f'Epoch {epoch+1}: Train Acc: {100.*correct/total:.2f}%, Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_acc': best_acc,
            }, save_path)
            print(f'Saved best model with accuracy: {best_acc:.2f}%')
    
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True
    
    # Load best model
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        best_acc = checkpoint['best_acc']
        print(f"\nLoaded best model with accuracy: {best_acc:.2f}%")
        
        # Update cache info
        cache_info = load_cache_info(cache_dir)
        cache_info[config.name] = {
            'accuracy': best_acc,
            'epochs': epochs,
            'model_path': save_path
        }
        save_cache_info(cache_dir, cache_info)
        print(f"Cached model info saved to {get_cache_info_path(cache_dir)}")
        
        # Check minimum accuracy
        if best_acc < min_accuracy:
            raise ModelAccuracyError(
                f"Model accuracy ({best_acc:.2f}%) is below minimum threshold ({min_accuracy}%). "
                f"Fine-tuning for more epochs or adjusting hyperparameters may be required."
            )
    
    return model


def evaluate_model(
    model: nn.Module, 
    data_loader: DataLoader, 
    device: torch.device
) -> float:
    """
    Evaluate model accuracy.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        device: Device for computation
    
    Returns:
        Accuracy percentage
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total


def validate_model_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    min_accuracy: float = MIN_ACCURACY_THRESHOLD,
    dataset_name: str = "dataset"
) -> float:
    """
    Evaluate model and raise error if accuracy is too low.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        device: Device for computation
        min_accuracy: Minimum required accuracy
        dataset_name: Name of dataset (for error message)
    
    Returns:
        Accuracy percentage
        
    Raises:
        ModelAccuracyError: If accuracy is below threshold
    """
    accuracy = evaluate_model(model, data_loader, device)
    
    if accuracy < min_accuracy:
        raise ModelAccuracyError(
            f"\n" + "="*60 + "\n"
            f"ERROR: Model accuracy ({accuracy:.2f}%) is below minimum threshold ({min_accuracy}%)\n"
            f"Dataset: {dataset_name}\n"
            f"\n"
            f"The model needs to be fine-tuned on this dataset before running benchmarks.\n"
            f"Options:\n"
            f"  1. Run with --stages train to fine-tune the model first\n"
            f"  2. Provide a pre-trained model via --model_path\n"
            f"  3. Check if a cached model exists in ./model_cache/\n"
            + "="*60
        )
    
    return accuracy


def list_cached_models(cache_dir: str = DEFAULT_CACHE_DIR) -> None:
    """Print information about all cached models."""
    cache_info = load_cache_info(cache_dir)
    
    print(f"\n{'='*60}")
    print("Cached Models")
    print(f"{'='*60}")
    
    if not cache_info:
        print("No cached models found.")
        print(f"Cache directory: {os.path.abspath(cache_dir)}")
    else:
        for dataset_name, info in cache_info.items():
            model_path = get_cached_model_path(dataset_name, cache_dir)
            exists = os.path.exists(model_path)
            status = "✓" if exists else "✗ (missing)"
            
            print(f"\n{dataset_name}:")
            print(f"  Status: {status}")
            if 'accuracy' in info:
                print(f"  Accuracy: {info['accuracy']:.2f}%")
            if 'epochs' in info:
                print(f"  Epochs: {info['epochs']}")
            print(f"  Path: {model_path}")
    
    print()
