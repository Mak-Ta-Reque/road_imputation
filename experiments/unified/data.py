"""
Data loading module for unified ROAD benchmark pipeline.
Supports CIFAR-10, Food-101, and ImageNet datasets.
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset, SubsetRandomSampler
from torchvision import datasets
from PIL import Image
import numpy as np
from typing import Optional, Tuple, List, Union

from .config import DatasetConfig, DATASET_CONFIGS


class Food101Dataset(Dataset):
    """Dataset loader for Food-101."""
    
    def __init__(self, root: str, train: bool = True, transform=None):
        super().__init__()
        self.root = root
        self.train = train
        self.transform = transform
        
        self.class_dict = self._load_classes()
        self.img_path_list = self._get_img_paths()
    
    def _load_classes(self) -> dict:
        """Load class name to index mapping."""
        classes = {}
        path = os.path.join(self.root, 'meta', 'classes.txt')
        if os.path.exists(path):
            with open(path, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    classes[line.strip('\n')] = i
        return classes
    
    def _get_img_paths(self) -> List[Tuple[str, str]]:
        """Get list of (label, image_path) tuples."""
        img_path_list = []
        split_file = 'train.txt' if self.train else 'test.txt'
        path = os.path.join(self.root, 'meta', split_file)
        
        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f:
                    label = line.split('/')[0]
                    img_path = line.strip('\n')
                    img_path_list.append((label, img_path))
        return img_path_list
    
    def __len__(self) -> int:
        return len(self.img_path_list)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        label, img_path = self.img_path_list[index]
        target = self.class_dict[label]
        
        full_path = os.path.join(self.root, 'images', f'{img_path}.jpg')
        img = Image.open(full_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, target


class ImageNetDataset(Dataset):
    """Dataset loader for ImageNet (standard folder structure)."""
    
    def __init__(self, root: str, train: bool = True, transform=None):
        super().__init__()
        split = 'train' if train else 'val'
        self.dataset = datasets.ImageNet(
            root, split=split, 
            transform=transform
        )
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[index]


def get_transforms(
    config: DatasetConfig, 
    train: bool = True,
    include_normalize: bool = True
) -> transforms.Compose:
    """
    Get appropriate transforms for a dataset.
    
    Args:
        config: Dataset configuration
        train: Whether this is for training (includes augmentation)
        include_normalize: Whether to include normalization
    
    Returns:
        Composed transforms
    """
    transform_list = []
    
    if train:
        if config.name == 'cifar10':
            transform_list.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            transform_list.extend([
                transforms.Resize(256),
                transforms.RandomCrop(config.image_size),
                transforms.RandomHorizontalFlip(),
            ])
    else:
        if config.name == 'cifar10':
            pass  # No resize needed for CIFAR
        else:
            transform_list.extend([
                transforms.Resize(256),
                transforms.CenterCrop(config.image_size),
            ])
    
    transform_list.append(transforms.ToTensor())
    
    if include_normalize:
        transform_list.append(
            transforms.Normalize(config.mean, config.std)
        )
    
    return transforms.Compose(transform_list)


def get_tensor_transform(config: DatasetConfig) -> transforms.Compose:
    """Get transform that only converts to tensor (for imputation)."""
    if config.name == 'cifar10':
        return transforms.Compose([transforms.ToTensor()])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor()
        ])


def get_normalize_transform(config: DatasetConfig) -> transforms.Normalize:
    """Get normalization transform only (applied after imputation)."""
    return transforms.Normalize(config.mean, config.std)


def get_dataset(
    dataset_name: str,
    data_path: str,
    train: bool = True,
    transform=None,
    tensor_only: bool = False
) -> Dataset:
    """
    Get dataset by name.
    
    Args:
        dataset_name: Name of dataset ('cifar10', 'food101', 'imagenet')
        data_path: Root path to dataset
        train: Whether to get training set
        transform: Optional custom transform
        tensor_only: If True, only apply ToTensor (for imputation pipeline)
    
    Returns:
        Dataset object
    """
    config = DATASET_CONFIGS[dataset_name]
    
    if transform is None:
        if tensor_only:
            transform = get_tensor_transform(config)
        else:
            transform = get_transforms(config, train=train)
    
    if dataset_name == 'cifar10':
        return datasets.CIFAR10(
            root=data_path, 
            train=train, 
            download=True, 
            transform=transform
        )
    elif dataset_name == 'cifar100':
        return datasets.CIFAR100(
            root=data_path, 
            train=train, 
            download=True, 
            transform=transform
        )
    elif dataset_name == 'food101':
        return Food101Dataset(
            root=data_path, 
            train=train, 
            transform=transform
        )
    elif dataset_name == 'imagenet':
        return ImageNetDataset(
            root=data_path, 
            train=train, 
            transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 8,
    subset_size: Optional[int] = None,
    seed: int = 42
) -> DataLoader:
    """
    Create a DataLoader with optional subset sampling.
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        subset_size: If specified, only use this many samples
        seed: Random seed for reproducible subset selection
    
    Returns:
        DataLoader object
    """
    if subset_size is not None and subset_size < len(dataset):
        # Create reproducible subset
        np.random.seed(seed)
        indices = np.random.permutation(len(dataset))[:subset_size]
        indices = indices.tolist()
        
        if shuffle:
            sampler = SubsetRandomSampler(indices)
            return DataLoader(
                dataset, 
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True
            )
        else:
            subset = Subset(dataset, indices)
            return DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )


def get_subset_dataset(
    dataset: Dataset,
    subset_size: int,
    seed: int = 42
) -> Subset:
    """
    Create a subset of a dataset with reproducible selection.
    
    Args:
        dataset: Original dataset
        subset_size: Number of samples to include
        seed: Random seed
    
    Returns:
        Subset dataset
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))[:subset_size].tolist()
    return Subset(dataset, indices)
