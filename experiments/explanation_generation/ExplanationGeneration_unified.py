"""
Unified Explanation Generation Script for Food101, CIFAR10, and ImageNet datasets.
Automatically loads ResNet50 models from torch hub, with fine-tuning if pretrained models aren't available.

Usage:
    python ExplanationGeneration_unified.py --dataset food101 --data_path /path/to/data --gpu True
    python ExplanationGeneration_unified.py --dataset cifar10 --data_path /path/to/data --gpu True
    python ExplanationGeneration_unified.py --dataset imagenet --data_path /path/to/data --gpu True
"""

import torch
torch.cuda.empty_cache()
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, Dataset
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from PIL import Image
from captum.attr import IntegratedGradients, NoiseTunnel, GuidedBackprop


# ============== Configuration ==============
DATASET_CONFIG = {
    'cifar10': {
        'num_classes': 10,
        'image_size': (32, 32),
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2470, 0.2435, 0.2616),
        'input_channels': 3,
    },
    'food101': {
        'num_classes': 101,
        'image_size': (224, 224),
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
        'input_channels': 3,
    },
    'imagenet': {
        'num_classes': 1000,
        'image_size': (224, 224),
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
        'input_channels': 3,
    }
}


# ============== Argument Parser ==============
def parse_args():
    parser = argparse.ArgumentParser(description='Unified Explanation Generation for Multiple Datasets')
    
    # Dataset and paths
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'food101', 'imagenet'],
                        help='Dataset to use: cifar10, food101, or imagenet')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the dataset')
    parser.add_argument('--save_path', type=str, default='./explanations',
                        help='Path to save explanations')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to a pretrained model checkpoint (optional)')
    parser.add_argument('--cache_dir', type=str, default='./model_cache',
                        help='Directory to cache downloaded models')
    
    # Training/Fine-tuning parameters
    parser.add_argument('--finetune', action='store_true',
                        help='Force fine-tuning even if pretrained model exists')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs for fine-tuning')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training and inference')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    
    # Explanation parameters
    parser.add_argument('--expl_method', type=str, default='ig',
                        choices=['ig', 'gb', 'ig_sg', 'gb_sg', 'ig_sq', 'gb_sq', 'ig_var', 'gb_var'],
                        help='Explanation method to use')
    parser.add_argument('--test', action='store_true',
                        help='Generate explanations for test set only')
    
    # Hardware
    parser.add_argument('--gpu', type=bool, default=True,
                        help='Use GPU if available')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU device ID to use')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


# ============== Data Loaders ==============
class Food101DataLoader(Dataset):
    """Dataset loader for Food-101."""
    def __init__(self, root, train=True, transform=None):
        super().__init__()
        self.root = root
        self.train = train
        self.transform = transform
        
        # Load class mapping
        self.class_dict = self._load_classes()
        self.img_path_list = self._get_img_paths()
    
    def _load_classes(self):
        classes = {}
        path = os.path.join(self.root, 'meta', 'classes.txt')
        if os.path.exists(path):
            with open(path, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    classes[line.strip('\n')] = i
        return classes
    
    def _get_img_paths(self):
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
    
    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, index):
        label, img_path = self.img_path_list[index]
        target = self.class_dict[label]
        
        full_path = os.path.join(self.root, 'images', f'{img_path}.jpg')
        img = Image.open(full_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, target


class ImageNetDataLoader(Dataset):
    """Dataset loader for ImageNet (standard folder structure)."""
    def __init__(self, root, train=True, transform=None):
        super().__init__()
        self.root = root
        self.train = train
        self.transform = transform
        
        split = 'train' if train else 'val'
        self.dataset = datasets.ImageFolder(os.path.join(root, split), transform=transform)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]


def get_transforms(dataset_name, train=True):
    """Get appropriate transforms for the dataset."""
    config = DATASET_CONFIG[dataset_name]
    image_size = config['image_size']
    mean = config['mean']
    std = config['std']
    
    if train:
        if dataset_name == 'cifar10':
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
    else:
        if dataset_name == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
    
    return transform


def get_dataset(dataset_name, data_path, train=True, transform=None):
    """Get the appropriate dataset based on name."""
    if dataset_name == 'cifar10':
        return datasets.CIFAR10(root=data_path, train=train, download=True, transform=transform)
    elif dataset_name == 'food101':
        return Food101DataLoader(root=data_path, train=train, transform=transform)
    elif dataset_name == 'imagenet':
        return ImageNetDataLoader(root=data_path, train=train, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ============== Model Loading ==============
def load_model_from_hub(dataset_name, num_classes, device, cache_dir='./model_cache'):
    """
    Load ResNet50 model from torch hub. 
    If pretrained weights for the specific dataset exist, use them.
    Otherwise, load ImageNet pretrained weights.
    """
    os.makedirs(cache_dir, exist_ok=True)
    torch.hub.set_dir(cache_dir)
    
    print(f"Loading ResNet50 model for {dataset_name}...")
    
    try:
        # Try loading from torch hub with new API (PyTorch >= 1.13)
        if dataset_name == 'imagenet':
            # For ImageNet, we can use the pretrained model directly
            model = torch.hub.load('pytorch/vision', 'resnet50', weights='IMAGENET1K_V2')
            print("Loaded pretrained ImageNet ResNet50 from torch hub")
        else:
            # For other datasets, load ImageNet pretrained and modify the classifier
            model = torch.hub.load('pytorch/vision', 'resnet50', weights='IMAGENET1K_V2')
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            print(f"Loaded ImageNet pretrained ResNet50 and modified for {num_classes} classes")
    except Exception as e:
        print(f"torch.hub.load failed: {e}")
        print("Falling back to torchvision.models...")
        
        try:
            # Try with new weights API
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        except:
            # Fall back to deprecated API
            model = models.resnet50(pretrained=True)
        
        if dataset_name != 'imagenet':
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        
        print("Loaded ResNet50 from torchvision.models")
    
    # Set ReLU inplace=False for gradient-based explanations
    _set_relu_inplace_false(model)
    
    model = model.to(device)
    return model


def _set_relu_inplace_false(model):
    """Set inplace=False for all ReLU layers (needed for some explanation methods)."""
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False


def load_custom_weights(model, model_path, device):
    """Load custom weights from a checkpoint file."""
    if model_path and os.path.exists(model_path):
        print(f"Loading custom weights from: {model_path}")
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
        
        print("Custom weights loaded successfully")
        return True
    return False


# ============== Fine-tuning ==============
def finetune_model(model, train_loader, val_loader, device, args, num_classes):
    """Fine-tune the model on the target dataset."""
    print(f"\n{'='*50}")
    print(f"Fine-tuning ResNet50 on {args.dataset}")
    print(f"{'='*50}")
    
    # Freeze early layers for faster training
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False
    
    # Only train the last layer and layer4
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_acc = 0.0
    save_path = os.path.join(args.cache_dir, f'{args.dataset}_resnet50_finetuned.pth')
    
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
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
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
        
        scheduler.step()
        
        # Validation
        if val_loader is not None:
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
    
    # Unfreeze all layers after fine-tuning
    for param in model.parameters():
        param.requires_grad = True
    
    # Load best model
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        print(f"\nLoaded best fine-tuned model with accuracy: {checkpoint['best_acc']:.2f}%")
    
    return model


def evaluate_model(model, data_loader, device):
    """Evaluate model accuracy."""
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
    
    model.train()
    return 100. * correct / total


# ============== Explanation Methods ==============
def normalize_map(s):
    """Normalize explanation map to [0, 1]."""
    epsilon = 1e-5
    return (s - np.min(s)) / (np.max(s) - np.min(s) + epsilon)


def attribute_image_features(model, algorithm, input, target, **kwargs):
    """Generic attribution function."""
    model.zero_grad()
    tensor_attributions = algorithm.attribute(input, target=target, **kwargs)
    return tensor_attributions


def IG(model, sample, target):
    """Integrated Gradients."""
    ig = IntegratedGradients(model)
    attr_ig, _ = attribute_image_features(
        model, ig, sample, target, 
        baselines=sample * 0, 
        return_convergence_delta=True
    )
    attribution = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
    return attribution


def IG_SG(model, sample, target, nt_type='smoothgrad'):
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


def GB(model, sample, target):
    """Guided Backpropagation."""
    gb = GuidedBackprop(model)
    attr_gb = attribute_image_features(model, gb, sample, target)
    attribution = np.transpose(attr_gb.squeeze().cpu().detach().numpy(), (1, 2, 0))
    return attribution


def GB_SG(model, sample, target, nt_type='smoothgrad'):
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


def get_explanation_method(expl_str):
    """Return explanation function based on string."""
    def compute_explanation(model, sample, target):
        if expl_str == "ig":
            return IG(model, sample, target)
        elif expl_str == "gb":
            return GB(model, sample, target)
        elif expl_str == "ig_sg":
            return IG_SG(model, sample, target)
        elif expl_str == "gb_sg":
            return GB_SG(model, sample, target)
        elif expl_str == "ig_sq":
            return IG_SG(model, sample, target, nt_type='smoothgrad_sq')
        elif expl_str == "gb_sq":
            return GB_SG(model, sample, target, nt_type='smoothgrad_sq')
        elif expl_str == "ig_var":
            return IG_SG(model, sample, target, nt_type='vargrad')
        elif expl_str == "gb_var":
            return GB_SG(model, sample, target, nt_type='vargrad')
        else:
            raise ValueError(f"Unknown explanation method: {expl_str}")
    return compute_explanation


# ============== Main Workflow ==============
def create_directories(save_path, expl_str):
    """Create directories for saving explanations."""
    save_expl_path = os.path.join(save_path, expl_str) if '_' in expl_str else os.path.join(save_path, f'{expl_str}_base')
    
    dirs = [
        os.path.join(save_expl_path, 'explanation', 'train'),
        os.path.join(save_expl_path, 'prediction', 'train'),
        os.path.join(save_expl_path, 'explanation', 'test'),
        os.path.join(save_expl_path, 'prediction', 'test'),
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    return save_expl_path


def generate_explanations(model, dataset, save_expl_path, split, device, get_expl, args):
    """Generate and save explanations for a dataset."""
    print(f"\nGenerating {args.expl_method} explanations for {split} set...")
    model.eval()
    
    start = time.time()
    for i_num in tqdm(range(len(dataset))):
        torch.cuda.empty_cache()
        
        sample, clss = dataset[i_num]
        sample = sample.unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(sample)
            _, predicted = torch.max(outputs.data, 1)
        
        expl = get_expl(model, sample, clss)
        
        # Save explanation and prediction
        np.save(os.path.join(save_expl_path, 'explanation', split, f'{i_num}.npy'), expl)
        np.save(os.path.join(save_expl_path, 'prediction', split, f'{i_num}.npy'), predicted.data[0].cpu().numpy())
    
    elapsed = time.time() - start
    print(f'Explanations for {split} set complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s')


def main():
    args = parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device setup
    if args.gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Get dataset configuration
    config = DATASET_CONFIG[args.dataset]
    num_classes = config['num_classes']
    
    # Create directories
    os.makedirs(args.cache_dir, exist_ok=True)
    save_expl_path = create_directories(args.save_path, args.expl_method)
    
    # Load transforms
    transform_train = get_transforms(args.dataset, train=True)
    transform_test = get_transforms(args.dataset, train=False)
    
    # Load datasets
    print(f"\nLoading {args.dataset} dataset from {args.data_path}...")
    trainset = get_dataset(args.dataset, args.data_path, train=True, transform=transform_train)
    testset = get_dataset(args.dataset, args.data_path, train=False, transform=transform_test)
    
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print(f'Train set: {len(trainset)} samples')
    print(f'Test set: {len(testset)} samples')
    
    # Load model
    model = load_model_from_hub(args.dataset, num_classes, device, args.cache_dir)
    
    # Try to load custom weights if provided
    weights_loaded = False
    if args.model_path:
        weights_loaded = load_custom_weights(model, args.model_path, device)
    
    # Check for cached fine-tuned model
    cached_model_path = os.path.join(args.cache_dir, f'{args.dataset}_resnet50_finetuned.pth')
    if not weights_loaded and os.path.exists(cached_model_path) and not args.finetune:
        print(f"Found cached fine-tuned model: {cached_model_path}")
        weights_loaded = load_custom_weights(model, cached_model_path, device)
    
    # Fine-tune if needed (for non-ImageNet datasets without custom weights)
    if args.dataset != 'imagenet' and (args.finetune or not weights_loaded):
        print("\nFine-tuning is required for this dataset...")
        model = finetune_model(model, trainloader, testloader, device, args, num_classes)
    
    # Evaluate model
    print("\nEvaluating model accuracy...")
    model.eval()
    test_acc = evaluate_model(model, testloader, device)
    print(f'Model accuracy on test set: {test_acc:.2f}%')
    
    # Get explanation method
    get_expl = get_explanation_method(args.expl_method)
    
    # Generate explanations
    if args.test:
        generate_explanations(model, testset, save_expl_path, 'test', device, get_expl, args)
    else:
        generate_explanations(model, trainset, save_expl_path, 'train', device, get_expl, args)
        generate_explanations(model, testset, save_expl_path, 'test', device, get_expl, args)
    
    print("\n" + "="*50)
    print("Explanation generation complete!")
    print(f"Saved to: {save_expl_path}")
    print("="*50)


if __name__ == '__main__':
    main()
