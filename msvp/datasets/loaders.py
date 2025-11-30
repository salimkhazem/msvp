"""
Dataset loaders for MNIST, FashionMNIST, and CIFAR10
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple, Optional
import os


def get_dataloaders(
    dataset_name: str,
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: str = './data',
    val_split: float = 0.1,
    download: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get train, validation, and test dataloaders for specified dataset.
    
    Args:
        dataset_name: One of 'mnist', 'fashion', 'cifar10'
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        data_dir: Directory to store/load datasets
        val_split: Fraction of training data to use for validation
        download: Whether to download dataset if not present
        
    Returns:
        train_loader, val_loader, test_loader
    """
    dataset_name = dataset_name.lower()
    
    # Define transforms based on dataset
    if dataset_name in ['mnist', 'fashion']:
        # Grayscale datasets (28x28)
        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        input_channels = 1
        num_classes = 10
        
        if dataset_name == 'mnist':
            train_dataset = datasets.MNIST(
                root=data_dir, train=True, download=download, transform=train_transform
            )
            test_dataset = datasets.MNIST(
                root=data_dir, train=False, download=download, transform=test_transform
            )
        else:  # fashion
            train_dataset = datasets.FashionMNIST(
                root=data_dir, train=True, download=download, transform=train_transform
            )
            test_dataset = datasets.FashionMNIST(
                root=data_dir, train=False, download=download, transform=test_transform
            )
    
    elif dataset_name == 'cifar10':
        # RGB dataset (32x32)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            )
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            )
        ])
        
        input_channels = 3
        num_classes = 10
        
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True, download=download, transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            root=data_dir, train=False, download=download, transform=test_transform
        )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: mnist, fashion, cifar10")
    
    # Split training into train and validation
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f"Dataset: {dataset_name}")
    print(f"  Train samples: {len(train_subset)}")
    print(f"  Val samples: {len(val_subset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Input channels: {input_channels}, Classes: {num_classes}")
    
    return train_loader, val_loader, test_loader


def get_dataset_info(dataset_name: str) -> dict:
    """
    Get metadata about a dataset.
    
    Args:
        dataset_name: One of 'mnist', 'fashion', 'cifar10'
        
    Returns:
        Dictionary with dataset metadata
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name in ['mnist', 'fashion']:
        return {
            'input_channels': 1,
            'num_classes': 10,
            'input_size': (28, 28),
            'name': 'MNIST' if dataset_name == 'mnist' else 'FashionMNIST'
        }
    elif dataset_name == 'cifar10':
        return {
            'input_channels': 3,
            'num_classes': 10,
            'input_size': (32, 32),
            'name': 'CIFAR-10'
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == '__main__':
    # Quick test
    print("Testing dataset loaders...\n")
    
    for dataset in ['mnist', 'fashion', 'cifar10']:
        print(f"\n{'='*50}")
        train_loader, val_loader, test_loader = get_dataloaders(
            dataset, batch_size=32, num_workers=2, download=True
        )
        
        # Test batch
        x, y = next(iter(train_loader))
        print(f"  Batch shape: {x.shape}, labels: {y.shape}")
        
        info = get_dataset_info(dataset)
        print(f"  Info: {info}")
    
    print("\n✓ All dataset loaders working!")
