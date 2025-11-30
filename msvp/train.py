"""
Main training script for Multi-Scale Visual Prompting experiments

Usage:
    python train.py --dataset cifar10 --model resnet18 --use_prompt --epochs 10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, OneCycleLR
import argparse
import random
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from datasets import get_dataloaders, get_dataset_info
from models import CNN, ResNet18, ViT
from utils import (
    train_epoch, validate, save_checkpoint, load_checkpoint,
    TrainingLogger, EarlyStopping, plot_training_curves,
    compute_confusion_matrix, plot_confusion_matrix
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(args, dataset_info):
    """Create model based on arguments."""
    model_kwargs = {
        'num_classes': dataset_info['num_classes'],
        'in_channels': dataset_info['input_channels'],
        'use_prompt': args.use_prompt,
        'fusion_type': args.fusion_type,
        'prompt_scales': args.prompt_scales,
    }
    
    if args.model == 'cnn':
        model_kwargs['input_size'] = dataset_info['input_size']
        model = CNN(**model_kwargs)
    elif args.model == 'resnet18':
        model_kwargs['input_size'] = dataset_info['input_size']
        model = ResNet18(**model_kwargs)
    elif args.model == 'vit':
        # ViT-specific parameters
        img_size = dataset_info['input_size'][0]
        patch_size = 7 if img_size == 28 else 4  # 7 for MNIST, 4 for CIFAR10
        model = ViT(
            num_classes=dataset_info['num_classes'],
            img_size=img_size,
            patch_size=patch_size,
            in_channels=dataset_info['input_channels'],
            embed_dim=192,
            depth=12,
            num_heads=3,
            mlp_ratio=4.0,
            dropout=args.dropout,
            use_prompt=args.use_prompt,
            fusion_type=args.fusion_type,
            prompt_scales=args.prompt_scales,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    return model


def get_optimizer(args, model):
    """Create optimizer based on arguments."""
    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    return optimizer


def get_scheduler(args, optimizer, steps_per_epoch):
    """Create learning rate scheduler based on arguments."""
    if args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.max_lr if args.max_lr else args.lr * 10,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")
    
    return scheduler


def main(args):
    """Main training loop."""
    
    # Set random seed
    set_seed(args.seed)
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{args.dataset}_{args.model}"
    if args.use_prompt:
        exp_name += f"_prompt_{args.fusion_type}_{args.prompt_scales}"
    exp_name += f"_{timestamp}"
    
    exp_dir = Path(args.save_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*70}\n")
    
    # Save configuration
    config_path = exp_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Configuration saved to {config_path}\n")
    
    # Get dataset info
    dataset_info = get_dataset_info(args.dataset)
    print(f"Dataset: {dataset_info['name']}")
    print(f"  Input: {dataset_info['input_channels']} channels, {dataset_info['input_size']}")
    print(f"  Classes: {dataset_info['num_classes']}\n")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        download=True
    )
    print()
    
    # Create model
    model = get_model(args, dataset_info)
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    # Multi-GPU support
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Use prompting: {args.use_prompt}")
    if args.use_prompt:
        print(f"  Fusion type: {args.fusion_type}")
        print(f"  Prompt scales: {args.prompt_scales}")
    print()
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(args, model)
    steps_per_epoch = len(train_loader)
    scheduler = get_scheduler(args, optimizer, steps_per_epoch)
    
    print(f"Optimizer: {args.optimizer}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"Scheduler: {args.scheduler}\n")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training logger
    logger = TrainingLogger(log_dir)
    
    # Early stopping (optional)
    early_stopping = EarlyStopping(patience=args.patience, mode='max') if args.early_stop else None
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    
    print(f"{'='*70}")
    print("Starting training...")
    print(f"{'='*70}\n")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"{'-'*70}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, args.print_freq
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        if args.scheduler == 'onecycle':
            # OneCycleLR steps per batch
            pass  # Already stepped in train_epoch
        else:
            scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        logger.log(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)
        logger.save()
        
        # Print summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            if args.save_best_only:
                save_checkpoint(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict() if args.multi_gpu else model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_acc': best_val_acc,
                        'config': vars(args)
                    },
                    save_dir=exp_dir,
                    filename='best_model.pth'
                )
                print(f"  ✓ New best model saved! (Val Acc: {best_val_acc:.2f}%)")
        
        # Early stopping
        if early_stopping and early_stopping(val_acc):
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    print(f"\n{'='*70}")
    print("Training completed!")
    print(f"  Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"{'='*70}\n")
    
    # Plot training curves
    plot_path = log_dir / 'training_curves.png'
    plot_training_curves(
        str(plot_path),
        logger.history['train_loss'],
        logger.history['val_loss'],
        logger.history['train_acc'],
        logger.history['val_acc'],
        title=f'Training Curves - {exp_name}'
    )
    
    # Final test evaluation
    print("Evaluating on test set...")
    test_loss, test_acc = validate(
        model, test_loader, criterion, device, desc='Testing'
    )
    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Acc: {test_acc:.2f}%\n")
    
    # Save test results
    results = {
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'test_acc': test_acc,
        'test_loss': test_loss
    }
    
    results_path = exp_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    print(f"\nAll outputs saved to:")
    print(f"  Checkpoints: {exp_dir}")
    print(f"  Logs: {log_dir}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train models with Multi-Scale Visual Prompting')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['mnist', 'fashion', 'cifar10'],
                       help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Data directory')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Model
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['cnn', 'resnet18', 'vit'],
                       help='Model architecture')
    parser.add_argument('--use_prompt', action='store_true',
                       help='Use multi-scale prompting')
    parser.add_argument('--fusion_type', type=str, default='add',
                       choices=['add', 'concat', 'gated'],
                       help='Fusion strategy for prompts')
    parser.add_argument('--prompt_scales', type=str, default='full',
                       choices=['global', 'global+mid', 'full'],
                       help='Which prompt scales to use')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--max_lr', type=float, default=None,
                       help='Max learning rate for OneCycleLR')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum for SGD')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['step', 'cosine', 'onecycle'],
                       help='Learning rate scheduler')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='./experiments/checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./experiments/logs',
                       help='Directory to save logs')
    parser.add_argument('--save_best_only', action='store_true', default=True,
                       help='Save only the best model')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--multi_gpu', action='store_true',
                       help='Use multiple GPUs if available')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--print_freq', type=int, default=50,
                       help='Print frequency')
    parser.add_argument('--early_stop', action='store_true',
                       help='Use early stopping')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    args = parser.parse_args()
    
    main(args)
