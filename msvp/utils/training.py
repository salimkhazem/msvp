"""
Training utilities and loops
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
from .metrics import compute_accuracy, AverageMeter


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    print_freq: int = 50
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch number
        print_freq: Print frequency
        
    Returns:
        Average loss and accuracy
    """
    model.train()
    
    loss_meter = AverageMeter('Loss')
    acc_meter = AverageMeter('Acc')
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        acc = compute_accuracy(outputs, targets)
        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc, inputs.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.2f}%'
        })
    
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    desc: str = 'Validation'
) -> Tuple[float, float]:
    """
    Validate model.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use
        desc: Description for progress bar
        
    Returns:
        Average loss and accuracy
    """
    model.eval()
    
    loss_meter = AverageMeter('Loss')
    acc_meter = AverageMeter('Acc')
    
    pbar = tqdm(val_loader, desc=desc)
    
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Metrics
        acc = compute_accuracy(outputs, targets)
        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc, inputs.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.2f}%'
        })
    
    return loss_meter.avg, acc_meter.avg


def save_checkpoint(
    state: dict,
    save_dir: str,
    filename: str = 'checkpoint.pth'
):
    """
    Save model checkpoint.
    
    Args:
        state: State dictionary containing model, optimizer, etc.
        save_dir: Directory to save checkpoint
        filename: Checkpoint filename
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = save_dir / filename
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = 'cuda'
) -> dict:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state
        device: Device to load checkpoint to
        
    Returns:
        Checkpoint dictionary with metadata
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best Val Acc: {checkpoint.get('best_val_acc', 'N/A'):.2f}%")
    
    return checkpoint


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if should stop training.
        
        Args:
            score: Current validation metric (loss or accuracy)
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:  # mode == 'max'
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\nEarly stopping triggered after {self.counter} epochs without improvement")
                return True
        
        return False


class TrainingLogger:
    """Log training metrics to file."""
    
    def __init__(self, log_dir: str, filename: str = 'training_log.json'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.log_dir / filename
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
    
    def log(self, epoch: int, train_loss: float, train_acc: float, 
            val_loss: float, val_acc: float, lr: float):
        """Log metrics for one epoch."""
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['learning_rates'].append(lr)
    
    def save(self):
        """Save history to file."""
        with open(self.filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load(self):
        """Load history from file."""
        if self.filepath.exists():
            with open(self.filepath, 'r') as f:
                self.history = json.load(f)


if __name__ == '__main__':
    print("Testing training utilities...")
    
    # Test EarlyStopping
    early_stop = EarlyStopping(patience=3, mode='max')
    scores = [0.8, 0.85, 0.83, 0.82, 0.81]
    for i, score in enumerate(scores):
        stop = early_stop(score)
        print(f"Epoch {i}: score={score:.2f}, stop={stop}, counter={early_stop.counter}")
    
    # Test TrainingLogger
    logger = TrainingLogger(log_dir='./test_logs')
    for epoch in range(5):
        logger.log(epoch, 0.5 - epoch*0.05, 80 + epoch*2, 0.6 - epoch*0.05, 75 + epoch*2, 0.001)
    logger.save()
    print(f"✓ Logger saved to {logger.filepath}")
    
    print("\n✓ All tests passed!")
