"""
Evaluation metrics for classification
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
from typing import Tuple


def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute top-1 accuracy.
    
    Args:
        outputs: Model predictions (B, num_classes)
        targets: Ground truth labels (B,)
        
    Returns:
        Accuracy as percentage
    """
    _, predicted = outputs.max(1)
    total = targets.size(0)
    correct = predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def compute_metrics(outputs: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    Compute comprehensive metrics.
    
    Args:
        outputs: Model predictions (B, num_classes)
        targets: Ground truth labels (B,)
        
    Returns:
        Dictionary of metrics
    """
    _, predicted = outputs.max(1)
    
    # Move to CPU and convert to numpy
    y_true = targets.cpu().numpy()
    y_pred = predicted.cpu().numpy()
    
    # Compute metrics
    accuracy = 100.0 * (y_pred == y_true).sum() / len(y_true)
    
    # Compute per-class and macro metrics
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
    }


def compute_confusion_matrix(outputs: torch.Tensor, targets: torch.Tensor, num_classes: int = 10) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        outputs: Model predictions (B, num_classes)
        targets: Ground truth labels (B,)
        num_classes: Number of classes
        
    Returns:
        Confusion matrix (num_classes, num_classes)
    """
    _, predicted = outputs.max(1)
    
    y_true = targets.cpu().numpy()
    y_pred = predicted.cpu().numpy()
    
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    return cm


def compute_per_class_accuracy(outputs: torch.Tensor, targets: torch.Tensor, num_classes: int = 10) -> np.ndarray:
    """
    Compute per-class accuracy.
    
    Args:
        outputs: Model predictions (B, num_classes)
        targets: Ground truth labels (B,)
        num_classes: Number of classes
        
    Returns:
        Array of per-class accuracies
    """
    cm = compute_confusion_matrix(outputs, targets, num_classes)
    
    # Avoid division by zero
    per_class_acc = np.zeros(num_classes)
    for i in range(num_classes):
        if cm[i].sum() > 0:
            per_class_acc[i] = cm[i, i] / cm[i].sum()
        else:
            per_class_acc[i] = 0.0
    
    return per_class_acc * 100.0  # Convert to percentage


class AverageMeter:
    """Compute and store the average and current value."""
    
    def __init__(self, name: str = ''):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f'{self.name}: {self.avg:.4f} (current: {self.val:.4f})'


if __name__ == '__main__':
    # Test metrics
    print("Testing metrics...")
    
    # Simulate predictions
    outputs = torch.randn(100, 10)
    targets = torch.randint(0, 10, (100,))
    
    # Test accuracy
    acc = compute_accuracy(outputs, targets)
    print(f"✓ Accuracy: {acc:.2f}%")
    
    # Test comprehensive metrics
    metrics = compute_metrics(outputs, targets)
    print(f"✓ Metrics: {metrics}")
    
    # Test confusion matrix
    cm = compute_confusion_matrix(outputs, targets)
    print(f"✓ Confusion matrix shape: {cm.shape}")
    assert cm.shape == (10, 10)
    
    # Test per-class accuracy
    per_class_acc = compute_per_class_accuracy(outputs, targets)
    print(f"✓ Per-class accuracy: {per_class_acc}")
    assert len(per_class_acc) == 10
    
    # Test AverageMeter
    meter = AverageMeter('loss')
    for i in range(10):
        meter.update(np.random.rand())
    print(f"✓ AverageMeter: {meter}")
    
    print("\n✓ All tests passed!")
