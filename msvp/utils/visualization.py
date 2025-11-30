"""
Visualization utilities for prompts, GradCAM, training curves, and confusion matrices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple
import cv2


# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def plot_training_curves(
    save_path: str,
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    title: str = 'Training Curves'
):
    """
    Plot training and validation curves.
    
    Args:
        save_path: Path to save figure
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        title: Figure title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'o-', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, val_losses, 's-', label='Val Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss vs Epoch', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'o-', label='Train Acc', linewidth=2, markersize=4)
    ax2.plot(epochs, val_accs, 's-', label='Val Acc', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy vs Epoch', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: str,
    class_names: Optional[List[str]] = None,
    title: str = 'Confusion Matrix',
    normalize: bool = True
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix (num_classes, num_classes)
        save_path: Path to save figure
        class_names: Optional class names
        title: Figure title
        normalize: Whether to normalize by row (true labels)
    """
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = '.2f'
    else:
        fmt = 'd'
    
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count' if normalize else 'Count'}
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def visualize_prompts(
    prompts: dict,
    save_path: str,
    target_size: int = 224,
    title: str = 'Learned Prompts'
):
    """
    Visualize learned prompts.
    
    Args:
        prompts: Dictionary with 'global', 'mid', 'local' prompt tensors
        save_path: Path to save figure
        target_size: Size to upsample prompts to
        title: Figure title
    """
    num_prompts = len(prompts)
    fig, axes = plt.subplots(1, num_prompts, figsize=(5 * num_prompts, 5))
    
    if num_prompts == 1:
        axes = [axes]
    
    prompt_names = ['global', 'mid', 'local']
    
    for idx, (name, prompt_tensor) in enumerate(prompts.items()):
        # prompt_tensor shape: (1, C, H, W)
        # Upsample to target size
        upsampled = F.interpolate(
            prompt_tensor,
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # (C, H, W)
        
        # Average across channels and normalize
        if upsampled.shape[0] > 1:
            prompt_vis = upsampled.mean(dim=0).cpu().numpy()  # (H, W)
        else:
            prompt_vis = upsampled.squeeze(0).cpu().numpy()  # (H, W)
        
        # Normalize to [0, 1]
        prompt_vis = (prompt_vis - prompt_vis.min()) / (prompt_vis.max() - prompt_vis.min() + 1e-8)
        
        # Plot
        im = axes[idx].imshow(prompt_vis, cmap='viridis')
        axes[idx].set_title(f'{name.capitalize()} Prompt\n({prompt_tensor.shape[2]}×{prompt_tensor.shape[3]} → {target_size}×{target_size})', 
                           fontsize=12, fontweight='bold')
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Prompt visualization saved to {save_path}")


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Args:
        model: Model to visualize
        target_layer: Target layer for CAM (e.g., last conv layer)
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Hook to save activations."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate GradCAM heatmap.
        
        Args:
            input_tensor: Input image (1, C, H, W)
            target_class: Target class index (if None, uses predicted class)
            
        Returns:
            Heatmap as numpy array (H, W)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


def visualize_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: nn.Module,
    save_path: str,
    class_idx: Optional[int] = None,
    title: str = 'GradCAM Visualization'
):
    """
    Visualize GradCAM heatmap overlaid on input image.
    
    Args:
        model: Model to visualize
        input_tensor: Input image (1, C, H, W)
        target_layer: Target layer for GradCAM
        save_path: Path to save figure
        class_idx: Target class (if None, uses predicted)
        title: Figure title
    """
    gradcam = GradCAM(model, target_layer)
    
    # Generate heatmap
    heatmap = gradcam.generate(input_tensor, class_idx)
    
    # Prepare input image
    img = input_tensor.squeeze(0).cpu().numpy()  # (C, H, W)
    if img.shape[0] == 1:
        # Grayscale
        img = img.squeeze(0)  # (H, W)
    else:
        # RGB
        img = img.transpose(1, 2, 0)  # (H, W, C)
    
    # Normalize image to [0, 1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Apply colormap
    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
    
    # Overlay
    if len(img.shape) == 2:
        # Convert grayscale to RGB for overlay
        img_rgb = np.stack([img, img, img], axis=-1)
    else:
        img_rgb = img
    
    overlay = 0.6 * img_rgb + 0.4 * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    if len(img.shape) == 2:
        axes[0].imshow(img, cmap='gray')
    else:
        axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title('GradCAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"GradCAM visualization saved to {save_path}")


if __name__ == '__main__':
    print("Testing visualization utilities...")
    
    # Test training curves
    train_losses = [0.5, 0.4, 0.3, 0.25, 0.2]
    val_losses = [0.6, 0.5, 0.4, 0.35, 0.3]
    train_accs = [80, 85, 88, 90, 92]
    val_accs = [75, 80, 83, 85, 87]
    
    plot_training_curves(
        './test_plots/training_curves.png',
        train_losses, val_losses, train_accs, val_accs
    )
    print("✓ Training curves plotted")
    
    # Test confusion matrix
    cm = np.random.randint(0, 100, (10, 10))
    plot_confusion_matrix(cm, './test_plots/confusion_matrix.png')
    print("✓ Confusion matrix plotted")
    
    # Test prompt visualization
    prompts = {
        'global': torch.randn(1, 3, 1, 1),
        'mid': torch.randn(1, 3, 4, 4),
        'local': torch.randn(1, 3, 8, 8)
    }
    visualize_prompts(prompts, './test_plots/prompts.png')
    print("✓ Prompts visualized")
    
    print("\n✓ All visualization tests passed!")
