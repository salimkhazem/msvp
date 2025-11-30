"""
Baseline CNN architecture with optional Multi-Scale Prompting
"""

import torch
import torch.nn as nn
from typing import Optional
from .prompting import MultiScalePrompting


class CNN(nn.Module):
    """
    Baseline CNN architecture for image classification.
    
    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels (1 for gray, 3 for RGB)
        use_prompt: Whether to use multi-scale prompting
        fusion_type: Fusion strategy if use_prompt=True
        prompt_scales: Which prompt scales to use
        input_size: Input spatial dimensions
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        use_prompt: bool = False,
        fusion_type: str = 'add',
        prompt_scales: str = 'full',
        input_size: tuple[int, int] = (32, 32)
    ):
        super().__init__()
        
        self.use_prompt = use_prompt
        self.in_channels = in_channels
        
        # Multi-scale prompting (optional)
        if use_prompt:
            self.prompting = MultiScalePrompting(
                in_channels=in_channels,
                fusion_type=fusion_type,
                prompt_scales=prompt_scales,
                input_size=input_size
            )
        
        # CNN backbone
        self.features = nn.Sequential(
            # Block 1: in_channels -> 32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # /2
            
            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # /4
            
            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # /8
            
            # Block 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Logits (B, num_classes)
        """
        # Apply prompting if enabled
        if self.use_prompt:
            x = self.prompting(x)
        
        # Feature extraction
        x = self.features(x)
        
        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature maps before global pooling (for GradCAM)."""
        if self.use_prompt:
            x = self.prompting(x)
        x = self.features(x)
        return x


if __name__ == '__main__':
    print("Testing CNN model...")
    
    # Test CIFAR10 (RGB)
    model_rgb = CNN(num_classes=10, in_channels=3, use_prompt=False)
    x_rgb = torch.randn(4, 3, 32, 32)
    out_rgb = model_rgb(x_rgb)
    print(f"✓ CNN (RGB, no prompt): input {x_rgb.shape} -> output {out_rgb.shape}")
    
    # Test with prompting
    model_rgb_prompt = CNN(num_classes=10, in_channels=3, use_prompt=True, input_size=(32, 32))
    out_rgb_prompt = model_rgb_prompt(x_rgb)
    print(f"✓ CNN (RGB, with prompt): input {x_rgb.shape} -> output {out_rgb_prompt.shape}")
    
    # Test MNIST (grayscale)
    model_gray = CNN(num_classes=10, in_channels=1, use_prompt=True, input_size=(28, 28))
    x_gray = torch.randn(4, 1, 28, 28)
    out_gray = model_gray(x_gray)
    print(f"✓ CNN (grayscale, with prompt): input {x_gray.shape} -> output {out_gray.shape}")
    
    # Count parameters
    params_baseline = sum(p.numel() for p in model_rgb.parameters())
    params_prompt = sum(p.numel() for p in model_rgb_prompt.parameters())
    print(f"\nParameters: baseline={params_baseline:,}, with_prompt={params_prompt:,}")
    print(f"Prompt overhead: {params_prompt - params_baseline:,} params")
    
    print("\n✓ All tests passed!")
