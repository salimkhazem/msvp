"""
ResNet18 architecture with optional Multi-Scale Prompting
"""

import torch
import torch.nn as nn
from typing import Optional
from .prompting import MultiScalePrompting


class BasicBlock(nn.Module):
    """Basic residual block for ResNet18."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    """
    ResNet18 architecture with optional Multi-Scale Prompting.
    
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
        
        # Multi-scale prompting (applied before conv1)
        if use_prompt:
            self.prompting = MultiScalePrompting(
                in_channels=in_channels,
                fusion_type=fusion_type,
                prompt_scales=prompt_scales,
                input_size=input_size
            )
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """Create a residual layer."""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
        # Apply prompting before first conv
        if self.use_prompt:
            x = self.prompting(x)
        
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling and classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature maps from layer4 (for GradCAM)."""
        if self.use_prompt:
            x = self.prompting(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x


if __name__ == '__main__':
    print("Testing ResNet18 model...")
    
    # Test CIFAR10 (RGB)
    model_rgb = ResNet18(num_classes=10, in_channels=3, use_prompt=False)
    x_rgb = torch.randn(4, 3, 32, 32)
    out_rgb = model_rgb(x_rgb)
    print(f"✓ ResNet18 (RGB, no prompt): input {x_rgb.shape} -> output {out_rgb.shape}")
    
    # Test with prompting
    model_rgb_prompt = ResNet18(num_classes=10, in_channels=3, use_prompt=True, input_size=(32, 32))
    out_rgb_prompt = model_rgb_prompt(x_rgb)
    print(f"✓ ResNet18 (RGB, with prompt): input {x_rgb.shape} -> output {out_rgb_prompt.shape}")
    
    # Test different fusion types
    for fusion in ['add', 'concat', 'gated']:
        model = ResNet18(num_classes=10, in_channels=3, use_prompt=True, 
                        fusion_type=fusion, input_size=(32, 32))
        out = model(x_rgb)
        print(f"✓ ResNet18 (fusion={fusion}): output {out.shape}")
    
    # Test MNIST (grayscale)
    model_gray = ResNet18(num_classes=10, in_channels=1, use_prompt=True, input_size=(28, 28))
    x_gray = torch.randn(4, 1, 28, 28)
    out_gray = model_gray(x_gray)
    print(f"✓ ResNet18 (grayscale, with prompt): input {x_gray.shape} -> output {out_gray.shape}")
    
    # Count parameters
    params_baseline = sum(p.numel() for p in model_rgb.parameters())
    params_prompt = sum(p.numel() for p in model_rgb_prompt.parameters())
    print(f"\nParameters: baseline={params_baseline:,}, with_prompt={params_prompt:,}")
    print(f"Prompt overhead: {params_prompt - params_baseline:,} params")
    
    print("\n✓ All tests passed!")
