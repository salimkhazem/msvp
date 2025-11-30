"""
Multi-Scale Visual Prompting Module

This module implements learnable visual prompts at multiple scales (global, mid, local)
that can be fused with input features using various strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class MultiScalePrompting(nn.Module):
    """
    Multi-Scale Visual Prompting module.
    
    Args:
        in_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        fusion_type: How to fuse prompts with input ('add', 'concat', 'gated')
        prompt_scales: Which scales to use ('global', 'global+mid', 'full')
        input_size: Input spatial size (H, W) - used for proper upsampling
    """
    
    def __init__(
        self,
        in_channels: int,
        fusion_type: Literal['add', 'concat', 'gated'] = 'add',
        prompt_scales: Literal['global', 'global+mid', 'full'] = 'full',
        input_size: tuple[int, int] = (32, 32)
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.fusion_type = fusion_type
        self.prompt_scales = prompt_scales
        self.input_size = input_size
        
        # Learnable prompts at different scales
        self.global_prompt = nn.Parameter(torch.randn(1, in_channels, 1, 1) * 0.01)
        
        if prompt_scales in ['global+mid', 'full']:
            self.mid_prompt = nn.Parameter(torch.randn(1, in_channels, 4, 4) * 0.01)
        else:
            self.register_parameter('mid_prompt', None)
            
        if prompt_scales == 'full':
            self.local_prompt = nn.Parameter(torch.randn(1, in_channels, 8, 8) * 0.01)
        else:
            self.register_parameter('local_prompt', None)
        
        # Fusion layers
        if fusion_type == 'concat':
            # Concatenation requires 1x1 conv to reduce channels back
            num_prompts = self._count_active_prompts()
            self.fusion_conv = nn.Conv2d(
                in_channels * (num_prompts + 1),  # input + prompts
                in_channels,
                kernel_size=1,
                bias=False
            )
        elif fusion_type == 'gated':
            # Gated fusion: learn a gate for each prompt
            self.gate = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.Sigmoid()
            )
        
        self._initialize_weights()
    
    def _count_active_prompts(self) -> int:
        """Count number of active prompt scales."""
        count = 1  # global always present
        if self.mid_prompt is not None:
            count += 1
        if self.local_prompt is not None:
            count += 1
        return count
    
    def _initialize_weights(self):
        """Initialize fusion layers with appropriate initialization."""
        if hasattr(self, 'fusion_conv'):
            nn.init.kaiming_normal_(self.fusion_conv.weight, mode='fan_out', nonlinearity='relu')
        
        if hasattr(self, 'gate'):
            for m in self.gate.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-scale prompting to input.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Prompted tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Upsample all prompts to input resolution
        prompts = []
        
        # Global prompt
        global_up = F.interpolate(
            self.global_prompt.expand(B, -1, -1, -1),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        prompts.append(global_up)
        
        # Mid prompt
        if self.mid_prompt is not None:
            mid_up = F.interpolate(
                self.mid_prompt.expand(B, -1, -1, -1),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            prompts.append(mid_up)
        
        # Local prompt
        if self.local_prompt is not None:
            local_up = F.interpolate(
                self.local_prompt.expand(B, -1, -1, -1),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            prompts.append(local_up)
        
        # Fuse prompts
        prompts_fused = sum(prompts) / len(prompts)  # Average all prompts
        
        # Apply fusion strategy
        if self.fusion_type == 'add':
            # Simple elementwise addition
            output = x + prompts_fused
            
        elif self.fusion_type == 'concat':
            # Concatenate and reduce with 1x1 conv
            concat_features = torch.cat([x] + prompts, dim=1)
            output = self.fusion_conv(concat_features)
            
        elif self.fusion_type == 'gated':
            # Gated fusion: x + gate * prompts
            gate = self.gate(prompts_fused)
            output = x + gate * prompts_fused
        
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        
        return output
    
    def get_prompts(self) -> dict[str, torch.Tensor]:
        """
        Get the learned prompts for visualization.
        
        Returns:
            Dictionary with prompt tensors
        """
        prompts = {'global': self.global_prompt.detach()}
        
        if self.mid_prompt is not None:
            prompts['mid'] = self.mid_prompt.detach()
        
        if self.local_prompt is not None:
            prompts['local'] = self.local_prompt.detach()
        
        return prompts


if __name__ == '__main__':
    # Quick test
    print("Testing MultiScalePrompting module...")
    
    # Test with RGB input (CIFAR10)
    x_rgb = torch.randn(4, 3, 32, 32)
    
    # Test different configurations
    configs = [
        ('add', 'full'),
        ('concat', 'full'),
        ('gated', 'full'),
        ('add', 'global+mid'),
        ('add', 'global'),
    ]
    
    for fusion, scales in configs:
        model = MultiScalePrompting(
            in_channels=3,
            fusion_type=fusion,
            prompt_scales=scales,
            input_size=(32, 32)
        )
        
        out = model(x_rgb)
        num_params = sum(p.numel() for p in model.parameters())
        
        print(f"✓ {fusion:8s} + {scales:12s}: output shape {out.shape}, params: {num_params}")
        assert out.shape == x_rgb.shape, "Output shape mismatch!"
    
    # Test with grayscale input (MNIST)
    x_gray = torch.randn(4, 1, 28, 28)
    model_gray = MultiScalePrompting(in_channels=1, input_size=(28, 28))
    out_gray = model_gray(x_gray)
    print(f"✓ Grayscale test: output shape {out_gray.shape}")
    assert out_gray.shape == x_gray.shape
    
    print("\n✓ All tests passed!")
