"""
Vision Transformer (ViT-Tiny) with optional Multi-Scale Prompting
"""

import torch
import torch.nn as nn
import math
from typing import Optional
from .prompting import MultiScalePrompting


class PatchEmbedding(nn.Module):
    """Convert image into patches and embed them."""
    
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
    
    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, n_patches_h, n_patches_w)
        x = self.proj(x)
        # Flatten patches: (B, embed_dim, n_patches_h, n_patches_w) -> (B, embed_dim, n_patches)
        x = x.flatten(2)
        # Transpose: (B, embed_dim, n_patches) -> (B, n_patches, embed_dim)
        x = x.transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """MLP block with GELU activation."""
    
    def __init__(self, in_features, hidden_features, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)
    
    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """
    Vision Transformer (Tiny variant) with optional Multi-Scale Prompting.
    
    Args:
        num_classes: Number of output classes
        img_size: Input image size (H=W assumed)
        patch_size: Patch size for patch embedding
        in_channels: Number of input channels (1 for gray, 3 for RGB)
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        dropout: Dropout rate
        use_prompt: Whether to use multi-scale prompting
        fusion_type: Fusion strategy if use_prompt=True
        prompt_scales: Which prompt scales to use
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_prompt: bool = False,
        fusion_type: str = 'add',
        prompt_scales: str = 'full',
    ):
        super().__init__()
        
        self.use_prompt = use_prompt
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Multi-scale prompting (applied before patch embedding)
        if use_prompt:
            self.prompting = MultiScalePrompting(
                in_channels=in_channels,
                fusion_type=fusion_type,
                prompt_scales=prompt_scales,
                input_size=(img_size, img_size)
            )
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        # Initialize patch embedding
        nn.init.kaiming_normal_(self.patch_embed.proj.weight, mode='fan_out', nonlinearity='relu')
        if self.patch_embed.proj.bias is not None:
            nn.init.constant_(self.patch_embed.proj.bias, 0)
        
        # Initialize cls token and pos embedding
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize classification head
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Logits (B, num_classes)
        """
        B = x.shape[0]
        
        # Apply prompting before patch embedding
        if self.use_prompt:
            x = self.prompting(x)
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_token, x], dim=1)  # (B, n_patches + 1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification: use class token
        x = self.norm(x)
        cls_token_final = x[:, 0]  # (B, embed_dim)
        x = self.head(cls_token_final)
        
        return x
    
    def get_attention_maps(self, x: torch.Tensor) -> list:
        """Get attention maps from all blocks (for visualization)."""
        attention_maps = []
        
        B = x.shape[0]
        
        if self.use_prompt:
            x = self.prompting(x)
        
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            # Extract attention from block (simplified - would need to modify block)
            x = block(x)
        
        return attention_maps


if __name__ == '__main__':
    print("Testing ViT model...")
    
    # Test CIFAR10 (RGB, 32x32)
    model_cifar = ViT(
        num_classes=10, 
        img_size=32, 
        patch_size=4, 
        in_channels=3,
        use_prompt=False
    )
    x_cifar = torch.randn(4, 3, 32, 32)
    out_cifar = model_cifar(x_cifar)
    print(f"✓ ViT (CIFAR10, no prompt): input {x_cifar.shape} -> output {out_cifar.shape}")
    
    # Test with prompting
    model_cifar_prompt = ViT(
        num_classes=10, 
        img_size=32, 
        patch_size=4, 
        in_channels=3,
        use_prompt=True
    )
    out_cifar_prompt = model_cifar_prompt(x_cifar)
    print(f"✓ ViT (CIFAR10, with prompt): input {x_cifar.shape} -> output {out_cifar_prompt.shape}")
    
    # Test MNIST (grayscale, 28x28)
    model_mnist = ViT(
        num_classes=10, 
        img_size=28, 
        patch_size=7,  # 28/7 = 4 patches per side
        in_channels=1,
        use_prompt=True
    )
    x_mnist = torch.randn(4, 1, 28, 28)
    out_mnist = model_mnist(x_mnist)
    print(f"✓ ViT (MNIST, with prompt): input {x_mnist.shape} -> output {out_mnist.shape}")
    
    # Test different fusion types
    for fusion in ['add', 'concat', 'gated']:
        model = ViT(
            num_classes=10, 
            img_size=32, 
            patch_size=4, 
            in_channels=3,
            use_prompt=True,
            fusion_type=fusion
        )
        out = model(x_cifar)
        print(f"✓ ViT (fusion={fusion}): output {out.shape}")
    
    # Count parameters
    params_baseline = sum(p.numel() for p in model_cifar.parameters())
    params_prompt = sum(p.numel() for p in model_cifar_prompt.parameters())
    print(f"\nParameters: baseline={params_baseline:,}, with_prompt={params_prompt:,}")
    print(f"Prompt overhead: {params_prompt - params_baseline:,} params")
    
    print("\n✓ All tests passed!")
