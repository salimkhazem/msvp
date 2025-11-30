"""
Configuration management for experiments
"""

from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class Config:
    """Configuration for training experiments."""
    
    # Dataset
    dataset: Literal['mnist', 'fashion', 'cifar10'] = 'cifar10'
    data_dir: str = './data'
    num_workers: int = 4
    
    # Model
    model: Literal['cnn', 'resnet18', 'vit'] = 'resnet18'
    use_prompt: bool = False
    fusion_type: Literal['add', 'concat', 'gated'] = 'add'
    prompt_scales: Literal['global', 'global+mid', 'full'] = 'full'
    
    # Training
    batch_size: int = 128
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: Literal['step', 'cosine', 'onecycle'] = 'cosine'
    
    # OneCycleLR specific
    max_lr: Optional[float] = None  # If None, uses lr
    
    # Optimization
    optimizer: Literal['adam', 'adamw', 'sgd'] = 'adam'
    momentum: float = 0.9  # For SGD
    
    # Regularization
    dropout: float = 0.1
    
    # Checkpointing
    save_dir: str = './experiments/checkpoints'
    log_dir: str = './experiments/logs'
    save_best_only: bool = True
    
    # Hardware
    device: str = 'cuda'
    multi_gpu: bool = False
    
    # Reproducibility
    seed: int = 42
    
    # Misc
    print_freq: int = 50
    val_freq: int = 1  # Validate every N epochs
    
    def __post_init__(self):
        """Validate and adjust configuration."""
        if self.max_lr is None:
            self.max_lr = self.lr * 10  # Common practice for OneCycleLR
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'dataset': self.dataset,
            'model': self.model,
            'use_prompt': self.use_prompt,
            'fusion_type': self.fusion_type,
            'prompt_scales': self.prompt_scales,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'scheduler': self.scheduler,
            'optimizer': self.optimizer,
            'seed': self.seed,
        }
    
    def __str__(self):
        """Pretty print configuration."""
        lines = ["Configuration:"]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


if __name__ == '__main__':
    # Test configuration
    config = Config()
    print(config)
    print("\n" + "="*50)
    
    # Test with custom values
    config_custom = Config(
        dataset='mnist',
        model='cnn',
        use_prompt=True,
        fusion_type='concat',
        epochs=20
    )
    print(config_custom)
