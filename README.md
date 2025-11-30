# Multi-Scale Visual Prompting (MS-VP) for Image Classification

A comprehensive PyTorch research codebase for exploring Multi-Scale Visual Prompting across multiple architectures and datasets.

## 🎯 Overview

This project implements **Multi-Scale Visual Prompting**, a parameter-efficient method for enhancing image classification models by learning visual prompts at multiple scales (global, mid-level, and local).

**Key Features:**
- 🔬 Three backbone architectures: CNN, ResNet18, ViT-Tiny
- 📊 Three benchmark datasets: MNIST, FashionMNIST, CIFAR-10
- 🎨 Multiple fusion strategies: Addition, Concatenation, Gated
- 🔍 Comprehensive ablation studies
- 📈 Publication-quality visualizations (GradCAM, prompts, metrics)
- ⚡ Multi-GPU training support

## 📁 Project Structure

```
msvp/
├── datasets/               # Dataset loaders
│   ├── __init__.py
│   └── loaders.py         # MNIST, FashionMNIST, CIFAR-10
├── models/                # Model architectures
│   ├── __init__.py
│   ├── prompting.py       # Multi-Scale Prompting module
│   ├── cnn.py            # Baseline CNN
│   ├── resnet.py         # ResNet18 with MS-VP
│   └── vit.py            # ViT-Tiny with MS-VP
├── utils/                 # Utilities
│   ├── __init__.py
│   ├── config.py         # Configuration management
│   ├── training.py       # Training loops
│   ├── metrics.py        # Evaluation metrics
│   └── visualization.py  # Plotting and GradCAM
├── experiments/           # Experiment outputs
│   ├── checkpoints/      # Model checkpoints
│   ├── logs/             # Training logs
│   └── plots/            # Visualizations
├── train.py              # Main training script
├── ablation_prompt_scale.py   # Ablation: prompt scales
├── ablation_fusion.py         # Ablation: fusion strategies
├── ablation_backbone.py       # Ablation: backbones
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
cd ./msvp
pip install -r requirements.txt
```

### Basic Training

Train ResNet18 on CIFAR-10 with Multi-Scale Prompting:

```bash
python train.py \
  --dataset cifar10 \
  --model resnet18 \
  --use_prompt \
  --epochs 10 \
  --batch_size 128 \
  --multi_gpu
```

Train baseline (without prompting):

```bash
python train.py \
  --dataset cifar10 \
  --model resnet18 \
  --epochs 10 \
  --batch_size 128 \
  --multi_gpu
```

### Available Options

**Datasets:** `mnist`, `fashion`, `cifar10`  
**Models:** `cnn`, `resnet18`, `vit`  
**Fusion Types:** `add`, `concat`, `gated`  
**Prompt Scales:** `global`, `global+mid`, `full`

### Multi-GPU Training

The codebase automatically uses all available GPUs when `--multi_gpu` is specified:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python train.py \
  --dataset cifar10 \
  --model resnet18 \
  --use_prompt \
  --multi_gpu
```

## 🔬 Ablation Studies

### 1. Prompt Scale Ablation

Compare global-only, global+mid, and full multi-scale prompting:

```bash
python ablation_prompt_scale.py \
  --dataset cifar10 \
  --model resnet18 \
  --epochs 10
```

**Output:** 
- Results table comparing configurations
- Bar plot: `experiments/plots/ablation_prompt_scale.png`
- CSV: `experiments/plots/ablation_prompt_scale_results.csv`

### 2. Fusion Strategy Ablation

Compare addition, concatenation, and gated fusion:

```bash
python ablation_fusion.py \
  --dataset cifar10 \
  --model resnet18 \
  --epochs 10
```

**Output:** 
- Comparison of fusion strategies
- Plots: `experiments/plots/ablation_fusion.png`
- CSV: `experiments/plots/ablation_fusion_results.csv`

### 3. Backbone Ablation

Compare CNN, ResNet18, and ViT across all datasets:

```bash
# Run all datasets (will take longer)
python ablation_backbone.py --epochs 10

# Or specific dataset
python ablation_backbone.py --dataset cifar10 --epochs 10
```

**Output:**
- 3×2 accuracy matrix (3 models × baseline/prompt)
- Plots: `experiments/plots/ablation_backbone.png`
- CSV: `experiments/plots/ablation_backbone_results.csv`

## 📊 Visualization

Visualizations are automatically generated during training and saved to `experiments/plots/`.

### Training Curves

Automatically saved after training to `experiments/logs/<exp_name>/training_curves.png`

### Prompt Visualization

Visualize learned prompts from a trained model:

```python
from models import ResNet18
from utils import visualize_prompts
import torch

model = ResNet18(in_channels=3, use_prompt=True)
model.load_state_dict(torch.load('path/to/checkpoint.pth')['model_state_dict'])

prompts = model.prompting.get_prompts()
visualize_prompts(prompts, save_path='prompts_viz.png')
```

### GradCAM Heatmaps

Compare attention patterns between baseline and MS-VP models:

```python
from utils import visualize_gradcam
from models import ResNet18
import torch

model = ResNet18(in_channels=3, use_prompt=True)
model.load_state_dict(torch.load('path/to/checkpoint.pth')['model_state_dict'])
model.eval()

# Get a sample image
x = torch.randn(1, 3, 32, 32).cuda()

# Target the last conv layer
target_layer = model.layer4[-1].conv2  # For ResNet18

visualize_gradcam(
    model, x, target_layer,
    save_path='gradcam.png',
    title='GradCAM: ResNet18 with MS-VP'
)
```

## 📈 Expected Results

### MNIST (10 epochs)
- **Baseline CNN**: ~98-99%
- **CNN + MS-VP**: ~99%+
- **ResNet18 + MS-VP**: ~99.5%+

### CIFAR-10 (10 epochs)
- **Baseline ResNet18**: ~88-90%
- **ResNet18 + MS-VP**: ~90-92%
- **ViT-Tiny + MS-VP**: ~85-88%

*Note: Results may vary. For publication, run 3 seeds and report mean ± std.*

## 🛠 Advanced Usage

### Custom Hyperparameters

```bash
python train.py \
  --dataset cifar10 \
  --model vit \
  --use_prompt \
  --fusion_type gated \
  --prompt_scales full \
  --epochs 50 \
  --batch_size 256 \
  --lr 1e-3 \
  --scheduler onecycle \
  --optimizer adamw \
  --weight_decay 5e-4 \
  --dropout 0.1 \
  --seed 42
```

### Resume Training

```python
from utils import load_checkpoint
import torch

checkpoint = load_checkpoint(
    'experiments/checkpoints/best_model.pth',
    model, optimizer, scheduler
)

start_epoch = checkpoint['epoch'] + 1
best_acc = checkpoint['best_val_acc']
```

## 🔧 Configuration Files

After each training run, configuration and results are saved:

- `experiments/checkpoints/<exp_name>/config.json` - Full configuration
- `experiments/checkpoints/<exp_name>/results.json` - Final metrics
- `experiments/checkpoints/<exp_name>/best_model.pth` - Best checkpoint
- `experiments/logs/<exp_name>/training_log.json` - Training history


