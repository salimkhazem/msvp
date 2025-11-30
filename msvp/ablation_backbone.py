"""
Ablation Study: Backbone Comparison

Compare different backbones with and without Multi-Scale Prompting:
- CNN
- ResNet18
- ViT-Tiny

Across all datasets: MNIST, FashionMNIST, CIFAR10
"""

import subprocess
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

# Set style
sns.set_style("whitegrid")


def run_experiment(dataset, model, use_prompt, epochs, batch_size, lr, seed):
    """Run a single experiment."""
    prompt_str = "with_prompt" if use_prompt else "baseline"
    exp_name = f"{dataset}_{model}_{prompt_str}"
    print(f"\n{'='*70}")
    print(f"Running: {exp_name}")
    print(f"{'='*70}\n")
    
    cmd = [
        'python', 'train.py',
        '--dataset', dataset,
        '--model', model,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--lr', str(lr),
        '--seed', str(seed),
        '--multi_gpu'
    ]
    
    if use_prompt:
        cmd.extend(['--use_prompt', '--fusion_type', 'add', '--prompt_scales', 'full'])
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
        return False


def collect_results(base_dir='./experiments/checkpoints'):
    """Collect results from all experiments."""
    results = []
    
    base_path = Path(base_dir)
    
    for exp_dir in base_path.iterdir():
        if not exp_dir.is_dir():
            continue
        
        results_file = exp_dir / 'results.json'
        config_file = exp_dir / 'config.json'
        
        if results_file.exists() and config_file.exists():
            with open(results_file, 'r') as f:
                res = json.load(f)
            
            with open(config_file, 'r') as f:
                cfg = json.load(f)
            
            results.append({
                'model': cfg['model'],
                'dataset': cfg['dataset'],
                'use_prompt': cfg.get('use_prompt', False),
                'val_acc': res['best_val_acc'],
                'test_acc': res['test_acc'],
                'test_loss': res['test_loss'],
                'best_epoch': res['best_epoch']
            })
    
    return pd.DataFrame(results)


def plot_results(df, save_path='./experiments/plots/ablation_backbone.png'):
    """Plot ablation results."""
    
    if len(df) == 0:
        print("No results found!")
        return
    
    # Create comparison matrix
    datasets = ['mnist', 'fashion', 'cifar10']
    models = ['cnn', 'resnet18', 'vit']
    
    # Create subplots for each dataset
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    dataset_labels = {
        'mnist': 'MNIST',
        'fashion': 'FashionMNIST',
        'cifar10': 'CIFAR-10'
    }
    
    model_labels = {
        'cnn': 'CNN',
        'resnet18': 'ResNet18',
        'vit': 'ViT-Tiny'
    }
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        
        # Filter data for this dataset
        df_dataset = df[df['dataset'] == dataset]
        
        if len(df_dataset) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(dataset_labels[dataset], fontsize=14, fontweight='bold')
            continue
        
        # Prepare data
        baseline_accs = []
        prompt_accs = []
        improvements = []
        
        for model in models:
            baseline = df_dataset[(df_dataset['model'] == model) & 
                                 (~df_dataset['use_prompt'])]
            prompt = df_dataset[(df_dataset['model'] == model) & 
                               (df_dataset['use_prompt'])]
            
            baseline_acc = baseline['test_acc'].mean() if len(baseline) > 0 else 0
            prompt_acc = prompt['test_acc'].mean() if len(prompt) > 0 else 0
            
            baseline_accs.append(baseline_acc)
            prompt_accs.append(prompt_acc)
            improvements.append(prompt_acc - baseline_acc)
        
        # Plot
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, baseline_accs, width, label='Baseline',
                      alpha=0.8, color='steelblue')
        bars2 = ax.bar(x + width/2, prompt_accs, width, label='With MS-VP',
                      alpha=0.8, color='coral')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%',
                           ha='center', va='bottom', fontsize=9)
        
        # Customize
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(dataset_labels[dataset], fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([model_labels[m] for m in models], fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Set y-axis range
        all_accs = baseline_accs + prompt_accs
        if max(all_accs) > 0:
            ax.set_ylim([0, min(100, max(all_accs) * 1.1)])
    
    plt.suptitle('Ablation Study: Backbone Comparison with/without Multi-Scale Prompting',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to {save_path}")
    
    # Print detailed table
    print("\n" + "="*90)
    print("Ablation Results: Backbone Comparison")
    print("="*90)
    
    for dataset in datasets:
        df_dataset = df[df['dataset'] == dataset]
        
        if len(df_dataset) == 0:
            continue
        
        print(f"\n{dataset_labels[dataset]}:")
        print("-"*90)
        print(f"{'Model':<15} {'Baseline (%)':<20} {'With MS-VP (%)':<20} {'Improvement':<15}")
        print("-"*90)
        
        for model in models:
            baseline = df_dataset[(df_dataset['model'] == model) & 
                                 (~df_dataset['use_prompt'])]
            prompt = df_dataset[(df_dataset['model'] == model) & 
                               (df_dataset['use_prompt'])]
            
            baseline_mean = baseline['test_acc'].mean() if len(baseline) > 0 else 0
            baseline_std = baseline['test_acc'].std() if len(baseline) > 0 else 0
            prompt_mean = prompt['test_acc'].mean() if len(prompt) > 0 else 0
            prompt_std = prompt['test_acc'].std() if len(prompt) > 0 else 0
            improvement = prompt_mean - baseline_mean
            
            print(f"{model_labels[model]:<15} "
                  f"{baseline_mean:>6.2f} ± {baseline_std:<5.2f}    "
                  f"{prompt_mean:>6.2f} ± {prompt_std:<5.2f}    "
                  f"{improvement:>+6.2f}%")
    
    print("="*90 + "\n")


def main(args):
    """Main ablation study."""
    
    datasets = ['mnist', 'fashion', 'cifar10'] if not args.dataset else [args.dataset]
    models = ['cnn', 'resnet18', 'vit']
    
    if not args.skip_training:
        print(f"\n{'='*70}")
        print("ABLATION STUDY: BACKBONE COMPARISON")
        print(f"{'='*70}\n")
        print(f"Datasets: {datasets}")
        print(f"Models: {models}")
        print(f"Epochs: {args.epochs}")
        print(f"Total experiments: {len(datasets) * len(models) * 2} (baseline + prompt)")
        print()
        
        for dataset in datasets:
            for model in models:
                # Baseline
                print(f"\n--- {dataset.upper()} / {model.upper()} / BASELINE ---")
                run_experiment(dataset, model, False, args.epochs, 
                             args.batch_size, args.lr, args.seed)
                
                # With prompting
                print(f"\n--- {dataset.upper()} / {model.upper()} / WITH MS-VP ---")
                run_experiment(dataset, model, True, args.epochs, 
                             args.batch_size, args.lr, args.seed)
    
    # Collect and plot results
    print("\n" + "="*70)
    print("Collecting results...")
    print("="*70 + "\n")
    
    df = collect_results()
    
    if len(df) > 0:
        plot_results(df, args.save_path)
        
        # Save results to CSV
        csv_path = Path(args.save_path).parent / 'ablation_backbone_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
    else:
        print("No results found! Make sure experiments completed successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ablation: Backbone Comparison')
    
    parser.add_argument('--dataset', type=str, default=None,
                       choices=['mnist', 'fashion', 'cifar10'],
                       help='Specific dataset (if None, runs all)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_path', type=str, 
                       default='./experiments/plots/ablation_backbone.png',
                       help='Path to save plot')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training, only collect results')
    
    args = parser.parse_args()
    
    main(args)
