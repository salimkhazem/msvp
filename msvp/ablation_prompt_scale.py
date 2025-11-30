"""
Ablation Study: Prompt Scale Comparison

Compare different prompt scale configurations:
1. Global only: (C, 1, 1)
2. Global + Mid: (C, 1, 1) + (C, 4, 4)
3. Full multi-scale: (C, 1, 1) + (C, 4, 4) + (C, 8, 8)
"""

import subprocess
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Set style
sns.set_style("whitegrid")


def run_experiment(dataset, model, prompt_scale, epochs, batch_size, lr, seed):
    """Run a single experiment."""
    exp_name = f"{dataset}_{model}_{prompt_scale}"
    print(f"\n{'='*70}")
    print(f"Running: {exp_name}")
    print(f"{'='*70}\n")
    
    cmd = [
        'python', 'train.py',
        '--dataset', dataset,
        '--model', model,
        '--use_prompt',
        '--prompt_scales', prompt_scale,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--lr', str(lr),
        '--seed', str(seed),
        '--multi_gpu'
    ]
    
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
            
            if cfg.get('use_prompt', False):
                results.append({
                    'prompt_scales': cfg['prompt_scales'],
                    'dataset': cfg['dataset'],
                    'model': cfg['model'],
                    'val_acc': res['best_val_acc'],
                    'test_acc': res['test_acc'],
                    'test_loss': res['test_loss'],
                    'best_epoch': res['best_epoch'],
                    'fusion_type': cfg.get('fusion_type', 'add')
                })
    
    return pd.DataFrame(results)


def plot_results(df, save_path='./experiments/plots/ablation_prompt_scale.png'):
    """Plot ablation results."""
    
    # Filter results for this ablation (add fusion only)
    df_filtered = df[df['fusion_type'] == 'add'].copy()
    
    if len(df_filtered) == 0:
        print("No results found!")
        return
    
    # Group by prompt_scales and compute mean/std
    grouped = df_filtered.groupby('prompt_scales').agg({
        'val_acc': ['mean', 'std'],
        'test_acc': ['mean', 'std']
    }).reset_index()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scales_order = ['global', 'global+mid', 'full']
    scales_labels = ['Global Only', 'Global + Mid', 'Full Multi-Scale']
    
    # Filter and sort
    grouped_sorted = grouped[grouped['prompt_scales'].isin(scales_order)]
    grouped_sorted['prompt_scales'] = pd.Categorical(
        grouped_sorted['prompt_scales'],
        categories=scales_order,
        ordered=True
    )
    grouped_sorted = grouped_sorted.sort_values('prompt_scales')
    
    x = range(len(grouped_sorted))
    val_means = grouped_sorted[('val_acc', 'mean')].values
    val_stds = grouped_sorted[('val_acc', 'std')].fillna(0).values
    test_means = grouped_sorted[('test_acc', 'mean')].values
    test_stds = grouped_sorted[('test_acc', 'std')].fillna(0).values
    
    # Plot bars
    width = 0.35
    ax.bar([i - width/2 for i in x], val_means, width, yerr=val_stds,
           label='Validation Accuracy', alpha=0.8, capsize=5)
    ax.bar([i + width/2 for i in x], test_means, width, yerr=test_stds,
           label='Test Accuracy', alpha=0.8, capsize=5)
    
    # Customize
    ax.set_xlabel('Prompt Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study: Prompt Scale Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scales_labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to {save_path}")
    
    # Print table
    print("\n" + "="*70)
    print("Ablation Results: Prompt Scales")
    print("="*70)
    print(f"{'Configuration':<20} {'Val Acc (%)':<15} {'Test Acc (%)':<15}")
    print("-"*70)
    
    for i, row in grouped_sorted.iterrows():
        scale = row['prompt_scales']
        val_mean = row[('val_acc', 'mean')]
        val_std = row[('val_acc', 'std')] if pd.notna(row[('val_acc', 'std')]) else 0
        test_mean = row[('test_acc', 'mean')]
        test_std = row[('test_acc', 'std')] if pd.notna(row[('test_acc', 'std')]) else 0
        
        scale_label = scales_labels[scales_order.index(scale)]
        print(f"{scale_label:<20} {val_mean:>6.2f} ± {val_std:<5.2f}  {test_mean:>6.2f} ± {test_std:<5.2f}")
    
    print("="*70 + "\n")


def main(args):
    """Main ablation study."""
    
    scales = ['global', 'global+mid', 'full']
    
    if not args.skip_training:
        print(f"\n{'='*70}")
        print("ABLATION STUDY: PROMPT SCALE COMPARISON")
        print(f"{'='*70}\n")
        print(f"Dataset: {args.dataset}")
        print(f"Model: {args.model}")
        print(f"Epochs: {args.epochs}")
        print(f"Configurations: {len(scales)}")
        print()
        
        for scale in scales:
            success = run_experiment(
                args.dataset,
                args.model,
                scale,
                args.epochs,
                args.batch_size,
                args.lr,
                args.seed
            )
            
            if not success:
                print(f"Warning: Experiment with {scale} failed!")
    
    # Collect and plot results
    print("\n" + "="*70)
    print("Collecting results...")
    print("="*70 + "\n")
    
    df = collect_results()
    
    if len(df) > 0:
        plot_results(df, args.save_path)
        
        # Save results to CSV
        csv_path = Path(args.save_path).parent / 'ablation_prompt_scale_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
    else:
        print("No results found! Make sure experiments completed successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ablation: Prompt Scale Comparison')
    
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['mnist', 'fashion', 'cifar10'],
                       help='Dataset to use')
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['cnn', 'resnet18', 'vit'],
                       help='Model to use')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_path', type=str, 
                       default='./experiments/plots/ablation_prompt_scale.png',
                       help='Path to save plot')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training, only collect results')
    
    args = parser.parse_args()
    
    main(args)
