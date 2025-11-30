"""
Ablation Study: Fusion Strategy Comparison

Compare different fusion strategies:
1. Add: x + prompts
2. Concat: Conv1x1(concat(x, prompts))
3. Gated: x + gate(prompts) * prompts
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


def run_experiment(dataset, model, fusion_type, epochs, batch_size, lr, seed):
    """Run a single experiment."""
    exp_name = f"{dataset}_{model}_fusion_{fusion_type}"
    print(f"\n{'='*70}")
    print(f"Running: {exp_name}")
    print(f"{'='*70}\n")
    
    cmd = [
        'python', 'train.py',
        '--dataset', dataset,
        '--model', model,
        '--use_prompt',
        '--fusion_type', fusion_type,
        '--prompt_scales', 'full',  # Use full multi-scale for fair comparison
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
            
            if cfg.get('use_prompt', False) and cfg.get('prompt_scales') == 'full':
                results.append({
                    'fusion_type': cfg['fusion_type'],
                    'dataset': cfg['dataset'],
                    'model': cfg['model'],
                    'val_acc': res['best_val_acc'],
                    'test_acc': res['test_acc'],
                    'test_loss': res['test_loss'],
                    'best_epoch': res['best_epoch']
                })
    
    return pd.DataFrame(results)


def plot_results(df, save_path='./experiments/plots/ablation_fusion.png'):
    """Plot ablation results."""
    
    if len(df) == 0:
        print("No results found!")
        return
    
    # Group by fusion_type and compute mean/std
    grouped = df.groupby('fusion_type').agg({
        'val_acc': ['mean', 'std'],
        'test_acc': ['mean', 'std']
    }).reset_index()
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    fusion_order = ['add', 'concat', 'gated']
    fusion_labels = ['Addition', 'Concatenation', 'Gated']
    
    # Filter and sort
    grouped_sorted = grouped[grouped['fusion_type'].isin(fusion_order)]
    grouped_sorted['fusion_type'] = pd.Categorical(
        grouped_sorted['fusion_type'],
        categories=fusion_order,
        ordered=True
    )
    grouped_sorted = grouped_sorted.sort_values('fusion_type')
    
    # Accuracy plot
    x = range(len(grouped_sorted))
    val_means = grouped_sorted[('val_acc', 'mean')].values
    val_stds = grouped_sorted[('val_acc', 'std')].fillna(0).values
    test_means = grouped_sorted[('test_acc', 'mean')].values
    test_stds = grouped_sorted[('test_acc', 'std')].fillna(0).values
    
    width = 0.35
    ax1.bar([i - width/2 for i in x], val_means, width, yerr=val_stds,
           label='Validation Accuracy', alpha=0.8, capsize=5)
    ax1.bar([i + width/2 for i in x], test_means, width, yerr=test_stds,
           label='Test Accuracy', alpha=0.8, capsize=5)
    
    ax1.set_xlabel('Fusion Strategy', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(fusion_labels, fontsize=11)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # Parameter analysis (simplified - would need actual model counting)
    # For visualization purposes
    param_counts = {
        'add': 100,  # Baseline (no extra params)
        'concat': 150,  # 1x1 conv adds parameters
        'gated': 130  # Gate network adds parameters
    }
    
    params = [param_counts[f] for f in grouped_sorted['fusion_type']]
    
    ax2.bar(x, params, alpha=0.8, color='steelblue')
    ax2.set_xlabel('Fusion Strategy', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Relative Parameters', fontsize=12, fontweight='bold')
    ax2.set_title('Parameter Overhead', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(fusion_labels, fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Ablation Study: Fusion Strategy Comparison', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to {save_path}")
    
    # Print table
    print("\n" + "="*70)
    print("Ablation Results: Fusion Strategies")
    print("="*70)
    print(f"{'Strategy':<20} {'Val Acc (%)':<15} {'Test Acc (%)':<15}")
    print("-"*70)
    
    for i, row in grouped_sorted.iterrows():
        fusion = row['fusion_type']
        val_mean = row[('val_acc', 'mean')]
        val_std = row[('val_acc', 'std')] if pd.notna(row[('val_acc', 'std')]) else 0
        test_mean = row[('test_acc', 'mean')]
        test_std = row[('test_acc', 'std')] if pd.notna(row[('test_acc', 'std')]) else 0
        
        fusion_label = fusion_labels[fusion_order.index(fusion)]
        print(f"{fusion_label:<20} {val_mean:>6.2f} ± {val_std:<5.2f}  {test_mean:>6.2f} ± {test_std:<5.2f}")
    
    print("="*70 + "\n")


def main(args):
    """Main ablation study."""
    
    fusion_types = ['add', 'concat', 'gated']
    
    if not args.skip_training:
        print(f"\n{'='*70}")
        print("ABLATION STUDY: FUSION STRATEGY COMPARISON")
        print(f"{'='*70}\n")
        print(f"Dataset: {args.dataset}")
        print(f"Model: {args.model}")
        print(f"Epochs: {args.epochs}")
        print(f"Configurations: {len(fusion_types)}")
        print()
        
        for fusion in fusion_types:
            success = run_experiment(
                args.dataset,
                args.model,
                fusion,
                args.epochs,
                args.batch_size,
                args.lr,
                args.seed
            )
            
            if not success:
                print(f"Warning: Experiment with {fusion} fusion failed!")
    
    # Collect and plot results
    print("\n" + "="*70)
    print("Collecting results...")
    print("="*70 + "\n")
    
    df = collect_results()
    
    if len(df) > 0:
        plot_results(df, args.save_path)
        
        # Save results to CSV
        csv_path = Path(args.save_path).parent / 'ablation_fusion_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
    else:
        print("No results found! Make sure experiments completed successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ablation: Fusion Strategy Comparison')
    
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
                       default='./experiments/plots/ablation_fusion.png',
                       help='Path to save plot')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training, only collect results')
    
    args = parser.parse_args()
    
    main(args)
