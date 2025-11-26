"""
visualize_results.py - Generate training and test result visualizations
Usage: python visualize_results.py <output_dir>
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 10


def plot_training_curves(metrics_df: pd.DataFrame, output_path: Path):
    """Plot training/dev loss and learning rate over epochs."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Loss curves
    epochs = metrics_df['epoch']
    ax1.plot(epochs, metrics_df['train_loss'], label='Train Loss', 
             linewidth=2, marker='o', markersize=3, alpha=0.7)
    ax1.plot(epochs, metrics_df['dev_loss'], label='Dev Loss', 
             linewidth=2, marker='s', markersize=3, alpha=0.7)
    
    # Mark best epoch
    best_idx = metrics_df['dev_loss'].idxmin()
    best_epoch = metrics_df.loc[best_idx, 'epoch']
    best_loss = metrics_df.loc[best_idx, 'dev_loss']
    ax1.axvline(best_epoch, color='red', linestyle='--', alpha=0.5, 
                label=f'Best (Epoch {best_epoch})')
    ax1.plot(best_epoch, best_loss, 'r*', markersize=15, 
             label=f'Min Dev Loss: {best_loss:.4f}')
    
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Learning rate
    ax2.plot(epochs, metrics_df['learning_rate'], 
             linewidth=2, color='green', marker='o', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved training curves to {output_path}")


def plot_binary_metrics(metrics_df: pd.DataFrame, output_path: Path):
    """Plot binary classification metrics as bar chart."""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    values = [float(metrics_df[m].iloc[0]) for m in metrics]
    
    colors = sns.color_palette("husl", len(metrics))
    bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Score')
    ax.set_title('Binary Classification Test Metrics', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved binary metrics to {output_path}")


def plot_multiclass_metrics(metrics_df: pd.DataFrame, output_path: Path):
    """Plot multiclass metrics with per-class breakdown."""
    
    # Separate overall accuracy from per-class metrics
    overall = metrics_df[metrics_df['class'] == 'overall']
    per_class = metrics_df[metrics_df['class'] != 'overall']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Overall accuracy
    acc_val = float(overall['accuracy'].iloc[0])
    ax1.bar(['Accuracy'], [acc_val], color='skyblue', 
            alpha=0.8, edgecolor='black', width=0.5)
    ax1.text(0, acc_val, f'{acc_val:.3f}', 
             ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel('Score')
    ax1.set_title('Overall Test Accuracy', fontsize=12, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Per-class metrics
    classes = per_class['class'].tolist()
    x = np.arange(len(classes))
    width = 0.25
    
    precision = [float(v) for v in per_class['precision']]
    recall = [float(v) for v in per_class['recall']]
    f1 = [float(v) for v in per_class['f1']]
    
    ax2.bar(x - width, precision, width, label='Precision', 
            alpha=0.8, edgecolor='black')
    ax2.bar(x, recall, width, label='Recall', 
            alpha=0.8, edgecolor='black')
    ax2.bar(x + width, f1, width, label='F1', 
            alpha=0.8, edgecolor='black')
    
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Score')
    ax2.set_title('Per-Class Test Metrics', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved multiclass metrics to {output_path}")


def plot_multilabel_metrics(metrics_df: pd.DataFrame, output_path: Path):
    """Plot multilabel metrics with per-label breakdown."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    labels = metrics_df['label'].tolist()
    x = np.arange(len(labels))
    width = 0.28
    
    precision = [float(v) for v in metrics_df['precision']]
    recall = [float(v) for v in metrics_df['recall']]
    f_beta = [float(v) for v in metrics_df['f_beta']]
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', 
                   alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, recall, width, label='Recall', 
                   alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, f_beta, width, label='F-beta', 
                   alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Label')
    ax.set_ylabel('Score')
    ax.set_title('Multilabel Classification Test Metrics', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved multilabel metrics to {output_path}")



def detect_task_type(test_metrics_path: Path) -> str:
    """Infer task type from test metrics CSV structure."""
    df = pd.read_csv(test_metrics_path)
    
    if 'metric' in df.columns:
        return 'binary'
    elif 'class' in df.columns:
        return 'multiclass'
    elif 'label' in df.columns:
        return 'multilabel'
    else:
        raise ValueError("Cannot detect task type from test metrics CSV")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize training metrics and test results"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to model output directory (e.g., is_loanword_gru_20251125_222946/)"
    )
    parser.add_argument(
        "--format",
        choices=['png', 'pdf', 'svg'],
        default='png',
        help="Output image format"
    )
    args = parser.parse_args()
    
    output_dir = args.output_dir
    
    if not output_dir.exists():
        print(f"Error: Directory {output_dir} does not exist")
        return
    
    # Load data
    metrics_file = output_dir / "metrics.csv"
    test_file = output_dir / "test_metrics.csv"
    
    if not metrics_file.exists():
        print(f"Error: {metrics_file} not found")
        return
    
    if not test_file.exists():
        print(f"Error: {test_file} not found")
        return
    
    print(f"Loading data from {output_dir}")
    metrics_df = pd.read_csv(metrics_file)
    test_df = pd.read_csv(test_file)
    
    # Convert string numbers to float
    metrics_df['train_loss'] = metrics_df['train_loss'].astype(float)
    metrics_df['dev_loss'] = metrics_df['dev_loss'].astype(float)
    metrics_df['learning_rate'] = metrics_df['learning_rate'].astype(float)
    
    # Detect task type
    task_type = detect_task_type(test_file)
    print(f"Detected task type: {task_type}")
    
    # Create visualizations directory
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    # Training curves
    plot_training_curves(
        metrics_df, 
        viz_dir / f"training_curves.{args.format}"
    )
    
    # Test metrics
    if task_type == "binary":
        plot_binary_metrics(
            test_df, 
            viz_dir / f"test_metrics.{args.format}"
        )
    elif task_type == "multiclass":
        plot_multiclass_metrics(
            test_df, 
            viz_dir / f"test_metrics.{args.format}"
        )
    elif task_type == "multilabel":
        plot_multilabel_metrics(
            test_df, 
            viz_dir / f"test_metrics.{args.format}"
        )
    
    print(f"\n✓ All visualizations saved to {viz_dir}")
    print("\nGenerated files:")
    print(f"  - training_curves.{args.format}")
    print(f"  - test_metrics.{args.format}")
    print(f"  - summary.txt")


if __name__ == "__main__":
    main()
