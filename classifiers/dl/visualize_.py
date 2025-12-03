import argparse
from pathlib import Path
from typing import Optional, Callable, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from classifiers.dl.models_ import create_model, CharLanguageModel


# Matplotlib config
sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'font.size': 10
})


# ============================================================================
# Utilities
# ============================================================================

def safe_import(module_name: str, package: str = None):
    """Safely import with helpful error message."""
    try:
        if package:
            return __import__(module_name, fromlist=[package])
        return __import__(module_name)
    except ImportError:
        print(f"Warning: {module_name} not installed")
        print(f"Install with: pip install {module_name}")
        return None


def recreate_model(checkpoint: dict, load_weights: bool = True):
    """Recreate model from checkpoint."""
    from classifiers.dl.models_ import create_model
    
    model_config = checkpoint['model_config']
    task_config = checkpoint['task_config']
    
    # CharLM encoder handling
    charlm_encoder = None
    if model_config.use_charlm:
        if "charlm_state" not in checkpoint or "charlm_config" not in checkpoint:
            print("Warning: Model uses CharLM but checkpoint missing charlm_state/charlm_config")
        else:
            charlm_config = checkpoint["charlm_config"]
            charlm_encoder = CharLanguageModel(charlm_config)
            charlm_encoder.load_state_dict(checkpoint["charlm_state"])
            charlm_encoder.eval()
            for param in charlm_encoder.parameters():
                param.requires_grad = False
            print("✓ Loaded CharLM encoder from checkpoint")
    
    model = create_model(model_config, task_config, charlm_encoder)
    
    if load_weights:
        model.load_state_dict(checkpoint['model_state'])
    
    model.eval()
    return model


def create_dummy_input(checkpoint: dict, batch_size: int = 4):
    """Create dummy input for model."""
    model_config = checkpoint['model_config']
    train_config = checkpoint['train_config']
    seq_len = train_config.max_seq_len
    
    if model_config.use_byt5:
        return torch.randn(batch_size, seq_len, model_config.byt5_dim)
    else:
        return torch.randint(0, model_config.vocab_size, (batch_size, seq_len))


def find_best_checkpoint(output_dir: Path) -> Optional[Path]:
    """Auto-find best checkpoint in models/ directory."""
    models_dir = output_dir / "models"
    if not models_dir.exists():
        return None
    
    checkpoints = list(models_dir.glob("*.pt"))
    if not checkpoints:
        return None
    
    # Find checkpoint with lowest loss in filename
    return min(
        checkpoints,
        key=lambda p: float(p.stem.split('loss')[-1]) 
        if 'loss' in p.stem else float('inf')
    )


# ============================================================================
# Training & Test Metrics Visualization
# ============================================================================

def plot_training_curves(metrics_df: pd.DataFrame, output_path: Path):
    """Plot training/dev loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = metrics_df['epoch']
    
    # Loss curves
    ax.plot(epochs, metrics_df['train_loss'], label='Train', 
            linewidth=2, marker='o', markersize=3, alpha=0.7)
    ax.plot(epochs, metrics_df['dev_loss'], label='Dev', 
            linewidth=2, marker='s', markersize=3, alpha=0.7)
    
    # Mark best epoch
    best_idx = metrics_df['dev_loss'].idxmin()
    best_epoch = metrics_df.loc[best_idx, 'epoch']
    best_loss = metrics_df.loc[best_idx, 'dev_loss']
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
    ax.plot(best_epoch, best_loss, 'r*', markersize=15, 
            label=f'Best (epoch {best_epoch}): {best_loss:.4f}')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss', fontweight='bold', fontsize=14)
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Saved training curves to {output_path}")


def plot_metrics_bar(metrics: dict, output_path: Path, title: str):
    """Generic bar plot for metrics."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    names = list(metrics.keys())
    values = list(metrics.values())
    colors = sns.color_palette("husl", len(names))
    
    bars = ax.bar(names, values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Score')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Saved {title.lower()} to {output_path}")


def plot_binary_metrics(df: pd.DataFrame, output_path: Path):
    """Plot binary classification metrics."""
    metrics = {m: float(df[m].iloc[0]) for m in ['accuracy', 'precision', 'recall', 'f1']}
    plot_metrics_bar(metrics, output_path, "Binary Classification Metrics")


def plot_multiclass_metrics(df: pd.DataFrame, output_path: Path):
    """Plot multiclass metrics."""
    overall = df[df['class'] == 'overall']
    per_class = df[df['class'] != 'overall']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Overall accuracy
    acc = float(overall['accuracy'].iloc[0])
    ax1.bar(['Accuracy'], [acc], color='skyblue', alpha=0.8, edgecolor='black')
    ax1.text(0, acc, f'{acc:.3f}', ha='center', va='bottom', 
             fontweight='bold', fontsize=12)
    ax1.set_ylim(0, 1.0)
    ax1.set_title('Overall Accuracy', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Per-class metrics
    classes = per_class['class'].tolist()
    x = np.arange(len(classes))
    width = 0.25
    
    for i, (metric, offset) in enumerate([('precision', -width), 
                                           ('recall', 0), 
                                           ('f1', width)]):
        values = [float(v) for v in per_class[metric]]
        ax2.bar(x + offset, values, width, label=metric.capitalize(), 
                alpha=0.8, edgecolor='black')
    
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Score')
    ax2.set_title('Per-Class Metrics', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Saved multiclass metrics to {output_path}")


def plot_multilabel_metrics(df: pd.DataFrame, output_path: Path):
    """Plot multilabel metrics."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    labels = df['label'].tolist()
    x = np.arange(len(labels))
    width = 0.28
    
    metrics_data = [
        ('precision', -width),
        ('recall', 0),
        ('f_beta', width)
    ]
    
    for metric, offset in metrics_data:
        values = [float(v) for v in df[metric]]
        bars = ax.bar(x + offset, values, width, label=metric.capitalize(), 
                      alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Label')
    ax.set_ylabel('Score')
    ax.set_title('Multilabel Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Saved multilabel metrics to {output_path}")


def detect_and_plot_test_metrics(test_file: Path, viz_dir: Path, fmt: str):
    """Auto-detect task type and plot test metrics."""
    df = pd.read_csv(test_file)
    
    # Detect task type
    if 'metric' in df.columns:
        task_type = 'binary'
        plot_binary_metrics(df, viz_dir / f"test_metrics.{fmt}")
    elif 'class' in df.columns:
        task_type = 'multiclass'
        plot_multiclass_metrics(df, viz_dir / f"test_metrics.{fmt}")
    elif 'label' in df.columns:
        task_type = 'multilabel'
        plot_multilabel_metrics(df, viz_dir / f"test_metrics.{fmt}")
    else:
        raise ValueError("Unknown test metrics format")
    
    return task_type


# ============================================================================
# Model Architecture Visualizations
# ============================================================================

def viz_simple_text(checkpoint: dict, output_path: Path):
    """Simple text architecture summary."""
    model = recreate_model(checkpoint)
    
    lines = [
        "=" * 80,
        "MODEL ARCHITECTURE",
        "=" * 80,
        str(model),
        "",
        "=" * 80,
        "PARAMETERS",
        "=" * 80,
        f"{'Train':^5} {'Name':<50} {'Shape':<30} {'Count':>12}",
        "-" * 80,
    ]
    
    total = trainable = 0
    for name, param in model.named_parameters():
        count = param.numel()
        marker = "  ✓  " if param.requires_grad else "  ✗  "
        lines.append(
            f"{marker} {name:<50} {str(list(param.shape)):<30} {count:>12,}"
        )
        total += count
        if param.requires_grad:
            trainable += count
    
    lines.extend([
        "-" * 80,
        f"Total:     {total:>12,}",
        f"Trainable: {trainable:>12,}",
        f"Frozen:    {total - trainable:>12,}",
        "=" * 80
    ])
    
    output_path.write_text("\n".join(lines), encoding='utf-8')
    print(f"✓ Saved simple architecture to {output_path}")


def viz_torchinfo(checkpoint: dict, output_path: Path):
    """Detailed torchinfo summary."""
    torchinfo = safe_import('torchinfo')
    if not torchinfo:
        return
     

    model = recreate_model(checkpoint)
    train_config = checkpoint['train_config']
    model_config = checkpoint['model_config']

    
    # Determine input shape
    batch_size = 32
    seq_len = train_config.max_seq_len
    
    if model_config.use_byt5:
        input_size = (batch_size, seq_len, model_config.byt5_dim)
        dtypes = [torch.float32]
    else:
        input_size = (batch_size, seq_len)
        dtypes = [torch.long]
    
    summary = torchinfo.summary(
        model,
        input_size=input_size,
        dtypes=dtypes,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        row_settings=["var_names"],
        verbose=1,
        depth=5,
    )
    
    output_path.write_text(str(summary), encoding='utf-8')
    print(f"✓ Saved torchinfo summary to {output_path}")
    print(summary)


def viz_tensorboard(checkpoint: dict, output_dir: Path):
    """TensorBoard graph export."""
    model = recreate_model(checkpoint)
    dummy_input = create_dummy_input(checkpoint)
    
    tb_dir = output_dir / 'tensorboard_logs'
    writer = SummaryWriter(tb_dir)
    
    try:
        writer.add_graph(model, dummy_input)
        writer.close()
        print(f"✓ TensorBoard logs saved to {tb_dir}")
        print(f"  Run: tensorboard --logdir {tb_dir}")
        print(f"  Open: http://localhost:6006")
    except Exception as e:
        print(f"✗ TensorBoard export failed: {e}")
        writer.close()



def viz_profile(checkpoint: dict, output_dir: Path):
    """PyTorch profiler analysis."""
    from torch.profiler import profile, ProfilerActivity, record_function
    
    model = recreate_model(checkpoint)
    dummy_input = create_dummy_input(checkpoint, batch_size=32)
    
    try:
        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with record_function("model_inference"):
                model(dummy_input)
        
        # Save results
        trace_path = output_dir / "trace.json"
        prof.export_chrome_trace(str(trace_path))
        
        profile_table = prof.key_averages().table(
            sort_by="cpu_time_total", row_limit=-1
        )
        (output_dir / "profile.txt").write_text(profile_table, encoding='utf-8')
        
        print(f"✓ Profile saved to {output_dir / 'profile.txt'}")
        print(f"✓ Chrome trace: {trace_path}")
        print("\nTop 10 operations:")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    except Exception as e:
        print(f"✗ Profiling failed: {e}")


def viz_custom_diagram(checkpoint: dict, output_path: Path):
    """Custom matplotlib architecture diagram."""
    model_config = checkpoint['model_config']
    task_config = checkpoint['task_config']
    
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.axis('off')
    
    components = []
    y = 0.95
    dy = 0.08
    
    # Build component list
    components.append(('Input', f'Sequence ({model_config.vocab_size} vocab)', y, 'lightgreen'))
    y -= dy
    
    # Encoder
    if model_config.use_byt5:
        components.append(('ByT5 Encoder', f'dim={model_config.byt5_dim}', y, 'lightyellow'))
        y -= dy
        components.append(('Projection', f'{model_config.byt5_dim}→{model_config.embed_dim}', y, 'lightblue'))
    elif model_config.use_charlm:
        components.append(('CharLM', f'dim={model_config.charlm_hidden_dim}', y, 'lightyellow'))
    else:
        components.append(('Embedding', f'{model_config.vocab_size}→{model_config.embed_dim}', y, 'lightblue'))
    y -= dy
    
    # Main model
    if model_config.model_type == 'gru':
        components.append(('BiGRU', f'{model_config.num_layers}×{model_config.hidden_dim}', y, 'lightcoral'))
    elif model_config.model_type == 'cnn':
        components.append(('CNN', f'kernels={model_config.kernel_sizes}', y, 'lightcoral'))
        if model_config.use_batch_norm:
            y -= dy
            components.append(('BatchNorm', '', y, 'lightgray'))
    y -= dy
    
    components.append(('Dropout', f'p={model_config.dropout}', y, 'lightgray'))
    y -= dy
    
    # Output
    out_dim = 1 if task_config.task_type == 'binary' else len(task_config.label_to_idx)
    components.append(('Output', f'{out_dim} units', y, 'lightgreen'))
    y -= dy
    
    act = {'binary': 'Sigmoid', 'multilabel': 'Sigmoid', 'multiclass': 'Softmax'}.get(
        task_config.task_type, 'None'
    )
    components.append((act, task_config.task_type, y, 'lightyellow'))
    
    # Draw
    for name, desc, y_pos, color in components:
        rect = plt.Rectangle((0.1, y_pos - 0.03), 0.8, 0.06,
                              facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(0.5, y_pos, name, ha='center', va='center', 
                fontsize=12, fontweight='bold')
        if desc:
            ax.text(0.5, y_pos - 0.015, desc, ha='center', va='center',
                    fontsize=8, style='italic')
        
        # Arrow
        if components.index((name, desc, y_pos, color)) < len(components) - 1:
            ax.arrow(0.5, y_pos - 0.04, 0, -0.03,
                     head_width=0.05, head_length=0.01, fc='black', ec='black')
    
    ax.text(0.5, 0.98, f'{model_config.model_type.upper()} Architecture',
            ha='center', va='top', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved architecture diagram to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize training, test results, and model architecture"
    )
    parser.add_argument("output_dir", type=Path, help="Model output directory")
    parser.add_argument("--checkpoint", type=Path, help="Model checkpoint (.pt)")
    parser.add_argument("--format", choices=['png', 'pdf', 'svg'], 
                        default='png', help="Image format")
    parser.add_argument(
        "--viz-method",
        choices=['all', 'simple', 'torchinfo', 'tensorboard', 'profile', 'diagram', 'none'],
        default='all',
        help="Model visualization method"
    )
    args = parser.parse_args()
    
    output_dir = args.output_dir
    if not output_dir.exists():
        print(f"Error: {output_dir} not found")
        return
    
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    print(f"{'='*80}")
    print(f"Visualizing: {output_dir}")
    print(f"{'='*80}\n")
    
    # Training metrics
    metrics_file = output_dir / "metrics.csv"
    if metrics_file.exists():
        print("[Training Metrics]")
        df = pd.read_csv(metrics_file)
        df = df.astype({'train_loss': float, 'dev_loss': float, 'learning_rate': float})
        plot_training_curves(df, viz_dir / f"training_curves.{args.format}")
    else:
        print("⊘ No training metrics found")
    
    # Test metrics
    test_file = output_dir / "test_metrics.csv"
    if test_file.exists():
        print("\n[Test Metrics]")
        task_type = detect_and_plot_test_metrics(test_file, viz_dir, args.format)
        print(f"  Task type: {task_type}")
    else:
        print("\n⊘ No test metrics found")
    
    # Model architecture

    if args.viz_method == 'none':
        print("\n⊘ Skipping model visualization (--viz-method none)")
    else:
        checkpoint_path = args.checkpoint or find_best_checkpoint(output_dir)
        
        if not checkpoint_path or not checkpoint_path.exists():
            print("\n⊘ No checkpoint found")
            print("  Use --checkpoint <path> to specify manually")
        else:
            print(f"\n[Model Architecture]")
            print(f"  Checkpoint: {checkpoint_path.name}")
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Define visualization methods
            methods = {
                'simple': (viz_simple_text, "model_architecture.txt"),
                'torchinfo': (viz_torchinfo, "model_summary_torchinfo.txt"),
                'tensorboard': (viz_tensorboard, None),
                'profile': (viz_profile, None),
                'diagram': (viz_custom_diagram, f"architecture_diagram.{args.format}"),
            }
            
            # Execute visualizations
            to_run = methods.items() if args.viz_method == 'all' else [(args.viz_method, methods[args.viz_method])]
            
            for name, (func, filename) in to_run:
                print(f"\n  [{name}]")
                try:
                    output_path = viz_dir / filename if filename else viz_dir
                    func(checkpoint, output_path)
                except Exception as e:
                    print(f"  ✗ Failed: {e}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✓ Visualizations saved to: {viz_dir}")
    print(f"{'='*80}")
    print("\nGenerated files:")
    for f in sorted(viz_dir.iterdir()):
        prefix = "  📁" if f.is_dir() else "  📄"
        print(f"{prefix} {f.name}")


if __name__ == "__main__":
    main()
