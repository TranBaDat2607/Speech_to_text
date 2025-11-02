"""
Plot metrics across all training batches
"""
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_all_batch_metrics(logs_dir: str = "./logs"):
    """Load metrics from all batch files (each file = 1 batch)"""
    pattern = str(Path(logs_dir) / "*_metrics.json")
    files = sorted(glob.glob(pattern))  # Sort by filename (timestamp)
    
    all_batches = []
    
    for batch_idx, file_path in enumerate(files):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                metrics = data.get('metrics', [])
                
                if metrics:
                    # Each file contains 1 batch's metrics
                    # Use the first (and only) entry
                    entry = metrics[0]
                    # Override batch number with sequential index
                    entry['batch'] = batch_idx
                    entry['filename'] = Path(file_path).name
                    all_batches.append(entry)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
    
    return all_batches


def plot_batch_progression(all_metrics, save_path: str = None):
    """Plot metrics across batches"""
    
    batches = []
    losses = []
    kl_losses = []
    ce_losses = []
    
    for entry in all_metrics:
        if 'batch' in entry and 'loss' in entry:
            batches.append(entry['batch'])
            losses.append(entry.get('loss', None))
            kl_losses.append(entry.get('kl_loss', None))
            ce_losses.append(entry.get('ce_loss', None))
    
    if not batches:
        print("No batch metrics found!")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Training Progress: Batch 0-{max(batches)}", fontsize=16)
    
    # Plot 1: Total Loss
    ax = axes[0, 0]
    valid_batches = [b for b, l in zip(batches, losses) if l is not None]
    valid_losses = [l for l in losses if l is not None]
    
    ax.plot(valid_batches, valid_losses, 'b-o', linewidth=2, markersize=6, label='Total Loss')
    ax.set_xlabel('Batch Number')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss per Batch')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if valid_losses:
        ax.axhline(y=np.mean(valid_losses), color='r', linestyle='--', alpha=0.5, label=f'Mean: {np.mean(valid_losses):.4f}')
    
    # Plot 2: KL Divergence Loss
    ax = axes[0, 1]
    valid_batches_kl = [b for b, l in zip(batches, kl_losses) if l is not None]
    valid_kl = [l for l in kl_losses if l is not None]
    
    if valid_kl:
        ax.plot(valid_batches_kl, valid_kl, 'r-o', linewidth=2, markersize=6, label='KL Loss')
        ax.set_xlabel('Batch Number')
        ax.set_ylabel('KL Loss')
        ax.set_title('KL Divergence Loss per Batch')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Plot 3: Cross-Entropy Loss
    ax = axes[1, 0]
    valid_batches_ce = [b for b, l in zip(batches, ce_losses) if l is not None]
    valid_ce = [l for l in ce_losses if l is not None]
    
    if valid_ce:
        ax.plot(valid_batches_ce, valid_ce, 'g-o', linewidth=2, markersize=6, label='CE Loss')
        ax.set_xlabel('Batch Number')
        ax.set_ylabel('CE Loss')
        ax.set_title('Cross-Entropy Loss per Batch')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Plot 4: All losses combined
    ax = axes[1, 1]
    if valid_losses:
        ax.plot(valid_batches, valid_losses, 'b-o', linewidth=2, markersize=4, label='Total', alpha=0.8)
    if valid_kl:
        ax.plot(valid_batches_kl, valid_kl, 'r--s', linewidth=1.5, markersize=4, label='KL', alpha=0.8)
    if valid_ce:
        ax.plot(valid_batches_ce, valid_ce, 'g--^', linewidth=1.5, markersize=4, label='CE', alpha=0.8)
    
    ax.set_xlabel('Batch Number')
    ax.set_ylabel('Loss')
    ax.set_title('All Losses Combined')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()


def print_summary(all_metrics):
    """Print summary statistics"""
    if not all_metrics:
        print("No metrics found")
        return
    
    batches = [m['batch'] for m in all_metrics]
    losses = [m['loss'] for m in all_metrics]
    kl_losses = [m.get('kl_loss') for m in all_metrics if 'kl_loss' in m]
    ce_losses = [m.get('ce_loss') for m in all_metrics if 'ce_loss' in m]
    
    print(f"\nTraining Summary:")
    print(f"  Batches trained: {len(batches)} (batch {min(batches)} to {max(batches)})")
    print(f"  Total samples: {(max(batches) + 1) * 100}")
    
    print(f"\nLoss Statistics:")
    print(f"  First batch loss: {losses[0]:.4f}")
    print(f"  Last batch loss:  {losses[-1]:.4f}")
    print(f"  Mean loss:        {np.mean(losses):.4f}")
    print(f"  Std loss:         {np.std(losses):.4f}")
    print(f"  Min loss:         {np.min(losses):.4f} (batch {batches[np.argmin(losses)]})")
    print(f"  Max loss:         {np.max(losses):.4f} (batch {batches[np.argmax(losses)]})")
    
    if len(losses) > 1:
        improvement = ((losses[0] - losses[-1]) / losses[0]) * 100
        trend = "decreasing ✓" if improvement > 0 else "increasing ✗" if improvement < 0 else "flat"
        print(f"  Change:           {improvement:+.2f}% ({trend})")
    
    if kl_losses:
        print(f"\nKL Loss: {kl_losses[0]:.4f} → {kl_losses[-1]:.4f}")
    if ce_losses:
        print(f"CE Loss: {ce_losses[0]:.4f} → {ce_losses[-1]:.4f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot metrics across all batches")
    parser.add_argument("--logs_dir", type=str, default="./logs", help="Directory with metrics files")
    parser.add_argument("--output", type=str, default="plots/all_batches_progress.png", help="Output file")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    
    args = parser.parse_args()
    
    print(f"Loading metrics from: {args.logs_dir}")
    all_metrics = load_all_batch_metrics(args.logs_dir)
    
    if not all_metrics:
        print("No metrics found!")
        return
    
    print(f"Found {len(all_metrics)} batch entries")
    
    print_summary(all_metrics)
    
    print("\nPlotting...")
    output_path = None if args.show else args.output
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    plot_batch_progression(all_metrics, save_path=output_path)


if __name__ == "__main__":
    main()
