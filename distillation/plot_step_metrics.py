"""
Plot step-level training metrics
Visualize loss progression within and across batches
"""
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_metrics_with_steps(logs_dir: str = "./logs"):
    """Load all metrics including step-level data"""
    pattern = str(Path(logs_dir) / "*_metrics.json")
    files = sorted(glob.glob(pattern))
    
    all_steps = []
    all_batches = []
    
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                metrics = data.get('metrics', [])
                
                for entry in metrics:
                    if 'step' in entry and 'global_step' in entry:
                        # Step-level entry
                        all_steps.append(entry)
                    elif 'batch' in entry and 'step' not in entry:
                        # Batch-level entry (average)
                        all_batches.append(entry)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
    
    # Sort by global_step / batch
    all_steps.sort(key=lambda x: x.get('global_step', 0))
    all_batches.sort(key=lambda x: x.get('batch', 0))
    
    return all_steps, all_batches


def plot_step_progression(all_steps, all_batches, save_path: str = None):
    """Plot step-level metrics with batch boundaries"""
    
    if not all_steps:
        print("No step-level metrics found!")
        return
    
    # Extract data
    global_steps = [s['global_step'] for s in all_steps]
    losses = [s.get('loss') for s in all_steps]
    kl_losses = [s.get('kl_loss') for s in all_steps]
    ce_losses = [s.get('ce_loss') for s in all_steps]
    batches = [s.get('batch', 0) for s in all_steps]
    
    # Find batch boundaries
    batch_boundaries = []
    current_batch = batches[0] if batches else 0
    for i, b in enumerate(batches):
        if b != current_batch:
            batch_boundaries.append(global_steps[i])
            current_batch = b
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle(f"Step-Level Training Progress ({len(all_steps)} steps, {len(all_batches)} batches)", 
                 fontsize=16)
    
    # Plot 1: Total Loss
    ax = axes[0]
    ax.plot(global_steps, losses, 'b-', linewidth=1, alpha=0.7, label='Step Loss')
    
    # Add batch boundaries
    for boundary in batch_boundaries:
        ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Add batch averages if available
    if all_batches:
        batch_steps = [b['batch'] * 100 for b in all_batches]  # Approximate global step
        batch_losses = [b['loss'] for b in all_batches]
        ax.plot(batch_steps, batch_losses, 'ro-', linewidth=2, markersize=6, 
                label='Batch Average', alpha=0.8)
    
    ax.set_xlabel('Global Step')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss per Step')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: KL Divergence Loss
    ax = axes[1]
    valid_kl = [(s, l) for s, l in zip(global_steps, kl_losses) if l is not None]
    if valid_kl:
        steps_kl, losses_kl = zip(*valid_kl)
        ax.plot(steps_kl, losses_kl, 'r-', linewidth=1, alpha=0.7, label='KL Loss')
        
        for boundary in batch_boundaries:
            ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        
        ax.set_xlabel('Global Step')
        ax.set_ylabel('KL Loss')
        ax.set_title('KL Divergence Loss per Step')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Plot 3: Cross-Entropy Loss
    ax = axes[2]
    valid_ce = [(s, l) for s, l in zip(global_steps, ce_losses) if l is not None]
    if valid_ce:
        steps_ce, losses_ce = zip(*valid_ce)
        ax.plot(steps_ce, losses_ce, 'g-', linewidth=1, alpha=0.7, label='CE Loss')
        
        for boundary in batch_boundaries:
            ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        
        ax.set_xlabel('Global Step')
        ax.set_ylabel('CE Loss')
        ax.set_title('Cross-Entropy Loss per Step')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()


def plot_per_batch_steps(all_steps, num_batches_to_show: int = 5, save_path: str = None):
    """Plot step progression within individual batches"""
    
    if not all_steps:
        print("No step-level metrics found!")
        return
    
    # Group by batch
    batch_data = {}
    for step in all_steps:
        batch = step.get('batch', 0)
        if batch not in batch_data:
            batch_data[batch] = []
        batch_data[batch].append(step)
    
    # Get first N batches
    batches_to_plot = sorted(batch_data.keys())[:num_batches_to_show]
    
    if not batches_to_plot:
        print("No batch data found!")
        return
    
    # Create subplots
    fig, axes = plt.subplots(len(batches_to_plot), 1, figsize=(12, 4 * len(batches_to_plot)))
    if len(batches_to_plot) == 1:
        axes = [axes]
    
    fig.suptitle(f"Step Progression Within Batches (First {len(batches_to_plot)} Batches)", 
                 fontsize=16)
    
    for idx, batch_num in enumerate(batches_to_plot):
        ax = axes[idx]
        batch_steps = batch_data[batch_num]
        
        step_indices = [s.get('step_in_epoch', i) for i, s in enumerate(batch_steps)]
        losses = [s.get('loss') for s in batch_steps]
        kl_losses = [s.get('kl_loss') for s in batch_steps]
        ce_losses = [s.get('ce_loss') for s in batch_steps]
        
        ax.plot(step_indices, losses, 'b-o', linewidth=2, markersize=4, label='Total Loss')
        if any(kl_losses):
            ax.plot(step_indices, kl_losses, 'r--s', linewidth=1.5, markersize=3, 
                   label='KL Loss', alpha=0.7)
        if any(ce_losses):
            ax.plot(step_indices, ce_losses, 'g--^', linewidth=1.5, markersize=3, 
                   label='CE Loss', alpha=0.7)
        
        ax.set_xlabel('Step in Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Batch {batch_num}: {len(batch_steps)} steps')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()


def print_summary(all_steps, all_batches):
    """Print summary statistics"""
    print(f"\n{'='*60}")
    print(f"Training Metrics Summary")
    print(f"{'='*60}")
    print(f"Total steps logged: {len(all_steps)}")
    print(f"Total batches logged: {len(all_batches)}")
    
    if all_steps:
        losses = [s['loss'] for s in all_steps if 'loss' in s]
        if losses:
            print(f"\nStep-level Loss Statistics:")
            print(f"  First step loss:  {losses[0]:.4f}")
            print(f"  Last step loss:   {losses[-1]:.4f}")
            print(f"  Mean loss:        {np.mean(losses):.4f}")
            print(f"  Std loss:         {np.std(losses):.4f}")
            print(f"  Min loss:         {np.min(losses):.4f}")
            print(f"  Max loss:         {np.max(losses):.4f}")
            
            if len(losses) > 1:
                improvement = ((losses[0] - losses[-1]) / losses[0]) * 100
                trend = "decreasing ✓" if improvement > 0 else "increasing ✗"
                print(f"  Change:           {improvement:+.2f}% ({trend})")
    
    if all_batches:
        batch_losses = [b['loss'] for b in all_batches if 'loss' in b]
        if batch_losses:
            print(f"\nBatch-level Loss Statistics:")
            print(f"  First batch loss: {batch_losses[0]:.4f}")
            print(f"  Last batch loss:  {batch_losses[-1]:.4f}")
            print(f"  Mean batch loss:  {np.mean(batch_losses):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Plot step-level training metrics")
    parser.add_argument("--logs_dir", type=str, default="./logs", 
                       help="Directory with metrics files")
    parser.add_argument("--output_global", type=str, default="plots/step_progression_global.png", 
                       help="Output file for global step progression")
    parser.add_argument("--output_batches", type=str, default="plots/step_progression_per_batch.png", 
                       help="Output file for per-batch step progression")
    parser.add_argument("--num_batches", type=int, default=5, 
                       help="Number of batches to show in detail")
    parser.add_argument("--show", action="store_true", 
                       help="Show plots interactively")
    
    args = parser.parse_args()
    
    print(f"Loading step-level metrics from: {args.logs_dir}")
    all_steps, all_batches = load_metrics_with_steps(args.logs_dir)
    
    if not all_steps and not all_batches:
        print("No metrics found!")
        return
    
    print(f"Found {len(all_steps)} step entries and {len(all_batches)} batch entries")
    
    print_summary(all_steps, all_batches)
    
    if all_steps:
        print("\nPlotting global step progression...")
        output_global = None if args.show else args.output_global
        if output_global:
            Path(output_global).parent.mkdir(parents=True, exist_ok=True)
        plot_step_progression(all_steps, all_batches, save_path=output_global)
        
        print("\nPlotting per-batch step progression...")
        output_batches = None if args.show else args.output_batches
        if output_batches:
            Path(output_batches).parent.mkdir(parents=True, exist_ok=True)
        plot_per_batch_steps(all_steps, num_batches_to_show=args.num_batches, 
                           save_path=output_batches)
    else:
        print("\nNo step-level metrics found. Enable 'log_step_metrics' in config.")


if __name__ == "__main__":
    main()
