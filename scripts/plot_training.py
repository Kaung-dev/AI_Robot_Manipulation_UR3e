#!/usr/bin/env python3
"""Plot BC training metrics from TensorBoard logs.

Usage:
    python3 scripts/plot_training.py logs/robomimic/Isaac-Lift-Pegboard-Franka-IK-Rel-v0/bc_rnn_franka_pegboard/20260518140009
    python3 scripts/plot_training.py <run_dir> --save  # saves PNG instead of showing
"""
import argparse
import glob
import os
import sys

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Install tensorboard: pip install tensorboard")
    sys.exit(1)

import matplotlib.pyplot as plt


def load_scalars(log_dir):
    """Load all scalar metrics from TensorBoard event files."""
    event_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files:
        print(f"No TensorBoard event files found in {log_dir}")
        sys.exit(1)

    metrics = {}
    for ef in event_files:
        ea = EventAccumulator(os.path.dirname(ef))
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            if tag not in metrics:
                metrics[tag] = (steps, values)
            else:
                # Merge if multiple event files
                metrics[tag] = (
                    metrics[tag][0] + steps,
                    metrics[tag][1] + values,
                )
    return metrics


def plot_metrics(metrics, run_dir, save=False):
    """Create a multi-panel figure of training metrics."""
    # Group metrics
    loss_keys = [k for k in metrics if "Loss" in k or "loss" in k or "Likelihood" in k]
    time_keys = [k for k in metrics if "Time" in k]
    lr_keys = [k for k in metrics if "lr" in k or "LR" in k]
    other_keys = [k for k in metrics if k not in loss_keys + time_keys + lr_keys]

    # Determine subplot layout
    panels = []
    if loss_keys:
        panels.append(("Loss / Likelihood", loss_keys))
    if lr_keys:
        panels.append(("Learning Rate", lr_keys))
    if other_keys:
        panels.append(("Other Metrics", other_keys))
    if time_keys:
        panels.append(("Timing", time_keys))

    if not panels:
        print("No metrics found to plot.")
        return

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), squeeze=False)
    fig.suptitle(f"BC Training: {os.path.basename(run_dir)}", fontsize=14, fontweight="bold")

    for idx, (title, keys) in enumerate(panels):
        ax = axes[idx, 0]
        for key in sorted(keys):
            steps, values = metrics[key]
            # Sort by step
            paired = sorted(zip(steps, values))
            steps, values = zip(*paired) if paired else ([], [])
            label = key.replace("Train/", "").replace("Valid/", "val_")
            ax.plot(steps, values, label=label, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        out_path = os.path.join(run_dir, "training_metrics.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot BC training metrics")
    parser.add_argument("run_dir", help="Path to the training run directory")
    parser.add_argument("--save", action="store_true", help="Save PNG instead of displaying")
    args = parser.parse_args()

    metrics = load_scalars(args.run_dir)
    print(f"Found {len(metrics)} metrics: {list(metrics.keys())}")
    plot_metrics(metrics, args.run_dir, save=args.save)


if __name__ == "__main__":
    main()
