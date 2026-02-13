"""
Plot learning curves from training logs.

This example demonstrates:
- Loading training logs
- Plotting learning curves
- Comparing training runs

Requirements:
    uv sync --extra all

Usage:
    uv run python examples/plotting/learning_curves.py logs_dir
"""

import argparse
import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available. Install with: uv sync --extra all")


def load_logs(log_dir: str) -> dict:
    """Load training logs from directory."""
    log_path = Path(log_dir)

    # Try to find log files
    log_files = list(log_path.glob("*.json")) + list(log_path.glob("*.jsonl"))

    if not log_files:
        raise FileNotFoundError(f"No log files found in {log_dir}")

    # Load first log file
    log_file = log_files[0]

    if log_file.suffix == ".jsonl":
        # Load JSONL (one JSON object per line)
        logs = []
        with open(log_file) as f:
            for line in f:
                logs.append(json.loads(line))
        return {"logs": logs}
    else:
        # Load JSON
        with open(log_file) as f:
            return json.load(f)


def plot_learning_curve(logs: list[dict], output_path: str):
    """Plot learning curve."""
    # Extract metrics
    steps = []
    rewards = []

    for entry in logs:
        if "step" in entry and "reward" in entry:
            steps.append(entry["step"])
            rewards.append(entry["reward"])
        elif "episode" in entry and "episode_return" in entry:
            steps.append(entry["episode"])
            rewards.append(entry["episode_return"])

    if not steps:
        print("⚠️  No training data found in logs")
        return

    # Compute rolling average
    window = min(100, len(rewards) // 10 + 1)
    if len(rewards) >= window:
        rolling_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        rolling_steps = steps[window - 1 :]
    else:
        rolling_avg = rewards
        rolling_steps = steps

    # Plot
    plt.figure(figsize=(12, 6))

    plt.plot(steps, rewards, alpha=0.2, label="Raw", color="blue")
    plt.plot(rolling_steps, rolling_avg, linewidth=2, label=f"Rolling Avg ({window})", color="red")

    plt.xlabel("Step")
    plt.ylabel("Episode Reward")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()

    print(f"✓ Saved: {output_path}")


def plot_metrics(logs: list[dict], output_path: str):
    """Plot multiple metrics."""
    # Extract all numeric metrics
    all_metrics = {}

    for entry in logs:
        for key, value in entry.items():
            if isinstance(value, (int, float)) and key != "step" and key != "episode":
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)

    if not all_metrics:
        print("⚠️  No metrics found in logs")
        return

    # Plot each metric
    num_metrics = len(all_metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 4 * num_metrics))

    if num_metrics == 1:
        axes = [axes]

    for ax, (metric, values) in zip(axes, all_metrics.items()):
        ax.plot(values)
        ax.set_ylabel(metric)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{metric} over time")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✓ Saved: {output_path}")


def main():
    """Plot learning curves."""
    if not MATPLOTLIB_AVAILABLE:
        return

    parser = argparse.ArgumentParser(description="Plot learning curves from logs")
    parser.add_argument(
        "log_dir",
        type=str,
        help="Directory containing training logs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots",
    )
    args = parser.parse_args()

    print("Learning Curve Plotter")
    print("=" * 80)

    # Load logs
    try:
        data = load_logs(args.log_dir)
        logs = data.get("logs", [data])
        print(f"✓ Loaded {len(logs)} log entries from {args.log_dir}")
    except Exception as e:
        print(f"❌ Failed to load logs: {e}")
        return

    # Setup output directory
    if args.output_dir is None:
        output_dir = Path(args.log_dir) / "plots"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {output_dir}")
    print()

    # Generate plots
    plot_learning_curve(logs, str(output_dir / "learning_curve.png"))
    plot_metrics(logs, str(output_dir / "metrics.png"))

    print("\n" + "=" * 80)
    print("Plots generated!")
    print(f"View plots in: {output_dir}")


if __name__ == "__main__":
    main()
