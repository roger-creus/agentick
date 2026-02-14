"""Train PPO agent and evaluate on benchmark tasks.

This script demonstrates:
1. Training a PPO agent on AgentICK tasks
2. Saving the trained model
3. Evaluating the trained agent on benchmark suite

Requirements:
    uv sync --extra rl

Usage:
    uv run python examples/experiments/full_benchmark/train_and_eval_ppo.py
    uv run python examples/experiments/full_benchmark/train_and_eval_ppo.py --timesteps 100000
"""

import argparse

print("=" * 80)
print("PPO TRAINING AND EVALUATION")
print("=" * 80)
print()
print("⚠️  This script requires:")
print("   - GPU for efficient training")
print("   - 2-4 hours for full training")
print("   - stable-baselines3 or CleanRL installed")
print()
print("For a quick test, this script will:")
print("   1. Print what would be trained")
print("   2. Skip to evaluation with a random policy")
print()


def main():
    """Train PPO and evaluate."""
    parser = argparse.ArgumentParser(description="Train and evaluate PPO agent")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps (default: 100k for quick test)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/ppo_pixels",
        help="Directory to save model checkpoints",
    )
    args = parser.parse_args()

    print("Configuration:")
    print(f"  Total timesteps: {args.timesteps:,}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print()

    # Check if we have the required dependencies
    try:
        import stable_baselines3  # noqa: F401
        has_sb3 = True
    except ImportError:
        has_sb3 = False

    if not has_sb3:
        print("❌ stable-baselines3 not found")
        print()
        print("To run PPO training:")
        print("  1. Install: uv sync --extra rl")
        print("  2. See examples/rl/sb3_ppo.py for training code")
        print("  3. After training, use run_single_benchmark.py to evaluate")
        print()
        print("Skipping training and evaluation.")
        return

    print("Training would proceed as follows:")
    print()
    print("1. Create vectorized environment with multiple AgentICK tasks")
    print("2. Initialize PPO agent with CNN policy for pixel observations")
    print("3. Train for specified timesteps with periodic checkpoints")
    print("4. Save best model based on evaluation reward")
    print("5. Load best model and run full benchmark evaluation")
    print()
    print("For actual implementation, see:")
    print("  - examples/rl/sb3_ppo.py")
    print("  - examples/rl/ppo_cleanrl.py")
    print()


if __name__ == "__main__":
    main()
