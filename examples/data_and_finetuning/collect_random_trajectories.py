"""
Collect random agent trajectories for dataset creation.

This example demonstrates:
- Collecting trajectories from a random policy
- Saving trajectories in various formats
- Useful for baseline comparisons or exploration

Requirements:
    uv sync

Usage:
    uv run python examples/data_and_finetuning/collect_random_trajectories.py
"""

import argparse
import json
from pathlib import Path

import numpy as np

import agentick
from agentick.data.collector import TrajectoryCollector


def main():
    """Collect random trajectories."""
    parser = argparse.ArgumentParser(description="Collect random agent trajectories")
    parser.add_argument(
        "--env",
        type=str,
        default="GoToGoal-v0",
        help="Environment ID",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of episodes to collect",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/random_trajectories",
        help="Output directory for trajectories",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default=None,
        help="Difficulty level (easy, medium, hard, expert)",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="ascii",
        help="Render mode for observations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    print("Collecting Random Trajectories")
    print("=" * 80)
    print(f"Environment: {args.env}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Output: {args.output_dir}")
    print(f"Difficulty: {args.difficulty or 'default'}")
    print(f"Render mode: {args.render_mode}")
    print()

    # Set seed
    np.random.seed(args.seed)

    # Create environment
    env_kwargs = {"render_mode": args.render_mode}
    if args.difficulty:
        env_kwargs["difficulty"] = args.difficulty

    env = agentick.make(args.env, **env_kwargs)

    # Create trajectory collector
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    collector = TrajectoryCollector(
        buffer_size=args.num_episodes + 10,
        collect_observations=True,
    )

    # Collect trajectories
    print("Collecting trajectories...")

    total_reward = 0
    total_steps = 0
    successes = 0

    for episode in range(args.num_episodes):
        collector.start_episode(metadata={"episode": episode})

        obs, info = env.reset(seed=args.seed + episode)

        done = False
        episode_reward = 0
        steps = 0

        while not done:
            # Random action
            action = env.action_space.sample()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            collector.add_step(obs, action, reward, done, info)

            obs = next_obs
            episode_reward += reward
            steps += 1

            # Safety limit
            if steps > 500:
                break

        collector.end_episode()

        total_reward += episode_reward
        total_steps += steps

        if info.get("success", False):
            successes += 1

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = total_reward / (episode + 1)
            avg_steps = total_steps / (episode + 1)
            success_rate = successes / (episode + 1)

            print(
                f"Episode {episode + 1:4d}/{args.num_episodes}: "
                f"Reward={avg_reward:.2f}, "
                f"Steps={avg_steps:.1f}, "
                f"Success={success_rate:.1%}"
            )

    env.close()

    # Save trajectories
    print("\nSaving trajectories...")

    # Save each trajectory as JSON
    trajectories = collector.get_trajectories()
    for i, traj in enumerate(trajectories):
        traj_path = output_dir / f"trajectory_{i:04d}.json"
        with open(traj_path, "w") as f:
            json.dump(traj.to_dict(), f, indent=2, default=str)

    print(f"✓ Saved {len(trajectories)} trajectories to {output_dir}")

    # Save summary statistics
    stats = {
        "num_episodes": args.num_episodes,
        "total_steps": total_steps,
        "avg_reward": total_reward / args.num_episodes,
        "avg_steps": total_steps / args.num_episodes,
        "success_rate": successes / args.num_episodes,
        "env_id": args.env,
        "difficulty": args.difficulty,
        "render_mode": args.render_mode,
        "seed": args.seed,
    }

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"✓ Saved stats: {stats_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)
    print(f"Episodes: {args.num_episodes}")
    print(f"Total steps: {total_steps}")
    print(f"Average reward: {stats['avg_reward']:.2f}")
    print(f"Average steps: {stats['avg_steps']:.1f}")
    print(f"Success rate: {stats['success_rate']:.1%}")
    print(f"\nData saved to: {output_dir}")

    print("\n💡 Next steps:")
    print("  - Use these trajectories as baselines")
    print("  - Export to HuggingFace format with export_to_huggingface.py")
    print("  - Analyze trajectory statistics")


if __name__ == "__main__":
    main()
