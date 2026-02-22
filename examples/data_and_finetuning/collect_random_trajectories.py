"""Collect random agent trajectories for dataset creation.

Supports both single-task (--env) and multi-task (--tasks) modes. Useful for
baseline comparisons and exploration data collection.

Requirements:
    uv sync

Usage:
    # Single task (backward-compatible)
    uv run python examples/data_and_finetuning/collect_random_trajectories.py \
        --env GoToGoal-v0 --num-episodes 50

    # Multi-task collection
    uv run python examples/data_and_finetuning/collect_random_trajectories.py \
        --tasks GoToGoal-v0 MazeNavigation-v0 --difficulties easy medium

    # All oracle tasks
    uv run python examples/data_and_finetuning/collect_random_trajectories.py \
        --tasks all --num-episodes 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _utils import ALL_DIFFICULTIES, resolve_tasks

import agentick
from agentick.data.collector import TrajectoryCollector


def collect_for_env(
    env_id: str,
    difficulty: str | None,
    num_episodes: int,
    render_mode: str,
    output_dir: Path,
    seed: int,
) -> dict:
    """Collect random trajectories for a single env/difficulty combo."""
    env_kwargs: dict = {"render_mode": render_mode}
    if difficulty:
        env_kwargs["difficulty"] = difficulty

    env = agentick.make(env_id, **env_kwargs)

    collector = TrajectoryCollector(
        buffer_size=num_episodes + 10,
        collect_observations=True,
    )

    total_reward = 0.0
    total_steps = 0
    successes = 0

    for episode in range(num_episodes):
        collector.start_episode(metadata={"episode": episode})
        obs, info = env.reset(seed=seed + episode)

        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            collector.add_step(obs, action, reward, done, info)
            obs = next_obs
            episode_reward += reward
            steps += 1

            if steps > 500:
                break

        collector.end_episode()
        total_reward += episode_reward
        total_steps += steps
        if info.get("success", False):
            successes += 1

    env.close()

    # Save trajectories
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories = collector.get_trajectories()
    for i, traj in enumerate(trajectories):
        traj_path = output_dir / f"trajectory_{i:04d}.json"
        with open(traj_path, "w") as f:
            json.dump(traj.to_dict(), f, indent=2, default=str)

    stats = {
        "env_id": env_id,
        "difficulty": difficulty,
        "num_episodes": num_episodes,
        "total_steps": total_steps,
        "avg_reward": total_reward / num_episodes,
        "avg_steps": total_steps / num_episodes,
        "success_rate": successes / num_episodes,
    }

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Collect random agent trajectories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Backward-compatible single-task mode
    parser.add_argument("--env", type=str, default=None, help="Single environment ID")

    # Multi-task mode
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Task names for multi-task collection (use 'all' for all oracle tasks)",
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=None,
        choices=ALL_DIFFICULTIES,
        help="Difficulty levels (default: easy for --env, all for --tasks)",
    )

    parser.add_argument(
        "--num-episodes", type=int, default=100, help="Episodes per task/difficulty"
    )
    parser.add_argument("--output-dir", type=str, default="data/random_trajectories", help="Output")
    parser.add_argument("--render-mode", type=str, default="ascii", help="Render mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    np.random.seed(args.seed)

    # Determine mode: single-task (--env) or multi-task (--tasks)
    if args.tasks is not None:
        # Multi-task mode
        if args.tasks == ["all"]:
            task_list = resolve_tasks(None)
        else:
            task_list = args.tasks
        difficulties = args.difficulties or ALL_DIFFICULTIES
    elif args.env is not None:
        # Single-task backward-compatible mode
        task_list = [args.env]
        difficulties = [args.difficulties[0]] if args.difficulties else [None]
    else:
        # Default: single GoToGoal-v0
        task_list = ["GoToGoal-v0"]
        difficulties = [args.difficulties[0]] if args.difficulties else [None]

    print("Collecting Random Trajectories")
    print("=" * 80)
    print(f"Tasks: {task_list}")
    print(f"Difficulties: {difficulties}")
    print(f"Episodes per combo: {args.num_episodes}")
    print(f"Output: {args.output_dir}")
    print()

    all_stats = []

    for task_id in task_list:
        for difficulty in difficulties:
            diff_label = difficulty or "default"
            print(f"\nCollecting {task_id} @ {diff_label}...")

            if difficulty:
                sub_dir = Path(args.output_dir) / task_id / difficulty
            else:
                sub_dir = Path(args.output_dir) / task_id

            try:
                stats = collect_for_env(
                    env_id=task_id,
                    difficulty=difficulty,
                    num_episodes=args.num_episodes,
                    render_mode=args.render_mode,
                    output_dir=sub_dir,
                    seed=args.seed,
                )
                print(
                    f"  {stats['num_episodes']} episodes: "
                    f"reward={stats['avg_reward']:.2f}, "
                    f"steps={stats['avg_steps']:.1f}, "
                    f"success={stats['success_rate']:.1%}"
                )
                all_stats.append(stats)
            except Exception as e:
                print(f"  ERROR: {e}")

    # Print summary
    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)
    print(f"Task/difficulty combos: {len(all_stats)}")
    if all_stats:
        total_episodes = sum(s["num_episodes"] for s in all_stats)
        avg_success = np.mean([s["success_rate"] for s in all_stats])
        print(f"Total episodes: {total_episodes}")
        print(f"Average success rate: {avg_success:.1%}")
    print(f"Data saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
