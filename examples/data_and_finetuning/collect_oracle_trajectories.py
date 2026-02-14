"""
Collect oracle agent demonstrations for supervised fine-tuning.

Demonstrates trajectory collection for imitation learning using the optimal oracle agent.
Runtime: ~1 minute for 10 trajectories

Usage:
    uv run python examples/data_and_finetuning/collect_oracle_trajectories.py
"""

import json
from pathlib import Path

import agentick
from agentick.benchmark.baselines import OracleAgent
from agentick.data.collector import TrajectoryCollector


def main():
    print("Oracle Trajectory Collection")
    print("=" * 80)

    # Create environment
    env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="state_dict")

    # Create oracle agent (uses privileged information for optimal policy)
    oracle = OracleAgent(env)

    # Create trajectory collector
    collector = TrajectoryCollector(
        buffer_size=100,
        collect_observations=True,
    )

    # Collect trajectories from oracle agent
    print("\nCollecting 10 oracle trajectories...")

    for i in range(10):
        # Start new episode
        collector.start_episode(metadata={"episode": i, "agent": "oracle"})

        # Reset environment
        obs, info = env.reset(seed=42 + i)

        done = False
        steps = 0
        while not done:
            # Oracle agent selects optimal action
            action = oracle.act(obs, info)

            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Record step
            collector.add_step(obs, action, reward, done, info)

            obs = next_obs
            steps += 1

            # Safety limit
            if steps > 500:
                break

        # Finish episode
        collector.end_episode()

        # Get last trajectory for stats
        traj = collector.trajectories[-1]
        print(
            f"  Episode {i + 1}: {traj.length} steps, "
            f"reward={traj.total_reward:.2f}, "
            f"success={traj.infos[-1].get('success', False) if traj.infos else False}"
        )

    # Save trajectories
    output_dir = Path("data/oracle_trajectories")
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = collector.get_trajectories()
    for i, traj in enumerate(trajectories):
        output_file = output_dir / f"trajectory_{i:03d}.json"
        with open(output_file, "w") as f:
            json.dump(traj.to_dict(), f, indent=2, default=str)

    print(f"\nSaved {len(trajectories)} trajectories to {output_dir}")

    # Print statistics
    stats = collector.get_stats()
    print("\nCollection Statistics:")
    print(f"  Total episodes: {stats['total_episodes']}")
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Mean reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Mean length: {stats['mean_length']:.1f}")

    print("\nNext steps:")
    print("  1. Run export_to_huggingface.py to convert to HF format")
    print("  2. Use trajectories for supervised fine-tuning")

    env.close()


if __name__ == "__main__":
    main()
