"""
Collect oracle agent demonstrations for supervised fine-tuning.

Demonstrates trajectory collection for imitation learning.
Runtime: ~2 minutes for 10 trajectories
"""

from pathlib import Path

import agentick
from agentick.data.collector import TrajectoryCollector


def main():
    print("Oracle Trajectory Collection")
    print("=" * 80)

    # Create environment
    env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="language")

    # Create trajectory collector
    collector = TrajectoryCollector(
        env,
        record_obs=True,
        record_video=False,
        record_actions=True,
        record_rewards=True,
    )

    # Collect trajectories from oracle agent
    print("\nCollecting 10 oracle trajectories...")

    trajectories = []
    for i in range(10):
        # Use oracle agent (optimal policy)
        from agentick.agents.oracle import OracleAgent

        oracle = OracleAgent(env)

        # Collect one trajectory
        trajectory = collector.collect_episode(oracle, seed=42 + i)

        trajectories.append(trajectory)

        print(
            f"  Episode {i + 1}: {len(trajectory.observations)} steps, "
            f"reward={trajectory.total_reward:.2f}, "
            f"success={trajectory.info.get('success', False)}"
        )

    # Save trajectories
    output_dir = Path("data/oracle_trajectories")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, traj in enumerate(trajectories):
        traj.save(output_dir / f"trajectory_{i:03d}.json")

    print(f"\nSaved {len(trajectories)} trajectories to {output_dir}")
    print("\nNext steps:")
    print("  1. Run export_to_huggingface.py to convert to HF format")
    print("  2. Run sft_with_trl.py to fine-tune a model")

    env.close()


if __name__ == "__main__":
    main()
