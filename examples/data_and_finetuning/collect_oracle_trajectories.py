"""
Collect oracle agent demonstrations for supervised fine-tuning.

Demonstrates trajectory collection for imitation learning using the optimal oracle agent.
Runtime: ~1 minute for 10 trajectories

Usage:
    uv run python examples/data_and_finetuning/collect_oracle_trajectories.py
"""

from pathlib import Path

import agentick
from agentick.data import DataCollector
from agentick.oracles import get_oracle


def main():
    print("Oracle Trajectory Collection")
    print("=" * 80)

    # Create environment
    env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="language")

    # Create oracle agent (uses privileged information for optimal policy)
    oracle = get_oracle("GoToGoal-v0", env)

    # Collect trajectories using the high-level DataCollector
    collector = DataCollector(env, oracle, record_modalities=["language"])
    print("\nCollecting 10 oracle trajectories...")
    dataset = collector.collect(num_episodes=10, seeds=range(10))

    # Print summary
    print(f"\nCollected {len(dataset.trajectories)} trajectories")
    for i, traj in enumerate(dataset.trajectories):
        success = traj.infos[-1].get("success", False) if traj.infos else False
        print(
            f"  Episode {i + 1}: {traj.length} steps, "
            f"reward={traj.total_reward:.2f}, success={success}"
        )

    # Save trajectories
    output_dir = Path("trajectories/oracle_gotogoal/")
    dataset.save(output_dir)
    print(f"\nSaved trajectories to {output_dir}")

    # Export to HuggingFace conversation format (for SFT)
    hf_dir = Path("trajectories/hf_conv/")
    dataset.export_to_huggingface(hf_dir, format="conversation")
    print(f"Exported HuggingFace dataset to {hf_dir}")

    print("\nNext steps:")
    print("  1. Run sft_with_trl.py to fine-tune a model on this data")
    print("  2. Or run export_to_huggingface.py for more export options")

    env.close()


if __name__ == "__main__":
    main()
