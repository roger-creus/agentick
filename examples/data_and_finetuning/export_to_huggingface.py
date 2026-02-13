"""
Export trajectories to HuggingFace Datasets format.

Prepares collected trajectories for LLM fine-tuning.
Runtime: <10 seconds
"""

from pathlib import Path

from agentick.data.collector import Trajectory
from agentick.data.formats import export_to_format


def main():
    print("Export Trajectories to HuggingFace Format")
    print("=" * 80)

    # Load trajectories
    traj_dir = Path("data/oracle_trajectories")
    if not traj_dir.exists():
        print(f"ERROR: {traj_dir} not found.")
        print("Run collect_oracle_trajectories.py first.")
        return

    traj_files = sorted(traj_dir.glob("trajectory_*.json"))
    print(f"\nFound {len(traj_files)} trajectory files")

    trajectories = []
    for traj_file in traj_files:
        traj = Trajectory.load(traj_file)
        trajectories.append(traj)
        print(
            f"  Loaded {traj_file.name}: {len(traj.observations)} steps, "
            f"reward={traj.total_reward:.2f}"
        )

    # Export to conversation format (for LLM fine-tuning)
    print("\nExporting to conversation format...")
    output_path = export_to_format(
        trajectories,
        output_path="data/oracle_conversations.jsonl",
        format_type="conversation",
    )

    print(f"Exported to {output_path}")

    # Export to HuggingFace Dataset format
    print("\nExporting to HuggingFace Dataset format...")
    output_path = export_to_format(
        trajectories,
        output_path="data/oracle_dataset",
        format_type="hf_dataset",
    )

    print(f"Exported to {output_path}")
    print("\nDataset ready for fine-tuning!")
    print("Next: Run sft_with_trl.py to fine-tune a model")


if __name__ == "__main__":
    main()
