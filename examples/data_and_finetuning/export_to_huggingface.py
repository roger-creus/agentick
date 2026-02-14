"""
Export trajectories to HuggingFace Datasets format.

Prepares collected trajectories for supervised fine-tuning with LLMs.
Runtime: <10 seconds

Usage:
    uv run python examples/data_and_finetuning/export_to_huggingface.py
"""

import json
from pathlib import Path


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
        with open(traj_file) as f:
            traj_data = json.load(f)
        trajectories.append(traj_data)
        print(
            f"  Loaded {traj_file.name}: {traj_data['length']} steps, "
            f"reward={traj_data['total_reward']:.2f}"
        )

    # Export to conversation format (for LLM fine-tuning)
    print("\nExporting to conversation format (JSONL)...")
    output_file = Path("data/oracle_conversations.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for i, traj in enumerate(trajectories):
            # Convert trajectory to conversation format
            conversation = {
                "id": f"trajectory_{i:04d}",
                "messages": [],
            }

            # Add system message
            conversation["messages"].append({
                "role": "system",
                "content": "You are an expert agent playing a navigation game. Navigate to the goal efficiently."
            })

            # Add trajectory steps as user/assistant pairs
            for j, (obs, action) in enumerate(zip(traj["observations"], traj["actions"])):
                # User message: observation
                conversation["messages"].append({
                    "role": "user",
                    "content": f"Step {j}: Observation: {obs}"
                })

                # Assistant message: action
                conversation["messages"].append({
                    "role": "assistant",
                    "content": f"Action: {action}"
                })

            # Write as JSONL
            f.write(json.dumps(conversation) + "\n")

    print(f"✓ Exported {len(trajectories)} conversations to {output_file}")

    # Create simple dataset summary
    summary_file = Path("data/dataset_summary.json")
    summary = {
        "num_trajectories": len(trajectories),
        "total_steps": sum(t["length"] for t in trajectories),
        "avg_reward": sum(t["total_reward"] for t in trajectories) / len(trajectories),
        "format": "JSONL conversation format",
        "output_file": str(output_file),
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Saved summary to {summary_file}")

    print("\n" + "=" * 80)
    print("EXPORT COMPLETE")
    print("=" * 80)
    print(f"Dataset: {output_file}")
    print(f"Format: JSONL (one conversation per line)")
    print(f"Total trajectories: {len(trajectories)}")
    print("\nNext steps:")
    print("  1. Load dataset with: datasets.load_dataset('json', data_files='...')")
    print("  2. Fine-tune with TRL or your preferred framework")
    print("  3. See sft_with_trl.py for example training code")


if __name__ == "__main__":
    main()
