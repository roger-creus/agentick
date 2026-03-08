"""
Create a leaderboard submission file.

Shows how to create submission.yaml for leaderboard evaluation.
Runtime: <1 second
"""

from pathlib import Path


def main():
    print("Create Leaderboard Submission")
    print("=" * 80)

    # Example submission for random agent
    submission_yaml = """# Agent Submission
agent_name: "RandomAgent-Baseline-v1"
author: "Your Name"
description: "Random agent baseline for comparison"
url: null
tags:
  - baseline
  - random
license: "MIT"
open_weights: true

agent_type: "code"
observation_mode: "language"

config:
  agent_class: "agentick.agents.random.RandomAgent"

suites:
  - "agentick-full-v2"
  # - "agentick-full-v2"  # Uncomment for full evaluation

hardware: "CPU"
estimated_cost: "$0 (deterministic)"
training_data: "None"
training_compute: "N/A"
"""

    # Save to file
    output_path = Path("submission.yaml")
    with open(output_path, "w") as f:
        f.write(submission_yaml)

    print(f"Created {output_path}")
    print("\nSubmission template created!")
    print("\nNext steps:")
    print("  1. Edit submission.yaml with your agent details")
    print(
        "  2. Run: uv run agentick evaluate --submission submission.yaml --suite agentick-full-v2"
    )
    print("  3. Or run: python examples/leaderboard/run_evaluation.py")


if __name__ == "__main__":
    main()
