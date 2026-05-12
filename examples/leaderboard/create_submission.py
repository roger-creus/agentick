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
    submission_yaml = '''# Agent Submission
name: "RandomAgent-Baseline-v1"
description: "Random action baseline for local smoke testing"
contact:
  github: "your-github-handle"
agent:
  model: "random-baseline"
  adapter: |
    def get_action(obs, env):
        return env.action_space.sample()

code: "https://github.com/roger-creus/agentick"
license: "MIT"
hardware: "CPU"
estimated_cost: "$0"
training_data: "None"
training_compute: "N/A"
'''

    # Save to file
    output_path = Path("submission.yaml")
    with open(output_path, "w") as f:
        f.write(submission_yaml)

    print(f"Created {output_path}")
    print("\nSubmission template created!")
    print("\nNext steps:")
    print("  1. Edit submission.yaml with your agent details")
    print(
        "  2. Run: uv run python examples/leaderboard/validate_submission.py submission.yaml"
    )
    print(
        "  3. Run: uv run python examples/leaderboard/run_evaluation.py "
        "submission.yaml --suite agentick-navigation-v2 --num-episodes 1"
    )


if __name__ == "__main__":
    main()
