"""
View and analyze leaderboard results.

This example demonstrates:
- Loading evaluation results
- Displaying formatted summaries
- Analyzing performance breakdowns

Requirements:
    uv sync

Usage:
    uv run python examples/leaderboard/view_results.py evaluation_results.json
"""

import argparse
import json

import numpy as np


def load_results(path: str) -> dict:
    """Load evaluation results."""
    with open(path) as f:
        return json.load(f)


def display_summary(data: dict):
    """Display summary statistics."""
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print(f"\nSubmission: {data.get('submission', 'Unknown')}")
    print(f"Model: {data.get('model', 'Unknown')}")
    print(f"Suite: {data.get('suite', 'Unknown')}")
    print(f"Episodes per task: {data.get('num_episodes', 'Unknown')}")

    if "elapsed_time" in data:
        print(f"Evaluation time: {data['elapsed_time']:.1f}s")


def display_task_breakdown(results: list[dict]):
    """Display per-task results."""
    print("\n" + "=" * 80)
    print("TASK BREAKDOWN")
    print("=" * 80)

    print(f"\n{'Task':<30} {'Reward':>12} {'Success':>10} {'Steps':>10}")
    print("-" * 80)

    for result in sorted(results, key=lambda x: x["mean_reward"], reverse=True):
        task = result["task"].replace("-v0", "")
        reward = result["mean_reward"]
        success = result["success_rate"]
        steps = result["mean_steps"]

        print(f"{task:<30} {reward:>11.2f}± {success:>8.1%} {steps:>10.1f}")


def display_capability_analysis(results: list[dict]):
    """Display capability analysis."""
    # Define capability categories
    categories = {
        "Navigation": ["GoToGoal", "MazeNavigation", "PreciseNavigation"],
        "Reasoning": ["ProgramSynthesis", "RuleInduction", "GraphColoring", "LightsOut"],
        "Memory": ["SequenceMemory", "DelayedGratification", "BacktrackPuzzle", "RecursiveRooms"],
        "Planning": ["KeyDoorPuzzle", "SokobanPush", "ShortestPath", "PackingPuzzle"],
        "Coordination": ["CooperativeTransport", "TagHunt", "Herding", "EmergentStrategy"],
        "Adaptation": [
            "FewShotAdaptation",
            "DistributionShift",
            "TaskInterference",
            "DynamicObstacles",
        ],
    }

    # Compute category scores
    category_scores = {}

    for category, task_list in categories.items():
        category_results = [r for r in results if any(task in r["task"] for task in task_list)]

        if category_results:
            rewards = [r["mean_reward"] for r in category_results]
            successes = [r["success_rate"] for r in category_results]

            category_scores[category] = {
                "reward": np.mean(rewards),
                "success": np.mean(successes),
                "count": len(category_results),
            }

    if not category_scores:
        return

    print("\n" + "=" * 80)
    print("CAPABILITY ANALYSIS")
    print("=" * 80)

    print(f"\n{'Capability':<20} {'Tasks':>8} {'Avg Reward':>12} {'Success Rate':>14}")
    print("-" * 80)

    for category, scores in sorted(
        category_scores.items(), key=lambda x: x[1]["success"], reverse=True
    ):
        print(
            f"{category:<20} {scores['count']:>8} {scores['reward']:>12.2f} {scores['success']:>13.1%}"
        )


def display_statistics(results: list[dict]):
    """Display overall statistics."""
    rewards = [r["mean_reward"] for r in results]
    successes = [r["success_rate"] for r in results]

    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)

    print(f"\nTotal tasks: {len(results)}")
    print(f"Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Median reward: {np.median(rewards):.2f}")
    print(f"Success rate: {np.mean(successes):.1%}")

    # Performance distribution
    excellent = sum(1 for s in successes if s >= 0.9)
    good = sum(1 for s in successes if 0.7 <= s < 0.9)
    fair = sum(1 for s in successes if 0.3 <= s < 0.7)
    poor = sum(1 for s in successes if s < 0.3)

    print("\nPerformance distribution:")
    print(f"  Excellent (≥90%): {excellent}")
    print(f"  Good (70-89%): {good}")
    print(f"  Fair (30-69%): {fair}")
    print(f"  Poor (<30%): {poor}")


def main():
    """View evaluation results."""
    parser = argparse.ArgumentParser(description="View leaderboard results")
    parser.add_argument(
        "results",
        type=str,
        help="Path to evaluation results JSON file",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Show detailed per-episode results",
    )
    args = parser.parse_args()

    print("Leaderboard Results Viewer")

    # Load results
    try:
        data = load_results(args.results)
    except Exception as e:
        print(f"❌ Failed to load results: {e}")
        return

    results = data.get("results", [])

    if not results:
        print("❌ No results found in file")
        return

    # Display views
    display_summary(data)
    display_task_breakdown(results)
    display_capability_analysis(results)
    display_statistics(results)

    # Detailed view
    if args.detail:
        print("\n" + "=" * 80)
        print("DETAILED RESULTS")
        print("=" * 80)

        for task_result in results:
            print(f"\n{task_result['task']}:")

            for episode in task_result["episodes"]:
                status = "✓" if episode["success"] else "✗"
                print(
                    f"  Episode {episode['episode']:2d}: {status} "
                    f"Reward={episode['reward']:.2f}, Steps={episode['steps']}"
                )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
