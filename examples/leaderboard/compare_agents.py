"""
Compare multiple agent submissions.

This example demonstrates:
- Loading multiple evaluation results
- Side-by-side comparison
- Statistical significance testing

Requirements:
    uv sync

Usage:
    uv run python examples/leaderboard/compare_agents.py results1.json results2.json
"""

import argparse
import json

import numpy as np


def load_results(path: str) -> dict:
    """Load evaluation results."""
    with open(path) as f:
        return json.load(f)


def main():
    """Compare agent results."""
    parser = argparse.ArgumentParser(description="Compare agent submissions")
    parser.add_argument(
        "results",
        nargs="+",
        help="Paths to evaluation result JSON files",
    )
    args = parser.parse_args()

    print("Agent Comparison")
    print("=" * 80)

    if len(args.results) < 2:
        print("❌ Need at least 2 result files to compare")
        print("Usage: python compare_agents.py results1.json results2.json [...]")
        return

    # Load all results
    all_data = []

    for path in args.results:
        try:
            data = load_results(path)
            all_data.append((path, data))
            print(f"✓ Loaded: {data.get('submission', 'Unknown')} from {path}")
        except Exception as e:
            print(f"❌ Failed to load {path}: {e}")

    if len(all_data) < 2:
        print("\n❌ Not enough valid result files")
        return

    print()

    # Extract results
    agents = []

    for path, data in all_data:
        name = data.get("submission", path)
        results = data.get("results", [])

        # Compute overall metrics
        rewards = [r["mean_reward"] for r in results]
        successes = [r["success_rate"] for r in results]

        agents.append(
            {
                "name": name,
                "model": data.get("model", "Unknown"),
                "results": results,
                "mean_reward": np.mean(rewards) if rewards else 0,
                "mean_success": np.mean(successes) if successes else 0,
                "num_tasks": len(results),
            }
        )

    # Overall comparison
    print("=" * 80)
    print("OVERALL COMPARISON")
    print("=" * 80)

    print(f"\n{'Agent':<30} {'Model':<25} {'Tasks':>7} {'Reward':>10} {'Success':>10}")
    print("-" * 100)

    for agent in sorted(agents, key=lambda x: x["mean_reward"], reverse=True):
        print(
            f"{agent['name']:<30} "
            f"{agent['model']:<25} "
            f"{agent['num_tasks']:>7} "
            f"{agent['mean_reward']:>10.2f} "
            f"{agent['mean_success']:>9.1%}"
        )

    # Task-by-task comparison
    print("\n" + "=" * 80)
    print("TASK-BY-TASK COMPARISON")
    print("=" * 80)

    # Get common tasks
    task_sets = [set(r["task"] for r in agent["results"]) for agent in agents]
    common_tasks = task_sets[0].intersection(*task_sets[1:])

    if not common_tasks:
        print("\n⚠️  No common tasks found across all agents")
        return

    print(f"\n{len(common_tasks)} common tasks\n")

    # Create comparison table
    header = f"{'Task':<30}"
    for agent in agents:
        header += f" {agent['name'][:15]:>15}"

    print(header)
    print("-" * (30 + 16 * len(agents)))

    for task in sorted(common_tasks):
        row = f"{task.replace('-v0', ''):<30}"

        for agent in agents:
            # Find task result
            task_result = next((r for r in agent["results"] if r["task"] == task), None)

            if task_result:
                success = task_result["success_rate"]
                row += f" {success:>14.1%}"
            else:
                row += f" {'N/A':>15}"

        print(row)

    # Head-to-head comparison (if exactly 2 agents)
    if len(agents) == 2:
        agent1, agent2 = agents

        print("\n" + "=" * 80)
        print(f"HEAD-TO-HEAD: {agent1['name']} vs {agent2['name']}")
        print("=" * 80)

        # Win/loss/tie
        wins = 0
        losses = 0
        ties = 0

        for task in common_tasks:
            result1 = next((r for r in agent1["results"] if r["task"] == task), None)
            result2 = next((r for r in agent2["results"] if r["task"] == task), None)

            if result1 and result2:
                if result1["success_rate"] > result2["success_rate"]:
                    wins += 1
                elif result1["success_rate"] < result2["success_rate"]:
                    losses += 1
                else:
                    ties += 1

        print(f"\n{agent1['name']}:")
        print(f"  Wins: {wins}")
        print(f"  Losses: {losses}")
        print(f"  Ties: {ties}")

        if wins + losses + ties > 0:
            win_rate = wins / (wins + losses + ties)
            print(f"  Win rate: {win_rate:.1%}")

    # Best/Worst tasks
    print("\n" + "=" * 80)
    print("BEST/WORST TASKS PER AGENT")
    print("=" * 80)

    for agent in agents:
        if not agent["results"]:
            continue

        print(f"\n{agent['name']}:")

        # Sort by success rate
        sorted_results = sorted(agent["results"], key=lambda x: x["success_rate"], reverse=True)

        print("  Best:")
        for result in sorted_results[:3]:
            task = result["task"].replace("-v0", "")
            success = result["success_rate"]
            print(f"    {task:<30} {success:>6.1%}")

        print("  Worst:")
        for result in sorted_results[-3:]:
            task = result["task"].replace("-v0", "")
            success = result["success_rate"]
            print(f"    {task:<30} {success:>6.1%}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
