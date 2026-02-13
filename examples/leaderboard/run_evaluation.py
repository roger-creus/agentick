"""
Run leaderboard evaluation locally.

This example demonstrates:
- Running full leaderboard evaluation
- Testing submission before uploading
- Generating evaluation results

Requirements:
    uv sync --extra all

Usage:
    uv run python examples/leaderboard/run_evaluation.py submission.yaml
"""

import argparse
import json
import time
from pathlib import Path

import yaml

import agentick


def load_submission(path: str) -> dict:
    """Load submission YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def create_agent_from_adapter(adapter_code: str, model: str):
    """Create agent function from adapter code."""
    # Execute adapter code in isolated namespace
    namespace = {
        "agentick": agentick,
    }

    exec(adapter_code, namespace)

    # Extract get_action function
    if "get_action" not in namespace:
        raise ValueError("Adapter code must define 'get_action' function")

    return namespace["get_action"]


def run_evaluation(
    agent_fn,
    task_id: str,
    num_episodes: int = 10,
    difficulty: str = None,
) -> dict:
    """Run evaluation on a single task."""
    env_kwargs = {"render_mode": "ascii"}
    if difficulty:
        env_kwargs["difficulty"] = difficulty

    env = agentick.make(task_id, **env_kwargs)

    results = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        done = False
        total_reward = 0
        steps = 0

        while not done:
            # Get action from agent
            try:
                action = agent_fn(obs, env)
            except Exception as e:
                print(f"    ⚠️  Agent error: {e}")
                action = env.action_space.sample()  # Fallback

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1

        results.append(
            {
                "episode": episode,
                "reward": total_reward,
                "steps": steps,
                "success": info.get("success", False),
            }
        )

    env.close()

    # Compute statistics
    import numpy as np

    rewards = [r["reward"] for r in results]
    steps = [r["steps"] for r in results]
    successes = [r["success"] for r in results]

    return {
        "task": task_id,
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_steps": np.mean(steps),
        "success_rate": np.mean(successes),
        "episodes": results,
    }


def main():
    """Run leaderboard evaluation."""
    parser = argparse.ArgumentParser(description="Run leaderboard evaluation locally")
    parser.add_argument(
        "submission",
        type=str,
        help="Path to submission YAML file",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="core",
        help="Task suite to evaluate on",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Episodes per task",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results",
    )
    args = parser.parse_args()

    print("Leaderboard Evaluation")
    print("=" * 80)

    # Load submission
    print(f"Loading submission: {args.submission}")

    try:
        submission = load_submission(args.submission)
    except Exception as e:
        print(f"❌ Failed to load submission: {e}")
        return

    print(f"  Name: {submission.get('name', 'Unknown')}")
    print(f"  Model: {submission.get('agent', {}).get('model', 'Unknown')}")
    print()

    # Create agent
    print("Creating agent from adapter...")

    try:
        adapter_code = submission["agent"]["adapter"]
        model = submission["agent"]["model"]
        agent_fn = create_agent_from_adapter(adapter_code, model)
        print("  ✓ Agent created")
    except Exception as e:
        print(f"  ❌ Failed to create agent: {e}")
        return

    # Get tasks from suite
    print(f"\nLoading task suite: {args.suite}")

    from agentick.leaderboard.suites import get_suite_tasks

    try:
        tasks = get_suite_tasks(args.suite)
        print(f"  ✓ {len(tasks)} tasks loaded")
    except Exception as e:
        print(f"  ❌ Failed to load suite: {e}")
        # Fallback to core tasks
        tasks = ["GoToGoal-v0", "MazeNavigation-v0", "KeyDoorPuzzle-v0"]
        print(f"  Using fallback tasks: {tasks}")

    # Run evaluation
    print(f"\nRunning evaluation ({args.num_episodes} episodes per task)...")
    print()

    all_results = []
    start_time = time.time()

    for i, task_id in enumerate(tasks, 1):
        print(f"  [{i}/{len(tasks)}] {task_id}...", end=" ", flush=True)

        try:
            result = run_evaluation(
                agent_fn,
                task_id,
                num_episodes=args.num_episodes,
            )

            all_results.append(result)

            print(f"✓ Reward={result['mean_reward']:.2f}, Success={result['success_rate']:.1%}")

        except Exception as e:
            print(f"❌ Error: {e}")

    elapsed = time.time() - start_time

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "submission": submission.get("name", "Unknown"),
                "model": submission.get("agent", {}).get("model", "Unknown"),
                "suite": args.suite,
                "num_episodes": args.num_episodes,
                "results": all_results,
                "elapsed_time": elapsed,
            },
            f,
            indent=2,
        )

    print(f"\n✓ Results saved to: {results_path}")

    # Print summary
    import numpy as np

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    mean_rewards = [r["mean_reward"] for r in all_results]
    success_rates = [r["success_rate"] for r in all_results]

    print(f"\nTasks evaluated: {len(all_results)}")
    print(f"Average reward: {np.mean(mean_rewards):.2f}")
    print(f"Average success rate: {np.mean(success_rates):.1%}")
    print(f"Time: {elapsed:.1f}s")

    print("\n💡 Next steps:")
    print("  - Review results in evaluation_results.json")
    print("  - If satisfied, submit to the official leaderboard")


if __name__ == "__main__":
    main()
