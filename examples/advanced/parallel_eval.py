"""
Parallel evaluation of agents across multiple tasks.

Demonstrates:
- Multi-process evaluation
- Vectorized environments
- Efficient batched evaluation
- Progress tracking
"""

import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np

import agentick
from agentick.benchmark.baselines import OracleAgent


def evaluate_single_task(
    task_id: str,
    difficulty: str,
    agent_fn: Callable,
    n_episodes: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Evaluate agent on a single task.

    Args:
        task_id: Task to evaluate
        difficulty: Difficulty level
        agent_fn: Function that returns agent instance
        n_episodes: Number of episodes
        seed: Random seed

    Returns:
        Dictionary with evaluation results
    """
    env = agentick.make(task_id, difficulty=difficulty, render_mode="ascii")
    agent = agent_fn(env)

    rewards = []
    lengths = []
    successes = []

    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode)
        total_reward = 0
        steps = 0

        for _ in range(100):
            action = agent.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        rewards.append(total_reward)
        lengths.append(steps)
        successes.append(info.get("success", False))

    env.close()

    return {
        "task_id": task_id,
        "difficulty": difficulty,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
        "n_episodes": n_episodes,
    }


def parallel_evaluate(
    task_ids: list[str],
    agent_fn: Callable,
    difficulty: str = "easy",
    n_episodes: int = 10,
    n_workers: int = 4,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Evaluate agent on multiple tasks in parallel.

    Args:
        task_ids: List of task IDs to evaluate
        agent_fn: Function that creates agent instance
        difficulty: Difficulty level
        n_episodes: Episodes per task
        n_workers: Number of parallel workers
        seed: Random seed

    Returns:
        List of results dictionaries
    """
    print(f"\nEvaluating on {len(task_ids)} tasks with {n_workers} workers...")
    print(f"Episodes per task: {n_episodes}")
    print()

    results = []
    start_time = time.time()

    # Create process pool
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(
                evaluate_single_task,
                task_id=task_id,
                difficulty=difficulty,
                agent_fn=agent_fn,
                n_episodes=n_episodes,
                seed=seed,
            ): task_id
            for task_id in task_ids
        }

        # Collect results as they complete
        for i, future in enumerate(as_completed(future_to_task), 1):
            task_id = future_to_task[future]

            try:
                result = future.result()
                results.append(result)

                # Show progress
                print(
                    f"[{i}/{len(task_ids)}] {task_id}: "
                    f"success_rate={result['success_rate']:.2%}, "
                    f"mean_reward={result['mean_reward']:.3f}"
                )

            except Exception as e:
                print(f"[{i}/{len(task_ids)}] {task_id}: ERROR - {e}")

    elapsed = time.time() - start_time

    print(f"\nCompleted in {elapsed:.1f} seconds")
    print(f"Average time per task: {elapsed / len(task_ids):.2f}s")

    return results


def create_oracle_agent(env):
    """Create oracle agent for given environment."""
    return OracleAgent(env)


def create_random_agent(env):
    """Create random agent."""

    class RandomAgent:
        def __init__(self, env):
            self.env = env

        def act(self, obs, info):
            return self.env.action_space.sample()

    return RandomAgent(env)


def demo_parallel_evaluation():
    """Demonstrate parallel evaluation."""
    print("\n" + "=" * 60)
    print("PARALLEL EVALUATION DEMO")
    print("=" * 60)

    # Select tasks to evaluate
    tasks = [
        "GoToGoal-v0",
        "MazeNavigation-v0",
        "KeyDoorPuzzle-v0",
        "PushBlock-v0",
        "MultiGoal-v0",
    ]

    # Run parallel evaluation
    results = parallel_evaluate(
        task_ids=tasks,
        agent_fn=create_oracle_agent,
        difficulty="easy",
        n_episodes=10,
        n_workers=4,
        seed=42,
    )

    # Display summary
    print("\n" + "=" * 60)
    print("SUMMARY RESULTS")
    print("=" * 60)

    for result in sorted(results, key=lambda x: x["success_rate"], reverse=True):
        print(f"\n{result['task_id']}:")
        print(f"  Success rate: {result['success_rate']:.2%}")
        print(f"  Mean reward:  {result['mean_reward']:.3f} ± {result['std_reward']:.3f}")
        print(f"  Mean length:  {result['mean_length']:.1f} steps")

    # Overall statistics
    overall_success = np.mean([r["success_rate"] for r in results])
    overall_reward = np.mean([r["mean_reward"] for r in results])

    print("\n" + "=" * 60)
    print("OVERALL PERFORMANCE")
    print("=" * 60)
    print(f"Average success rate: {overall_success:.2%}")
    print(f"Average reward: {overall_reward:.3f}")


def compare_agents_parallel():
    """Compare multiple agents in parallel."""
    print("\n" + "=" * 60)
    print("PARALLEL AGENT COMPARISON")
    print("=" * 60)

    tasks = ["GoToGoal-v0", "MazeNavigation-v0", "KeyDoorPuzzle-v0"]

    agents = {"Oracle": create_oracle_agent, "Random": create_random_agent}

    all_results = {}

    for agent_name, agent_fn in agents.items():
        print(f"\nEvaluating {agent_name}...")

        results = parallel_evaluate(
            task_ids=tasks,
            agent_fn=agent_fn,
            difficulty="easy",
            n_episodes=5,
            n_workers=3,
            seed=42,
        )

        all_results[agent_name] = results

    # Compare results
    print("\n" + "=" * 60)
    print("AGENT COMPARISON")
    print("=" * 60)

    for task in tasks:
        print(f"\n{task}:")

        for agent_name, results in all_results.items():
            task_result = next((r for r in results if r["task_id"] == task), None)

            if task_result:
                print(
                    f"  {agent_name:10s}: success={task_result['success_rate']:.2%}, "
                    f"reward={task_result['mean_reward']:.3f}"
                )


def benchmark_speedup():
    """Benchmark parallel vs sequential evaluation."""
    print("\n" + "=" * 60)
    print("SPEEDUP BENCHMARK")
    print("=" * 60)

    tasks = ["GoToGoal-v0", "MazeNavigation-v0", "KeyDoorPuzzle-v0"]

    # Sequential evaluation (1 worker)
    print("\nSequential (1 worker)...")
    start = time.time()
    parallel_evaluate(
        task_ids=tasks,
        agent_fn=create_oracle_agent,
        n_episodes=5,
        n_workers=1,
        seed=42,
    )
    sequential_time = time.time() - start

    # Parallel evaluation (4 workers)
    print("\nParallel (4 workers)...")
    start = time.time()
    parallel_evaluate(
        task_ids=tasks,
        agent_fn=create_oracle_agent,
        n_episodes=5,
        n_workers=4,
        seed=42,
    )
    parallel_time = time.time() - start

    # Show speedup
    speedup = sequential_time / parallel_time

    print("\n" + "=" * 60)
    print("SPEEDUP ANALYSIS")
    print("=" * 60)
    print(f"Sequential time: {sequential_time:.2f}s")
    print(f"Parallel time:   {parallel_time:.2f}s")
    print(f"Speedup:         {speedup:.2f}x")
    print(f"Efficiency:      {speedup / 4:.1%} (ideal: 100%)")


def main():
    """Run all parallel evaluation demos."""
    print("\n" + "=" * 70)
    print("PARALLEL EVALUATION EXAMPLES")
    print("=" * 70)

    demo_parallel_evaluation()
    compare_agents_parallel()
    benchmark_speedup()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("\n1. Use parallel evaluation to speed up multi-task benchmarks")
    print("2. Adjust n_workers based on available CPU cores")
    print("3. Parallel evaluation scales well for independent evaluations")
    print("4. Typical speedup: 2-3x with 4 workers (depends on task)")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
