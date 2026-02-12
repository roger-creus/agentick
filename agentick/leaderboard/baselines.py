"""Pre-computed baseline results for normalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


def run_random_baseline(
    task_name: str,
    difficulty: str = "medium",
    n_episodes: int = 50,
    seeds: list[int] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run random agent on a task to establish baseline performance.

    Args:
        task_name: Name of the task
        difficulty: Difficulty level
        n_episodes: Number of episodes to run
        seeds: Optional list of seeds (if None, generates sequential seeds)
        verbose: Whether to show progress bar

    Returns:
        Dictionary with mean_return, std_return, episode_returns
    """
    import agentick

    if seeds is None:
        seeds = list(range(n_episodes))

    env = agentick.make(task_name, difficulty=difficulty)

    episode_returns = []
    success_flags = []

    iterator = tqdm(seeds, desc=f"Random baseline: {task_name}") if verbose else seeds

    for seed in iterator:
        obs, info = env.reset(seed=seed)
        done = False
        episode_return = 0.0

        while not done:
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            done = terminated or truncated

        episode_returns.append(episode_return)
        success_flags.append(info.get("success", False))

    return {
        "task_name": task_name,
        "difficulty": difficulty,
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "success_rate": float(np.mean(success_flags)),
        "episode_returns": episode_returns,
        "n_episodes": len(episode_returns),
    }


def run_greedy_baseline(
    task_name: str,
    difficulty: str = "medium",
    n_episodes: int = 50,
    seeds: list[int] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run greedy heuristic agent (moves toward goal).

    Args:
        task_name: Name of the task
        difficulty: Difficulty level
        n_episodes: Number of episodes to run
        seeds: Optional list of seeds
        verbose: Whether to show progress bar

    Returns:
        Dictionary with performance metrics
    """
    import agentick
    from agentick.benchmark.baselines import GreedyAgent

    if seeds is None:
        seeds = list(range(n_episodes))

    env = agentick.make(task_name, difficulty=difficulty, render_mode="state_dict")
    agent = GreedyAgent()

    episode_returns = []
    success_flags = []

    iterator = tqdm(seeds, desc=f"Greedy baseline: {task_name}") if verbose else seeds

    for seed in iterator:
        obs, info = env.reset(seed=seed)
        done = False
        episode_return = 0.0

        while not done:
            # Greedy action
            state_dict = env.render()
            valid_actions = info.get("valid_actions", list(range(env.action_space.n)))
            action = agent.act(obs, valid_actions, state_dict)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            done = terminated or truncated

        episode_returns.append(episode_return)
        success_flags.append(info.get("success", False))

    return {
        "task_name": task_name,
        "difficulty": difficulty,
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "success_rate": float(np.mean(success_flags)),
        "episode_returns": episode_returns,
        "n_episodes": len(episode_returns),
    }


def run_oracle_baseline(
    task_name: str,
    difficulty: str = "medium",
    n_episodes: int = 20,
    seeds: list[int] | None = None,
    verbose: bool = True,
) -> dict[str, Any] | None:
    """
    Run optimal BFS/A* agent to establish oracle performance.

    Note: Only works on tractable tasks. May timeout or fail on complex tasks.

    Args:
        task_name: Name of the task
        difficulty: Difficulty level
        n_episodes: Number of episodes to run
        seeds: Optional list of seeds
        verbose: Whether to show progress bar

    Returns:
        Dictionary with performance metrics, or None if oracle fails
    """
    import agentick
    from agentick.benchmark.baselines import OracleAgent

    if seeds is None:
        seeds = list(range(n_episodes))

    try:
        env = agentick.make(task_name, difficulty=difficulty)
    except Exception as e:
        if verbose:
            print(f"Failed to create environment for {task_name}: {e}")
        return None

    episode_returns = []
    success_flags = []

    iterator = tqdm(seeds, desc=f"Oracle baseline: {task_name}") if verbose else seeds

    for seed in iterator:
        try:
            obs, info = env.reset(seed=seed)

            # Create oracle agent with environment
            agent = OracleAgent(env=env)

            done = False
            episode_return = 0.0
            max_steps = 1000  # Safety limit

            step_count = 0
            while not done and step_count < max_steps:
                # Oracle action
                action = agent.act(obs, info.get("valid_actions"), None)

                if action is None:
                    # Oracle failed to find path
                    break

                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                done = terminated or truncated
                step_count += 1

            if done and info.get("success", False):
                episode_returns.append(episode_return)
                success_flags.append(True)
            else:
                # Oracle failed on this seed
                if verbose:
                    print(f"Oracle failed on seed {seed}")

        except Exception as e:
            if verbose:
                print(f"Oracle error on seed {seed}: {e}")
            continue

    if len(episode_returns) == 0:
        if verbose:
            print(f"Oracle failed on all episodes for {task_name}")
        return None

    return {
        "task_name": task_name,
        "difficulty": difficulty,
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "success_rate": float(np.mean(success_flags)),
        "episode_returns": episode_returns,
        "n_episodes": len(episode_returns),
    }


def compute_baselines_for_suite(
    suite_name: str,
    output_dir: str | Path = "leaderboard_data/baselines",
    run_random: bool = True,
    run_greedy: bool = True,
    run_oracle: bool = True,
    n_episodes_random: int = 50,
    n_episodes_oracle: int = 20,
) -> dict[str, dict[str, Any]]:
    """
    Compute all baselines for a benchmark suite.

    Args:
        suite_name: Name of the suite (e.g., "agentick-full-v1")
        output_dir: Directory to save baseline results
        run_random: Whether to run random baseline
        run_greedy: Whether to run greedy baseline
        run_oracle: Whether to run oracle baseline
        n_episodes_random: Number of episodes for random/greedy
        n_episodes_oracle: Number of episodes for oracle

    Returns:
        Dictionary mapping task_name to baselines
    """
    from agentick.leaderboard.suites import get_suite

    suite = get_suite(suite_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baselines = {}

    print(f"\n=== Computing baselines for {suite.display_name} ===\n")

    for task_name in suite.tasks:
        print(f"\nTask: {task_name}")
        baselines[task_name] = {}

        # Random baseline
        if run_random:
            random_result = run_random_baseline(
                task_name,
                difficulty=suite.difficulty,
                n_episodes=n_episodes_random,
                seeds=list(suite.eval_seeds[:n_episodes_random]),
            )
            baselines[task_name]["random"] = random_result
            baselines[task_name]["random_baseline"] = random_result["mean_return"]

        # Greedy baseline
        if run_greedy:
            greedy_result = run_greedy_baseline(
                task_name,
                difficulty=suite.difficulty,
                n_episodes=n_episodes_random,
                seeds=list(suite.eval_seeds[:n_episodes_random]),
            )
            baselines[task_name]["greedy"] = greedy_result

        # Oracle baseline (for tractable tasks)
        if run_oracle:
            oracle_result = run_oracle_baseline(
                task_name,
                difficulty=suite.difficulty,
                n_episodes=n_episodes_oracle,
                seeds=list(suite.eval_seeds[:n_episodes_oracle]),
            )
            if oracle_result is not None:
                baselines[task_name]["oracle"] = oracle_result
                baselines[task_name]["optimal_return"] = oracle_result["mean_return"]
            else:
                # Use greedy or heuristic estimate
                if "greedy" in baselines[task_name]:
                    baselines[task_name]["optimal_return"] = (
                        baselines[task_name]["greedy"]["mean_return"] * 1.5
                    )
                else:
                    baselines[task_name]["optimal_return"] = (
                        baselines[task_name]["random_baseline"] * 10.0
                    )

    # Save to JSON
    output_file = output_dir / f"{suite_name}_baselines.json"
    with open(output_file, "w") as f:
        json.dump(baselines, f, indent=2, sort_keys=True)

    print(f"\n✓ Baselines saved to {output_file}")

    return baselines


def load_baselines(baseline_file: str | Path) -> dict[str, dict[str, float]]:
    """
    Load baselines from JSON file.

    Args:
        baseline_file: Path to baseline JSON file

    Returns:
        Dictionary mapping task_name to {random_baseline, optimal_return}
    """
    with open(baseline_file) as f:
        data = json.load(f)

    # Extract just random_baseline and optimal_return for each task
    baselines = {}
    for task_name, task_data in data.items():
        baselines[task_name] = {
            "random_baseline": task_data.get("random_baseline", 0.0),
            "optimal_return": task_data.get("optimal_return", 1.0),
        }

    return baselines
