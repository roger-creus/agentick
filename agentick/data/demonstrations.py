"""Generate demonstration trajectories from oracle/optimal policies."""

from __future__ import annotations

from typing import Any

import numpy as np

from agentick.data.collector import Trajectory, TrajectoryCollector


def collect_oracle_trajectories(
    env_id: str,
    num_episodes: int,
    difficulty: str = "easy",
    render_mode: str = "language",
    oracle_policy: Any | None = None,
    max_steps: int = 1000,
    seed: int = 42,
) -> list[Trajectory]:
    """
    Collect demonstration trajectories using oracle policy.

    Args:
        env_id: Environment identifier
        num_episodes: Number of episodes to collect
        difficulty: Difficulty level
        render_mode: Render mode for observations
        oracle_policy: Oracle policy (if None, uses env's oracle if available)
        max_steps: Maximum steps per episode
        seed: Random seed

    Returns:
        List of collected trajectories

    Example:
        >>> trajectories = collect_oracle_trajectories(
        ...     "GoToGoal-v0",
        ...     num_episodes=100,
        ...     difficulty="easy",
        ...     render_mode="language",
        ... )
        >>> print(f"Collected {len(trajectories)} trajectories")
    """
    import agentick

    # Create environment
    env = agentick.make(env_id, difficulty=difficulty, render_mode=render_mode)

    # Create collector
    collector = TrajectoryCollector(buffer_size=num_episodes + 10)

    # Set seed
    rng = np.random.default_rng(seed)

    successful_episodes = 0
    attempted_episodes = 0

    while successful_episodes < num_episodes and attempted_episodes < num_episodes * 3:
        # Reset environment
        episode_seed = rng.integers(0, 2**31)
        obs, info = env.reset(seed=episode_seed)

        # Start trajectory collection
        collector.start_episode(
            metadata={
                "env_id": env_id,
                "difficulty": difficulty,
                "seed": episode_seed,
                "oracle": True,
            }
        )

        # Use oracle policy if available
        if oracle_policy is None:
            # Try to get oracle from environment
            if hasattr(env.unwrapped, "get_oracle_action"):
                policy = env.unwrapped.get_oracle_action
            else:
                # Fallback: random policy
                def policy(obs, info):
                    return env.action_space.sample()
        else:
            policy = oracle_policy

        # Run episode
        done = False
        truncated = False
        steps = 0

        while not (done or truncated) and steps < max_steps:
            # Get action from oracle
            try:
                action = policy(obs, info)
            except Exception:
                # Fallback to random if oracle fails
                action = env.action_space.sample()

            # Step environment
            next_obs, reward, done, truncated, info = env.step(action)

            # Record step
            collector.add_step(obs, action, reward, done or truncated, info)

            obs = next_obs
            steps += 1

        # End trajectory
        collector.end_episode()

        # Track statistics
        attempted_episodes += 1
        if collector.trajectories[-1].total_reward > 0:
            successful_episodes += 1

    env.close()

    return collector.get_trajectories()


def collect_random_trajectories(
    env_id: str,
    num_episodes: int,
    difficulty: str = "easy",
    render_mode: str = "language",
    max_steps: int = 1000,
    seed: int = 42,
) -> list[Trajectory]:
    """
    Collect random baseline trajectories.

    Useful for DPO preference pairs (oracle=preferred, random=rejected).

    Args:
        env_id: Environment identifier
        num_episodes: Number of episodes to collect
        difficulty: Difficulty level
        render_mode: Render mode
        max_steps: Maximum steps per episode
        seed: Random seed

    Returns:
        List of trajectories from random policy
    """
    import agentick

    env = agentick.make(env_id, difficulty=difficulty, render_mode=render_mode)
    collector = TrajectoryCollector(buffer_size=num_episodes + 10)

    rng = np.random.default_rng(seed)

    for i in range(num_episodes):
        episode_seed = rng.integers(0, 2**31)
        obs, info = env.reset(seed=episode_seed)

        collector.start_episode(
            metadata={
                "env_id": env_id,
                "difficulty": difficulty,
                "seed": episode_seed,
                "random": True,
            }
        )

        done = False
        truncated = False
        steps = 0

        while not (done or truncated) and steps < max_steps:
            action = env.action_space.sample()
            next_obs, reward, done, truncated, info = env.step(action)
            collector.add_step(obs, action, reward, done or truncated, info)
            obs = next_obs
            steps += 1

        collector.end_episode()

    env.close()

    return collector.get_trajectories()


def create_preference_pairs(
    env_id: str,
    num_pairs: int,
    difficulty: str = "easy",
    render_mode: str = "language",
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Create preference pairs for DPO/preference learning.

    Each pair contains:
    - preferred: Oracle trajectory
    - rejected: Random trajectory
    Both from the same environment seed.

    Args:
        env_id: Environment identifier
        num_pairs: Number of pairs to create
        difficulty: Difficulty level
        render_mode: Render mode
        seed: Random seed

    Returns:
        List of preference pairs

    Example:
        >>> pairs = create_preference_pairs("GoToGoal-v0", num_pairs=100)
        >>> print(f"Preferred reward: {pairs[0]['preferred'].total_reward}")
        >>> print(f"Rejected reward: {pairs[0]['rejected'].total_reward}")
    """
    import agentick

    env = agentick.make(env_id, difficulty=difficulty, render_mode=render_mode)
    rng = np.random.default_rng(seed)

    pairs = []

    for i in range(num_pairs):
        episode_seed = rng.integers(0, 2**31)

        # Collect oracle trajectory
        oracle_collector = TrajectoryCollector(buffer_size=1)
        obs, info = env.reset(seed=episode_seed)
        oracle_collector.start_episode(metadata={"oracle": True, "seed": episode_seed})

        # Get oracle policy
        if hasattr(env.unwrapped, "get_oracle_action"):
            policy = env.unwrapped.get_oracle_action
        else:

            def policy(obs, info):
                return env.action_space.sample()

        done = False
        while not done:
            action = policy(obs, info)
            next_obs, reward, done, truncated, info = env.step(action)
            oracle_collector.add_step(obs, action, reward, done or truncated, info)
            obs = next_obs
            done = done or truncated

        oracle_collector.end_episode()

        # Collect random trajectory (same seed)
        random_collector = TrajectoryCollector(buffer_size=1)
        obs, info = env.reset(seed=episode_seed)
        random_collector.start_episode(metadata={"random": True, "seed": episode_seed})

        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, done, truncated, info = env.step(action)
            random_collector.add_step(obs, action, reward, done or truncated, info)
            obs = next_obs
            done = done or truncated

        random_collector.end_episode()

        # Create pair
        pairs.append(
            {
                "preferred": oracle_collector.trajectories[0],
                "rejected": random_collector.trajectories[0],
                "seed": episode_seed,
            }
        )

    env.close()

    return pairs
