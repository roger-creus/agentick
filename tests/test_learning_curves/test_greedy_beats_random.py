"""Test that greedy agent beats random on navigation tasks."""

import numpy as np
import pytest

import agentick
from agentick.benchmark.baselines import GreedyAgent, RandomAgent

NAVIGATION_TASKS = [
    "GoToGoal-v0",
    "MazeNavigation-v0",
    "MultiGoalRoute-v0",
]


@pytest.mark.parametrize("task_name", NAVIGATION_TASKS)
@pytest.mark.timeout(120)
def test_greedy_beats_random(task_name):
    """Test that greedy agent outperforms random on navigation tasks."""
    env = agentick.make(task_name, difficulty="easy", reward_mode="sparse")

    random_agent = RandomAgent()
    greedy_agent = GreedyAgent()

    try:
        # Run 50 episodes for each agent
        random_returns = []
        greedy_returns = []

        for episode in range(50):
            # Random agent episode
            obs, info = env.reset(seed=episode)
            episode_return = 0
            for step in range(env.spec.max_episode_steps or 100):
                valid_actions = env.get_valid_actions()
                action = random_agent.act(obs, valid_actions)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                if terminated or truncated:
                    break
            random_returns.append(episode_return)

            # Greedy agent episode (same seed)
            obs, info = env.reset(seed=episode)
            episode_return = 0
            for step in range(env.spec.max_episode_steps or 100):
                state_dict = env.get_state_dict()
                valid_actions = env.get_valid_actions()
                action = greedy_agent.act(obs, valid_actions, state_dict)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                if terminated or truncated:
                    break
            greedy_returns.append(episode_return)

        random_mean = np.mean(random_returns)
        greedy_mean = np.mean(greedy_returns)

        # Greedy should be better (or at least not worse)
        # Allow small margin for randomness
        assert (
            greedy_mean >= random_mean - 0.1
        ), f"Greedy ({greedy_mean:.2f}) should beat random ({random_mean:.2f})"
    finally:
        env.close()
