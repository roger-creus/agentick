"""Transfer learning evaluator for world model testing.

Tests agent's ability to adapt to modified environment mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TransferResult:
    """Result from transfer learning evaluation."""

    baseline_performance: float
    transfer_performance: float
    adaptation_speed: float  # Steps to reach baseline performance
    final_gap: float  # Performance gap after adaptation period


class TransferEvaluator:
    """Evaluate agent's ability to adapt to modified mechanics."""

    def __init__(
        self,
        base_env_factory,
        modified_env_factory,
        n_episodes: int = 10,
        adaptation_episodes: int = 5,
    ):
        """
        Initialize transfer evaluator.

        Args:
            base_env_factory: Factory for base environment
            modified_env_factory: Factory for modified environment
            n_episodes: Number of episodes for evaluation
            adaptation_episodes: Number of episodes for adaptation
        """
        self.base_env_factory = base_env_factory
        self.modified_env_factory = modified_env_factory
        self.n_episodes = n_episodes
        self.adaptation_episodes = adaptation_episodes

    def evaluate_transfer(
        self,
        agent,
        seed: int | None = None,
    ) -> TransferResult:
        """
        Evaluate transfer to modified environment.

        Args:
            agent: Agent to evaluate (with act() and optionally adapt() methods)
            seed: Random seed

        Returns:
            TransferResult with adaptation metrics
        """
        rng = np.random.default_rng(seed)

        # Measure baseline performance on original env
        baseline_rewards = []
        for ep in range(self.n_episodes):
            env = self.base_env_factory()
            obs, _ = env.reset(seed=rng.integers(0, 10000))
            episode_reward = 0.0
            done = False

            while not done:
                if hasattr(agent, "act"):
                    action = agent.act(obs)
                else:
                    action = env.action_space.sample()

                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated

            baseline_rewards.append(episode_reward)

        baseline_performance = np.mean(baseline_rewards)

        # Measure transfer performance on modified env
        transfer_rewards = []
        steps_to_baseline = float("inf")

        for ep in range(self.adaptation_episodes):
            env = self.modified_env_factory()
            obs, _ = env.reset(seed=rng.integers(0, 10000))
            episode_reward = 0.0
            done = False
            step_count = 0

            while not done:
                if hasattr(agent, "act"):
                    action = agent.act(obs)
                else:
                    action = env.action_space.sample()

                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                step_count += 1
                done = terminated or truncated

            transfer_rewards.append(episode_reward)

            # Check if agent reached baseline performance
            if episode_reward >= baseline_performance * 0.9 and steps_to_baseline == float("inf"):
                steps_to_baseline = (ep + 1) * step_count

            # Allow agent to adapt after episode
            if hasattr(agent, "adapt_to_transfer"):
                agent.adapt_to_transfer(env)

        transfer_performance = np.mean(transfer_rewards)
        final_gap = baseline_performance - transfer_performance

        # Normalize adaptation speed (lower is better)
        max_steps = self.adaptation_episodes * 100  # Assume max 100 steps per episode
        adaptation_speed = (
            steps_to_baseline / max_steps if steps_to_baseline != float("inf") else 1.0
        )

        return TransferResult(
            baseline_performance=baseline_performance,
            transfer_performance=transfer_performance,
            adaptation_speed=adaptation_speed,
            final_gap=final_gap,
        )

    def evaluate_few_shot_transfer(
        self,
        agent,
        n_demonstrations: int = 3,
        seed: int | None = None,
    ) -> TransferResult:
        """
        Evaluate few-shot transfer with demonstrations.

        Agent sees n_demonstrations in modified environment, then must perform.

        Args:
            agent: Agent with observe_demonstration() and act() methods
            n_demonstrations: Number of demo episodes to show
            seed: Random seed

        Returns:
            TransferResult with few-shot adaptation metrics
        """
        rng = np.random.default_rng(seed)

        # Measure baseline
        baseline_rewards = []
        for ep in range(self.n_episodes):
            env = self.base_env_factory()
            obs, _ = env.reset(seed=rng.integers(0, 10000))
            episode_reward = 0.0
            done = False

            while not done:
                action = agent.act(obs) if hasattr(agent, "act") else env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated

            baseline_rewards.append(episode_reward)

        baseline_performance = np.mean(baseline_rewards)

        # Provide demonstrations in modified environment
        for demo_idx in range(n_demonstrations):
            env = self.modified_env_factory()
            obs, _ = env.reset(seed=rng.integers(0, 10000))
            trajectory = []
            done = False

            while not done:
                # Oracle/optimal action for demonstration
                action = env.action_space.sample()  # Placeholder: should be optimal
                next_obs, reward, terminated, truncated, info = env.step(action)
                trajectory.append((obs, action, reward, next_obs, terminated or truncated, info))
                obs = next_obs
                done = terminated or truncated

            # Agent observes demonstration
            if hasattr(agent, "observe_demonstration"):
                agent.observe_demonstration(trajectory)

        # Test transfer performance
        transfer_rewards = []
        for ep in range(self.n_episodes):
            env = self.modified_env_factory()
            obs, _ = env.reset(seed=rng.integers(0, 10000))
            episode_reward = 0.0
            done = False

            while not done:
                action = agent.act(obs) if hasattr(agent, "act") else env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated

            transfer_rewards.append(episode_reward)

        transfer_performance = np.mean(transfer_rewards)
        final_gap = baseline_performance - transfer_performance

        return TransferResult(
            baseline_performance=baseline_performance,
            transfer_performance=transfer_performance,
            adaptation_speed=0.0,  # N/A for few-shot
            final_gap=final_gap,
        )
