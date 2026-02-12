"""Counterfactual reasoning evaluator for world model testing.

Tests agent's ability to reason about alternate outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CounterfactualResult:
    """Result from counterfactual reasoning evaluation."""

    accuracy: float  # Fraction of correct counterfactual predictions
    mean_error: float  # Average prediction error


class CounterfactualEvaluator:
    """Evaluate agent's counterfactual reasoning ability."""

    def __init__(self, env_factory, n_tests: int = 10, trajectory_length: int = 10):
        """
        Initialize counterfactual evaluator.

        Args:
            env_factory: Callable that creates environment instances
            n_tests: Number of counterfactual tests
            trajectory_length: Length of trajectories to generate
        """
        self.env_factory = env_factory
        self.n_tests = n_tests
        self.trajectory_length = trajectory_length

    def evaluate_counterfactual_prediction(
        self,
        agent,
        seed: int | None = None,
    ) -> CounterfactualResult:
        """
        Evaluate counterfactual prediction.

        Agent is shown a trajectory and asked "what would have happened if
        action X was taken instead of action Y at step T?"

        Args:
            agent: Agent with predict_counterfactual(trajectory, step, alt_action) method
            seed: Random seed

        Returns:
            CounterfactualResult with prediction accuracy
        """
        rng = np.random.default_rng(seed)
        errors = []
        correct = 0
        total = 0

        for test_idx in range(self.n_tests):
            env = self.env_factory()
            obs, _ = env.reset(seed=rng.integers(0, 10000))

            # Collect factual trajectory
            factual_trajectory = [(obs, None, None, None)]
            for _ in range(self.trajectory_length):
                action = env.action_space.sample()
                next_obs, reward, terminated, truncated, _ = env.step(action)
                factual_trajectory.append((obs, action, reward, next_obs))
                obs = next_obs

                if terminated or truncated:
                    break

            if len(factual_trajectory) < 3:
                continue

            # Choose intervention point
            intervention_step = rng.integers(1, len(factual_trajectory) - 1)
            factual_action = factual_trajectory[intervention_step][1]

            # Choose alternative action
            alt_action = env.action_space.sample()
            while alt_action == factual_action:
                alt_action = env.action_space.sample()

            # Generate counterfactual trajectory by replaying with alt action
            env2 = self.env_factory()
            env2.reset(seed=rng.integers(0, 10000))

            # Replay up to intervention point
            for step in range(intervention_step):
                if factual_trajectory[step][1] is not None:
                    env2.step(factual_trajectory[step][1])

            # Take alternative action
            alt_obs, alt_reward, alt_terminated, alt_truncated, _ = env2.step(alt_action)
            true_counterfactual = alt_obs

            # Ask agent to predict counterfactual outcome
            history = factual_trajectory[:intervention_step]

            if hasattr(agent, "predict_counterfactual"):
                predicted_counterfactual = agent.predict_counterfactual(
                    history, intervention_step, alt_action
                )
            else:
                # Fallback: predict same as factual
                predicted_counterfactual = factual_trajectory[intervention_step][3]

            # Compute error
            error = self._compute_prediction_error(predicted_counterfactual, true_counterfactual)
            errors.append(error)

            if error < 0.1:  # Threshold for "correct"
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0
        mean_error = np.mean(errors) if errors else 1.0

        return CounterfactualResult(
            accuracy=accuracy,
            mean_error=mean_error,
        )

    def evaluate_causal_attribution(
        self,
        agent,
        seed: int | None = None,
    ) -> dict[str, float]:
        """
        Evaluate causal attribution.

        Agent must identify which actions in a trajectory were causally
        responsible for the final outcome.

        Args:
            agent: Agent with attribute_causality(trajectory) -> list[float] method
            seed: Random seed

        Returns:
            Dict with attribution quality metrics
        """
        rng = np.random.default_rng(seed)
        attribution_errors = []

        for test_idx in range(self.n_tests):
            env = self.env_factory()
            obs, _ = env.reset(seed=rng.integers(0, 10000))

            # Collect trajectory
            trajectory = []
            for _ in range(self.trajectory_length):
                action = env.action_space.sample()
                next_obs, reward, terminated, truncated, info = env.step(action)
                trajectory.append((obs, action, reward, next_obs, terminated or truncated, info))
                obs = next_obs

                if terminated or truncated:
                    break

            if len(trajectory) < 2:
                continue

            # Ground truth: compute actual causal importance of each action
            # (by replaying with that action removed/changed)
            true_importance = []
            baseline_outcome = trajectory[-1][2]  # Final reward

            for step_idx in range(len(trajectory)):
                # Replay with this action changed to a different action
                env2 = self.env_factory()
                env2.reset(seed=rng.integers(0, 10000))

                counterfactual_reward = 0.0
                for s in range(len(trajectory)):
                    if s == step_idx:
                        # Use a different random action
                        action = env2.action_space.sample()
                        while action == trajectory[s][1]:
                            action = env2.action_space.sample()
                    else:
                        action = trajectory[s][1]

                    _, reward, terminated, truncated, _ = env2.step(action)
                    counterfactual_reward += reward

                    if terminated or truncated:
                        break

                # Importance is the difference in outcome when this action is changed
                # Higher difference = more causal importance
                importance = abs(baseline_outcome - counterfactual_reward)
                true_importance.append(importance)

            # Normalize
            true_importance = np.array(true_importance)
            if true_importance.sum() > 0:
                true_importance = true_importance / true_importance.sum()

            # Ask agent to attribute causality
            if hasattr(agent, "attribute_causality"):
                predicted_importance = agent.attribute_causality(trajectory)
            else:
                # Fallback: uniform attribution
                predicted_importance = np.ones(len(trajectory)) / len(trajectory)

            # Compute error (KL divergence or L2)
            error = np.sum((predicted_importance - true_importance) ** 2)
            attribution_errors.append(error)

        return {
            "mean_attribution_error": np.mean(attribution_errors) if attribution_errors else 1.0,
            "median_attribution_error": (
                np.median(attribution_errors) if attribution_errors else 1.0
            ),
        }

    def _compute_prediction_error(self, predicted: Any, actual: Any) -> float:
        """Compute prediction error between predicted and actual observations."""
        if isinstance(predicted, str) and isinstance(actual, str):
            return 0.0 if predicted == actual else 1.0

        elif isinstance(predicted, np.ndarray) and isinstance(actual, np.ndarray):
            if predicted.shape != actual.shape:
                return 1.0
            mse = np.mean((predicted.astype(float) - actual.astype(float)) ** 2)
            max_val = 255.0 if predicted.dtype == np.uint8 else 1.0
            return min(mse / (max_val**2), 1.0)

        elif isinstance(predicted, dict) and isinstance(actual, dict):
            common_keys = set(predicted.keys()) & set(actual.keys())
            if not common_keys:
                return 1.0
            errors = [0.0 if predicted[k] == actual[k] else 1.0 for k in common_keys]
            return np.mean(errors)

        else:
            return 0.0 if predicted == actual else 1.0
