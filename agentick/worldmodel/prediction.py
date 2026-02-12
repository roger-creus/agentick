"""State prediction evaluator for world model testing.

Tests agent's ability to predict future states given action sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PredictionResult:
    """Result from state prediction evaluation."""

    accuracy: float
    mean_error: float
    predictions: list[Any]
    ground_truth: list[Any]


class StatePredictionEvaluator:
    """Evaluate agent's ability to predict future states."""

    def __init__(self, env_factory, n_tests: int = 10, prediction_horizon: int = 5):
        """
        Initialize state prediction evaluator.

        Args:
            env_factory: Callable that creates environment instances
            n_tests: Number of prediction tests to run
            prediction_horizon: Number of steps ahead to predict
        """
        self.env_factory = env_factory
        self.n_tests = n_tests
        self.prediction_horizon = prediction_horizon

    def evaluate_masked_prediction(
        self,
        agent,
        n_choices: int = 4,
        seed: int | None = None,
    ) -> PredictionResult:
        """
        Evaluate masked state prediction (multiple choice).

        Agent is shown a sequence of states with some masked, and must predict
        the masked states from multiple choices.

        Args:
            agent: Agent with predict_state(observations, actions) method
            n_choices: Number of choices per masked state
            seed: Random seed

        Returns:
            PredictionResult with accuracy and predictions
        """
        rng = np.random.default_rng(seed)
        correct = 0
        total = 0
        predictions = []
        ground_truth = []

        for test_idx in range(self.n_tests):
            env = self.env_factory()
            obs, _ = env.reset(seed=rng.integers(0, 10000))

            # Collect trajectory
            trajectory = [(obs, None)]
            for _ in range(self.prediction_horizon):
                action = env.action_space.sample()
                obs, _, terminated, truncated, _ = env.step(action)
                trajectory.append((obs, action))
                if terminated or truncated:
                    break

            if len(trajectory) < 3:
                continue

            # Mask random state
            mask_idx = rng.integers(1, len(trajectory) - 1)
            masked_state = trajectory[mask_idx][0]

            # Generate choices (correct + distractors)
            choices = [masked_state]
            for _ in range(n_choices - 1):
                # Sample distractor from other states
                distractor_idx = rng.choice([i for i in range(len(trajectory)) if i != mask_idx])
                choices.append(trajectory[distractor_idx][0])
            rng.shuffle(choices)

            # Get agent prediction
            context = trajectory[:mask_idx]
            if hasattr(agent, "predict_state_multiple_choice"):
                prediction_idx = agent.predict_state_multiple_choice(context, choices)
            else:
                # Fallback: random guess
                prediction_idx = rng.integers(0, n_choices)

            correct_idx = choices.index(masked_state)
            predictions.append(prediction_idx)
            ground_truth.append(correct_idx)

            if prediction_idx == correct_idx:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0
        return PredictionResult(
            accuracy=accuracy,
            mean_error=1.0 - accuracy,
            predictions=predictions,
            ground_truth=ground_truth,
        )

    def evaluate_free_form_prediction(
        self,
        agent,
        seed: int | None = None,
    ) -> PredictionResult:
        """
        Evaluate free-form state prediction.

        Agent must predict the exact next state given history.

        Args:
            agent: Agent with predict_next_state(observations, actions) method
            seed: Random seed

        Returns:
            PredictionResult with mean error
        """
        rng = np.random.default_rng(seed)
        errors = []
        predictions = []
        ground_truth = []

        for test_idx in range(self.n_tests):
            env = self.env_factory()
            obs, _ = env.reset(seed=rng.integers(0, 10000))

            # Collect trajectory
            trajectory = [obs]
            actions = []
            for _ in range(self.prediction_horizon):
                action = env.action_space.sample()
                actions.append(action)
                obs, _, terminated, truncated, _ = env.step(action)
                trajectory.append(obs)
                if terminated or truncated:
                    break

            if len(trajectory) < 2:
                continue

            # Predict next state from history
            history_obs = trajectory[:-1]
            true_next = trajectory[-1]

            if hasattr(agent, "predict_next_state"):
                predicted_next = agent.predict_next_state(history_obs, actions)
            else:
                # Fallback: return last observed state
                predicted_next = history_obs[-1]

            # Compute error (depends on observation type)
            error = self._compute_prediction_error(predicted_next, true_next)
            errors.append(error)
            predictions.append(predicted_next)
            ground_truth.append(true_next)

        mean_error = np.mean(errors) if errors else 1.0
        return PredictionResult(
            accuracy=1.0 - mean_error,
            mean_error=mean_error,
            predictions=predictions,
            ground_truth=ground_truth,
        )

    def _compute_prediction_error(self, predicted: Any, actual: Any) -> float:
        """Compute prediction error between predicted and actual observations."""
        if isinstance(predicted, str) and isinstance(actual, str):
            # Text: Levenshtein-like ratio
            if predicted == actual:
                return 0.0
            return 1.0  # Simplified: 0 if match, 1 if mismatch

        elif isinstance(predicted, np.ndarray) and isinstance(actual, np.ndarray):
            # Pixel/state: normalized MSE
            if predicted.shape != actual.shape:
                return 1.0
            mse = np.mean((predicted.astype(float) - actual.astype(float)) ** 2)
            # Normalize by max possible error
            max_val = 255.0 if predicted.dtype == np.uint8 else 1.0
            return min(mse / (max_val**2), 1.0)

        elif isinstance(predicted, dict) and isinstance(actual, dict):
            # Structured: compare keys
            common_keys = set(predicted.keys()) & set(actual.keys())
            if not common_keys:
                return 1.0

            errors = []
            for key in common_keys:
                if predicted[key] == actual[key]:
                    errors.append(0.0)
                else:
                    errors.append(1.0)
            return np.mean(errors)

        else:
            # Fallback: exact match
            return 0.0 if predicted == actual else 1.0
