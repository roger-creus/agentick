"""Change detection evaluator for world model testing.

Tests agent's ability to detect when environment dynamics change mid-episode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ChangeDetectionResult:
    """Result from change detection evaluation."""

    detection_accuracy: float  # Fraction of changes correctly detected
    mean_detection_latency: float  # Steps between change and detection
    false_positive_rate: float  # Fraction of no-change episodes with false alarms


class ChangeDetectionEvaluator:
    """Evaluate agent's ability to detect mid-episode dynamics changes."""

    def __init__(
        self,
        env_factory,
        change_env_factory,
        n_tests: int = 20,
        episode_length: int = 100,
    ):
        """
        Initialize change detection evaluator.

        Args:
            env_factory: Factory for normal environment
            change_env_factory: Factory that creates env with changeable dynamics
            n_tests: Number of test episodes
            episode_length: Max steps per episode
        """
        self.env_factory = env_factory
        self.change_env_factory = change_env_factory
        self.n_tests = n_tests
        self.episode_length = episode_length

    def evaluate_change_detection(
        self,
        agent,
        seed: int | None = None,
    ) -> ChangeDetectionResult:
        """
        Evaluate change detection ability.

        Args:
            agent: Agent with detect_change(observations) method
            seed: Random seed

        Returns:
            ChangeDetectionResult with detection metrics
        """
        rng = np.random.default_rng(seed)

        detections = []
        detection_latencies = []
        false_positives = 0

        # Test episodes WITH changes
        n_change_episodes = self.n_tests // 2
        for test_idx in range(n_change_episodes):
            env = self.change_env_factory()
            obs, _ = env.reset(seed=rng.integers(0, 10000))

            # Randomly choose when to change dynamics
            change_step = rng.integers(10, self.episode_length - 10)
            observations = [obs]
            detected_change = False
            detection_step = None

            for step in range(self.episode_length):
                # Apply change at designated step
                if step == change_step and hasattr(env.unwrapped, "change_dynamics"):
                    env.unwrapped.change_dynamics(rng)

                action = env.action_space.sample()
                obs, _, terminated, truncated, _ = env.step(action)
                observations.append(obs)

                # Check if agent detects change
                if step >= change_step and not detected_change:
                    if hasattr(agent, "detect_change"):
                        change_detected = agent.detect_change(observations)
                    else:
                        # Fallback: never detect
                        change_detected = False

                    if change_detected:
                        detected_change = True
                        detection_step = step

                if terminated or truncated:
                    break

            if detected_change:
                detections.append(True)
                detection_latencies.append(detection_step - change_step)
            else:
                detections.append(False)

        # Test episodes WITHOUT changes (for false positive rate)
        n_no_change_episodes = self.n_tests - n_change_episodes
        for test_idx in range(n_no_change_episodes):
            env = self.env_factory()
            obs, _ = env.reset(seed=rng.integers(0, 10000))
            observations = [obs]
            false_alarm = False

            for step in range(self.episode_length):
                action = env.action_space.sample()
                obs, _, terminated, truncated, _ = env.step(action)
                observations.append(obs)

                # Check if agent falsely detects change
                if hasattr(agent, "detect_change"):
                    if agent.detect_change(observations):
                        false_alarm = True
                        break

                if terminated or truncated:
                    break

            if false_alarm:
                false_positives += 1

        # Compute metrics
        detection_accuracy = sum(detections) / len(detections) if detections else 0.0
        mean_latency = np.mean(detection_latencies) if detection_latencies else float("inf")
        false_positive_rate = (
            false_positives / n_no_change_episodes if n_no_change_episodes > 0 else 0.0
        )

        return ChangeDetectionResult(
            detection_accuracy=detection_accuracy,
            mean_detection_latency=mean_latency,
            false_positive_rate=false_positive_rate,
        )

    def evaluate_change_localization(
        self,
        agent,
        seed: int | None = None,
    ) -> dict[str, float]:
        """
        Evaluate ability to localize WHEN change occurred.

        Agent must pinpoint the exact step when dynamics changed.

        Args:
            agent: Agent with localize_change(observations) -> int method
            seed: Random seed

        Returns:
            Dict with localization accuracy metrics
        """
        rng = np.random.default_rng(seed)
        errors = []

        for test_idx in range(self.n_tests):
            env = self.change_env_factory()
            obs, _ = env.reset(seed=rng.integers(0, 10000))

            change_step = rng.integers(10, self.episode_length - 10)
            observations = [obs]

            for step in range(self.episode_length):
                if step == change_step and hasattr(env.unwrapped, "change_dynamics"):
                    env.unwrapped.change_dynamics(rng)

                action = env.action_space.sample()
                obs, _, terminated, truncated, _ = env.step(action)
                observations.append(obs)

                if terminated or truncated:
                    break

            # Ask agent to localize change
            if hasattr(agent, "localize_change"):
                predicted_step = agent.localize_change(observations)
            else:
                # Fallback: random guess
                predicted_step = rng.integers(0, len(observations))

            error = abs(predicted_step - change_step)
            errors.append(error)

        return {
            "mean_localization_error": np.mean(errors),
            "median_localization_error": np.median(errors),
            "max_localization_error": np.max(errors),
        }
