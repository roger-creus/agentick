"""Composite reward that combines multiple reward signals."""

from __future__ import annotations

from collections.abc import Callable


class CompositeReward:
    """
    Combine multiple reward signals with weights.

    Useful for combining extrinsic + intrinsic rewards, or multiple shaped rewards.
    """

    def __init__(
        self,
        reward_functions: list[Callable],
        weights: list[float] | None = None,
        names: list[str] | None = None,
    ):
        """
        Initialize composite reward.

        Args:
            reward_functions: List of reward functions
            weights: Optional weights for each reward (default: equal weights)
            names: Optional names for each reward component
        """
        self.reward_functions = reward_functions
        self.weights = weights or [1.0] * len(reward_functions)
        self.names = names or [f"reward_{i}" for i in range(len(reward_functions))]

        if len(self.weights) != len(self.reward_functions):
            raise ValueError("Number of weights must match number of reward functions")

        if len(self.names) != len(self.reward_functions):
            raise ValueError("Number of names must match number of reward functions")

        self.reward_history: dict[str, list[float]] = {name: [] for name in self.names}

    def compute_reward(self, *args, **kwargs) -> float:
        """
        Compute composite reward.

        Args:
            *args, **kwargs: Arguments passed to each reward function

        Returns:
            Weighted sum of all rewards
        """
        total_reward = 0.0

        for fn, weight, name in zip(self.reward_functions, self.weights, self.names):
            component_reward = fn(*args, **kwargs)
            weighted_reward = weight * component_reward
            total_reward += weighted_reward

            # Track component rewards
            self.reward_history[name].append(component_reward)

        return total_reward

    def get_component_rewards(self, *args, **kwargs) -> dict[str, float]:
        """
        Get breakdown of component rewards without combining.

        Args:
            *args, **kwargs: Arguments passed to each reward function

        Returns:
            Dict mapping reward names to values
        """
        components = {}

        for fn, name in zip(self.reward_functions, self.names):
            components[name] = fn(*args, **kwargs)

        return components

    def get_weighted_components(self, *args, **kwargs) -> dict[str, float]:
        """
        Get weighted component rewards.

        Args:
            *args, **kwargs: Arguments passed to each reward function

        Returns:
            Dict mapping reward names to weighted values
        """
        components = {}

        for fn, weight, name in zip(self.reward_functions, self.weights, self.names):
            components[name] = weight * fn(*args, **kwargs)

        return components

    def reset(self) -> None:
        """Reset reward history."""
        self.reward_history = {name: [] for name in self.names}

    def get_statistics(self) -> dict[str, dict[str, float]]:
        """
        Get statistics for each reward component.

        Returns:
            Dict mapping reward names to statistics
        """
        import numpy as np

        stats = {}

        for name, history in self.reward_history.items():
            if history:
                stats[name] = {
                    "mean": float(np.mean(history)),
                    "std": float(np.std(history)),
                    "min": float(np.min(history)),
                    "max": float(np.max(history)),
                    "sum": float(np.sum(history)),
                }
            else:
                stats[name] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "sum": 0.0,
                }

        return stats

    def set_weights(self, weights: list[float]) -> None:
        """
        Update reward weights.

        Args:
            weights: New weights for each reward component
        """
        if len(weights) != len(self.reward_functions):
            raise ValueError("Number of weights must match number of reward functions")

        self.weights = weights

    def set_weight(self, name: str, weight: float) -> None:
        """
        Update weight for specific reward component.

        Args:
            name: Name of reward component
            weight: New weight value
        """
        if name not in self.names:
            raise ValueError(f"Unknown reward component: {name}")

        idx = self.names.index(name)
        self.weights[idx] = weight

    def add_reward_function(
        self,
        fn: Callable,
        weight: float = 1.0,
        name: str | None = None,
    ) -> None:
        """
        Add new reward function to composite.

        Args:
            fn: Reward function to add
            weight: Weight for this reward
            name: Name for this reward component
        """
        if name is None:
            name = f"reward_{len(self.reward_functions)}"

        self.reward_functions.append(fn)
        self.weights.append(weight)
        self.names.append(name)
        self.reward_history[name] = []

    def remove_reward_function(self, name: str) -> None:
        """
        Remove reward function from composite.

        Args:
            name: Name of reward component to remove
        """
        if name not in self.names:
            raise ValueError(f"Unknown reward component: {name}")

        idx = self.names.index(name)
        self.reward_functions.pop(idx)
        self.weights.pop(idx)
        self.names.pop(idx)
        del self.reward_history[name]
