"""Intrinsic motivation rewards for exploration.

Provides exploration bonuses and curiosity-driven rewards.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


class ExplorationBonus:
    """
    Exploration bonus based on state visit counts.

    Rewards visiting novel states more than frequently visited states.
    """

    def __init__(
        self,
        bonus_scale: float = 1.0,
        decay_type: str = "inverse",  # "inverse" or "sqrt"
    ):
        """
        Initialize exploration bonus.

        Args:
            bonus_scale: Scaling factor for bonus reward
            decay_type: How bonus decays with visit count ("inverse" or "sqrt")
        """
        self.bonus_scale = bonus_scale
        self.decay_type = decay_type
        self.visit_counts: dict[Any, int] = defaultdict(int)

    def reset(self) -> None:
        """Reset visit counts for new episode or agent."""
        self.visit_counts.clear()

    def compute_bonus(self, state: dict[str, Any]) -> float:
        """
        Compute exploration bonus for state.

        Args:
            state: Current state

        Returns:
            Exploration bonus reward
        """
        # Convert state to hashable key
        state_key = self._state_to_key(state)

        # Increment visit count
        self.visit_counts[state_key] += 1
        count = self.visit_counts[state_key]

        # Compute bonus based on visit count
        if self.decay_type == "inverse":
            bonus = self.bonus_scale / count
        elif self.decay_type == "sqrt":
            bonus = self.bonus_scale / np.sqrt(count)
        else:
            bonus = self.bonus_scale if count == 1 else 0.0

        return bonus

    def _state_to_key(self, state: dict[str, Any]) -> tuple:
        """Convert state dict to hashable key for counting state visits."""
        components = []

        # Agent position and orientation
        if "agent" in state and hasattr(state["agent"], "position"):
            components.append(("agent_pos", state["agent"].position))
            if hasattr(state["agent"], "orientation"):
                components.append(("agent_ori", state["agent"].orientation.value))

        # Grid state (terrain and objects)
        if "grid" in state:
            grid = state["grid"]
            # Hash terrain and objects arrays
            terrain_bytes = grid.terrain.tobytes()
            objects_bytes = grid.objects.tobytes()
            # Use hash for efficiency (full grid would be too large)
            components.append(("terrain_hash", hash(terrain_bytes)))
            components.append(("objects_hash", hash(objects_bytes)))

            # Include object colors if present
            if hasattr(grid, "object_colors"):
                colors_bytes = grid.object_colors.tobytes()
                components.append(("colors_hash", hash(colors_bytes)))

        # Inventory state
        if "agent" in state and hasattr(state["agent"], "inventory"):
            inventory = state["agent"].inventory
            # Hash inventory by entity types
            inv_types = tuple(sorted(item.entity_type for item in inventory))
            components.append(("inventory", inv_types))

        return tuple(components)


class CuriosityReward:
    """
    Curiosity-driven reward based on prediction error.

    Rewards states where agent's world model prediction error is high.
    """

    def __init__(
        self,
        reward_scale: float = 1.0,
        prediction_window: int = 10,
    ):
        """
        Initialize curiosity reward.

        Args:
            reward_scale: Scaling factor for curiosity reward
            prediction_window: Number of recent predictions to track
        """
        self.reward_scale = reward_scale
        self.prediction_window = prediction_window
        self.prediction_errors: list[float] = []

    def reset(self) -> None:
        """Reset for new episode."""
        self.prediction_errors.clear()

    def compute_reward(
        self,
        predicted_state: Any,
        actual_state: Any,
    ) -> float:
        """
        Compute curiosity reward based on prediction error.

        Args:
            predicted_state: Agent's predicted next state
            actual_state: Actual next state

        Returns:
            Curiosity reward (higher when prediction is more wrong)
        """
        # Compute prediction error
        error = self._compute_error(predicted_state, actual_state)

        # Track errors
        self.prediction_errors.append(error)
        if len(self.prediction_errors) > self.prediction_window:
            self.prediction_errors.pop(0)

        # Normalize by average recent error
        avg_error = np.mean(self.prediction_errors) if self.prediction_errors else 1.0
        normalized_error = error / max(avg_error, 1e-6)

        return self.reward_scale * normalized_error

    def _compute_error(self, predicted: Any, actual: Any) -> float:
        """Compute prediction error."""
        if isinstance(predicted, np.ndarray) and isinstance(actual, np.ndarray):
            if predicted.shape != actual.shape:
                return 1.0
            return float(np.mean((predicted - actual) ** 2))

        elif isinstance(predicted, dict) and isinstance(actual, dict):
            # Compare dict keys
            common_keys = set(predicted.keys()) & set(actual.keys())
            if not common_keys:
                return 1.0
            mismatches = sum(1 for k in common_keys if predicted[k] != actual[k])
            return mismatches / len(common_keys)

        else:
            return 0.0 if predicted == actual else 1.0


class InformationGainReward:
    """
    Information gain reward for active learning.

    Rewards actions that maximize information gain about the environment.
    """

    def __init__(
        self,
        reward_scale: float = 1.0,
    ):
        """
        Initialize information gain reward.

        Args:
            reward_scale: Scaling factor for reward
        """
        self.reward_scale = reward_scale
        self.state_observations: dict[Any, int] = defaultdict(int)
        self.transition_counts: dict[tuple[Any, Any], int] = defaultdict(int)

    def reset(self) -> None:
        """Reset for new agent."""
        self.state_observations.clear()
        self.transition_counts.clear()

    def compute_reward(
        self,
        state: dict[str, Any],
        action: int,
        next_state: dict[str, Any],
    ) -> float:
        """
        Compute information gain reward.

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state

        Returns:
            Information gain reward
        """
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        transition_key = (state_key, action, next_state_key)

        # Track observations
        self.state_observations[state_key] += 1
        self.transition_counts[transition_key] += 1

        # Information gain: -log(p(transition))
        # Higher reward for rare transitions
        transition_count = self.transition_counts[transition_key]
        state_count = self.state_observations[state_key]

        prob = transition_count / max(state_count, 1)
        info_gain = -np.log(max(prob, 1e-10))

        return self.reward_scale * info_gain

    def _state_to_key(self, state: dict[str, Any]) -> tuple:
        """Convert state to hashable key."""
        if "agent" in state and hasattr(state["agent"], "position"):
            return state["agent"].position
        return tuple(sorted(state.items()))


class NoveltyReward:
    """
    Novelty reward based on state dissimilarity.

    Rewards states that are dissimilar to previously visited states.
    """

    def __init__(
        self,
        reward_scale: float = 1.0,
        memory_size: int = 100,
        threshold: float = 0.1,
    ):
        """
        Initialize novelty reward.

        Args:
            reward_scale: Scaling factor
            memory_size: Number of recent states to remember
            threshold: Similarity threshold for novelty
        """
        self.reward_scale = reward_scale
        self.memory_size = memory_size
        self.threshold = threshold
        self.state_memory: list[dict[str, Any]] = []

    def reset(self) -> None:
        """Reset state memory."""
        self.state_memory.clear()

    def compute_reward(self, state: dict[str, Any]) -> float:
        """
        Compute novelty reward.

        Args:
            state: Current state

        Returns:
            Novelty reward (higher for novel states)
        """
        if not self.state_memory:
            novelty = 1.0
        else:
            # Compute minimum distance to memorized states
            distances = [self._state_distance(state, mem_state) for mem_state in self.state_memory]
            min_distance = min(distances)
            novelty = min(min_distance, 1.0)

        # Add to memory
        self.state_memory.append(state.copy() if isinstance(state, dict) else state)
        if len(self.state_memory) > self.memory_size:
            self.state_memory.pop(0)

        # Reward if above threshold
        return self.reward_scale * novelty if novelty > self.threshold else 0.0

    def _state_distance(self, state1: dict[str, Any], state2: dict[str, Any]) -> float:
        """Compute distance between states."""
        if "agent" in state1 and "agent" in state2:
            pos1 = state1["agent"].position if hasattr(state1["agent"], "position") else (0, 0)
            pos2 = state2["agent"].position if hasattr(state2["agent"], "position") else (0, 0)
            return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
        return 1.0 if state1 != state2 else 0.0
