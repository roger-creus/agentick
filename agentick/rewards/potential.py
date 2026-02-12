"""Potential-based reward shaping.

Implements provably policy-preserving reward shaping using potential functions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


class PotentialBasedReward:
    """
    Potential-based reward shaping (Ng et al., 1999).

    Adds shaped reward: F(s, s') = γ * Φ(s') - Φ(s)
    where Φ is a potential function.

    This preserves optimal policy while providing denser feedback.
    """

    def __init__(
        self,
        potential_fn: Callable[[dict[str, Any]], float],
        gamma: float = 0.99,
    ):
        """
        Initialize potential-based reward shaper.

        Args:
            potential_fn: Function that maps state to potential value
            gamma: Discount factor
        """
        self.potential_fn = potential_fn
        self.gamma = gamma
        self.last_potential = None

    def reset(self, initial_state: dict[str, Any]) -> None:
        """
        Reset for new episode.

        Args:
            initial_state: Initial state of environment
        """
        self.last_potential = self.potential_fn(initial_state)

    def shape_reward(
        self,
        reward: float,
        next_state: dict[str, Any],
        terminated: bool,
    ) -> float:
        """
        Add shaped reward to base reward.

        Args:
            reward: Base reward from environment
            next_state: Next state after action
            terminated: Whether episode terminated

        Returns:
            Shaped reward = base + F(s, s')
        """
        if self.last_potential is None:
            raise RuntimeError("Must call reset() before shaping rewards")

        next_potential = self.potential_fn(next_state)

        # F(s, s') = γ * Φ(s') - Φ(s)
        # If terminated, future potential is 0
        shaped_reward = (
            reward + self.gamma * (next_potential if not terminated else 0.0) - self.last_potential
        )

        self.last_potential = next_potential if not terminated else None

        return shaped_reward


def manhattan_distance_potential(goal_position: tuple[int, int]) -> Callable:
    """
    Create potential function based on negative Manhattan distance to goal.

    Args:
        goal_position: (x, y) coordinates of goal

    Returns:
        Potential function
    """

    def potential(state: dict[str, Any]) -> float:
        if "agent" in state and hasattr(state["agent"], "position"):
            agent_pos = state["agent"].position
            distance = abs(agent_pos[0] - goal_position[0]) + abs(agent_pos[1] - goal_position[1])
            return -distance
        return 0.0

    return potential


def euclidean_distance_potential(goal_position: tuple[int, int]) -> Callable:
    """
    Create potential function based on negative Euclidean distance to goal.

    Args:
        goal_position: (x, y) coordinates of goal

    Returns:
        Potential function
    """

    def potential(state: dict[str, Any]) -> float:
        if "agent" in state and hasattr(state["agent"], "position"):
            agent_pos = state["agent"].position
            distance = np.sqrt(
                (agent_pos[0] - goal_position[0]) ** 2 + (agent_pos[1] - goal_position[1]) ** 2
            )
            return -distance
        return 0.0

    return potential


def subgoal_potential(subgoals: list[tuple[int, int]]) -> Callable:
    """
    Create potential function based on number of remaining subgoals.

    Args:
        subgoals: List of (x, y) subgoal positions to visit in order

    Returns:
        Potential function
    """

    def potential(state: dict[str, Any]) -> float:
        if "agent" not in state or not hasattr(state["agent"], "position"):
            return 0.0

        agent_pos = state["agent"].position

        # Find how many subgoals are completed
        completed = 0
        for subgoal in subgoals:
            if agent_pos == subgoal:
                completed += 1
            else:
                break  # Must be completed in order

        # Potential is number of completed subgoals
        return float(completed)

    return potential


def inventory_potential(target_items: list[str]) -> Callable:
    """
    Create potential function based on number of target items in inventory.

    Args:
        target_items: List of item types to collect

    Returns:
        Potential function
    """

    def potential(state: dict[str, Any]) -> float:
        if "agent" not in state or not hasattr(state["agent"], "inventory"):
            return 0.0

        inventory = state["agent"].inventory
        collected = sum(1 for item in inventory if item.entity_type in target_items)

        return float(collected)

    return potential


def composite_potential(*potential_fns: Callable, weights: list[float] | None = None) -> Callable:
    """
    Create composite potential function from multiple potentials.

    Args:
        *potential_fns: Potential functions to combine
        weights: Optional weights for each potential (default: equal weights)

    Returns:
        Combined potential function
    """
    if weights is None:
        weights = [1.0] * len(potential_fns)
    elif len(weights) != len(potential_fns):
        raise ValueError("Number of weights must match number of potential functions")

    def potential(state: dict[str, Any]) -> float:
        return sum(w * fn(state) for w, fn in zip(weights, potential_fns))

    return potential
