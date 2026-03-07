"""Action space and action masking for gridworld environments."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from agentick.core.types import ActionType, Direction, Position


class ActionSpace:
    """
    Discrete action space with action masking support.

    Provides mapping between action integers and semantic actions,
    and supports computing valid actions based on environment state.
    """

    # Default action set (6 basic actions)
    BASIC_ACTIONS = [
        ActionType.NOOP,
        ActionType.MOVE_UP,
        ActionType.MOVE_DOWN,
        ActionType.MOVE_LEFT,
        ActionType.MOVE_RIGHT,
        ActionType.INTERACT,
    ]

    # Extended action set with orientation (9 actions)
    EXTENDED_ACTIONS = BASIC_ACTIONS + [
        ActionType.ROTATE_LEFT,
        ActionType.ROTATE_RIGHT,
        ActionType.MOVE_FORWARD,
    ]

    # Action names for language interfaces
    ACTION_NAMES = {
        ActionType.NOOP: "noop",
        ActionType.MOVE_UP: "move_up",
        ActionType.MOVE_DOWN: "move_down",
        ActionType.MOVE_LEFT: "move_left",
        ActionType.MOVE_RIGHT: "move_right",
        ActionType.INTERACT: "interact",
        ActionType.ROTATE_LEFT: "rotate_left",
        ActionType.ROTATE_RIGHT: "rotate_right",
        ActionType.MOVE_FORWARD: "move_forward",
    }

    # Reverse mapping
    NAME_TO_ACTION = {name: action for action, name in ACTION_NAMES.items()}

    def __init__(self, actions: list[ActionType] | None = None, extended: bool = False):
        """
        Initialize action space.

        Args:
            actions: List of allowed actions. If None, uses BASIC_ACTIONS or EXTENDED_ACTIONS.
            extended: If True and actions is None, use EXTENDED_ACTIONS.
        """
        if actions is None:
            actions = self.EXTENDED_ACTIONS if extended else self.BASIC_ACTIONS

        self.actions = actions
        self.n_actions = len(actions)

        # Create gymnasium space
        self.gym_space = gym.spaces.Discrete(self.n_actions)

        # Mapping from action index to ActionType
        self.idx_to_action = {i: action for i, action in enumerate(actions)}
        self.action_to_idx = {action: i for i, action in enumerate(actions)}

    def get_action_type(self, action_idx: int) -> ActionType:
        """Get ActionType for a given action index."""
        return self.idx_to_action[action_idx]

    def get_action_idx(self, action_type: ActionType) -> int:
        """Get action index for a given ActionType."""
        return self.action_to_idx[action_type]

    def get_action_name(self, action_idx: int) -> str:
        """Get human-readable name for an action."""
        action_type = self.get_action_type(action_idx)
        return self.ACTION_NAMES[action_type]

    def parse_action_name(self, name: str) -> int:
        """Parse action name to action index."""
        name = name.lower().strip().replace(" ", "_")
        if name in self.NAME_TO_ACTION:
            action_type = self.NAME_TO_ACTION[name]
            if action_type in self.action_to_idx:
                return self.action_to_idx[action_type]
        raise ValueError(f"Unknown action name: {name}")

    def get_all_action_names(self) -> list[str]:
        """Get list of all action names in order."""
        return [self.get_action_name(i) for i in range(self.n_actions)]

    def contains(self, action_type: ActionType) -> bool:
        """Check if action type is in this action space."""
        return action_type in self.action_to_idx

    def sample(self, rng: np.random.Generator | None = None) -> int:
        """Sample a random action."""
        if rng is None:
            return self.gym_space.sample()
        return rng.integers(0, self.n_actions)


def compute_action_mask(
    action_space: ActionSpace,
    agent_pos: Position,
    grid_walkable: np.ndarray,
    can_interact: bool = False,
) -> np.ndarray:
    """
    Compute binary action mask indicating valid actions.

    Args:
        action_space: The action space
        agent_pos: Agent position (x, y)
        grid_walkable: Boolean array indicating walkable cells
        can_interact: Whether INTERACT action is valid

    Returns:
        Boolean array of shape (n_actions,) where True = valid action
    """
    mask = np.zeros(action_space.n_actions, dtype=bool)
    x, y = agent_pos
    h, w = grid_walkable.shape

    for i in range(action_space.n_actions):
        action_type = action_space.get_action_type(i)

        if action_type == ActionType.NOOP:
            mask[i] = True

        elif action_type == ActionType.MOVE_UP:
            if y > 0 and grid_walkable[y - 1, x]:
                mask[i] = True

        elif action_type == ActionType.MOVE_DOWN:
            if y < h - 1 and grid_walkable[y + 1, x]:
                mask[i] = True

        elif action_type == ActionType.MOVE_LEFT:
            if x > 0 and grid_walkable[y, x - 1]:
                mask[i] = True

        elif action_type == ActionType.MOVE_RIGHT:
            if x < w - 1 and grid_walkable[y, x + 1]:
                mask[i] = True

        elif action_type == ActionType.INTERACT:
            mask[i] = True  # always valid; no-op if nothing interactable

        elif action_type in (ActionType.ROTATE_LEFT, ActionType.ROTATE_RIGHT):
            mask[i] = True

        elif action_type == ActionType.MOVE_FORWARD:
            # This would need orientation info, default to True
            mask[i] = True

    return mask


def get_move_delta(action_type: ActionType) -> Position | None:
    """
    Get (dx, dy) movement delta for a movement action.

    Args:
        action_type: The action type

    Returns:
        (dx, dy) tuple, or None if not a movement action
    """
    deltas = {
        ActionType.MOVE_UP: (0, -1),
        ActionType.MOVE_DOWN: (0, 1),
        ActionType.MOVE_LEFT: (-1, 0),
        ActionType.MOVE_RIGHT: (1, 0),
    }
    return deltas.get(action_type)


def is_movement_action(action_type: ActionType) -> bool:
    """Check if action is a movement action."""
    return action_type in (
        ActionType.MOVE_UP,
        ActionType.MOVE_DOWN,
        ActionType.MOVE_LEFT,
        ActionType.MOVE_RIGHT,
        ActionType.MOVE_FORWARD,
    )


def action_to_direction(action_type: ActionType) -> Direction | None:
    """
    Convert movement action to Direction.

    Args:
        action_type: The action type

    Returns:
        Direction, or None if not a directional movement action
    """
    mapping = {
        ActionType.MOVE_UP: Direction.NORTH,
        ActionType.MOVE_DOWN: Direction.SOUTH,
        ActionType.MOVE_LEFT: Direction.WEST,
        ActionType.MOVE_RIGHT: Direction.EAST,
    }
    return mapping.get(action_type)
