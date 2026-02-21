"""Feature extraction for converting state_dict observations to flat vectors.

For RL training with state observations (non-image).
"""

import numpy as np
from gymnasium import spaces


def extract_state_features(state_dict: dict, grid_size: tuple[int, int] = (15, 15)) -> np.ndarray:
    """Extract flat feature vector from state_dict observation.

    Args:
        state_dict: Dict with keys 'grid', 'agent', 'entities', 'info'
        grid_size: Maximum grid size (for padding)

    Returns:
        Flat numpy array of features
    """
    features = []

    # Grid features (flatten terrain, objects, agents)
    if "grid" in state_dict:
        grid = state_dict["grid"]

        # Terrain (pad/crop to grid_size)
        terrain = np.array(grid.get("terrain", []))
        if terrain.size > 0:
            h, w = terrain.shape
            padded = np.zeros(grid_size)
            padded[: min(h, grid_size[0]), : min(w, grid_size[1])] = terrain[
                : grid_size[0], : grid_size[1]
            ]
            features.append(padded.flatten())
        else:
            features.append(np.zeros(grid_size[0] * grid_size[1]))

        # Objects (pad/crop)
        objects = np.array(grid.get("objects", []))
        if objects.size > 0:
            h, w = objects.shape
            padded = np.zeros(grid_size)
            padded[: min(h, grid_size[0]), : min(w, grid_size[1])] = objects[
                : grid_size[0], : grid_size[1]
            ]
            features.append(padded.flatten())
        else:
            features.append(np.zeros(grid_size[0] * grid_size[1]))

    # Agent features
    if "agent" in state_dict:
        agent = state_dict["agent"]

        # Position (normalized)
        pos = agent.get("position", (0, 0))
        features.append(np.array([pos[0] / grid_size[1], pos[1] / grid_size[0]]))

        # Orientation (one-hot: north, south, east, west)
        orientation = agent.get("orientation", "north")
        orientation_map = {"north": 0, "south": 1, "east": 2, "west": 3}
        ori_idx = orientation_map.get(orientation, 0)
        ori_vec = np.zeros(4)
        ori_vec[ori_idx] = 1
        features.append(ori_vec)

        # Energy, health
        features.append(
            np.array(
                [
                    agent.get("energy", 1.0),
                    agent.get("health", 1.0),
                ]
            )
        )

        # Inventory (count, simplified)
        inventory = agent.get("inventory", [])
        features.append(np.array([len(inventory)]))

    # Info features
    if "info" in state_dict:
        info = state_dict["info"]

        # Step count (normalized)
        step_count = info.get("step_count", 0)
        max_steps = info.get("max_steps", 100)
        features.append(np.array([step_count / max(max_steps, 1)]))

    # Concatenate all features
    return np.concatenate([f.flatten() for f in features]).astype(np.float32)


def get_state_feature_space(grid_size: tuple[int, int] = (15, 15)) -> spaces.Box:
    """Get observation space for state features.

    Args:
        grid_size: Maximum grid size

    Returns:
        Gymnasium Box space
    """
    # Calculate feature dimension
    # terrain: grid_size[0] * grid_size[1]
    # objects: grid_size[0] * grid_size[1]
    # position: 2
    # orientation: 4
    # energy, health: 2
    # inventory count: 1
    # step_count: 1

    feature_dim = (
        grid_size[0] * grid_size[1] * 2  # terrain + objects
        + 2  # position
        + 4  # orientation one-hot
        + 2  # energy, health
        + 1  # inventory count
        + 1  # step count
    )

    return spaces.Box(low=-1.0, high=1.0, shape=(feature_dim,), dtype=np.float32)
