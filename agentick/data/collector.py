"""Trajectory collection for dataset generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Trajectory:
    """A complete episode trajectory."""

    observations: list[Any] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    infos: list[dict] = field(default_factory=list)
    total_reward: float = 0.0
    length: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(
        self,
        observation: Any,
        action: int,
        reward: float,
        done: bool,
        info: dict,
    ):
        """Add a step to the trajectory."""
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.infos.append(info)
        self.total_reward += reward
        self.length += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert trajectory to dictionary."""
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "infos": self.infos,
            "total_reward": self.total_reward,
            "length": self.length,
            "metadata": self.metadata,
        }


class TrajectoryCollector:
    """
    Collect trajectories from environment rollouts.

    Supports multiple observation modalities and efficient buffering.

    Example:
        >>> collector = TrajectoryCollector(buffer_size=1000)
        >>>
        >>> # Collect trajectories
        >>> for episode in range(100):
        ...     obs, info = env.reset()
        ...     collector.start_episode(metadata={"episode": episode})
        ...
        ...     done = False
        ...     while not done:
        ...         action = policy(obs)
        ...         obs, reward, done, truncated, info = env.step(action)
        ...         collector.add_step(obs, action, reward, done or truncated, info)
        ...
        ...     collector.end_episode()
        >>>
        >>> # Export trajectories
        >>> trajectories = collector.get_trajectories()
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        collect_observations: bool = True,
    ):
        """
        Initialize trajectory collector.

        Args:
            buffer_size: Maximum number of trajectories to keep in memory
            collect_observations: Whether to collect observations (can be large)
        """
        self.buffer_size = buffer_size
        self.collect_observations = collect_observations

        self.trajectories: list[Trajectory] = []
        self.current_trajectory: Trajectory | None = None

        # Statistics
        self.total_episodes = 0
        self.total_steps = 0

    def start_episode(self, metadata: dict[str, Any] | None = None):
        """
        Start collecting a new episode.

        Args:
            metadata: Optional metadata to attach to episode
        """
        self.current_trajectory = Trajectory(metadata=metadata or {})

    def add_step(
        self,
        observation: Any,
        action: int,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ):
        """
        Add a step to the current trajectory.

        Args:
            observation: Environment observation
            action: Action taken
            reward: Reward received
            done: Whether episode ended
            info: Info dict
        """
        if self.current_trajectory is None:
            raise ValueError("Must call start_episode() before add_step()")

        # Optionally skip observations to save memory
        obs_to_store = observation if self.collect_observations else None

        self.current_trajectory.add_step(obs_to_store, action, reward, done, info)
        self.total_steps += 1

    def end_episode(self):
        """Finish current episode and add to buffer."""
        if self.current_trajectory is None:
            return

        # Add to trajectories buffer
        self.trajectories.append(self.current_trajectory)
        self.total_episodes += 1

        # Enforce buffer size limit
        if len(self.trajectories) > self.buffer_size:
            self.trajectories.pop(0)

        self.current_trajectory = None

    def get_trajectories(
        self,
        min_reward: float | None = None,
        max_length: int | None = None,
    ) -> list[Trajectory]:
        """
        Get collected trajectories with optional filtering.

        Args:
            min_reward: Minimum total reward to include
            max_length: Maximum episode length to include

        Returns:
            List of trajectories matching criteria
        """
        trajectories = self.trajectories

        if min_reward is not None:
            trajectories = [t for t in trajectories if t.total_reward >= min_reward]

        if max_length is not None:
            trajectories = [t for t in trajectories if t.length <= max_length]

        return trajectories

    def get_stats(self) -> dict[str, Any]:
        """Get collection statistics."""
        if len(self.trajectories) == 0:
            return {
                "total_episodes": 0,
                "total_steps": 0,
                "mean_reward": 0.0,
                "mean_length": 0.0,
            }

        rewards = [t.total_reward for t in self.trajectories]
        lengths = [t.length for t in self.trajectories]

        return {
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "buffered_episodes": len(self.trajectories),
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_length": np.mean(lengths),
            "std_length": np.std(lengths),
        }

    def save(self, path: str | Path):
        """
        Save trajectories to disk.

        Args:
            path: File path (will save as .npz)
        """
        import numpy as np

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to numpy-friendly format
        data = {
            "trajectories": [t.to_dict() for t in self.trajectories],
            "stats": self.get_stats(),
        }

        np.savez_compressed(path, **data)

    def load(self, path: str | Path):
        """
        Load trajectories from disk.

        Args:
            path: File path
        """
        data = np.load(path, allow_pickle=True)
        trajectories_data = data["trajectories"]

        self.trajectories = []
        for traj_dict in trajectories_data:
            traj = Trajectory(**traj_dict)
            self.trajectories.append(traj)

    def clear(self):
        """Clear all collected trajectories."""
        self.trajectories.clear()
        self.current_trajectory = None
        self.total_episodes = 0
        self.total_steps = 0
