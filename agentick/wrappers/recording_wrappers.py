"""Episode recording wrappers."""

import json
import time
from typing import Any

import gymnasium as gym
import numpy as np


class EpisodeRecorder(gym.Wrapper):
    """
    Record full episodes with ALL modalities simultaneously.

    Records:
    - Observations in text, language, pixel, and state_dict formats
    - Action names and indices
    - Reward breakdowns (if available in info)
    - Timestamps (relative to episode start)
    - Full info dicts
    """

    def __init__(
        self,
        env,
        save_path=None,
        record_all_modalities=True,
        include_pixels=True,
        include_language=True,
    ):
        super().__init__(env)
        self.save_path = save_path
        self.record_all_modalities = record_all_modalities
        self.include_pixels = include_pixels
        self.include_language = include_language
        self.episode_data = []
        self.current_episode = []
        self.episode_start_time = None

    def reset(self, **kwargs):
        """Reset environment and start recording a new episode.

        Args:
            **kwargs: Arguments passed to the wrapped environment's reset().

        Returns:
            tuple: (observation, info) from the wrapped environment.
        """
        # Save previous episode
        if self.current_episode:
            self.episode_data.append(
                {
                    "steps": self.current_episode,
                    "duration": time.time() - self.episode_start_time
                    if self.episode_start_time
                    else 0,
                }
            )
            if self.save_path:
                self._save()

        self.current_episode = []
        self.episode_start_time = time.time()

        obs, info = self.env.reset(**kwargs)

        # Record reset with all modalities
        step_data = {
            "type": "reset",
            "timestamp": 0.0,
            "observation": self._record_observation(obs, info),
            "info": self._serialize_info(info),
        }

        self.current_episode.append(step_data)
        return obs, info

    def step(self, action):
        """Execute action and record the transition.

        Args:
            action: Action to execute.

        Returns:
            tuple: (observation, reward, terminated, truncated, info).
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Get action name
        action_name = info.get("action_name", f"action_{action}")

        # Get reward breakdown
        reward_breakdown = info.get("reward_breakdown", {"total": reward})

        step_data = {
            "type": "step",
            "timestamp": time.time() - self.episode_start_time,
            "action": {
                "index": int(action),
                "name": action_name,
            },
            "reward": {
                "total": float(reward),
                "breakdown": reward_breakdown,
            },
            "observation": self._record_observation(obs, info),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": self._serialize_info(info),
        }

        self.current_episode.append(step_data)
        return obs, reward, terminated, truncated, info

    def _record_observation(self, obs, info):
        """Record observation in multiple modalities."""
        obs_data = {}

        # The primary observation (whatever render mode is active)
        if isinstance(obs, np.ndarray):
            if obs.ndim == 3:  # Pixel observation
                obs_data["pixels"] = {
                    "shape": obs.shape,
                    "dtype": str(obs.dtype),
                    "data": obs.tolist() if self.include_pixels else None,
                }
            else:
                obs_data["array"] = {
                    "shape": obs.shape,
                    "dtype": str(obs.dtype),
                    "data": obs.tolist(),
                }
        elif isinstance(obs, str):
            obs_data["text"] = obs
        elif isinstance(obs, dict):
            obs_data["structured"] = obs

        # If recording all modalities, get other render modes
        if self.record_all_modalities and hasattr(self.env, "render"):
            try:
                # Get state dict
                if hasattr(self.env, "get_state_dict"):
                    obs_data["state_dict"] = self.env.get_state_dict()

                # Get language description (if not already primary)
                if "text" not in obs_data and self.include_language:
                    if hasattr(self.env.unwrapped, "_get_language_obs"):
                        lang_obs = self.env.unwrapped._get_language_obs()
                        obs_data["language"] = lang_obs
            except Exception:
                # If modality not available, skip it
                pass

        return obs_data

    def _serialize_info(self, info: dict[str, Any]) -> dict[str, Any]:
        """Serialize info dict, handling non-serializable types."""
        serialized = {}
        for key, value in info.items():
            try:
                if isinstance(value, (int, float, str, bool, type(None))):
                    serialized[key] = value
                elif isinstance(value, (list, tuple)):
                    serialized[key] = list(value)
                elif isinstance(value, dict):
                    serialized[key] = self._serialize_info(value)
                elif isinstance(value, np.ndarray):
                    serialized[key] = {
                        "_type": "ndarray",
                        "shape": value.shape,
                        "dtype": str(value.dtype),
                    }
                else:
                    serialized[key] = str(value)
            except Exception:
                serialized[key] = f"<non-serializable: {type(value).__name__}>"
        return serialized

    def _save(self):
        """Save episode data to file."""
        with open(self.save_path, "w") as f:
            json.dump(self.episode_data, f, indent=2)


class TrajectoryWrapper(gym.Wrapper):
    """Store trajectories for offline RL / imitation learning."""

    def __init__(self, env):
        super().__init__(env)
        self.trajectories = []
        self.current_trajectory = {"observations": [], "actions": [], "rewards": []}

    def reset(self, **kwargs):
        """Reset environment and start a new trajectory.

        Args:
            **kwargs: Arguments passed to the wrapped environment's reset().

        Returns:
            tuple: (observation, info) from the wrapped environment.
        """
        if self.current_trajectory["actions"]:
            self.trajectories.append(self.current_trajectory)
        self.current_trajectory = {"observations": [], "actions": [], "rewards": []}
        obs, info = self.env.reset(**kwargs)
        self.current_trajectory["observations"].append(obs)
        return obs, info

    def step(self, action):
        """Execute action and record it in the current trajectory.

        Args:
            action: Action to execute.

        Returns:
            tuple: (observation, reward, terminated, truncated, info).
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_trajectory["actions"].append(action)
        self.current_trajectory["rewards"].append(reward)
        self.current_trajectory["observations"].append(obs)
        return obs, reward, terminated, truncated, info

    def get_trajectories(self):
        """Get all recorded trajectories.

        Returns:
            list: List of trajectory dictionaries, each with 'observations',
                'actions', and 'rewards' keys.
        """
        return self.trajectories
