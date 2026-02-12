"""Vectorized environments for parallel episode execution.

Provides both synchronous and asynchronous vectorized environments
compatible with Gymnasium's VectorEnv interface.
"""

from __future__ import annotations

from typing import Any

from gymnasium.vector import AsyncVectorEnv as GymAsyncVectorEnv
from gymnasium.vector import SyncVectorEnv as GymSyncVectorEnv

import agentick


class SyncVectorAgentickEnv(GymSyncVectorEnv):
    """
    Synchronous vectorized Agentick environment.

    Runs N environments in a single process, stepping them sequentially.
    Fast for simple environments on CPU.

    Example:
        >>> env = SyncVectorAgentickEnv(
        ...     num_envs=8,
        ...     task_id="GoToGoal-v0",
        ...     difficulty="easy",
        ...     render_mode="rgb_array",
        ... )
        >>> obs, info = env.reset()
        >>> obs.shape  # (8, H, W, 3) for pixel observations
    """

    def __init__(
        self,
        num_envs: int,
        task_id: str,
        difficulty: str = "easy",
        render_mode: str = "rgb_array",
        **kwargs: Any,
    ):
        """
        Initialize synchronous vectorized environment.

        Args:
            num_envs: Number of parallel environments
            task_id: Task identifier (e.g., "GoToGoal-v0")
            difficulty: Difficulty level
            render_mode: Render mode for observations
            **kwargs: Additional arguments passed to make()
        """
        self.task_id = task_id
        self.difficulty = difficulty
        self.render_mode_str = render_mode
        self.env_kwargs = kwargs

        def make_env():
            def _init():
                return agentick.make(
                    task_id,
                    difficulty=difficulty,
                    render_mode=render_mode,
                    **kwargs,
                )

            return _init

        env_fns = [make_env() for _ in range(num_envs)]

        super().__init__(env_fns)

    def reset(
        self,
        seed: int | list[int] | None = None,
        options: dict | None = None,
    ):
        """
        Reset all environments.

        Args:
            seed: Seed(s) for environments (if int, broadcast to all)
            options: Optional reset options

        Returns:
            Tuple of (observations, infos)
        """
        # Handle seed broadcasting
        if isinstance(seed, int):
            seeds = [seed + i for i in range(self.num_envs)]
        elif seed is None:
            seeds = [None] * self.num_envs
        else:
            seeds = seed

        return super().reset(seed=seeds, options=options)


class AsyncVectorAgentickEnv(GymAsyncVectorEnv):
    """
    Asynchronous vectorized Agentick environment.

    Runs N environments across multiple processes for true parallelism.
    Better for environments with significant computation per step.

    Example:
        >>> env = AsyncVectorAgentickEnv(
        ...     num_envs=8,
        ...     task_id="KeyDoorPuzzle-v0",
        ...     difficulty="medium",
        ...     render_mode="state_dict",
        ... )
        >>> obs, info = env.reset()
    """

    def __init__(
        self,
        num_envs: int,
        task_id: str,
        difficulty: str = "easy",
        render_mode: str = "rgb_array",
        **kwargs: Any,
    ):
        """
        Initialize asynchronous vectorized environment.

        Args:
            num_envs: Number of parallel environments
            task_id: Task identifier
            difficulty: Difficulty level
            render_mode: Render mode for observations
            **kwargs: Additional arguments passed to make()
        """
        self.task_id = task_id
        self.difficulty = difficulty
        self.render_mode_str = render_mode
        self.env_kwargs = kwargs

        def make_env(rank: int):
            def _init():
                env = agentick.make(
                    task_id,
                    difficulty=difficulty,
                    render_mode=render_mode,
                    **kwargs,
                )
                # Seed each environment differently
                env.reset(seed=rank)
                return env

            return _init

        env_fns = [make_env(i) for i in range(num_envs)]

        super().__init__(env_fns)

    def reset(
        self,
        seed: int | list[int] | None = None,
        options: dict | None = None,
    ):
        """
        Reset all environments.

        Args:
            seed: Seed(s) for environments
            options: Optional reset options

        Returns:
            Tuple of (observations, infos)
        """
        # Handle seed broadcasting
        if isinstance(seed, int):
            seeds = [seed + i for i in range(self.num_envs)]
        elif seed is None:
            seeds = [None] * self.num_envs
        else:
            seeds = seed

        return super().reset(seed=seeds, options=options)


def make_vec_env(
    task_id: str,
    num_envs: int = 8,
    difficulty: str = "easy",
    vec_env_cls: type = SyncVectorAgentickEnv,
    render_mode: str = "rgb_array",
    **kwargs: Any,
) -> SyncVectorAgentickEnv | AsyncVectorAgentickEnv:
    """
    Create a vectorized environment.

    Args:
        task_id: Task identifier
        num_envs: Number of parallel environments
        difficulty: Difficulty level
        vec_env_cls: Vectorized env class (Sync or Async)
        render_mode: Render mode
        **kwargs: Additional environment arguments

    Returns:
        Vectorized environment instance

    Example:
        >>> # Synchronous (single process)
        >>> env = make_vec_env("GoToGoal-v0", num_envs=8)
        >>>
        >>> # Asynchronous (multiprocess)
        >>> env = make_vec_env(
        ...     "GoToGoal-v0",
        ...     num_envs=8,
        ...     vec_env_cls=AsyncVectorAgentickEnv,
        ... )
    """
    return vec_env_cls(
        num_envs=num_envs,
        task_id=task_id,
        difficulty=difficulty,
        render_mode=render_mode,
        **kwargs,
    )
