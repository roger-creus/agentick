"""Helper interface for RL agents."""


class RLInterface:
    """Helper for RL agents: standard gym-compatible obs/action spaces."""

    @staticmethod
    def make_vectorized_env(task_name, n_envs=8, **kwargs):
        """Create vectorized environment for parallel training."""
        import gymnasium as gym

        import agentick

        def make_env(seed):
            def _init():
                env = agentick.make(task_name, seed=seed, **kwargs)
                return env

            return _init

        envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(n_envs)])
        return envs
