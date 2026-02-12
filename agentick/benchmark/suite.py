"""Benchmark suite definitions."""

import agentick


def get_suite(name: str = "full", difficulty: str = "medium", **kwargs) -> list:
    """Get predefined benchmark suite."""
    return agentick.make_suite(name, difficulty=difficulty, **kwargs)


class BenchmarkRunner:
    """Runs agent on suite and collects metrics."""

    def __init__(self, suite: str = "full", n_episodes: int = 100, seeds: list[int] | None = None):
        self.suite = suite
        self.n_episodes = n_episodes
        self.seeds = seeds or list(range(n_episodes))

    def evaluate(self, agent_fn, difficulty: str = "medium"):
        """
        Evaluate agent on suite.

        Args:
            agent_fn: Function that takes (obs, info) and returns action
            difficulty: Difficulty level

        Returns:
            Dictionary of results
        """
        envs = get_suite(self.suite, difficulty=difficulty)
        results = {}

        for env in envs:
            task_name = env.task.name
            episode_returns = []
            successes = []

            for seed in self.seeds[: self.n_episodes]:
                obs, info = env.reset(seed=seed)
                episode_return = 0.0
                done = False

                while not done:
                    action = agent_fn(obs, info)
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_return += reward
                    done = terminated or truncated

                episode_returns.append(episode_return)
                successes.append(info.get("success", False))

            results[task_name] = {
                "mean_return": sum(episode_returns) / len(episode_returns),
                "success_rate": sum(successes) / len(successes),
            }

        return results
