"""Training callbacks for evaluation, curriculum, and checkpointing."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


class EvalCallback:
    """
    Callback for periodic evaluation during training.

    Runs evaluation on test suite at regular intervals and logs metrics.
    """

    def __init__(
        self,
        eval_env_factory: Callable,
        eval_frequency: int = 10000,
        n_eval_episodes: int = 10,
        logger: Any | None = None,
    ):
        """
        Initialize evaluation callback.

        Args:
            eval_env_factory: Factory function that creates eval environment
            eval_frequency: Evaluate every N training steps
            n_eval_episodes: Number of episodes per evaluation
            logger: Optional logger for metrics
        """
        self.eval_env_factory = eval_env_factory
        self.eval_frequency = eval_frequency
        self.n_eval_episodes = n_eval_episodes
        self.logger = logger
        self.step_count = 0
        self.eval_history: list[dict[str, float]] = []

    def on_step(self, agent: Any, step: int) -> dict[str, float]:
        """
        Called after each training step.

        Args:
            agent: Agent being trained
            step: Current training step

        Returns:
            Dict of eval metrics if evaluation ran, else empty dict
        """
        self.step_count += 1

        if self.step_count % self.eval_frequency == 0:
            return self.evaluate(agent, step)

        return {}

    def evaluate(self, agent: Any, step: int) -> dict[str, float]:
        """
        Run evaluation.

        Args:
            agent: Agent to evaluate
            step: Current training step

        Returns:
            Dict of evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        success_count = 0

        for episode_idx in range(self.n_eval_episodes):
            env = self.eval_env_factory()
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done:
                if hasattr(agent, "predict"):
                    action, _ = agent.predict(obs, deterministic=True)
                elif hasattr(agent, "act"):
                    action = agent.act(obs)
                else:
                    action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated

                if terminated and info.get("success", False):
                    success_count += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        metrics = {
            "eval/mean_reward": float(np.mean(episode_rewards)),
            "eval/std_reward": float(np.std(episode_rewards)),
            "eval/mean_length": float(np.mean(episode_lengths)),
            "eval/success_rate": float(success_count / self.n_eval_episodes),
            "eval/step": step,
        }

        self.eval_history.append({"step": step, **metrics})

        if self.logger:
            for key, value in metrics.items():
                self.logger.log(key, value, step)

        return metrics


class CheckpointCallback:
    """
    Callback for saving model checkpoints.

    Saves best model and periodic checkpoints during training.
    """

    def __init__(
        self,
        save_dir: str | Path,
        save_frequency: int = 10000,
        save_best: bool = True,
        metric: str = "eval/mean_reward",
        logger: Any | None = None,
    ):
        """
        Initialize checkpoint callback.

        Args:
            save_dir: Directory to save checkpoints
            save_frequency: Save every N steps
            save_best: Whether to save best model
            metric: Metric to use for best model (higher is better)
            logger: Optional logger
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_frequency = save_frequency
        self.save_best = save_best
        self.metric = metric
        self.logger = logger

        self.best_metric = float("-inf")
        self.step_count = 0

    def on_step(
        self,
        agent: Any,
        step: int,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """
        Called after each training step.

        Args:
            agent: Agent being trained
            step: Current training step
            metrics: Optional metrics dict
        """
        self.step_count += 1

        # Periodic checkpoint
        if self.step_count % self.save_frequency == 0:
            checkpoint_path = self.save_dir / f"checkpoint_{step}.pt"
            self._save_agent(agent, checkpoint_path)

            if self.logger:
                self.logger.log("checkpoint/saved", 1, step)

        # Best model checkpoint
        if self.save_best and metrics and self.metric in metrics:
            current_metric = metrics[self.metric]

            if current_metric > self.best_metric:
                self.best_metric = current_metric
                best_path = self.save_dir / "best_model.pt"
                self._save_agent(agent, best_path)

                # Save metadata
                metadata_path = self.save_dir / "best_model_metadata.json"
                metadata = {
                    "step": step,
                    "metric": self.metric,
                    "value": float(current_metric),
                }
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                if self.logger:
                    self.logger.log("checkpoint/best_updated", 1, step)
                    self.logger.log("checkpoint/best_metric", current_metric, step)

    def _save_agent(self, agent: Any, path: Path) -> None:
        """Save agent to file."""
        if hasattr(agent, "save"):
            agent.save(str(path))
        elif hasattr(agent, "state_dict"):
            import torch

            torch.save(agent.state_dict(), path)
        else:
            # Can't save this agent type
            pass
