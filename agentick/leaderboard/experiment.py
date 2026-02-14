"""Experiment runner for quick evaluations and testing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import yaml

import agentick


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    name: str
    description: str = ""
    tasks: list[str] = None
    agent_type: str = "random"
    num_seeds: int = 3
    episodes_per_seed: int = 1
    render_mode: str = "rgb_array"
    timeout: int = 100
    record_video: bool = True
    record_traces: bool = True

    def __post_init__(self):
        if self.tasks is None:
            self.tasks = ["GoToGoal-v0"]


class ExperimentRunner:
    """Simple experiment runner for testing and quick evaluations."""

    def __init__(
        self,
        config_path: str | None = None,
        config: ExperimentConfig | None = None,
        output_dir: str = "results",
    ):
        """
        Initialize experiment runner.

        Args:
            config_path: Path to YAML config file
            config: ExperimentConfig object (alternative to config_path)
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if config_path is not None:
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)
            self.config = ExperimentConfig(**config_dict)
        elif config is not None:
            self.config = config
        else:
            raise ValueError("Must provide either config_path or config")

    def _create_agent(self, agent_type: str) -> Any:
        """Create agent based on type."""
        if agent_type == "random":
            # Random agent
            class RandomAgent:
                def act(self, obs, env):
                    return env.action_space.sample()

            return RandomAgent()
        elif agent_type in ("greedy", "oracle"):
            # Greedy/oracle agent (moves towards goal if visible)
            class GreedyAgent:
                def act(self, obs, env):
                    # Simple heuristic: try forward first, then random
                    return 2  # forward action

            return GreedyAgent()
        elif agent_type == "ppo":
            raise ValueError(
                "PPO agent requires a trained model. "
                "Train first with: uv run python examples/experiments/full_benchmark/train_and_eval_ppo.py"
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def run(self) -> list[dict]:
        """Run experiment and return results."""
        results = []
        trace_paths = []

        print(f"Experiment: {self.config.name}")
        print(f"Tasks: {len(self.config.tasks)}")
        print(f"Seeds: {self.config.num_seeds}")
        print(f"Episodes per seed: {self.config.episodes_per_seed}")
        print()

        # Create agent
        agent = self._create_agent(self.config.agent_type)

        # Run on each task
        for task_id in self.config.tasks:
            print(f"Running {task_id}...")

            # Record video for first episode of first seed per task
            task_video_recorded = False

            for seed in range(self.config.num_seeds):
                for episode in range(self.config.episodes_per_seed):
                    # Create environment
                    env = agentick.make(task_id, render_mode=self.config.render_mode)

                    # Wrap with video recorder for first episode per task
                    if self.config.record_video and not task_video_recorded:
                        video_dir = self.output_dir / "videos"
                        video_dir.mkdir(parents=True, exist_ok=True)
                        env = gym.wrappers.RecordVideo(
                            env,
                            video_folder=str(video_dir),
                            name_prefix=f"{self.config.name}_{task_id}",
                            episode_trigger=lambda x: x == 0,  # Only record first episode
                        )
                        task_video_recorded = True

                    obs, info = env.reset(seed=seed)

                    # Initialize interaction trace
                    trace = []
                    if self.config.record_traces and seed == 0 and episode == 0:
                        trace.append(f"Task: {task_id}, Seed: {seed}")
                        trace.append(f"Initial observation: {obs}")

                    # Run episode
                    total_reward = 0
                    steps = 0
                    done = False

                    while not done and steps < self.config.timeout:
                        action = agent.act(obs, env)
                        obs, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated
                        total_reward += reward

                        # Record interaction trace
                        if self.config.record_traces and seed == 0 and episode == 0:
                            trace.append(f"Step {steps}: action={action}, reward={reward:.2f}")

                        steps += 1

                    # Save interaction trace for sample episode
                    if self.config.record_traces and seed == 0 and episode == 0:
                        trace.append(f"Episode ended: total_reward={total_reward:.2f}, steps={steps}")
                        trace.append(f"Success: {info.get('success', False)}")

                        trace_dir = self.output_dir / "traces"
                        trace_dir.mkdir(parents=True, exist_ok=True)
                        trace_path = trace_dir / f"{self.config.name}_{task_id}_trace.txt"
                        with open(trace_path, "w") as f:
                            f.write("\n".join(trace))
                        trace_paths.append(trace_path)

                    # Record result
                    result = {
                        "task": task_id,
                        "seed": seed,
                        "episode": episode,
                        "total_reward": float(total_reward),
                        "steps": int(steps),
                        "success": bool(info.get("success", False)),
                    }
                    results.append(result)

                    env.close()

        # Save results
        results_path = self.output_dir / f"{self.config.name}_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {results_path}")

        if self.config.record_video:
            print(f"Videos saved to: {self.output_dir / 'videos'}")

        if self.config.record_traces:
            print(f"Interaction traces saved to: {self.output_dir / 'traces'}")

        return results
