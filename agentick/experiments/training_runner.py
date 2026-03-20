"""Training benchmark runner for PPO pixel-based training across all tasks."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from agentick.experiments.config import ExperimentConfig

# Task-to-category mapping derived from registry subpackage structure (6 categories)
TASK_CATEGORIES: dict[str, str] = {
    # Navigation (8)
    "GoToGoal-v0": "navigation",
    "MazeNavigation-v0": "navigation",
    "ShortestPath-v0": "navigation",
    "DynamicObstacles-v0": "navigation",
    "CuriosityMaze-v0": "navigation",
    "RecursiveRooms-v0": "navigation",
    "TimingChallenge-v0": "navigation",
    "InstructionFollowing-v0": "navigation",
    # Planning (9)
    "SokobanPush-v0": "planning",
    "KeyDoorPuzzle-v0": "planning",
    "BacktrackPuzzle-v0": "planning",
    "TileSorting-v0": "planning",
    "PackingPuzzle-v0": "planning",
    "PreciseNavigation-v0": "planning",
    "RecipeAssembly-v0": "planning",
    "ToolUse-v0": "planning",
    "ResourceManagement-v0": "planning",
    # Reasoning (8)
    "SwitchCircuit-v0": "reasoning",
    "RuleInduction-v0": "reasoning",
    "LightsOut-v0": "reasoning",
    "GraphColoring-v0": "reasoning",
    "SymbolMatching-v0": "reasoning",
    "ProgramSynthesis-v0": "reasoning",
    "TaskInterference-v0": "reasoning",
    "DeceptiveReward-v0": "reasoning",
    # Memory (4)
    "SequenceMemory-v0": "memory",
    "DelayedGratification-v0": "memory",
    "TreasureHunt-v0": "memory",
    "FogOfWarExploration-v0": "memory",
    # Generalization (3)
    "FewShotAdaptation-v0": "generalization",
    "DistributionShift-v0": "generalization",
    "NoisyObservation-v0": "generalization",
    # Multi-Agent (5)
    "CooperativeTransport-v0": "multi_agent",
    "TagHunt-v0": "multi_agent",
    "ChaseEvade-v0": "multi_agent",
    "Herding-v0": "multi_agent",
    "EmergentStrategy-v0": "multi_agent",
}

ALL_CATEGORIES = [
    "navigation",
    "planning",
    "reasoning",
    "memory",
    "generalization",
    "multi_agent",
]


def get_task_category(task_name: str) -> str:
    """Get category for a task, falling back to 'unknown'."""
    return TASK_CATEGORIES.get(task_name, "unknown")


class TrainingBenchmarkRunner:
    """Run PPO training benchmark across tasks and difficulties.

    Trains a separate PPO agent (CnnPolicy, pixel obs) for each (task, difficulty)
    combination, evaluates periodically, and saves models, metrics, and videos.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        if config.training is None:
            raise ValueError("ExperimentConfig.training must be set for training runs")
        self.training_config = config.training
        self.start_time: float | None = None
        self.end_time: float | None = None

    def run(
        self, resume_from: str | Path | None = None, output_dir: str | Path | None = None
    ) -> Path:
        """Run training benchmark over all tasks x difficulties.

        Args:
            resume_from: Path to previous run directory to resume from.

        Returns:
            Path to the output directory.
        """
        self.start_time = time.time()

        # Create or resume output directory
        if resume_from:
            output_dir = Path(resume_from)
            checkpoint = self._load_checkpoint(output_dir)
        elif output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = self._load_checkpoint(output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.config.output_dir) / f"{self.config.name}_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = None

        # Save config
        self.config.to_yaml(output_dir / "config.yaml")

        # Collect metadata
        metadata = self._collect_metadata()
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Resolve tasks and difficulties
        task_names = self._resolve_tasks()
        difficulties = self.config.difficulties

        # Build run matrix
        all_runs = [(t, d) for t in task_names for d in difficulties]
        completed = set()
        if checkpoint:
            completed = set(tuple(r) for r in checkpoint.get("completed_runs", []))
        remaining = [r for r in all_runs if r not in completed]

        # Load existing results
        all_results: dict[str, dict[str, Any]] = {}
        if checkpoint:
            all_results = checkpoint.get("results", {})

        # Seeds are now per-task-difficulty; computed in the loop below
        explicit_seeds = self.config.seeds

        print(f"PPO Training Benchmark: {self.config.name}")
        print(f"  Tasks: {len(task_names)}, Difficulties: {len(difficulties)}")
        print(f"  Total runs: {len(all_runs)}, Remaining: {len(remaining)}")
        print(f"  Timesteps per run: {self.training_config.total_timesteps:,}")
        print(f"  Parallel envs: {self.training_config.n_envs}")
        print(f"  Reward mode: {self.config.reward_mode}")
        render_mode = self.config.render_modes[0] if self.config.render_modes else "rgb_array"
        print(f"  Render mode: {render_mode}")
        print(f"  Output: {output_dir}")
        print()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            main_task = progress.add_task(
                "[cyan]Training benchmark",
                total=len(all_runs),
                completed=len(completed),
            )

            for task_name, difficulty in remaining:
                progress.update(
                    main_task,
                    description=f"[cyan]{task_name} ({difficulty})",
                )

                run_dir = output_dir / "per_task" / task_name / difficulty
                run_dir.mkdir(parents=True, exist_ok=True)

                # Per-task-difficulty seed from train split
                if explicit_seeds is not None:
                    seed = explicit_seeds[0]
                else:
                    from agentick.leaderboard.seeds import generate_task_seeds

                    seed = generate_task_seeds(task_name, difficulty, "train", 1)[0]

                try:
                    metrics = self._train_single_task(
                        task_name, difficulty, seed, run_dir, output_dir
                    )
                except Exception as e:
                    print(f"\n  [!] Failed: {task_name} ({difficulty}): {e}")
                    metrics = {
                        "error": str(e),
                        "mean_return": 0.0,
                        "success_rate": 0.0,
                        "training_curve": [],
                    }

                # Store results
                key = f"{task_name}_{difficulty}"
                all_results[key] = {
                    "task": task_name,
                    "difficulty": difficulty,
                    "category": get_task_category(task_name),
                    **metrics,
                }

                # Save per-run metrics
                with open(run_dir / "metrics.json", "w") as f:
                    json.dump(all_results[key], f, indent=2, default=_json_default)

                completed.add((task_name, difficulty))
                progress.update(main_task, advance=1)

                # Checkpoint
                self._save_checkpoint(
                    output_dir,
                    {
                        "completed_runs": [list(r) for r in completed],
                        "results": all_results,
                    },
                )

        self.end_time = time.time()

        # Merge results with any existing summary (from other SLURM jobs
        # writing to the same shared output directory).
        merged_results = dict(all_results)
        merged_tasks = list(task_names)
        ts_path = output_dir / "training_summary.json"
        if ts_path.exists():
            try:
                with open(ts_path) as f:
                    existing = json.load(f)
                for key, val in existing.get("results", {}).items():
                    if key not in merged_results:
                        merged_results[key] = val
                for t in existing.get("tasks", []):
                    if t not in merged_tasks:
                        merged_tasks.append(t)
            except (json.JSONDecodeError, OSError):
                pass  # corrupted file, overwrite

        # Build and save summary using merged results
        summary = self._build_summary(merged_results, merged_tasks, difficulties)
        summary["total_time_seconds"] = self.end_time - self.start_time

        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=_json_default)

        # Build training_summary.json (for plotting)
        render_mode = (
            self.config.render_modes[0] if self.config.render_modes else "rgb_array"
        )
        training_summary = {
            "config_name": self.config.name,
            "reward_mode": self.config.reward_mode,
            "render_mode": render_mode,
            "total_timesteps": self.training_config.total_timesteps,
            "tasks": merged_tasks,
            "difficulties": difficulties,
            "results": merged_results,
        }
        with open(output_dir / "training_summary.json", "w") as f:
            json.dump(training_summary, f, indent=2, default=_json_default)

        # Clean checkpoint on success (missing_ok for concurrent job safety)
        ckpt_path = output_dir / ".checkpoint.json"
        ckpt_path.unlink(missing_ok=True)

        # Auto-generate plots
        print("\nGenerating training plots...")
        try:
            from agentick.visualization.experiment_plots import ExperimentPlotter

            plotter = ExperimentPlotter(output_dir)
            plotter.plot_all()
            print(f"  Figures saved to: {output_dir / 'figures'}")
        except Exception as e:
            print(f"  Warning: Failed to generate plots: {e}")

        print(f"\nTraining benchmark complete: {output_dir}")
        print(f"  Mean success rate: {summary.get('overall_success_rate', 0):.2%}")
        print(f"  Total time: {summary['total_time_seconds'] / 60:.1f} min")
        print("\nTensorBoard:")
        print(f"  tensorboard --logdir {output_dir / 'tensorboard'}")

        return output_dir

    def _train_single_task(
        self,
        task_name: str,
        difficulty: str,
        seed: int,
        run_dir: Path,
        output_dir: Path,
    ) -> dict[str, Any]:
        """Train PPO on a single (task, difficulty) and return metrics."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import (
            CheckpointCallback,
            EvalCallback,
        )

        tc = self.training_config
        hp = self.config.agent.hyperparameters

        # -- Create training environments --
        train_env = self._make_vec_env(task_name, difficulty, seed, tc.n_envs, is_training=True)

        # -- Create eval environment using official eval seeds --
        eval_dir = run_dir / "eval"
        eval_dir.mkdir(exist_ok=True)

        from agentick.leaderboard.seeds import generate_task_seeds

        eval_seeds = list(generate_task_seeds(task_name, difficulty, "eval", 25))
        eval_env = self._make_vec_env(
            task_name, difficulty, eval_seeds, len(eval_seeds), is_training=False,
        )

        # -- Callbacks --
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / "model"),
            log_path=str(eval_dir),
            eval_freq=max(tc.eval_frequency // tc.n_envs, 1),
            n_eval_episodes=1,  # 1 episode per env × 25 envs = 25 eval episodes
            deterministic=True,
            verbose=0,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=max(tc.checkpoint_frequency // tc.n_envs, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo",
            verbose=0,
        )

        # -- Create PPO --
        # Use tensorboard logging only if tensorboard is installed
        try:
            import tensorboard  # noqa: F401
            tb_log_dir = output_dir / "tensorboard"
            tb_log_dir.mkdir(exist_ok=True)
            tb_log_arg: str | None = str(tb_log_dir)
        except ImportError:
            tb_log_arg = None
        tb_log_name = f"{task_name}_{difficulty}"

        model = PPO(
            policy=hp.get("policy", "CnnPolicy"),
            env=train_env,
            n_steps=hp.get("n_steps", 128),
            batch_size=hp.get("batch_size", 256),
            n_epochs=hp.get("n_epochs", 4),
            learning_rate=hp.get("learning_rate", 2.5e-4),
            clip_range=hp.get("clip_range", 0.2),
            ent_coef=hp.get("ent_coef", 0.01),
            vf_coef=hp.get("vf_coef", 0.5),
            max_grad_norm=hp.get("max_grad_norm", 0.5),
            gamma=hp.get("gamma", 0.99),
            gae_lambda=hp.get("gae_lambda", 0.95),
            tensorboard_log=tb_log_arg,
            device=tc.device,
            seed=seed,
            verbose=0,
        )

        # -- Train --
        model.learn(
            total_timesteps=tc.total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            tb_log_name=tb_log_name,
        )

        # -- Extract training curve from eval logs --
        training_curve = self._extract_training_curve(eval_dir)

        # -- Run final evaluation on eval seeds --
        final_metrics = self._run_final_eval(
            model_path=run_dir / "model" / "best_model.zip"
            if tc.save_best_model and (run_dir / "model" / "best_model.zip").exists()
            else None,
            model=model,
            task_name=task_name,
            difficulty=difficulty,
            seeds=eval_seeds,
            video_dir=run_dir / "videos" if self.config.record_videos else None,
        )

        # Cleanup
        train_env.close()
        eval_env.close()

        return {
            **final_metrics,
            "training_curve": training_curve,
        }

    def _make_vec_env(
        self,
        task_name: str,
        difficulty: str,
        seed: int | list[int],
        n_envs: int,
        is_training: bool,
    ):
        """Create a DummyVecEnv with Atari preprocessing + VecTransposeImage.

        Args:
            seed: A single seed (envs get seed, seed+1, ...) or a list of seeds
                  (one per env, n_envs must equal len(seed)).
        """
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

        from agentick.wrappers.atari_preprocessing import make_atari_env

        render_mode = self.config.render_modes[0] if self.config.render_modes else "rgb_array"

        # Build per-env seed list
        if isinstance(seed, list):
            env_seeds = seed
        else:
            env_seeds = [seed + i for i in range(n_envs)]

        def make_env(env_seed: int):
            def _init():
                env = make_atari_env(
                    task_name,
                    seed=env_seed,
                    difficulty=difficulty,
                    reward_mode=self.config.reward_mode,
                    render_mode=render_mode,
                )
                env = Monitor(env)
                return env

            return _init

        env_fns = [make_env(s) for s in env_seeds]
        vec_env = DummyVecEnv(env_fns)
        vec_env = VecTransposeImage(vec_env)
        return vec_env

    def _run_final_eval(
        self,
        model_path: Path | None,
        model: Any,
        task_name: str,
        difficulty: str,
        seeds: list[int],
        video_dir: Path | None,
    ) -> dict[str, Any]:
        """Run final evaluation on all eval seeds."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

        from agentick.wrappers.atari_preprocessing import make_atari_env

        # Load best model if available
        if model_path and model_path.exists():
            eval_model = PPO.load(str(model_path))
        else:
            eval_model = model

        render_mode = self.config.render_modes[0] if self.config.render_modes else "rgb_array"

        returns = []
        lengths = []
        successes = []

        # Evaluate one episode per eval seed
        for seed in seeds:
            def make_eval_env(s=seed):
                env = make_atari_env(
                    task_name,
                    seed=s,
                    difficulty=difficulty,
                    reward_mode=self.config.reward_mode,
                    render_mode=render_mode,
                )
                env = Monitor(env)
                return env

            vec_env = DummyVecEnv([make_eval_env])
            vec_env = VecTransposeImage(vec_env)

            obs = vec_env.reset()
            done = False
            ep_return = 0.0
            ep_length = 0
            ep_success = False

            while not done:
                action, _ = eval_model.predict(obs, deterministic=True)
                obs, reward, dones, infos = vec_env.step(action)
                ep_return += float(reward[0])
                ep_length += 1
                done = dones[0]
                if done and infos[0].get("success", False):
                    ep_success = True

            returns.append(ep_return)
            lengths.append(ep_length)
            successes.append(ep_success)
            vec_env.close()

        # Record a concatenated video from 5 random eval seeds
        if video_dir:
            import random

            from agentick.experiments._video_utils import _has_ffmpeg, _save_gif, _save_mp4

            n_video_seeds = min(5, len(seeds))
            video_seeds = random.sample(seeds, n_video_seeds)

            all_frames: list[np.ndarray] = []
            separator_n = 10  # black frames between episodes

            for vid_seed in video_seeds:
                def make_video_env(s=vid_seed):
                    env = make_atari_env(
                        task_name,
                        seed=s,
                        difficulty=difficulty,
                        reward_mode=self.config.reward_mode,
                        render_mode="rgb_array",
                    )
                    env = Monitor(env)
                    return env

                vid_env = DummyVecEnv([make_video_env])
                vid_env = VecTransposeImage(vid_env)
                obs = vid_env.reset()
                done = False
                while not done:
                    action, _ = eval_model.predict(obs, deterministic=True)
                    # Capture frame (VecTransposeImage changes channel order, get from inner env)
                    frame = vid_env.venv.envs[0].render()
                    if isinstance(frame, np.ndarray):
                        all_frames.append(frame)
                    obs, _, dones, _ = vid_env.step(action)
                    done = dones[0]
                vid_env.close()

                # Add black separator frames between episodes
                if all_frames:
                    h, w = all_frames[-1].shape[:2]
                    sep = np.zeros((h, w, 3), dtype=np.uint8)
                    all_frames.extend([sep] * separator_n)

            # Remove trailing separator
            if len(all_frames) > separator_n:
                all_frames = all_frames[:-separator_n]

            if all_frames:
                video_dir.mkdir(parents=True, exist_ok=True)
                name = f"eval_{task_name}_{difficulty}"
                try:
                    if _has_ffmpeg():
                        _save_mp4(all_frames, video_dir / f"{name}.mp4", fps=10)
                    else:
                        _save_gif(all_frames, video_dir / f"{name}.gif", fps=10)
                except Exception as e:
                    print(f"  Warning: Failed to save video {name}: {e}")

        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "success_rate": float(np.mean(successes)),
            "mean_length": float(np.mean(lengths)),
            "eval_returns": [float(r) for r in returns],
        }

    def _extract_training_curve(self, eval_dir: Path) -> list[dict[str, float]]:
        """Read SB3's evaluations.npz to extract timesteps vs mean reward."""
        npz_path = eval_dir / "evaluations.npz"
        if not npz_path.exists():
            return []

        data = np.load(str(npz_path))
        timesteps = data["timesteps"]
        results = data["results"]  # shape: (n_evals, n_episodes)

        curve = []
        for i, ts in enumerate(timesteps):
            ep_returns = results[i]
            curve.append(
                {
                    "timestep": int(ts),
                    "mean_reward": float(np.mean(ep_returns)),
                    "std_reward": float(np.std(ep_returns)),
                    "success_rate": float(np.mean(ep_returns > 0)),
                }
            )
        return curve

    def _resolve_tasks(self) -> list[str]:
        """Resolve task names from config."""
        if isinstance(self.config.tasks, list):
            return self.config.tasks

        suite_name = self.config.tasks
        if suite_name == "full":
            from agentick.tasks.registry import list_tasks

            return list_tasks()
        else:
            # Delegate to the same logic as ExperimentRunner
            from agentick.experiments.runner import ExperimentRunner

            runner = ExperimentRunner.__new__(ExperimentRunner)
            runner.config = self.config
            return runner._resolve_tasks()

    def _build_summary(
        self,
        all_results: dict[str, Any],
        task_names: list[str],
        difficulties: list[str],
    ) -> dict[str, Any]:
        """Build summary statistics from all run results."""
        success_rates = []
        mean_returns = []
        per_difficulty: dict[str, list[float]] = {d: [] for d in difficulties}
        per_category: dict[str, list[float]] = {c: [] for c in ALL_CATEGORIES}

        for key, res in all_results.items():
            sr = res.get("success_rate", 0.0)
            mr = res.get("mean_return", 0.0)
            success_rates.append(sr)
            mean_returns.append(mr)

            diff = res.get("difficulty", "")
            if diff in per_difficulty:
                per_difficulty[diff].append(sr)

            cat = res.get("category", "unknown")
            if cat in per_category:
                per_category[cat].append(sr)

        summary = {
            "overall_success_rate": float(np.mean(success_rates)) if success_rates else 0.0,
            "overall_mean_return": float(np.mean(mean_returns)) if mean_returns else 0.0,
            "n_tasks": len(task_names),
            "n_difficulties": len(difficulties),
            "n_total_runs": len(all_results),
            "per_difficulty_success_rate": {
                d: float(np.mean(rates)) if rates else 0.0 for d, rates in per_difficulty.items()
            },
            "per_category_success_rate": {
                c: float(np.mean(rates)) if rates else 0.0
                for c, rates in per_category.items()
                if rates
            },
        }
        return summary

    def _collect_metadata(self) -> dict[str, Any]:
        """Collect metadata about the run environment."""
        import os
        import platform

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "config_name": self.config.name,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        }

        try:
            git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            metadata["git_hash"] = git_hash
        except Exception:
            metadata["git_hash"] = None

        try:
            import agentick

            metadata["agentick_version"] = agentick.__version__
        except Exception:
            metadata["agentick_version"] = "unknown"

        try:
            import stable_baselines3

            metadata["sb3_version"] = stable_baselines3.__version__
        except Exception:
            metadata["sb3_version"] = "unknown"

        try:
            import torch

            metadata["torch_version"] = torch.__version__
            metadata["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                metadata["cuda_device"] = torch.cuda.get_device_name(0)
        except Exception:
            metadata["torch_version"] = "unknown"
            metadata["cuda_available"] = False

        return metadata

    def _save_checkpoint(self, output_dir: Path, data: dict[str, Any]) -> None:
        """Save checkpoint for crash recovery."""
        ckpt_path = output_dir / ".checkpoint.json"
        with open(ckpt_path, "w") as f:
            json.dump(data, f, indent=2, default=_json_default)

    def _load_checkpoint(self, output_dir: Path) -> dict[str, Any] | None:
        """Load checkpoint if it exists."""
        ckpt_path = output_dir / ".checkpoint.json"
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                return json.load(f)
        return None


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
