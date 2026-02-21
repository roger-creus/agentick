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

# Task-to-category mapping derived from registry subpackage structure
TASK_CATEGORIES: dict[str, str] = {
    "GoToGoal-v0": "navigation",
    "MazeNavigation-v0": "navigation",
    "DynamicObstacles-v0": "navigation",
    "FogOfWar-v0": "navigation",
    "MultiGoalRoute-v0": "navigation",
    "KeyDoorPuzzle-v0": "memory",
    "SequenceMemory-v0": "memory",
    "BreadcrumbTrail-v0": "memory",
    "BacktrackPuzzle-v0": "memory",
    "DelayedGratification-v0": "memory",
    "SokobanPush-v0": "reasoning",
    "SwitchCircuit-v0": "reasoning",
    "RuleInduction-v0": "reasoning",
    "CausalChain-v0": "reasoning",
    "SymbolMatching-v0": "reasoning",
    "MultiRoomEscape-v0": "skill",
    "ResourceManagement-v0": "skill",
    "RecipeAssembly-v0": "skill",
    "ToolUse-v0": "skill",
    "EmergentStrategy-v0": "skill",
    "PreciseNavigation-v0": "control",
    "ChaseEvade-v0": "control",
    "Herding-v0": "control",
    "TimingChallenge-v0": "control",
    "GraphColoring-v0": "combinatorial",
    "LightsOut-v0": "combinatorial",
    "PackingPuzzle-v0": "combinatorial",
    "TileSorting-v0": "combinatorial",
    "DeceptiveReward-v0": "adversarial",
    "DistributionShift-v0": "adversarial",
    "NoisyObservation-v0": "adversarial",
    "FewShotAdaptation-v0": "meta",
    "TaskInterference-v0": "meta",
    "CompetitiveTag-v0": "multi_agent",
    "CooperativeTransport-v0": "multi_agent",
    "InstructionFollowing-v0": "compositional",
    "ProgramSynthesis-v0": "compositional",
    "RecursiveRooms-v0": "compositional",
}

ALL_CATEGORIES = [
    "navigation",
    "memory",
    "reasoning",
    "skill",
    "control",
    "combinatorial",
    "adversarial",
    "meta",
    "multi_agent",
    "compositional",
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

        seeds = self.config.seeds
        if seeds is None:
            rng = np.random.default_rng(42)
            seeds = rng.integers(0, 1_000_000, size=self.config.n_seeds).tolist()
        seed = seeds[0]

        print(f"PPO Training Benchmark: {self.config.name}")
        print(f"  Tasks: {len(task_names)}, Difficulties: {len(difficulties)}")
        print(f"  Total runs: {len(all_runs)}, Remaining: {len(remaining)}")
        print(f"  Timesteps per run: {self.training_config.total_timesteps:,}")
        print(f"  Parallel envs: {self.training_config.n_envs}")
        print(f"  Reward mode: {self.config.reward_mode}")
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

        # Build and save summary
        summary = self._build_summary(all_results, task_names, difficulties)
        summary["total_time_seconds"] = self.end_time - self.start_time

        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=_json_default)

        # Build training_summary.json (for plotting)
        training_summary = {
            "config_name": self.config.name,
            "reward_mode": self.config.reward_mode,
            "total_timesteps": self.training_config.total_timesteps,
            "tasks": task_names,
            "difficulties": difficulties,
            "results": all_results,
        }
        with open(output_dir / "training_summary.json", "w") as f:
            json.dump(training_summary, f, indent=2, default=_json_default)

        # Clean checkpoint on success
        ckpt_path = output_dir / ".checkpoint.json"
        if ckpt_path.exists():
            ckpt_path.unlink()

        # Auto-generate plots
        print("\nGenerating training plots...")
        try:
            from agentick.visualization.training_plots import TrainingBenchmarkPlotter

            plotter = TrainingBenchmarkPlotter(output_dir)
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

        # -- Create eval environment --
        eval_dir = run_dir / "eval"
        eval_dir.mkdir(exist_ok=True)
        eval_env = self._make_vec_env(task_name, difficulty, seed + 1000, 1, is_training=False)

        # -- Callbacks --
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / "model"),
            log_path=str(eval_dir),
            eval_freq=max(tc.eval_frequency // tc.n_envs, 1),
            n_eval_episodes=tc.n_eval_episodes,
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
        tb_log_dir = output_dir / "tensorboard"
        tb_log_dir.mkdir(exist_ok=True)
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
            tensorboard_log=str(tb_log_dir),
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

        # -- Run final evaluation with video --
        final_metrics = self._run_final_eval(
            model_path=run_dir / "model" / "best_model.zip"
            if tc.save_best_model and (run_dir / "model" / "best_model.zip").exists()
            else None,
            model=model,
            task_name=task_name,
            difficulty=difficulty,
            seed=seed + 2000,
            video_dir=run_dir / "videos" if self.config.record_videos else None,
            n_episodes=tc.n_eval_episodes,
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
        seed: int,
        n_envs: int,
        is_training: bool,
    ):
        """Create a DummyVecEnv with Atari preprocessing + VecTransposeImage."""
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

        from agentick.wrappers.atari_preprocessing import make_atari_env

        def make_env(env_seed: int):
            def _init():
                env = make_atari_env(
                    task_name,
                    seed=env_seed,
                    difficulty=difficulty,
                    reward_mode=self.config.reward_mode,
                )
                env = Monitor(env)
                return env

            return _init

        env_fns = [make_env(seed + i) for i in range(n_envs)]
        vec_env = DummyVecEnv(env_fns)
        vec_env = VecTransposeImage(vec_env)
        return vec_env

    def _run_final_eval(
        self,
        model_path: Path | None,
        model: Any,
        task_name: str,
        difficulty: str,
        seed: int,
        video_dir: Path | None,
        n_episodes: int,
    ) -> dict[str, Any]:
        """Run final evaluation episodes and optionally record video."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

        from agentick.wrappers.atari_preprocessing import make_atari_env

        # Load best model if available
        if model_path and model_path.exists():
            eval_model = PPO.load(str(model_path))
        else:
            eval_model = model

        # Create eval env
        def make_env():
            env = make_atari_env(
                task_name,
                seed=seed,
                difficulty=difficulty,
                reward_mode=self.config.reward_mode,
            )
            env = Monitor(env)
            return env

        vec_env = DummyVecEnv([make_env])
        vec_env = VecTransposeImage(vec_env)

        # Optionally wrap with video recorder
        if video_dir:
            from stable_baselines3.common.vec_env import VecVideoRecorder

            video_dir.mkdir(parents=True, exist_ok=True)
            vec_env = VecVideoRecorder(
                vec_env,
                str(video_dir),
                record_video_trigger=lambda x: x == 0,
                video_length=500,
                name_prefix=f"eval_{task_name}_{difficulty}",
            )

        returns = []
        lengths = []
        successes = []

        for ep in range(n_episodes):
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
            curve.append(
                {
                    "timestep": int(ts),
                    "mean_reward": float(np.mean(results[i])),
                    "std_reward": float(np.std(results[i])),
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
