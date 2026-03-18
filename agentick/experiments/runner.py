"""Experiment runner with crash-safe execution and parallel processing."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from agentick.agents.backends.base import ModelBackend
from agentick.experiments.config import ExperimentConfig


def _run_task_worker(args: tuple) -> tuple[str, dict[str, Any]]:
    """
    Worker function for parallel task execution.

    Args:
        args: Tuple of (task_name, config, seeds, output_dir)

    Returns:
        Tuple of (task_name, task_results)
    """
    task_name, config, seeds, output_dir = args

    from agentick.leaderboard.seeds import generate_task_seeds

    task_results = {
        "task_name": task_name,
        "per_difficulty": {},
    }

    for difficulty in config.difficulties:
        # Per-task-difficulty seeds (unless explicit seeds provided)
        if seeds is not None:
            diff_seeds = seeds
        else:
            diff_seeds = list(generate_task_seeds(
                task_name, difficulty, getattr(config, "split", "eval"), config.n_seeds
            ))

        difficulty_results = {
            "difficulty": difficulty,
            "episodes": [],
            "metrics": {},
        }

        # Create episode directory
        episodes_dir = Path(output_dir) / "per_task" / task_name / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)

        # Run episodes
        for seed_idx, seed in enumerate(diff_seeds):
            for ep_idx in range(config.n_episodes):
                # Create environment
                import agentick

                env = agentick.make(
                    task_name,
                    difficulty=difficulty,
                    render_mode=config.render_modes[0] if config.render_modes else None,
                )

                # Run episode
                obs, info = env.reset(seed=seed)

                # Create oracle if needed
                _oracle = None
                if config.agent.type == "oracle":
                    from agentick.oracles import get_oracle
                    try:
                        _oracle = get_oracle(task_name, env)
                        _oracle.reset(obs, info)
                    except ValueError:
                        pass

                trajectory = {
                    "seed": seed,
                    "seed_idx": seed_idx,
                    "episode_idx": ep_idx,
                    "total_reward": 0.0,
                    "episode_length": 0,
                    "success": False,
                }

                terminated = False
                truncated = False
                step_count = 0
                total_reward = 0.0

                while not (terminated or truncated):
                    if _oracle is not None:
                        action = _oracle.act(obs, info)
                    else:
                        action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    if _oracle is not None:
                        _oracle.update(obs, info)
                    total_reward += reward
                    step_count += 1

                trajectory["total_reward"] = float(total_reward)
                trajectory["episode_length"] = int(step_count)
                trajectory["success"] = bool(info.get("success", False))

                # Save episode data
                if config.record_trajectories:
                    episode_file = episodes_dir / f"diff_{difficulty}_seed_{seed_idx}_ep_{ep_idx}.json"
                    with open(episode_file, "w") as f:
                        json.dump(trajectory, f, indent=2)

                episode_data = {
                    "seed": int(seed),
                    "episode_idx": int(ep_idx),
                    "return": float(total_reward),
                    "length": int(step_count),
                    "success": bool(trajectory["success"]),
                }
                difficulty_results["episodes"].append(episode_data)

                env.close()

        # Compute metrics for this difficulty
        episodes = difficulty_results["episodes"]
        if episodes:
            returns = [ep["return"] for ep in episodes]
            lengths = [ep["length"] for ep in episodes]
            successes = [ep["success"] for ep in episodes]

            metrics = {}
            if "mean_return" in config.metrics:
                metrics["mean_return"] = float(np.mean(returns))
            if "success_rate" in config.metrics:
                metrics["success_rate"] = float(np.mean(successes))
            if "mean_length" in config.metrics:
                metrics["mean_length"] = float(np.mean(lengths))

            difficulty_results["metrics"] = metrics

        task_results["per_difficulty"][difficulty] = difficulty_results

    # Compute aggregate task metrics
    all_episodes = []
    for diff_results in task_results["per_difficulty"].values():
        all_episodes.extend(diff_results["episodes"])

    if all_episodes:
        returns = [ep["return"] for ep in all_episodes]
        lengths = [ep["length"] for ep in all_episodes]
        successes = [ep["success"] for ep in all_episodes]

        aggregate_metrics = {}
        if "mean_return" in config.metrics:
            aggregate_metrics["mean_return"] = float(np.mean(returns))
        if "success_rate" in config.metrics:
            aggregate_metrics["success_rate"] = float(np.mean(successes))
        if "mean_length" in config.metrics:
            aggregate_metrics["mean_length"] = float(np.mean(lengths))

        task_results["aggregate_metrics"] = aggregate_metrics

    return task_name, task_results


class ExperimentResults:
    """Container for experiment results."""

    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: Path,
        metadata: dict[str, Any],
        summary: dict[str, Any],
        per_task_results: dict[str, Any],
    ):
        self.config = config
        self.output_dir = output_dir
        self.metadata = metadata
        self.summary = summary
        self.per_task_results = per_task_results

    def save(self) -> None:
        """Save results to disk.

        When multiple SLURM jobs share the same output directory (per-task
        splitting), each job writes its own per_task/ data and merges its
        summary into the existing summary.json.
        """
        # Save config — only write if not already present, so the first
        # job's config (with the original name) is preserved.
        config_path = self.output_dir / "config.yaml"
        if not config_path.exists():
            self.config.to_yaml(config_path)

        # Save metadata (overwrite is fine — metadata is per-job)
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        # Save per-task results (each task writes its own subdir, no collision)
        # Merge per_difficulty with existing metrics.json so partial reruns
        # (e.g. only missing difficulties) don't overwrite complete results.
        for task_name, task_results in self.per_task_results.items():
            task_dir = self.output_dir / "per_task" / task_name
            task_dir.mkdir(parents=True, exist_ok=True)

            metrics_path = task_dir / "metrics.json"
            merged = task_results
            if metrics_path.exists():
                try:
                    with open(metrics_path) as f:
                        existing = json.load(f)
                    existing_pd = existing.get("per_difficulty", {})
                    new_pd = task_results.get("per_difficulty", {})
                    # Keep existing difficulties, overwrite with new ones
                    existing_pd.update(new_pd)
                    merged = dict(existing)
                    merged["per_difficulty"] = existing_pd
                    # Recompute aggregate metrics from all difficulties
                    all_episodes = []
                    for diff_data in existing_pd.values():
                        all_episodes.extend(diff_data.get("episodes", []))
                    if all_episodes:
                        returns = [ep["return"] for ep in all_episodes]
                        successes = [ep["success"] for ep in all_episodes]
                        lengths = [ep["length"] for ep in all_episodes]
                        merged["aggregate_metrics"] = {
                            "mean_return": float(np.mean(returns)),
                            "success_rate": float(np.mean(successes)),
                            "mean_length": float(np.mean(lengths)),
                        }
                except (json.JSONDecodeError, OSError, KeyError):
                    merged = task_results

            with open(metrics_path, "w") as f:
                json.dump(merged, f, indent=2)

        # Merge summary with existing (other SLURM jobs may have written theirs)
        summary_path = self.output_dir / "summary.json"
        merged_summary = dict(self.summary)
        if summary_path.exists():
            try:
                with open(summary_path) as f:
                    existing = json.load(f)
                # Accumulate time
                existing_time = existing.get("total_time_seconds", 0)
                our_time = merged_summary.get("total_time_seconds", 0)
                merged_summary["total_time_seconds"] = existing_time + our_time
            except (json.JSONDecodeError, OSError):
                pass

        # Recompute summary from all per_task/ dirs on disk
        per_task_dir = self.output_dir / "per_task"
        if per_task_dir.exists():
            all_returns = []
            all_successes = []
            all_lengths = []
            for task_dir in per_task_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                metrics_path = task_dir / "metrics.json"
                if not metrics_path.exists():
                    continue
                try:
                    with open(metrics_path) as f:
                        task_data = json.load(f)
                    agg = task_data.get("aggregate_metrics", {})
                    if "mean_return" in agg:
                        all_returns.append(agg["mean_return"])
                    if "success_rate" in agg:
                        all_successes.append(agg["success_rate"])
                    if "mean_length" in agg:
                        all_lengths.append(agg["mean_length"])
                except (json.JSONDecodeError, OSError):
                    continue

            if all_returns:
                merged_summary["mean_return"] = float(np.mean(all_returns))
                merged_summary["std_return"] = float(np.std(all_returns))
            if all_successes:
                merged_summary["success_rate"] = float(np.mean(all_successes))
            if all_lengths:
                merged_summary["mean_length"] = float(np.mean(all_lengths))

        with open(summary_path, "w") as f:
            json.dump(merged_summary, f, indent=2)

    @classmethod
    def load(cls, output_dir: str | Path) -> ExperimentResults:
        """Load results from disk."""
        output_dir = Path(output_dir)

        config = ExperimentConfig.from_yaml(output_dir / "config.yaml")

        with open(output_dir / "metadata.json") as f:
            metadata = json.load(f)

        with open(output_dir / "summary.json") as f:
            summary = json.load(f)

        # Load per-task results
        per_task_results = {}
        per_task_dir = output_dir / "per_task"
        if per_task_dir.exists():
            for task_dir in per_task_dir.iterdir():
                if task_dir.is_dir():
                    metrics_path = task_dir / "metrics.json"
                    if metrics_path.exists():
                        with open(metrics_path) as f:
                            per_task_results[task_dir.name] = json.load(f)

        return cls(config, output_dir, metadata, summary, per_task_results)


_API_BACKENDS = frozenset({"openai", "gemini"})


class ExperimentRunner:
    """Run experiments with progress tracking and crash safety."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.start_time: float | None = None
        self.end_time: float | None = None

        # Create agent if config specifies LLM/VLM
        self.agent = None
        self._agent_type = config.agent.type
        if config.agent.type in ("llm", "vlm"):
            from agentick.agents.factory import create_agent

            self.agent = create_agent(config.agent)

        # Create cost tracker for API-based agents
        self.cost_tracker = None
        if self._is_api_agent():
            from agentick.leaderboard.cost_tracker import CostTracker

            model_name = config.agent.hyperparameters.get("model", "unknown")
            self.cost_tracker = CostTracker(model_name)

    def _is_api_agent(self) -> bool:
        """Check if the agent uses an API backend."""
        if self.config.agent.type not in ("llm", "vlm"):
            return False
        backend = self.config.agent.hyperparameters.get("backend", "openai")
        return backend in _API_BACKENDS

    def run(
        self,
        resume_from: str | Path | None = None,
        n_parallel: int = 1,
        output_dir: str | Path | None = None,
    ) -> ExperimentResults:
        """
        Run the experiment with crash-safe checkpoint support.

        Args:
            resume_from: Path to previous run directory to resume from
            n_parallel: Number of tasks to run in parallel (1 = sequential)

        Returns:
            ExperimentResults with all data
        """
        # LLM/VLM agents are not picklable -- force sequential execution
        # (vLLM batching is handled within _run_task via BatchedEpisodeRunner)
        if self.agent is not None and n_parallel > 1:
            print("Note: LLM/VLM agents require sequential execution, setting n_parallel=1")
            n_parallel = 1

        # Check for resume
        if resume_from:
            checkpoint = self._load_checkpoint(Path(resume_from))
            if checkpoint:
                print(f"Resuming from checkpoint: {resume_from}")
                return self._resume_experiment(checkpoint)

        self.start_time = time.time()

        # Create output directory
        if output_dir:
            output_dir = Path(output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.config.output_dir) / f"{self.config.name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect metadata
        metadata = self._collect_metadata()
        metadata["start_time"] = self.start_time

        # Record agent info in metadata
        if self.agent is not None:
            metadata["agent_name"] = self.agent.name
            metadata["agent_type"] = self.config.agent.type
            metadata["observation_modes"] = self.agent.observation_modes

        # Resolve task list
        task_names = self._resolve_tasks()

        # Seeds are now generated per-task-difficulty via generate_task_seeds()
        # If explicit seeds are provided, use those (legacy support)
        seeds = self.config.seeds

        # Save initial checkpoint
        checkpoint_data = {
            "output_dir": str(output_dir),
            "metadata": metadata,
            "seeds": seeds,
            "task_names": task_names,
            "completed_tasks": [],
            "per_task_results": {},
        }
        self._save_checkpoint(output_dir, checkpoint_data)

        # Run experiments
        print(f"Running experiment: {self.config.name}")
        print(f"Tasks: {len(task_names)}, Difficulties: {len(self.config.difficulties)}")
        print(
            f"Seeds: {self.config.n_seeds}, Episodes per task/difficulty: {self.config.n_episodes}"
        )
        total_episodes = (
            len(task_names)
            * len(self.config.difficulties)
            * self.config.n_seeds
            * self.config.n_episodes
        )
        print(f"Total episodes: {total_episodes}")

        per_task_results = {}

        # Use parallel execution if requested
        if n_parallel > 1:
            # Parallel execution with multiprocessing
            print(f"Using {n_parallel} parallel workers")

            with mp.Pool(processes=n_parallel) as pool:
                # Prepare worker arguments
                worker_args = [
                    (task_name, self.config, seeds, output_dir) for task_name in task_names
                ]

                # Run tasks in parallel with progress bar
                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                ) as progress:
                    main_task = progress.add_task(
                        f"[cyan]Running {self.config.name}", total=len(task_names)
                    )

                    # Use imap for progress updates
                    for task_name, task_results in pool.imap(_run_task_worker, worker_args):
                        per_task_results[task_name] = task_results
                        progress.update(main_task, advance=1)

                        # Save checkpoint after each task completes
                        checkpoint_data["completed_tasks"].append(task_name)
                        checkpoint_data["per_task_results"] = per_task_results
                        self._save_checkpoint(output_dir, checkpoint_data)
        else:
            # Sequential execution
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                main_task = progress.add_task(
                    f"[cyan]Running {self.config.name}", total=len(task_names)
                )

                for task_name in task_names:
                    task_results = self._run_task(task_name, seeds, output_dir, progress, main_task)
                    per_task_results[task_name] = task_results
                    progress.update(main_task, advance=1)

                    # Save checkpoint after each task
                    checkpoint_data["completed_tasks"].append(task_name)
                    checkpoint_data["per_task_results"] = per_task_results
                    self._save_checkpoint(output_dir, checkpoint_data)

        self.end_time = time.time()

        # Compute summary metrics
        summary = self._compute_summary(per_task_results)
        summary["total_time_seconds"] = self.end_time - self.start_time
        summary["total_episodes"] = total_episodes

        # Include aggregate agent stats
        if self.agent is not None:
            summary["agent_stats"] = self.agent.get_stats()

        # Create results object
        results = ExperimentResults(
            config=self.config,
            output_dir=output_dir,
            metadata=metadata,
            summary=summary,
            per_task_results=per_task_results,
        )

        # Save results
        results.save()

        # Clean up checkpoint file on successful completion
        checkpoint_path = output_dir / ".checkpoint.json"
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        # Auto-generate plots
        print("\nGenerating visualizations...")
        try:
            from agentick.visualization.experiment_plots import ExperimentPlotter

            plotter = ExperimentPlotter(output_dir)
            plotter.plot_all()
            print(f"  Figures saved to: {plotter.figures_dir}")
        except Exception as e:
            print(f"  Warning: Failed to generate plots: {e}")

        # Print cost report for API agents
        if self.cost_tracker is not None:
            self.cost_tracker.print_report()

        print(f"\n✓ Experiment complete: {output_dir}")
        print(f"  Mean return: {summary.get('mean_return', 0):.3f}")
        print(f"  Success rate: {summary.get('success_rate', 0):.2%}")
        print("\nOutput locations:")
        print(f"  Results: {output_dir / 'summary.json'}")
        print(f"  Figures: {output_dir / 'figures'}")
        if (output_dir / "videos").exists():
            print(f"  Videos: {output_dir / 'videos'}")
        print("\nTo explore results interactively:")
        print("  jupyter notebook examples/notebooks/02_analyze_experiment.ipynb")

        return results

    def _collect_metadata(self) -> dict[str, Any]:
        """Collect metadata about the run."""
        import platform as _platform

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "config_name": self.config.name,
        }

        # Get git hash
        try:
            git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            metadata["git_hash"] = git_hash
        except Exception:
            metadata["git_hash"] = None

        # Get package version
        try:
            import agentick

            metadata["agentick_version"] = agentick.__version__
        except Exception:
            metadata["agentick_version"] = "unknown"

        # System info
        metadata["python_version"] = _platform.python_version()
        metadata["platform"] = _platform.platform()
        metadata["cpu_count"] = os.cpu_count()

        # Agent info
        hp = self.config.agent.hyperparameters
        metadata["agent_type"] = self.config.agent.type
        metadata["agent_name"] = hp.get("agent_name", self.config.name)
        metadata["model"] = hp.get("model", None)
        metadata["backend"] = hp.get("backend", None)
        metadata["observation_modes"] = hp.get("observation_modes", [])
        metadata["harness"] = hp.get("harness", None)

        return metadata

    def _save_checkpoint(self, output_dir: Path, checkpoint_data: dict[str, Any]) -> None:
        """Save checkpoint for crash recovery."""
        checkpoint_path = output_dir / ".checkpoint.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

    def _load_checkpoint(self, output_dir: Path) -> dict[str, Any] | None:
        """Load checkpoint if exists."""
        checkpoint_path = output_dir / ".checkpoint.json"
        if checkpoint_path.exists():
            with open(checkpoint_path) as f:
                return json.load(f)
        return None

    def _resume_experiment(self, checkpoint: dict[str, Any]) -> ExperimentResults:
        """Resume experiment from checkpoint."""
        output_dir = Path(checkpoint["output_dir"])
        metadata = checkpoint["metadata"]
        seeds = checkpoint["seeds"]
        task_names = checkpoint["task_names"]
        completed_tasks = set(checkpoint["completed_tasks"])
        per_task_results = checkpoint["per_task_results"]

        self.start_time = metadata["start_time"]

        # Resume from where we left off
        remaining_tasks = [t for t in task_names if t not in completed_tasks]

        if remaining_tasks:
            print(f"Resuming: {len(completed_tasks)} tasks done, {len(remaining_tasks)} remaining")

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                main_task = progress.add_task(
                    f"[cyan]Resuming {self.config.name}",
                    total=len(task_names),
                    completed=len(completed_tasks),
                )

                for task_name in remaining_tasks:
                    task_results = self._run_task(task_name, seeds, output_dir, progress, main_task)
                    per_task_results[task_name] = task_results
                    progress.update(main_task, advance=1)

                    # Save checkpoint after each task
                    completed_tasks.add(task_name)
                    checkpoint["completed_tasks"] = list(completed_tasks)
                    checkpoint["per_task_results"] = per_task_results
                    self._save_checkpoint(output_dir, checkpoint)

        self.end_time = time.time()

        # Compute summary metrics
        total_episodes = (
            len(task_names)
            * len(self.config.difficulties)
            * self.config.n_seeds
            * self.config.n_episodes
        )
        summary = self._compute_summary(per_task_results)
        summary["total_time_seconds"] = self.end_time - self.start_time
        summary["total_episodes"] = total_episodes

        # Create results object
        results = ExperimentResults(
            config=self.config,
            output_dir=output_dir,
            metadata=metadata,
            summary=summary,
            per_task_results=per_task_results,
        )

        # Save results
        results.save()

        # Clean up checkpoint
        checkpoint_path = output_dir / ".checkpoint.json"
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        print(f"\n✓ Experiment complete: {output_dir}")
        print(f"  Mean return: {summary.get('mean_return', 0):.3f}")
        print(f"  Success rate: {summary.get('success_rate', 0):.2%}")

        return results

    def _resolve_tasks(self) -> list[str]:
        """Resolve task names from config."""
        if isinstance(self.config.tasks, list):
            return self.config.tasks

        # Handle suite names
        suite_name = self.config.tasks

        if suite_name == "full":
            from agentick.tasks.registry import list_tasks

            return list_tasks()

        # Map suite names to task lists from the suites module
        from agentick.leaderboard.suites import (
            GENERALIZATION_TASKS,
            MEMORY_TASKS,
            MULTIAGENT_TASKS,
            NAVIGATION_TASKS,
            PLANNING_TASKS,
            REASONING_TASKS,
        )

        suite_map = {
            "navigation": NAVIGATION_TASKS,
            "planning": PLANNING_TASKS,
            "reasoning": REASONING_TASKS,
            "memory": MEMORY_TASKS,
            "generalization": GENERALIZATION_TASKS,
            "multi_agent": MULTIAGENT_TASKS,
        }

        if suite_name in suite_map:
            return suite_map[suite_name]

        # Assume it's a single task name
        return [suite_name]

    def _run_task(
        self,
        task_name: str,
        seeds: list[int] | None,
        output_dir: Path,
        progress: Progress,
        main_task_id: Any,
    ) -> dict[str, Any]:
        """Run all episodes for a single task."""
        from agentick.leaderboard.seeds import generate_task_seeds

        task_results = {
            "task_name": task_name,
            "per_difficulty": {},
        }

        for difficulty in self.config.difficulties:
            # Per-task-difficulty seeds (unless explicit seeds provided)
            if seeds is not None:
                diff_seeds = seeds
            else:
                diff_seeds = list(generate_task_seeds(
                    task_name, difficulty, self.config.split, self.config.n_seeds
                ))
            difficulty_results = {
                "difficulty": difficulty,
                "episodes": [],
                "metrics": {},
            }

            # Create episode directory
            episodes_dir = output_dir / "per_task" / task_name / "episodes"
            episodes_dir.mkdir(parents=True, exist_ok=True)

            # Determine render mode: agent needs take priority.
            # For multimodal agents, prefer rgb_array as the primary env render mode
            # since text modes can be obtained via render_in_mode().
            if self.agent is not None:
                if "rgb_array" in self.agent.observation_modes:
                    render_mode = "rgb_array"
                else:
                    render_mode = self.agent.observation_modes[0]
            elif self.config.render_modes:
                render_mode = self.config.render_modes[0]
            else:
                render_mode = None

            # For oracle/random, use a simple render mode
            if render_mode is None and self._agent_type in ("oracle", "random"):
                render_mode = "ascii"  # lightweight for non-visual agents

            # Use batched execution for any backend with a real generate_batch()
            use_batched = (
                self.agent is not None
                and type(self.agent.backend).generate_batch
                is not ModelBackend.generate_batch
            )

            if use_batched:
                # Batched execution via BatchedEpisodeRunner
                self._run_episodes_batched(
                    task_name,
                    difficulty,
                    diff_seeds,
                    render_mode,
                    episodes_dir,
                    None,
                    difficulty_results,
                )
            else:
                # Sequential execution
                for seed_idx, seed in enumerate(diff_seeds):
                    for ep_idx in range(self.config.n_episodes):
                        # Create environment
                        import agentick

                        env = agentick.make(
                            task_name,
                            difficulty=difficulty,
                            render_mode=render_mode,
                        )

                        # Run episode
                        episode_data = self._run_episode(
                            env,
                            seed,
                            seed_idx,
                            ep_idx,
                            episodes_dir,
                            None,
                            task_name=task_name,
                            difficulty=difficulty,
                        )
                        difficulty_results["episodes"].append(episode_data)

                        env.close()

            # Compute metrics for this difficulty
            difficulty_results["metrics"] = self._compute_metrics(difficulty_results["episodes"])

            # Record concatenated video (5 random seeds replayed, no extra LLM calls)
            if self.config.record_videos:
                video_dir = output_dir / "videos" / task_name / difficulty
                self._record_concatenated_video(
                    task_name, difficulty, difficulty_results["episodes"], video_dir,
                )

            task_results["per_difficulty"][difficulty] = difficulty_results

        # Compute aggregate task metrics
        all_episodes = []
        for diff_results in task_results["per_difficulty"].values():
            all_episodes.extend(diff_results["episodes"])

        task_results["aggregate_metrics"] = self._compute_metrics(all_episodes)

        return task_results

    def _run_episodes_batched(
        self,
        task_name: str,
        difficulty: str,
        seeds: list[int],
        render_mode: str | None,
        episodes_dir: Path,
        video_dir: Path | None,
        difficulty_results: dict[str, Any],
    ) -> None:
        """Run episodes in batches using BatchedEpisodeRunner."""
        from agentick.agents.harness import HARNESS_REGISTRY
        from agentick.experiments.batched_runner import BatchedEpisodeRunner

        assert self.agent is not None

        hp = self.config.agent.hyperparameters
        harness_name = hp.get("harness", "markovian_zero_shot")
        harness_cls = HARNESS_REGISTRY[harness_name]

        # Only pass kwargs that the harness class's __init__ actually accepts
        import inspect

        harness_kwargs: dict[str, Any] = {}
        try:
            sig = inspect.signature(harness_cls.__init__)
            valid_params = set(sig.parameters.keys()) - {"self"}
        except (ValueError, TypeError):
            valid_params = set()
        for key in ("max_context_tokens", "diff_mode", "max_response_chars", "max_tokens"):
            if key in hp and key in valid_params:
                harness_kwargs[key] = hp[key]

        runner = BatchedEpisodeRunner(
            backend=self.agent.backend,
            harness_cls=harness_cls,
            harness_kwargs=harness_kwargs,
            obs_modes=self.agent.observation_modes,
            cost_tracker=self.cost_tracker,
        )

        # Batch ALL seeds x episodes together — vLLM handles them in one pass
        import agentick

        envs = []
        batch_seeds = []
        batch_seed_indices = []
        batch_ep_indices = []

        for seed_idx, seed in enumerate(seeds):
            for ep_idx in range(self.config.n_episodes):
                env = agentick.make(
                    task_name,
                    difficulty=difficulty,
                    render_mode=render_mode,
                )
                envs.append(env)
                batch_seeds.append(seed)
                batch_seed_indices.append(seed_idx)
                batch_ep_indices.append(ep_idx)

        n_total = len(envs)
        print(
            f"\n  Batched {n_total} episodes "
            f"({len(seeds)} seeds x {self.config.n_episodes} eps) "
            f"| {task_name} | {difficulty}"
        )

        batch_results = runner.run_batch(
            envs=envs,
            seeds=batch_seeds,
            seed_indices=batch_seed_indices,
            episode_indices=batch_ep_indices,
        )

        for i, result in enumerate(batch_results):
            episode_data: dict[str, Any] = {
                "seed": result.seed,
                "episode_idx": result.episode_idx,
                "return": result.total_reward,
                "length": result.episode_length,
                "success": result.success,
                "agent_stats": result.agent_stats,
                "steps": result.steps,
            }
            difficulty_results["episodes"].append(episode_data)

            # Save trajectory
            if self.config.record_trajectories:
                traj = {
                    "seed": result.seed,
                    "seed_idx": result.seed_idx,
                    "episode_idx": result.episode_idx,
                    "steps": result.steps,
                    "total_reward": result.total_reward,
                    "episode_length": result.episode_length,
                    "success": result.success,
                }
                ep_file = (
                    episodes_dir
                    / f"seed_{result.seed_idx}_ep_{result.episode_idx}.json"
                )
                with open(ep_file, "w") as f:
                    json.dump(traj, f, indent=2)

            # Save agent trace (LLM observations, responses, reasoning)
            if result.call_log:
                trace = {
                    "metadata": {
                        "task": task_name,
                        "difficulty": difficulty,
                        "seed": result.seed,
                        "config_name": self.config.name,
                        "model_id": self.config.agent.hyperparameters.get(
                            "model", ""
                        ),
                        "harness": self.config.agent.hyperparameters.get(
                            "harness", ""
                        ),
                        "observation_modes": self.agent.observation_modes,
                        "success": result.success,
                        "total_reward": result.total_reward,
                        "episode_length": result.episode_length,
                        "total_tokens": result.agent_stats.get(
                            "total_tokens", 0
                        ),
                    },
                    "steps": [],
                }
                for j, log_entry in enumerate(result.call_log):
                    step_data = dict(log_entry)
                    if j < len(result.steps):
                        step_data["reward"] = result.steps[j]["reward"]
                        step_data["terminated"] = result.steps[j]["terminated"]
                        step_data["truncated"] = result.steps[j]["truncated"]
                    trace["steps"].append(step_data)

                trace_dir = episodes_dir.parent / "traces" / difficulty
                trace_dir.mkdir(parents=True, exist_ok=True)
                trace_file = (
                    trace_dir
                    / f"seed_{result.seed_idx}_ep_{result.episode_idx}.json"
                )
                with open(trace_file, "w") as f:
                    json.dump(trace, f, indent=2)

            status = "SUCCESS" if result.success else "FAIL"
            print(
                f"    ep[{i}] {status} | {result.episode_length} steps | "
                f"reward={result.total_reward:.2f}"
            )

        # Print cumulative cost (already updated per-step in BatchedEpisodeRunner)
        if self.cost_tracker is not None:
            print(f"  Cumulative API cost: ${self.cost_tracker.get_total_cost():.4f}")

        # Clean up envs
        for env in envs:
            env.close()

    def _save_video(
        self,
        frames: list[np.ndarray],
        video_dir: Path,
        seed_idx: int,
        ep_idx: int,
    ) -> None:
        """Save episode frames as video (mp4 if ffmpeg available, else gif)."""
        from agentick.experiments._video_utils import _has_ffmpeg, _save_gif, _save_mp4

        name = f"seed_{seed_idx}_ep_{ep_idx}"
        try:
            if _has_ffmpeg():
                _save_mp4(frames, video_dir / f"{name}.mp4", fps=10)
            else:
                _save_gif(frames, video_dir / f"{name}.gif", fps=10)
        except Exception as e:
            print(f"  Warning: Failed to save video {name}: {e}")

    def _record_concatenated_video(
        self,
        task_name: str,
        difficulty: str,
        episodes: list[dict[str, Any]],
        video_dir: Path,
        n_video_seeds: int = 5,
    ) -> None:
        """Record a single video concatenating replayed episodes from multiple seeds.

        Replays recorded actions on fresh envs with render_mode='rgb_array',
        so no extra LLM/API calls are needed.
        """
        import random

        import agentick
        from agentick.experiments._video_utils import _has_ffmpeg, _save_gif, _save_mp4

        # Collect episodes that have step data (with actions for replay)
        seed_to_ep: dict[int, dict[str, Any]] = {}
        for ep in episodes:
            seed = ep.get("seed")
            if seed is not None and seed not in seed_to_ep and ep.get("steps"):
                seed_to_ep[seed] = ep

        if not seed_to_ep:
            print(f"  Warning: No replayable episodes for video ({task_name}/{difficulty})")
            return

        available = list(seed_to_ep.keys())
        selected = random.sample(available, min(n_video_seeds, len(available)))

        all_frames: list[np.ndarray] = []
        separator_n = 10  # black frames between episodes

        for seed in selected:
            ep = seed_to_ep[seed]
            actions = [step["action"] for step in ep["steps"]]

            env = agentick.make(task_name, difficulty=difficulty, render_mode="rgb_array")
            obs, info = env.reset(seed=seed)

            frame = env.render()
            if isinstance(frame, np.ndarray):
                all_frames.append(frame)

            for action in actions:
                obs, reward, terminated, truncated, info = env.step(action)
                frame = env.render()
                if isinstance(frame, np.ndarray):
                    all_frames.append(frame)
                if terminated or truncated:
                    break

            env.close()

            # Add black separator frames between episodes
            if all_frames:
                h, w = all_frames[-1].shape[:2]
                sep = np.zeros((h, w, 3), dtype=np.uint8)
                all_frames.extend([sep] * separator_n)

        # Remove trailing separator
        if len(all_frames) > separator_n:
            all_frames = all_frames[:-separator_n]

        if not all_frames:
            return

        video_dir.mkdir(parents=True, exist_ok=True)
        name = f"{task_name}_{difficulty}"
        try:
            if _has_ffmpeg():
                _save_mp4(all_frames, video_dir / f"{name}.mp4", fps=10)
            else:
                _save_gif(all_frames, video_dir / f"{name}.gif", fps=10)
            print(f"  Saved video: {video_dir / name}.mp4 ({len(selected)} seeds)")
        except Exception as e:
            print(f"  Warning: Failed to save video {name}: {e}")

    def _inject_secondary_obs(self, env: Any, info: dict[str, Any]) -> None:
        """Inject secondary renderings into info for multimodal agents."""
        if self.agent is None or len(self.agent.observation_modes) <= 1:
            return
        render_mode = env.render_mode
        for mode in self.agent.observation_modes:
            if mode != render_mode and mode in ("language", "ascii", "language_structured"):
                info[f"obs_{mode}"] = env.render_in_mode(mode)

    def _run_episode(
        self,
        env: Any,
        seed: int,
        seed_idx: int,
        ep_idx: int,
        episodes_dir: Path,
        video_dir: Path | None = None,
        task_name: str = "unknown",
        difficulty: str = "unknown",
    ) -> dict[str, Any]:
        """Run a single episode."""
        is_agent = self.agent is not None

        # Create oracle agent if needed (per-episode, needs env)
        oracle_agent = None
        if self._agent_type == "oracle":
            from agentick.oracles import get_oracle
            try:
                oracle_agent = get_oracle(task_name, env)
            except ValueError:
                print(f"  Warning: No oracle for {task_name}, using random actions")

        if is_agent:
            print(
                f"\n{'=' * 60}\n"
                f"  Episode: {task_name} | {difficulty} | "
                f"seed={seed} (#{seed_idx}) ep={ep_idx}\n"
                f"{'=' * 60}"
            )

        obs, info = env.reset(seed=seed)
        self._inject_secondary_obs(env, info)

        if oracle_agent is not None:
            oracle_agent.reset(obs, info)

        # Reset agent state at episode start
        if is_agent:
            self.agent.reset()

        # Collect frames for video recording
        frames: list[np.ndarray] = []
        if video_dir is not None:
            frame = env.render_in_mode("rgb_array")
            if isinstance(frame, np.ndarray):
                frames.append(frame)

        trajectory = {
            "seed": seed,
            "seed_idx": seed_idx,
            "episode_idx": ep_idx,
            "steps": [],
            "total_reward": 0.0,
            "episode_length": 0,
            "success": False,
        }

        terminated = False
        truncated = False
        step_count = 0
        total_reward = 0.0

        while not (terminated or truncated):
            # Get action from agent or fall back to random
            if is_agent:
                action = self.agent.act(obs, info)
            elif oracle_agent is not None:
                action = oracle_agent.act(obs, info)
            else:
                action = env.action_space.sample()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            if oracle_agent is not None:
                oracle_agent.update(obs, info)
            self._inject_secondary_obs(env, info)

            # Collect video frame
            if video_dir is not None:
                frame = env.render_in_mode("rgb_array")
                if isinstance(frame, np.ndarray):
                    frames.append(frame)

            total_reward += reward
            step_count += 1

            # Per-step agent progress
            if is_agent and self.agent.call_log:
                last = self.agent.call_log[-1]
                obs_preview = last["observation"][:120].replace("\n", " | ")
                resp_preview = last["response"][:100].replace("\n", " ")
                reasoning_line = ""
                if last.get("reasoning"):
                    r = last["reasoning"][:100].replace("\n", " ")
                    reasoning_line = f"    Reasoning: {r}...\n"
                print(
                    f"  Step {step_count}: "
                    f"{last['action_name']} (={last['parsed_action']}) "
                    f"-> r={reward:.2f}  "
                    f"({last['latency']:.2f}s, "
                    f"{last['input_tokens']}+{last['output_tokens']} tok)\n"
                    f"    Obs: {obs_preview}...\n"
                    f"{reasoning_line}"
                    f"    Raw: {resp_preview}"
                )

            if self.config.record_trajectories or self.agent is not None:
                step_data = {
                    "step": step_count,
                    "action": int(action),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
                trajectory["steps"].append(step_data)

        trajectory["total_reward"] = float(total_reward)
        trajectory["episode_length"] = int(step_count)
        trajectory["success"] = bool(info.get("success", False))

        if is_agent:
            status = "SUCCESS" if trajectory["success"] else "FAIL"
            ep_latency = sum(e["latency"] for e in self.agent.call_log)
            ep_tokens = sum(e["input_tokens"] + e["output_tokens"] for e in self.agent.call_log)
            print(
                f"  --- {status} | {step_count} steps | "
                f"reward={total_reward:.2f} | "
                f"{ep_latency:.1f}s | {ep_tokens} tokens"
            )

            # Update cost tracker
            if self.cost_tracker is not None:
                for entry in self.agent.call_log:
                    self.cost_tracker.add_call(
                        input_tokens=entry.get("input_tokens", 0),
                        output_tokens=entry.get("output_tokens", 0),
                    )
                print(f"  Cumulative API cost: ${self.cost_tracker.get_total_cost():.4f}")

        # Save episode data
        if self.config.record_trajectories:
            episode_file = episodes_dir / f"diff_{difficulty}_seed_{seed_idx}_ep_{ep_idx}.json"
            with open(episode_file, "w") as f:
                json.dump(trajectory, f, indent=2)

        # Save video
        if video_dir is not None and frames:
            self._save_video(frames, video_dir, seed_idx, ep_idx)

        # Save agent trace
        if self.agent is not None and self.agent.call_log:
            trace = {
                "metadata": {
                    "task": task_name,
                    "difficulty": difficulty,
                    "seed": seed,
                    "config_name": self.config.name,
                    "model_id": self.config.agent.hyperparameters.get("model", ""),
                    "harness": self.config.agent.hyperparameters.get("harness", ""),
                    "observation_modes": self.agent.observation_modes,
                    "success": trajectory["success"],
                    "total_reward": trajectory["total_reward"],
                    "episode_length": trajectory["episode_length"],
                    "total_tokens": sum(
                        e.get("input_tokens", 0) + e.get("output_tokens", 0)
                        for e in self.agent.call_log
                    ),
                },
                "system_prompt": getattr(self.agent, "_system_prompt", ""),
                "steps": [],
            }
            for i, log_entry in enumerate(self.agent.call_log):
                step_data = dict(log_entry)
                # Merge reward/terminated/truncated from trajectory
                if i < len(trajectory["steps"]):
                    traj_step = trajectory["steps"][i]
                    step_data["reward"] = traj_step["reward"]
                    step_data["terminated"] = traj_step["terminated"]
                    step_data["truncated"] = traj_step["truncated"]
                trace["steps"].append(step_data)

            trace_dir = episodes_dir.parent / "traces" / difficulty
            trace_dir.mkdir(parents=True, exist_ok=True)
            with open(trace_dir / f"seed_{seed_idx}_ep_{ep_idx}.json", "w") as f:
                json.dump(trace, f, indent=2)

        # Build episode result
        episode_result: dict[str, Any] = {
            "seed": seed,
            "episode_idx": ep_idx,
            "return": total_reward,
            "length": step_count,
            "success": trajectory["success"],
            "steps": trajectory["steps"],
        }

        # Include agent stats if available
        if self.agent is not None:
            stats = self.agent.get_stats()
            episode_result["agent_stats"] = stats

        return episode_result

    def _compute_metrics(self, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute metrics from episodes."""
        if not episodes:
            return {}

        returns = [ep["return"] for ep in episodes]
        lengths = [ep["length"] for ep in episodes]
        successes = [ep["success"] for ep in episodes]

        metrics = {}

        if "mean_return" in self.config.metrics:
            metrics["mean_return"] = float(np.mean(returns))
        if "std_return" in self.config.metrics:
            metrics["std_return"] = float(np.std(returns))
        if "median_return" in self.config.metrics:
            metrics["median_return"] = float(np.median(returns))
        if "min_return" in self.config.metrics:
            metrics["min_return"] = float(np.min(returns))
        if "max_return" in self.config.metrics:
            metrics["max_return"] = float(np.max(returns))

        if "success_rate" in self.config.metrics:
            metrics["success_rate"] = float(np.mean(successes))

        if "mean_length" in self.config.metrics:
            metrics["mean_length"] = float(np.mean(lengths))

        # Agent-specific metrics (aggregated from episode-level agent_stats)
        agent_stats_list = [ep.get("agent_stats") for ep in episodes if ep.get("agent_stats")]
        if agent_stats_list:
            if "mean_latency" in self.config.metrics:
                latencies = [s["mean_latency"] for s in agent_stats_list if s.get("mean_latency")]
                if latencies:
                    metrics["mean_latency"] = float(np.mean(latencies))
            if "total_tokens" in self.config.metrics:
                metrics["total_tokens"] = sum(s.get("total_tokens", 0) for s in agent_stats_list)
            if "total_api_calls" in self.config.metrics:
                metrics["total_api_calls"] = sum(s.get("total_calls", 0) for s in agent_stats_list)
            if "total_cost_usd" in self.config.metrics and self.cost_tracker is not None:
                total_input = sum(s.get("total_input_tokens", 0) for s in agent_stats_list)
                total_output = sum(s.get("total_output_tokens", 0) for s in agent_stats_list)
                pricing = self.cost_tracker.pricing
                cost = (
                    (total_input / 1_000_000) * pricing["input"]
                    + (total_output / 1_000_000) * pricing["output"]
                )
                metrics["total_cost_usd"] = cost

        return metrics

    def _compute_summary(self, per_task_results: dict[str, Any]) -> dict[str, Any]:
        """Compute summary metrics across all tasks."""
        all_returns = []
        all_successes = []
        all_lengths = []

        for task_results in per_task_results.values():
            metrics = task_results.get("aggregate_metrics", {})
            if "mean_return" in metrics:
                all_returns.append(metrics["mean_return"])
            if "success_rate" in metrics:
                all_successes.append(metrics["success_rate"])
            if "mean_length" in metrics:
                all_lengths.append(metrics["mean_length"])

        summary = {}

        if all_returns:
            summary["mean_return"] = float(np.mean(all_returns))
            summary["std_return"] = float(np.std(all_returns))

        if all_successes:
            summary["success_rate"] = float(np.mean(all_successes))

        if all_lengths:
            summary["mean_length"] = float(np.mean(all_lengths))

        # Aggregate cost across all tasks
        if self.cost_tracker is not None:
            summary["total_cost_usd"] = self.cost_tracker.get_total_cost()
            summary["cost_report"] = self.cost_tracker.get_report()

        return summary


def run_experiment(config: ExperimentConfig) -> ExperimentResults:
    """
    Run an experiment from config.

    Args:
        config: Experiment configuration

    Returns:
        ExperimentResults
    """
    runner = ExperimentRunner(config)
    return runner.run()
