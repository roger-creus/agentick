"""Main evaluation engine for running agent submissions on benchmark suites."""

from __future__ import annotations

import platform
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

import agentick
from agentick.leaderboard.baselines import load_baselines
from agentick.leaderboard.cost_tracker import CostTracker
from agentick.leaderboard.integrity import compute_result_hash, verify_reproducibility
from agentick.leaderboard.result import EpisodeResult, EvaluationResult
from agentick.leaderboard.scoring import compute_score_from_results
from agentick.leaderboard.submission import SubmissionSpec
from agentick.leaderboard.suites import BenchmarkSuite, get_suite


class LeaderboardEvaluator:
    """
    Main evaluation engine for Agentick leaderboard.

    Runs agents on benchmark suites and produces evaluation results.
    """

    def __init__(
        self,
        verbose: bool = True,
        save_trajectories: bool = False,
    ):
        """
        Initialize evaluator.

        Args:
            verbose: Whether to show progress bars
            save_trajectories: Whether to save full episode trajectories
        """
        self.verbose = verbose
        self.save_trajectories = save_trajectories

    def _load_agent(self, submission: SubmissionSpec) -> Any:
        """
        Load agent from submission spec.

        Args:
            submission: Submission specification

        Returns:
            Agent instance
        """
        if submission.agent_type == "api":
            from agentick.leaderboard.adapters.api_adapter import APIAgent

            return APIAgent(**submission.config, observation_mode=submission.observation_mode)

        elif submission.agent_type == "code":
            from agentick.leaderboard.adapters.code_adapter import CodeAgent

            return CodeAgent(**submission.config)

        elif submission.agent_type == "huggingface":
            from agentick.leaderboard.adapters.huggingface_adapter import HuggingFaceAgent

            return HuggingFaceAgent(
                **submission.config, observation_mode=submission.observation_mode
            )

        elif submission.agent_type == "local_weights":
            from agentick.leaderboard.adapters.local_weights_adapter import LocalWeightsAgent

            return LocalWeightsAgent(
                **submission.config, observation_mode=submission.observation_mode
            )

        elif submission.agent_type == "docker":
            from agentick.leaderboard.adapters.docker_adapter import DockerAgent

            return DockerAgent(**submission.config)

        elif submission.agent_type == "git_repo":
            from agentick.leaderboard.adapters.git_adapter import GitRepoAgent

            return GitRepoAgent(**submission.config)

        else:
            raise ValueError(f"Unknown agent type: {submission.agent_type}")

    def evaluate(
        self,
        submission: SubmissionSpec,
        suite: str | BenchmarkSuite,
        output_dir: str | Path | None = None,
        verify_reproducibility_flag: bool = False,
    ) -> EvaluationResult:
        """
        Run full evaluation on a benchmark suite.

        Args:
            submission: Agent submission specification
            suite: Suite name or BenchmarkSuite instance
            output_dir: Directory to save results
            verify_reproducibility_flag: Whether to verify reproducibility

        Returns:
            EvaluationResult
        """
        # Get suite
        if isinstance(suite, str):
            suite = get_suite(suite)

        if self.verbose:
            print(f"\n=== Evaluating {submission.agent_name} on {suite.display_name} ===\n")

        start_time = datetime.now()

        # Load agent
        agent = self._load_agent(submission)

        # Initialize cost tracker for API agents
        cost_tracker = None
        if submission.agent_type == "api":
            model_name = submission.config.get("model", "unknown")
            cost_tracker = CostTracker(model_name)

        # Run evaluation
        episodes = []
        task_results = {}

        # Progress bar over tasks
        tasks_iter = tqdm(suite.tasks, desc="Tasks") if self.verbose else suite.tasks

        for task_name in tasks_iter:
            if self.verbose:
                print(f"\n  Task: {task_name}")

            # Run this task on all seeds
            task_episodes = []

            seeds_iter = (
                tqdm(suite.eval_seeds, desc="  Seeds", leave=False)
                if self.verbose
                else suite.eval_seeds
            )

            for seed in seeds_iter:
                # Create environment
                env = agentick.make(
                    task_name,
                    difficulty=suite.difficulty,
                    render_mode=submission.observation_mode,
                )

                # Override max_steps if suite specifies
                if suite.max_steps_override:
                    env.max_steps = suite.max_steps_override

                # Reset
                obs, info = env.reset(seed=seed)
                agent.reset()

                # Run episode
                done = False
                episode_return = 0.0
                steps = 0
                trajectory = [] if self.save_trajectories else None

                while not done:
                    # Get action
                    action = agent.act(obs, info)

                    # Save trajectory
                    if self.save_trajectories:
                        trajectory.append(
                            {
                                "step": steps,
                                "observation": str(obs),
                                "action": action,
                            }
                        )

                    # Step
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_return += reward
                    steps += 1
                    done = terminated or truncated

                # Record episode
                episode = EpisodeResult(
                    task_name=task_name,
                    difficulty=suite.difficulty,
                    seed=seed,
                    episode_return=episode_return,
                    steps=steps,
                    success=info.get("success", False),
                    trajectory=trajectory,
                )

                task_episodes.append(episode)
                episodes.append(episode)

            # Collect task-level results
            task_results[task_name] = {
                "difficulty": suite.difficulty,
                "episode_returns": [ep.episode_return for ep in task_episodes],
                "success_flags": [ep.success for ep in task_episodes],
            }

        # Collect API stats if applicable
        total_api_calls = None
        total_tokens = None
        estimated_cost = None

        if cost_tracker and hasattr(agent, "get_statistics"):
            stats = agent.get_statistics()
            total_api_calls = stats.get(
                "total_calls", agent.total_calls if hasattr(agent, "total_calls") else None
            )
            total_tokens = stats.get(
                "total_tokens", agent.total_tokens if hasattr(agent, "total_tokens") else None
            )

            if total_tokens:
                cost_tracker.add_call(total_tokens=total_tokens)
                estimated_cost = cost_tracker.get_total_cost()

        # Load baselines
        baselines_path = Path("leaderboard_data/baselines") / f"{suite.name}_baselines.json"
        if baselines_path.exists():
            baselines = load_baselines(baselines_path)
        else:
            # Use dummy baselines
            baselines = {
                task_name: {"random_baseline": 0.0, "optimal_return": 1.0}
                for task_name in suite.tasks
            }

        # Compute scores
        aggregate_score = compute_score_from_results(task_results, baselines)

        # Verify reproducibility if requested
        reproducibility_verified = False
        reproducibility_delta = None

        # Build result
        end_time = datetime.now()
        wall_time = (end_time - start_time).total_seconds()

        result = EvaluationResult(
            submission=submission,
            suite_name=suite.name,
            suite_version=suite.version,
            suite_hash=suite.compute_hash(),
            started_at=start_time,
            completed_at=end_time,
            wall_time_seconds=wall_time,
            agentick_score=aggregate_score.agentick_score,
            agentick_score_ci=aggregate_score.agentick_score_ci,
            per_capability={k: v.model_dump() for k, v in aggregate_score.per_capability.items()},
            per_task={k: v.model_dump() for k, v in aggregate_score.per_task.items()},
            episodes=episodes,
            reproducibility_verified=reproducibility_verified,
            reproducibility_delta=reproducibility_delta,
            evaluator_version=agentick.__version__,
            result_hash="",  # Will be set below
            hardware_info={
                "platform": platform.system(),
                "python_version": platform.python_version(),
            },
            total_api_calls=total_api_calls,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost,
        )

        # Compute result hash
        result.result_hash = compute_result_hash(result)

        # Verify reproducibility if requested
        if verify_reproducibility_flag:
            is_reproducible, delta = verify_reproducibility(result)
            result.reproducibility_verified = is_reproducible
            result.reproducibility_delta = delta

        # Save result
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            result_file = output_dir / f"{submission.agent_name}_{suite.name}.json"
            result.to_json(result_file)

            if self.verbose:
                print(f"\n✓ Results saved to {result_file}")

        # Print summary
        if self.verbose:
            print(f"\n{result.get_summary()}")

        return result
