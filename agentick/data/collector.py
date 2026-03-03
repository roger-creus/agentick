"""Trajectory collection and dataset generation.

Provides two levels of API:
- ``TrajectoryCollector``: low-level step-by-step recording
- ``DataCollector``: high-level env+agent wrapper with multi-modality,
  one-call HuggingFace export, and Hub push
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np


@dataclass
class Trajectory:
    """A complete episode trajectory."""

    observations: list[Any] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    infos: list[dict] = field(default_factory=list)
    total_reward: float = 0.0
    length: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(
        self,
        observation: Any,
        action: int,
        reward: float,
        done: bool,
        info: dict,
    ):
        """Add a step to the trajectory."""
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.infos.append(info)
        self.total_reward += reward
        self.length += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert trajectory to dictionary."""
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "infos": self.infos,
            "total_reward": self.total_reward,
            "length": self.length,
            "metadata": self.metadata,
        }


class TrajectoryCollector:
    """
    Collect trajectories from environment rollouts.

    Supports multiple observation modalities and efficient buffering.

    Example:
        >>> collector = TrajectoryCollector(buffer_size=1000)
        >>>
        >>> # Collect trajectories
        >>> for episode in range(100):
        ...     obs, info = env.reset()
        ...     collector.start_episode(metadata={"episode": episode})
        ...
        ...     done = False
        ...     while not done:
        ...         action = policy(obs)
        ...         obs, reward, done, truncated, info = env.step(action)
        ...         collector.add_step(obs, action, reward, done or truncated, info)
        ...
        ...     collector.end_episode()
        >>>
        >>> # Export trajectories
        >>> trajectories = collector.get_trajectories()
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        collect_observations: bool = True,
    ):
        """
        Initialize trajectory collector.

        Args:
            buffer_size: Maximum number of trajectories to keep in memory
            collect_observations: Whether to collect observations (can be large)
        """
        self.buffer_size = buffer_size
        self.collect_observations = collect_observations

        self.trajectories: list[Trajectory] = []
        self.current_trajectory: Trajectory | None = None

        # Statistics
        self.total_episodes = 0
        self.total_steps = 0

    def start_episode(self, metadata: dict[str, Any] | None = None):
        """
        Start collecting a new episode.

        Args:
            metadata: Optional metadata to attach to episode
        """
        self.current_trajectory = Trajectory(metadata=metadata or {})

    def add_step(
        self,
        observation: Any,
        action: int,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ):
        """
        Add a step to the current trajectory.

        Args:
            observation: Environment observation
            action: Action taken
            reward: Reward received
            done: Whether episode ended
            info: Info dict
        """
        if self.current_trajectory is None:
            raise ValueError("Must call start_episode() before add_step()")

        # Optionally skip observations to save memory
        obs_to_store = observation if self.collect_observations else None

        self.current_trajectory.add_step(obs_to_store, action, reward, done, info)
        self.total_steps += 1

    def end_episode(self):
        """Finish current episode and add to buffer."""
        if self.current_trajectory is None:
            return

        # Add to trajectories buffer
        self.trajectories.append(self.current_trajectory)
        self.total_episodes += 1

        # Enforce buffer size limit
        if len(self.trajectories) > self.buffer_size:
            self.trajectories.pop(0)

        self.current_trajectory = None

    def get_trajectories(
        self,
        min_reward: float | None = None,
        max_length: int | None = None,
    ) -> list[Trajectory]:
        """
        Get collected trajectories with optional filtering.

        Args:
            min_reward: Minimum total reward to include
            max_length: Maximum episode length to include

        Returns:
            List of trajectories matching criteria
        """
        trajectories = self.trajectories

        if min_reward is not None:
            trajectories = [t for t in trajectories if t.total_reward >= min_reward]

        if max_length is not None:
            trajectories = [t for t in trajectories if t.length <= max_length]

        return trajectories

    def get_stats(self) -> dict[str, Any]:
        """Get collection statistics."""
        if len(self.trajectories) == 0:
            return {
                "total_episodes": 0,
                "total_steps": 0,
                "mean_reward": 0.0,
                "mean_length": 0.0,
            }

        rewards = [t.total_reward for t in self.trajectories]
        lengths = [t.length for t in self.trajectories]

        return {
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "buffered_episodes": len(self.trajectories),
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_length": np.mean(lengths),
            "std_length": np.std(lengths),
        }

    def save(self, path: str | Path):
        """
        Save trajectories to disk.

        Args:
            path: File path (will save as .npz)
        """
        import numpy as np

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to numpy-friendly format
        data = {
            "trajectories": [t.to_dict() for t in self.trajectories],
            "stats": self.get_stats(),
        }

        np.savez_compressed(path, **data)

    def save_json(self, path: str | Path):
        """
        Save trajectories to JSON format.

        Args:
            path: File path (will save as .json)
        """
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert observations to serializable format
        def make_serializable(obj):
            """Convert numpy arrays and other non-serializable objects to lists."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(make_serializable(item) for item in obj)
            return obj

        # Convert trajectories to JSON-serializable format
        trajectories_data = []
        for traj in self.trajectories:
            traj_dict = traj.to_dict()
            traj_dict = make_serializable(traj_dict)
            trajectories_data.append(traj_dict)

        data = {
            "trajectories": trajectories_data,
            "stats": make_serializable(self.get_stats()),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str | Path):
        """
        Load trajectories from disk.

        Args:
            path: File path
        """
        data = np.load(path, allow_pickle=True)
        trajectories_data = data["trajectories"]

        self.trajectories = []
        for traj_dict in trajectories_data:
            traj = Trajectory(**traj_dict)
            self.trajectories.append(traj)

    def clear(self):
        """Clear all collected trajectories."""
        self.trajectories.clear()
        self.current_trajectory = None
        self.total_episodes = 0
        self.total_steps = 0


# ---------------------------------------------------------------------------
# High-level DataCollector + CollectedDataset
# ---------------------------------------------------------------------------


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_make_serializable(i) for i in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted
    return obj


@runtime_checkable
class AgentProtocol(Protocol):
    """Minimal interface for agents used by DataCollector."""

    def act(self, obs: Any, info: dict[str, Any]) -> int: ...


@dataclass
class MultiModalStep:
    """A single step with multi-modal observations."""

    observations: dict[str, Any]
    action: int
    reward: float
    done: bool
    info: dict[str, Any]
    reasoning: str | None = None


@dataclass
class MultiModalTrajectory:
    """Full episode with multi-modal observations."""

    steps: list[MultiModalStep] = field(default_factory=list)
    total_reward: float = 0.0
    success: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def actions(self) -> list[int]:
        return [s.action for s in self.steps]

    @property
    def rewards(self) -> list[float]:
        return [s.reward for s in self.steps]


class CollectedDataset:
    """Dataset of multi-modal trajectories with export helpers.

    Returned by :meth:`DataCollector.collect`.  Provides ``save``,
    ``export_to_huggingface``, and ``push_to_hub`` methods.
    """

    def __init__(
        self,
        trajectories: list[MultiModalTrajectory],
        task_id: str,
        modalities: list[str],
    ) -> None:
        self.trajectories = trajectories
        self.task_id = task_id
        self.modalities = modalities

    def __len__(self) -> int:
        return len(self.trajectories)

    @property
    def num_steps(self) -> int:
        return sum(t.length for t in self.trajectories)

    @property
    def success_rate(self) -> float:
        if not self.trajectories:
            return 0.0
        return sum(1 for t in self.trajectories if t.success) / len(self.trajectories)

    def stats(self) -> dict[str, Any]:
        rewards = [t.total_reward for t in self.trajectories]
        lengths = [t.length for t in self.trajectories]
        return {
            "num_episodes": len(self.trajectories),
            "num_steps": self.num_steps,
            "success_rate": self.success_rate,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        }

    # -- Persistence ----------------------------------------------------------

    def save(self, path: str | Path, save_pixels: bool = False) -> Path:
        """Save dataset to a directory as JSON files.

        Args:
            path: Directory to write files.
            save_pixels: If ``True``, include raw pixel arrays (rgb_array).
                This can produce very large files.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        meta = {
            "task_id": self.task_id,
            "modalities": self.modalities,
            "stats": _make_serializable(self.stats()),
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Save trajectories as JSONL
        with open(path / "trajectories.jsonl", "w") as f:
            for traj in self.trajectories:
                record = self._traj_to_record(traj, skip_pixels=not save_pixels)
                f.write(json.dumps(record) + "\n")

        return path

    @staticmethod
    def load(path: str | Path) -> CollectedDataset:
        """Load a dataset previously saved with :meth:`save`."""
        path = Path(path)
        with open(path / "meta.json") as f:
            meta = json.load(f)

        trajectories: list[MultiModalTrajectory] = []
        with open(path / "trajectories.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                steps = []
                for s in rec["steps"]:
                    steps.append(MultiModalStep(
                        observations=s["observations"],
                        action=s["action"],
                        reward=s["reward"],
                        done=s["done"],
                        info=s.get("info", {}),
                        reasoning=s.get("reasoning"),
                    ))
                trajectories.append(MultiModalTrajectory(
                    steps=steps,
                    total_reward=rec["total_reward"],
                    success=rec["success"],
                    metadata=rec.get("metadata", {}),
                ))

        return CollectedDataset(
            trajectories=trajectories,
            task_id=meta["task_id"],
            modalities=meta["modalities"],
        )

    # -- HuggingFace export ---------------------------------------------------

    def export_to_huggingface(
        self,
        output_path: str | Path,
        format: Literal["conversation", "decision", "trajectory"] = "conversation",
        observation_mode: str | None = None,
    ) -> Path:
        """Export to HuggingFace Datasets format.

        Args:
            output_path: Directory to write the dataset.
            format: ``"conversation"`` for chat SFT, ``"decision"`` for BC
                (obs, action, reward, next_obs, done) tuples,
                ``"trajectory"`` for raw episodes.
            observation_mode: Which modality to use for observations.
                Defaults to first text modality or first available.

        Returns:
            Path to the saved dataset.
        """
        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError(
                "datasets package required: pip install datasets"
            )

        output_path = Path(output_path)

        if observation_mode is None:
            for m in ("language", "ascii", "language_structured"):
                if m in self.modalities:
                    observation_mode = m
                    break
            if observation_mode is None:
                observation_mode = self.modalities[0]

        if format == "conversation":
            rows = self._to_conversation_rows(observation_mode)
        elif format == "decision":
            rows = self._to_decision_rows(observation_mode)
        else:
            rows = self._to_trajectory_rows(observation_mode)

        ds = Dataset.from_list(rows)
        ds.save_to_disk(str(output_path))
        return output_path

    def push_to_hub(
        self,
        repo_id: str,
        format: Literal["conversation", "decision", "trajectory"] = "conversation",
        observation_mode: str | None = None,
        private: bool = False,
    ) -> str:
        """Push dataset directly to HuggingFace Hub.

        Args:
            repo_id: HuggingFace repo (e.g. ``"user/agentick-oracle-data"``).
            format: Export format.
            observation_mode: Observation modality.
            private: Create private repo.

        Returns:
            URL of the uploaded dataset.
        """
        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError(
                "datasets package required: pip install datasets"
            )

        if observation_mode is None:
            for m in ("language", "ascii", "language_structured"):
                if m in self.modalities:
                    observation_mode = m
                    break
            if observation_mode is None:
                observation_mode = self.modalities[0]

        if format == "conversation":
            rows = self._to_conversation_rows(observation_mode)
        elif format == "decision":
            rows = self._to_decision_rows(observation_mode)
        else:
            rows = self._to_trajectory_rows(observation_mode)

        ds = Dataset.from_list(rows)
        ds.push_to_hub(repo_id, private=private)
        return f"https://huggingface.co/datasets/{repo_id}"

    # -- Internal row builders ------------------------------------------------

    def _traj_to_record(
        self, traj: MultiModalTrajectory, skip_pixels: bool = False,
    ) -> dict[str, Any]:
        steps_data = []
        for s in traj.steps:
            obs = {}
            for k, v in s.observations.items():
                if skip_pixels and k in ("rgb_array", "rgb_array_flat", "rgb_array_2d"):
                    continue
                obs[k] = _make_serializable(v)
            step_rec: dict[str, Any] = {
                "observations": obs,
                "action": int(s.action),
                "reward": float(s.reward),
                "done": s.done,
            }
            if s.reasoning is not None:
                step_rec["reasoning"] = s.reasoning
            steps_data.append(step_rec)
        return {
            "steps": steps_data,
            "total_reward": float(traj.total_reward),
            "success": traj.success,
            "metadata": _make_serializable(traj.metadata),
        }

    def _to_conversation_rows(self, obs_mode: str) -> list[dict[str, Any]]:
        """Build chat-format rows for LLM SFT."""
        rows = []
        system_msg = (
            f"You are an expert agent playing the {self.task_id} task. "
            "Given the observation, respond with the best action number."
        )
        for traj in self.trajectories:
            messages = [{"role": "system", "content": system_msg}]
            for step in traj.steps:
                obs_text = step.observations.get(obs_mode, "")
                if not isinstance(obs_text, str):
                    obs_text = json.dumps(_make_serializable(obs_text))
                messages.append({"role": "user", "content": obs_text})

                action_text = str(step.action)
                if step.reasoning:
                    action_text = f"{step.reasoning}\nAction: {step.action}"
                messages.append({"role": "assistant", "content": action_text})

            rows.append({
                "messages": messages,
                "task_id": self.task_id,
                "total_reward": float(traj.total_reward),
                "success": traj.success,
            })
        return rows

    def _to_decision_rows(self, obs_mode: str) -> list[dict[str, Any]]:
        """Build (obs, action, reward, next_obs, done) rows for BC/offline RL."""
        rows = []
        for traj in self.trajectories:
            for i, step in enumerate(traj.steps):
                obs = step.observations.get(obs_mode, "")
                if not isinstance(obs, str):
                    obs = json.dumps(_make_serializable(obs))
                next_obs = ""
                if i + 1 < len(traj.steps):
                    next_obs = traj.steps[i + 1].observations.get(obs_mode, "")
                    if not isinstance(next_obs, str):
                        next_obs = json.dumps(_make_serializable(next_obs))
                rows.append({
                    "observation": obs,
                    "action": int(step.action),
                    "reward": float(step.reward),
                    "next_observation": next_obs,
                    "done": step.done,
                    "task_id": self.task_id,
                    "episode_id": id(traj),
                })
        return rows

    def _to_trajectory_rows(self, obs_mode: str) -> list[dict[str, Any]]:
        """Build per-episode rows with full trajectory data."""
        rows = []
        for traj in self.trajectories:
            observations = []
            for step in traj.steps:
                obs = step.observations.get(obs_mode, "")
                if not isinstance(obs, str):
                    obs = json.dumps(_make_serializable(obs))
                observations.append(obs)
            rows.append({
                "observations": observations,
                "actions": [int(a) for a in traj.actions],
                "rewards": [float(r) for r in traj.rewards],
                "total_reward": float(traj.total_reward),
                "length": traj.length,
                "success": traj.success,
                "task_id": self.task_id,
                "metadata": _make_serializable(traj.metadata),
            })
        return rows


class DataCollector:
    """High-level data collector that records any agent on any env.

    Example::

        from agentick.oracles import get_oracle
        import agentick

        env = agentick.make("GoToGoal-v0", difficulty="medium")
        oracle = get_oracle("GoToGoal-v0", env)
        collector = DataCollector(env, oracle, record_modalities=["ascii", "language"])
        dataset = collector.collect(num_episodes=20, seeds=range(20))
        dataset.save("trajectories/oracle_goto/")
        dataset.export_to_huggingface("trajectories/hf/", format="conversation")
    """

    def __init__(
        self,
        env: Any,
        agent: AgentProtocol,
        record_modalities: list[str] | None = None,
        record_reasoning: bool = True,
        record_actions: bool = True,
        record_rewards: bool = True,
        record_dones: bool = True,
        record_info: bool = False,
        record_videos: bool = False,
    ) -> None:
        """
        Args:
            env: A Gymnasium-compatible agentick environment.
            agent: Agent with ``.act(obs, info) -> int`` method.
            record_modalities: Observation modalities to record.
                Defaults to ``["language"]``.
            record_reasoning: Capture ``agent.last_reasoning`` if available.
            record_actions: Record actions (always True for useful data).
            record_rewards: Record rewards.
            record_dones: Record done flags.
            record_info: Record full info dicts (can be large).
            record_videos: Record rgb_array frames as a separate video list.
        """
        self.env = env
        self.agent = agent
        self.record_modalities = record_modalities or ["language"]
        self.record_reasoning = record_reasoning
        self.record_actions = record_actions
        self.record_rewards = record_rewards
        self.record_dones = record_dones
        self.record_info = record_info
        self.record_videos = record_videos

    def _get_task_id(self) -> str:
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "task") and hasattr(unwrapped.task, "name"):
            return unwrapped.task.name
        spec = getattr(self.env, "spec", None)
        if spec and hasattr(spec, "id"):
            return spec.id
        return "unknown"

    def _render_modalities(self) -> dict[str, Any]:
        """Render the current env state in all requested modalities."""
        unwrapped = self.env.unwrapped
        observations: dict[str, Any] = {}
        for mode in self.record_modalities:
            if hasattr(unwrapped, "render_in_mode"):
                observations[mode] = unwrapped.render_in_mode(mode)
            elif mode == self.env.render_mode:
                observations[mode] = self.env.render()
        return observations

    def collect(
        self,
        num_episodes: int = 10,
        seeds: list[int] | range | None = None,
        difficulty: str | None = None,
        show_progress: bool = True,
    ) -> CollectedDataset:
        """Collect trajectories by running the agent on the env.

        Args:
            num_episodes: Number of episodes to collect.
            seeds: Per-episode seeds. If shorter than *num_episodes*,
                remaining episodes use sequential seeds.
            difficulty: If given, recreate the env with this difficulty
                (requires ``agentick.make``).
            show_progress: Show tqdm progress bar.

        Returns:
            A :class:`CollectedDataset` with export methods.
        """
        env = self.env

        # Optionally recreate env with different difficulty
        if difficulty is not None:
            import agentick
            task_id = self._get_task_id()
            env = agentick.make(
                task_id, difficulty=difficulty,
                render_mode=env.render_mode,
            )

        # Resolve seeds
        if seeds is None:
            seed_list = list(range(num_episodes))
        else:
            seed_list = list(seeds)
            while len(seed_list) < num_episodes:
                seed_list.append(seed_list[-1] + 1 if seed_list else 0)

        task_id = self._get_task_id()
        trajectories: list[MultiModalTrajectory] = []

        # Progress bar
        iterator: Any = range(num_episodes)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc=f"Collecting {task_id}", unit="ep")
            except ImportError:
                pass

        for i in iterator:
            seed = seed_list[i]
            traj = self._run_episode(env, seed, task_id)
            trajectories.append(traj)

        if difficulty is not None and env is not self.env:
            env.close()

        return CollectedDataset(
            trajectories=trajectories,
            task_id=task_id,
            modalities=self.record_modalities,
        )

    def _run_episode(
        self, env: Any, seed: int, task_id: str,
    ) -> MultiModalTrajectory:
        """Run a single episode and return the trajectory."""
        obs, info = env.reset(seed=seed)

        # Reset agent if it supports it
        if hasattr(self.agent, "reset"):
            self.agent.reset(obs, info)

        # Record initial observations
        observations = self._render_modalities()

        traj = MultiModalTrajectory(metadata={
            "task_id": task_id,
            "seed": seed,
            "difficulty": info.get("difficulty", ""),
        })

        done = False
        truncated = False
        while not (done or truncated):
            action = self.agent.act(obs, info)

            # Capture reasoning
            reasoning = None
            if self.record_reasoning and hasattr(self.agent, "last_reasoning"):
                reasoning = self.agent.last_reasoning

            obs, reward, done, truncated, info = env.step(action)

            # Render after step for next-state observations
            post_obs = self._render_modalities()

            step = MultiModalStep(
                observations=observations,
                action=int(action),
                reward=float(reward),
                done=done or truncated,
                info=dict(info) if self.record_info else {},
                reasoning=reasoning,
            )
            traj.steps.append(step)
            traj.total_reward += reward

            # Current post-step obs become next step's observations
            observations = post_obs

        traj.success = info.get("success", False)
        return traj
