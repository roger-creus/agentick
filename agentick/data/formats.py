"""Export trajectories to various formats for fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from agentick.data.collector import Trajectory


def export_to_format(
    trajectories: list[Trajectory],
    output_path: str | Path,
    format_type: Literal["jsonl", "hf_dataset", "d4rl", "conversation"],
    **kwargs: Any,
) -> Path:
    """
    Export trajectories to specified format.

    Args:
        trajectories: List of trajectories to export
        output_path: Output file/directory path
        format_type: Export format
        **kwargs: Format-specific arguments

    Returns:
        Path to exported data

    Supported formats:
        - jsonl: JSON Lines format (one episode per line)
        - hf_dataset: HuggingFace Datasets format
        - d4rl: D4RL-style HDF5 format for offline RL
        - conversation: Chat format for LLM fine-tuning
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format_type == "jsonl":
        return _export_jsonl(trajectories, output_path, **kwargs)
    elif format_type == "hf_dataset":
        return _export_hf_dataset(trajectories, output_path, **kwargs)
    elif format_type == "d4rl":
        return _export_d4rl(trajectories, output_path, **kwargs)
    elif format_type == "conversation":
        return _export_conversation(trajectories, output_path, **kwargs)
    else:
        raise ValueError(f"Unknown format: {format_type}")


def _export_jsonl(
    trajectories: list[Trajectory],
    output_path: Path,
    **kwargs: Any,
) -> Path:
    """Export to JSON Lines format."""
    with open(output_path, "w") as f:
        for traj in trajectories:
            # Convert trajectory to serializable format
            data = {
                "actions": [int(a) for a in traj.actions],
                "rewards": [float(r) for r in traj.rewards],
                "total_reward": float(traj.total_reward),
                "length": int(traj.length),
                "metadata": traj.metadata,
            }

            # Include observations if they're strings or dicts (not arrays)
            if traj.observations and isinstance(traj.observations[0], (str, dict)):
                data["observations"] = traj.observations

            f.write(json.dumps(data) + "\n")

    return output_path


def _export_hf_dataset(
    trajectories: list[Trajectory],
    output_path: Path,
    **kwargs: Any,
) -> Path:
    """Export to HuggingFace Datasets format."""
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("datasets package required. Install with: uv sync --extra finetune")

    # Flatten trajectories into steps
    episodes = []
    for i, traj in enumerate(trajectories):
        episode_data = {
            "episode_id": i,
            "actions": traj.actions,
            "rewards": traj.rewards,
            "total_reward": traj.total_reward,
            "length": traj.length,
        }

        # Add metadata fields
        for key, value in traj.metadata.items():
            episode_data[f"meta_{key}"] = value

        episodes.append(episode_data)

    # Create dataset
    dataset = Dataset.from_list(episodes)

    # Save
    dataset.save_to_disk(str(output_path))

    return output_path


def _export_d4rl(
    trajectories: list[Trajectory],
    output_path: Path,
    **kwargs: Any,
) -> Path:
    """Export to D4RL-style HDF5 format."""
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required. Install with: uv sync --extra all")

    # Concatenate all trajectories
    all_observations = []
    all_actions = []
    all_rewards = []
    all_dones = []
    all_timeouts = []

    for traj in trajectories:
        # Pad or convert observations to consistent format
        if traj.observations:
            all_observations.extend(traj.observations)

        all_actions.extend(traj.actions)
        all_rewards.extend(traj.rewards)
        all_dones.extend(traj.dones)

        # Timeouts (terminal states)
        timeouts = [False] * len(traj.actions)
        if traj.dones[-1]:
            timeouts[-1] = True
        all_timeouts.extend(timeouts)

    # Create HDF5 file
    with h5py.File(output_path, "w") as f:
        f.create_dataset("actions", data=np.array(all_actions, dtype=np.int32))
        f.create_dataset("rewards", data=np.array(all_rewards, dtype=np.float32))
        f.create_dataset("terminals", data=np.array(all_dones, dtype=bool))
        f.create_dataset("timeouts", data=np.array(all_timeouts, dtype=bool))

        # Observations (if numeric)
        if all_observations and isinstance(all_observations[0], np.ndarray):
            f.create_dataset("observations", data=np.array(all_observations))

    return output_path


def _export_conversation(
    trajectories: list[Trajectory],
    output_path: Path,
    system_prompt: str | None = None,
    **kwargs: Any,
) -> Path:
    """Export to conversation format for chat model fine-tuning."""
    if system_prompt is None:
        system_prompt = (
            "You are an AI agent playing a grid-based navigation game. "
            "Given the current state, choose the best action."
        )

    conversations = []

    for traj in trajectories:
        messages = [{"role": "system", "content": system_prompt}]

        for i, (obs, action) in enumerate(zip(traj.observations, traj.actions)):
            # User message (observation)
            if isinstance(obs, str):
                obs_text = obs
            elif isinstance(obs, dict):
                obs_text = json.dumps(obs)
            else:
                obs_text = f"Observation {i}"

            messages.append({"role": "user", "content": obs_text})

            # Assistant message (action)
            action_text = f"Action: {action}"
            messages.append({"role": "assistant", "content": action_text})

        conversations.append({"messages": messages, **traj.metadata})

    # Save as JSONL
    with open(output_path, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")

    return output_path
