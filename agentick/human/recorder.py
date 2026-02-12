"""Human data recorder for collecting human performance data."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


class HumanDataRecorder:
    """
    Record human play sessions for baseline analysis.

    Records:
    - Actions and timing
    - Episode outcomes
    - Optional demographic data
    - Anonymized participant IDs
    """

    def __init__(
        self,
        save_dir: str | Path = "human_data",
        participant_id: str | None = None,
    ):
        """
        Initialize human data recorder.

        Args:
            save_dir: Directory to save recorded data
            participant_id: Optional participant ID (generates random if None)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.participant_id = participant_id or str(uuid.uuid4())[:8]
        self.session_id = str(uuid.uuid4())[:8]
        self.session_start_time = datetime.now()

        self.episodes: list[dict[str, Any]] = []
        self.demographics: dict[str, Any] = {}

    def record_episode(
        self,
        task_name: str,
        difficulty: str,
        episode_stats: dict[str, Any],
        trajectory: list[tuple] | None = None,
    ) -> None:
        """
        Record one episode.

        Args:
            task_name: Name of task
            difficulty: Difficulty level
            episode_stats: Episode statistics dict
            trajectory: Optional full trajectory data
        """
        episode_data = {
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "task_name": task_name,
            "difficulty": difficulty,
            "timestamp": datetime.now().isoformat(),
            **episode_stats,
        }

        if trajectory:
            # Store trajectory separately for large data
            filename = (
                f"{self.participant_id}_{self.session_id}_"
                f"episode_{len(self.episodes)}_trajectory.json"
            )
            trajectory_file = self.save_dir / filename
            with open(trajectory_file, "w") as f:
                # Convert trajectory to serializable format
                serializable_trajectory = [
                    {
                        "obs": str(t[0])[:100],  # Truncate obs
                        "action": int(t[1]) if t[1] is not None else None,
                        "reward": float(t[2]),
                        "terminated": bool(t[3]),
                        "truncated": bool(t[4]),
                    }
                    for t in trajectory
                ]
                json.dump(serializable_trajectory, f, indent=2)
            episode_data["trajectory_file"] = str(trajectory_file)

        self.episodes.append(episode_data)

    def collect_demographics(self, demographics: dict[str, Any]) -> None:
        """
        Collect optional demographic information.

        Args:
            demographics: Dict with demographic data
                (e.g., age, gaming_experience, puzzle_experience)
        """
        self.demographics = {
            "participant_id": self.participant_id,
            "collected_at": datetime.now().isoformat(),
            **demographics,
        }

    def save_session(self) -> Path:
        """
        Save session data to file.

        Returns:
            Path to saved session file
        """
        session_data = {
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "session_start": self.session_start_time.isoformat(),
            "session_end": datetime.now().isoformat(),
            "demographics": self.demographics,
            "episodes": self.episodes,
            "n_episodes": len(self.episodes),
        }

        # Save session file
        session_file = self.save_dir / f"{self.participant_id}_{self.session_id}_session.json"
        with open(session_file, "w") as f:
            json.dump(session_data, f, indent=2)

        # Also append to aggregate file
        aggregate_file = self.save_dir / "all_sessions.jsonl"
        with open(aggregate_file, "a") as f:
            json.dump(session_data, f)
            f.write("\n")

        return session_file

    def get_session_summary(self) -> dict[str, Any]:
        """
        Get summary statistics for current session.

        Returns:
            Dict with session summary
        """
        if not self.episodes:
            return {"n_episodes": 0}

        total_reward = sum(ep.get("total_reward", 0.0) for ep in self.episodes)
        total_steps = sum(ep.get("step_count", 0) for ep in self.episodes)
        successes = sum(1 for ep in self.episodes if ep.get("success", False))
        total_duration = sum(ep.get("duration", 0.0) for ep in self.episodes)

        return {
            "n_episodes": len(self.episodes),
            "total_reward": total_reward,
            "mean_reward": total_reward / len(self.episodes),
            "total_steps": total_steps,
            "mean_steps": total_steps / len(self.episodes),
            "success_rate": successes / len(self.episodes),
            "total_duration": total_duration,
            "mean_duration": total_duration / len(self.episodes),
        }


def create_survey_form() -> dict[str, Any]:
    """
    Create basic demographic survey form.

    Returns:
        Dict with survey questions
    """
    return {
        "age_range": {
            "question": "What is your age range?",
            "type": "choice",
            "choices": ["<18", "18-24", "25-34", "35-44", "45-54", "55+"],
        },
        "gaming_experience": {
            "question": "How much gaming experience do you have?",
            "type": "scale",
            "min": 1,
            "max": 5,
            "labels": {1: "None", 3: "Moderate", 5: "Expert"},
        },
        "puzzle_experience": {
            "question": "How much puzzle-solving experience do you have?",
            "type": "scale",
            "min": 1,
            "max": 5,
            "labels": {1: "None", 3: "Moderate", 5: "Expert"},
        },
        "keyboard_proficiency": {
            "question": "How comfortable are you with keyboard controls?",
            "type": "scale",
            "min": 1,
            "max": 5,
            "labels": {1: "Not comfortable", 3: "Moderate", 5: "Very comfortable"},
        },
    }


def load_session_data(save_dir: str | Path) -> list[dict[str, Any]]:
    """
    Load all session data from directory.

    Args:
        save_dir: Directory containing session files

    Returns:
        List of session data dicts
    """
    save_dir = Path(save_dir)
    aggregate_file = save_dir / "all_sessions.jsonl"

    if not aggregate_file.exists():
        return []

    sessions = []
    with open(aggregate_file) as f:
        for line in f:
            if line.strip():
                sessions.append(json.loads(line))

    return sessions
