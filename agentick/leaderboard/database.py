"""JSON-file based leaderboard database."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from agentick.leaderboard.result import EvaluationResult


class LeaderboardDatabase:
    """
    Simple JSON-file based database for leaderboard entries.

    No server needed - just files on disk.
    """

    def __init__(self, data_dir: str | Path = "leaderboard_data"):
        """
        Initialize database.

        Args:
            data_dir: Root directory for leaderboard data
        """
        self.data_dir = Path(data_dir)
        self.entries_dir = self.data_dir / "entries"
        self.rankings_dir = self.data_dir / "rankings"
        self.history_dir = self.data_dir / "history"

        # Create directories
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        self.rankings_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def add_entry(self, result: EvaluationResult) -> None:
        """
        Add evaluation result to database.

        Args:
            result: Evaluation result to add
        """
        # Generate entry filename
        agent_name = result.submission.agent_name
        suite_name = result.suite_name
        filename = f"{agent_name}_{suite_name}.json"

        # Save entry
        entry_path = self.entries_dir / filename
        result.to_json(entry_path)

    def get_entries(self, suite_name: str | None = None) -> list[EvaluationResult]:
        """
        Get all entries, optionally filtered by suite.

        Args:
            suite_name: Optional suite name to filter by

        Returns:
            List of evaluation results
        """
        entries = []

        for entry_file in self.entries_dir.glob("*.json"):
            result = EvaluationResult.from_json(entry_file)

            if suite_name is None or result.suite_name == suite_name:
                entries.append(result)

        return entries

    def save_rankings(self, suite_name: str, rankings: list[dict[str, Any]]) -> None:
        """
        Save pre-computed rankings for a suite.

        Args:
            suite_name: Suite name
            rankings: List of ranked entries
        """
        ranking_file = self.rankings_dir / f"{suite_name}.json"

        with open(ranking_file, "w") as f:
            json.dump(rankings, f, indent=2, default=str)

    def load_rankings(self, suite_name: str) -> list[dict[str, Any]]:
        """
        Load rankings for a suite.

        Args:
            suite_name: Suite name

        Returns:
            List of ranked entries
        """
        ranking_file = self.rankings_dir / f"{suite_name}.json"

        if not ranking_file.exists():
            return []

        with open(ranking_file) as f:
            return json.load(f)

    def create_snapshot(self) -> None:
        """Create historical snapshot of current leaderboard state."""
        timestamp = datetime.now().strftime("%Y-%m-%d")
        snapshot_file = self.history_dir / f"{timestamp}.json"

        # Collect all entries
        all_entries = {}
        for entry_file in self.entries_dir.glob("*.json"):
            all_entries[entry_file.stem] = json.load(open(entry_file))

        # Save snapshot
        with open(snapshot_file, "w") as f:
            json.dump(all_entries, f, indent=2, default=str)
