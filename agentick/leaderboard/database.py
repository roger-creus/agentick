"""Thin JSON I/O wrapper for leaderboard entries."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any  # noqa: I001

# Default path relative to repo root
_DEFAULT_ENTRIES_PATH = Path(__file__).resolve().parents[2] / "leaderboard_data" / "entries.json"


def _default_entries_path() -> Path:
    """Return the default entries.json path (repo_root/leaderboard_data/entries.json)."""
    return _DEFAULT_ENTRIES_PATH


def load_entries(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load all leaderboard entries from the JSON file.

    Args:
        path: Path to entries.json. Defaults to leaderboard_data/entries.json.

    Returns:
        List of entry dicts. Empty list if file does not exist yet.
    """
    path = Path(path) if path else _default_entries_path()
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("entries", [])


def save_entries(entries: list[dict[str, Any]], path: str | Path | None = None) -> None:
    """Write all entries to the JSON file.

    Args:
        entries: List of entry dicts.
        path: Path to entries.json. Defaults to leaderboard_data/entries.json.
    """
    path = Path(path) if path else _default_entries_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"entries": entries, "updated_at": datetime.now().isoformat()}, f, indent=2)


def add_entry(entry: dict[str, Any], path: str | Path | None = None) -> None:
    """Add a single entry to the leaderboard.

    Required keys in *entry*:
        agent_name, author, description, agent_type, observation_mode,
        harness, model, open_weights, date,
        scores: {agentick_score, agentick_score_ci, per_category, per_task},
        metadata (optional dict)

    Args:
        entry: The entry dict to add.
        path: Path to entries.json. Defaults to leaderboard_data/entries.json.

    Raises:
        ValueError: If required keys are missing.
    """
    required_keys = {
        "agent_name",
        "author",
        "description",
        "agent_type",
        "observation_mode",
        "harness",
        "model",
        "open_weights",
        "date",
        "scores",
    }
    missing = required_keys - set(entry.keys())
    if missing:
        raise ValueError(f"Entry is missing required keys: {missing}")

    # per_task is the only strictly required score field; aggregates may be
    # absent for partial entries (single task/difficulty submissions).
    if "per_task" not in entry.get("scores", {}):
        raise ValueError("Entry scores dict must contain 'per_task'")

    entries = load_entries(path)
    entries.append(entry)
    save_entries(entries, path)


def get_entries(
    suite_name: str | None = None, path: str | Path | None = None
) -> list[dict[str, Any]]:
    """Get entries, optionally filtered by suite name stored in metadata.

    Args:
        suite_name: If provided, only return entries whose
            metadata.suite_name matches.
        path: Path to entries.json.

    Returns:
        Filtered (or full) list of entry dicts.
    """
    entries = load_entries(path)
    if suite_name is None:
        return entries
    return [
        e
        for e in entries
        if e.get("metadata", {}).get("suite_name") == suite_name
    ]
