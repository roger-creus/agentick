"""Deterministic per-task-difficulty seed generation for train/eval splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def generate_task_seeds(
    task_name: str, difficulty: str, split: str, n_seeds: int
) -> tuple[int, ...]:
    """Generate deterministic seeds for a (task, difficulty, split) triple.

    Args:
        task_name: e.g. "GoToGoal-v0"
        difficulty: "easy" | "medium" | "hard" | "expert"
        split: "train" | "eval"
        n_seeds: Number of seeds to generate (2000 for train, 25 for eval)

    Returns:
        Tuple of deterministic seeds in range [0, 2^31)
    """
    key = f"{task_name}::{difficulty}::{split}"
    hash_int = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(hash_int)
    return tuple(int(x) for x in rng.integers(0, 2**31, size=n_seeds))


def get_train_seeds(task_name: str, difficulty: str) -> tuple[int, ...]:
    """Get the standard 2000 training seeds for a (task, difficulty) pair."""
    return generate_task_seeds(task_name, difficulty, "train", 2000)


def get_eval_seeds(task_name: str, difficulty: str) -> tuple[int, ...]:
    """Get the standard 25 evaluation seeds for a (task, difficulty) pair."""
    return generate_task_seeds(task_name, difficulty, "eval", 25)


def export_seeds_to_json(output_path: str | Path) -> None:
    """Export all official eval seeds to JSON for verification.

    Writes per-task-difficulty eval seeds for all registered tasks.
    """
    from agentick.tasks.registry import list_tasks

    difficulties = ["easy", "medium", "hard", "expert"]
    seeds_data = {}

    for task_name in sorted(list_tasks()):
        task_data = {}
        for diff in difficulties:
            seeds = get_eval_seeds(task_name, diff)
            task_data[diff] = {
                "n_seeds": len(seeds),
                "seeds": list(seeds),
                "hash": hashlib.sha256(json.dumps(list(seeds)).encode()).hexdigest(),
            }
        seeds_data[task_name] = task_data

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(seeds_data, f, indent=2, sort_keys=True)


def verify_seeds(
    task_name: str, difficulty: str, split: str, seeds: tuple[int, ...]
) -> bool:
    """Verify that provided seeds match the deterministically generated ones."""
    expected = generate_task_seeds(task_name, difficulty, split, len(seeds))
    return seeds == expected
