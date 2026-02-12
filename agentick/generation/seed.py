"""Seed management for reproducible procedural generation."""

from __future__ import annotations

from typing import Literal

import numpy as np


class SeedManager:
    """Manage seeds for reproducible generation."""

    # Registry of fixed seeds for benchmark evaluation
    # Each task has a set of fixed seeds per difficulty level
    BENCHMARK_SEEDS = {
        "GoToGoal-v0": {
            "easy": list(range(1000, 1100)),
            "medium": list(range(2000, 2100)),
            "hard": list(range(3000, 3100)),
            "expert": list(range(4000, 4100)),
        },
        "KeyDoorPuzzle-v0": {
            "easy": list(range(1100, 1200)),
            "medium": list(range(2100, 2200)),
            "hard": list(range(3100, 3200)),
            "expert": list(range(4100, 4200)),
        },
        # Add more tasks as they are implemented
    }

    def __init__(self, base_seed: int = 42):
        """
        Initialize seed manager.

        Args:
            base_seed: Base seed for sequence generation
        """
        self.base_seed = base_seed
        self.sequence = np.random.SeedSequence(base_seed)

    def spawn(self, n: int = 1) -> list[np.random.Generator]:
        """
        Spawn n independent random generators.

        Args:
            n: Number of generators to spawn

        Returns:
            List of independent RNG instances
        """
        child_sequences = self.sequence.spawn(n)
        return [np.random.default_rng(seq) for seq in child_sequences]

    def get_generator(self, stage: str | int) -> np.random.Generator:
        """
        Get a generator for a specific stage of generation.

        Args:
            stage: Stage identifier (can be string or int)

        Returns:
            RNG for that stage
        """
        # Create sub-seed based on stage
        if isinstance(stage, str):
            stage_hash = hash(stage) % (2**31)
        else:
            stage_hash = stage

        child_seq = self.sequence.spawn(1)[0]
        child_seq.state = child_seq.state + stage_hash

        return np.random.default_rng(child_seq)

    @classmethod
    def get_benchmark_seeds(
        cls,
        task_id: str,
        difficulty: Literal["easy", "medium", "hard", "expert"],
        num_seeds: int = 100,
    ) -> list[int]:
        """
        Get fixed benchmark seeds for a task and difficulty.

        Args:
            task_id: Task identifier
            difficulty: Difficulty level
            num_seeds: Number of seeds to return

        Returns:
            List of fixed seeds for benchmark evaluation
        """
        if task_id in cls.BENCHMARK_SEEDS:
            if difficulty in cls.BENCHMARK_SEEDS[task_id]:
                seeds = cls.BENCHMARK_SEEDS[task_id][difficulty]
                return seeds[:num_seeds]

        # Fallback: generate deterministic seeds based on task and difficulty
        base = hash(f"{task_id}_{difficulty}") % (2**31)
        return list(range(base, base + num_seeds))

    @classmethod
    def register_task_seeds(
        cls,
        task_id: str,
        difficulty_seeds: dict[str, list[int]],
    ):
        """
        Register fixed seeds for a task.

        Args:
            task_id: Task identifier
            difficulty_seeds: Dict mapping difficulty to seed lists
        """
        cls.BENCHMARK_SEEDS[task_id] = difficulty_seeds


def get_benchmark_seed(
    task_id: str,
    difficulty: str,
    instance_idx: int = 0,
) -> int:
    """
    Get a specific benchmark seed.

    Args:
        task_id: Task identifier
        difficulty: Difficulty level
        instance_idx: Instance index (0-99 typically)

    Returns:
        Seed for that specific instance
    """
    seeds = SeedManager.get_benchmark_seeds(task_id, difficulty, num_seeds=instance_idx + 1)
    return seeds[instance_idx]


def create_seed_sequence(base_seed: int, num_stages: int) -> list[int]:
    """
    Create a sequence of seeds for multi-stage generation.

    Args:
        base_seed: Starting seed
        num_stages: Number of stages

    Returns:
        List of seeds, one per stage
    """
    seq = np.random.SeedSequence(base_seed)
    child_seqs = seq.spawn(num_stages)
    return [int(cs.generate_state(1)[0]) for cs in child_seqs]
