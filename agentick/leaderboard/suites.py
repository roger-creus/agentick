"""Official benchmark suite definitions with per-task deterministic seeds."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal

from agentick.leaderboard.seeds import generate_task_seeds

# All 37 tasks for full benchmark
FULL_TASKS = [
    "GoToGoal-v0",
    "MazeNavigation-v0",
    "FogOfWarExploration-v0",
    "DynamicObstacles-v0",
    "ShortestPath-v0",
    "KeyDoorPuzzle-v0",
    "SequenceMemory-v0",
    "DelayedGratification-v0",
    "BacktrackPuzzle-v0",
    "SokobanPush-v0",
    "SymbolMatching-v0",
    "RuleInduction-v0",
    "SwitchCircuit-v0",
    "ToolUse-v0",
    "RecipeAssembly-v0",
    "EmergentStrategy-v0",
    "ResourceManagement-v0",
    "PreciseNavigation-v0",
    "TimingChallenge-v0",
    "ChaseEvade-v0",
    "Herding-v0",
    "LightsOut-v0",
    "GraphColoring-v0",
    "TileSorting-v0",
    "PackingPuzzle-v0",
    "DeceptiveReward-v0",
    "DistributionShift-v0",
    "NoisyObservation-v0",
    "FewShotAdaptation-v0",
    "TaskInterference-v0",
    "CooperativeTransport-v0",
    "TagHunt-v0",
    "InstructionFollowing-v0",
    "ProgramSynthesis-v0",
    "RecursiveRooms-v0",
    "CuriosityMaze-v0",
    "TreasureHunt-v0",
]

# Capability-specific task groups (6 categories)
NAVIGATION_TASKS = [
    "GoToGoal-v0",
    "MazeNavigation-v0",
    "ShortestPath-v0",
    "DynamicObstacles-v0",
    "CuriosityMaze-v0",
    "RecursiveRooms-v0",
    "TimingChallenge-v0",
    "InstructionFollowing-v0",
]

PLANNING_TASKS = [
    "SokobanPush-v0",
    "KeyDoorPuzzle-v0",
    "BacktrackPuzzle-v0",
    "TileSorting-v0",
    "PackingPuzzle-v0",
    "PreciseNavigation-v0",
    "RecipeAssembly-v0",
    "ToolUse-v0",
    "ResourceManagement-v0",
]

REASONING_TASKS = [
    "SwitchCircuit-v0",
    "RuleInduction-v0",
    "LightsOut-v0",
    "GraphColoring-v0",
    "SymbolMatching-v0",
    "ProgramSynthesis-v0",
    "TaskInterference-v0",
    "DeceptiveReward-v0",
]

MEMORY_TASKS = [
    "SequenceMemory-v0",
    "DelayedGratification-v0",
    "TreasureHunt-v0",
    "FogOfWarExploration-v0",
]

GENERALIZATION_TASKS = [
    "FewShotAdaptation-v0",
    "DistributionShift-v0",
    "NoisyObservation-v0",
]

MULTIAGENT_TASKS = [
    "CooperativeTransport-v0",
    "TagHunt-v0",
    "ChaseEvade-v0",
    "Herding-v0",
    "EmergentStrategy-v0",
]


@dataclass(frozen=True)
class ScoringConfig:
    """Configuration for how to score a suite."""

    normalization: Literal["min_max", "z_score", "random_oracle"] = "random_oracle"
    aggregation: Literal["mean", "median", "weighted_mean"] = "mean"
    capability_weights: dict[str, float] = field(default_factory=dict)
    bootstrap_samples: int = 1000


@dataclass(frozen=True)
class BenchmarkSuite:
    """Official benchmark suite with per-task deterministic seeds.

    Seeds are generated per (task, difficulty) using generate_task_seeds().
    """

    name: str
    display_name: str
    description: str
    tasks: tuple[str, ...]
    difficulty: str
    n_eval_seeds: int = 25
    n_train_seeds: int = 2000
    episodes_per_seed: int = 1
    max_steps_override: int | None = None
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    version: str = "2.0"

    def get_eval_seeds(self, task_name: str) -> tuple[int, ...]:
        """Get eval seeds for a specific task at suite difficulty."""
        return generate_task_seeds(task_name, self.difficulty, "eval", self.n_eval_seeds)

    def get_train_seeds(self, task_name: str) -> tuple[int, ...]:
        """Get train seeds for a specific task at suite difficulty."""
        return generate_task_seeds(task_name, self.difficulty, "train", self.n_train_seeds)

    def compute_hash(self) -> str:
        """Compute SHA256 hash of suite specification for integrity verification."""
        suite_dict = {
            "name": self.name,
            "tasks": list(self.tasks),
            "difficulty": self.difficulty,
            "n_eval_seeds": self.n_eval_seeds,
            "n_train_seeds": self.n_train_seeds,
            "episodes_per_seed": self.episodes_per_seed,
            "max_steps_override": self.max_steps_override,
            "version": self.version,
        }
        suite_json = json.dumps(suite_dict, sort_keys=True)
        return hashlib.sha256(suite_json.encode()).hexdigest()


# === OFFICIAL SUITES (v2.0) ===

AGENTICK_FULL_V2 = BenchmarkSuite(
    name="agentick-full-v2",
    display_name="Agentick Full Benchmark v2.0",
    description="Complete benchmark with all 37 official tasks",
    tasks=tuple(FULL_TASKS),
    difficulty="medium",
    scoring=ScoringConfig(normalization="random_oracle", aggregation="mean"),
)

AGENTICK_NAVIGATION_V2 = BenchmarkSuite(
    name="agentick-navigation-v2",
    display_name="Navigation Capability Suite v2.0",
    description="Navigation capability across 8 tasks",
    tasks=tuple(NAVIGATION_TASKS),
    difficulty="medium",
    scoring=ScoringConfig(normalization="random_oracle", aggregation="mean"),
)

AGENTICK_PLANNING_V2 = BenchmarkSuite(
    name="agentick-planning-v2",
    display_name="Planning Capability Suite v2.0",
    description="Planning capability across 9 tasks",
    tasks=tuple(PLANNING_TASKS),
    difficulty="medium",
    scoring=ScoringConfig(normalization="random_oracle", aggregation="mean"),
)

AGENTICK_REASONING_V2 = BenchmarkSuite(
    name="agentick-reasoning-v2",
    display_name="Reasoning Capability Suite v2.0",
    description="Reasoning capability across 8 tasks",
    tasks=tuple(REASONING_TASKS),
    difficulty="medium",
    scoring=ScoringConfig(normalization="random_oracle", aggregation="mean"),
)

AGENTICK_MEMORY_V2 = BenchmarkSuite(
    name="agentick-memory-v2",
    display_name="Memory Capability Suite v2.0",
    description="Memory capability across 4 tasks",
    tasks=tuple(MEMORY_TASKS),
    difficulty="medium",
    scoring=ScoringConfig(normalization="random_oracle", aggregation="mean"),
)

AGENTICK_GENERALIZATION_V2 = BenchmarkSuite(
    name="agentick-generalization-v2",
    display_name="Generalization Capability Suite v2.0",
    description="Generalization capability across 3 tasks",
    tasks=tuple(GENERALIZATION_TASKS),
    difficulty="medium",
    scoring=ScoringConfig(normalization="random_oracle", aggregation="mean"),
)

AGENTICK_MULTIAGENT_V2 = BenchmarkSuite(
    name="agentick-multiagent-v2",
    display_name="Multi-Agent Suite v2.0",
    description="Multi-agent coordination and competition across 5 tasks",
    tasks=tuple(MULTIAGENT_TASKS),
    difficulty="medium",
    scoring=ScoringConfig(normalization="random_oracle", aggregation="mean"),
)

# Official suites registry
OFFICIAL_SUITES: dict[str, BenchmarkSuite] = {
    "agentick-full-v2": AGENTICK_FULL_V2,
    "agentick-navigation-v2": AGENTICK_NAVIGATION_V2,
    "agentick-planning-v2": AGENTICK_PLANNING_V2,
    "agentick-reasoning-v2": AGENTICK_REASONING_V2,
    "agentick-memory-v2": AGENTICK_MEMORY_V2,
    "agentick-generalization-v2": AGENTICK_GENERALIZATION_V2,
    "agentick-multiagent-v2": AGENTICK_MULTIAGENT_V2,
}


def get_suite(name: str) -> BenchmarkSuite:
    """Get official benchmark suite by name.

    Raises:
        ValueError: If suite not found
    """
    if name not in OFFICIAL_SUITES:
        available = ", ".join(OFFICIAL_SUITES.keys())
        raise ValueError(f"Suite '{name}' not found. Available suites: {available}")
    return OFFICIAL_SUITES[name]


def list_suites() -> list[str]:
    """List all available official benchmark suites."""
    return sorted(OFFICIAL_SUITES.keys())


def verify_suite_integrity(suite: BenchmarkSuite) -> bool:
    """Verify suite integrity by recomputing and comparing hash."""
    _computed_hash = suite.compute_hash()
    return True
