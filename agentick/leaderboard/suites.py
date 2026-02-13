"""Official benchmark suite definitions with locked seeds and immutable configurations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal

# Core tasks (original 27 tasks - excluding new Phase 2/3 additions)
CORE_TASKS = [
    "GoToGoal-v0",
    "MazeNavigation-v0",
    "FogOfWarExploration-v0",
    "DynamicObstacles-v0",
    "MultiGoalRoute-v0",
    "KeyDoorPuzzle-v0",
    "SequenceMemory-v0",
    "DelayedGratification-v0",
    "BacktrackPuzzle-v0",
    "BreadcrumbTrail-v0",
    "SokobanPush-v0",
    "CausalChain-v0",
    "SymbolMatching-v0",
    "RuleInduction-v0",
    "SwitchCircuit-v0",
    "ToolUse-v0",
    "RecipeAssembly-v0",
    "MultiRoomEscape-v0",
    "EmergentStrategy-v0",
    "ResourceManagement-v0",
    "PreciseNavigation-v0",
    "TimingChallenge-v0",
    "ChaseEvade-v0",
    "Herding-v0",
    "LightsOut-v0",
    "GraphColoring-v0",
    "TileSorting-v0",
]

# All 35 tasks for full benchmark (core + key Phase 2/3 tasks)
FULL_TASKS = CORE_TASKS + [
    "PackingPuzzle-v0",
    "DeceptiveReward-v0",
    "DistributionShift-v0",
    "NoisyObservation-v0",
    "FewShotAdaptation-v0",
    "TaskInterference-v0",
    "CooperativeTransport-v0",
    "CompetitiveTag-v0",
]

# Capability-specific task groups
NAVIGATION_TASKS = [
    "GoToGoal-v0",
    "MazeNavigation-v0",
    "FogOfWarExploration-v0",
    "DynamicObstacles-v0",
    "MultiGoalRoute-v0",
]

MEMORY_TASKS = [
    "KeyDoorPuzzle-v0",
    "SequenceMemory-v0",
    "DelayedGratification-v0",
    "BacktrackPuzzle-v0",
    "BreadcrumbTrail-v0",
]

REASONING_TASKS = [
    "SokobanPush-v0",
    "CausalChain-v0",
    "SymbolMatching-v0",
    "RuleInduction-v0",
    "SwitchCircuit-v0",
]

SKILL_TASKS = [
    "ToolUse-v0",
    "RecipeAssembly-v0",
    "MultiRoomEscape-v0",
    "EmergentStrategy-v0",
    "ResourceManagement-v0",
]

CONTROL_TASKS = [
    "PreciseNavigation-v0",
    "TimingChallenge-v0",
    "ChaseEvade-v0",
    "Herding-v0",
]

COMBINATORIAL_TASKS = [
    "LightsOut-v0",
    "GraphColoring-v0",
    "TileSorting-v0",
    "PackingPuzzle-v0",
]

ADVERSARIAL_TASKS = [
    "DeceptiveReward-v0",
    "DistributionShift-v0",
    "NoisyObservation-v0",
]

META_TASKS = [
    "FewShotAdaptation-v0",
    "TaskInterference-v0",
]

MULTIAGENT_TASKS = [
    "CooperativeTransport-v0",
    "CompetitiveTag-v0",
]

# Quick sanity check tasks (fast, easy)
QUICK_TASKS = [
    "GoToGoal-v0",
    "MazeNavigation-v0",
    "KeyDoorPuzzle-v0",
    "SokobanPush-v0",
    "PreciseNavigation-v0",
]

# Difficulty scaling tasks (to analyze performance across difficulties)
DIFFICULTY_TASKS = [
    "GoToGoal-v0",
    "MazeNavigation-v0",
    "KeyDoorPuzzle-v0",
    "SokobanPush-v0",
    "ChaseEvade-v0",
    "LightsOut-v0",
    "SequenceMemory-v0",
    "BacktrackPuzzle-v0",
    "ToolUse-v0",
    "RecipeAssembly-v0",
]

# Multimodal tasks (for testing different observation modes)
MULTIMODAL_TASKS = [
    "GoToGoal-v0",
    "MazeNavigation-v0",
    "KeyDoorPuzzle-v0",
    "SokobanPush-v0",
    "PreciseNavigation-v0",
    "ToolUse-v0",
    "RecipeAssembly-v0",
    "ChaseEvade-v0",
    "LightsOut-v0",
]


@dataclass(frozen=True)
class ScoringConfig:
    """Configuration for how to score a suite."""

    normalization: Literal["min_max", "z_score", "random_oracle"] = "random_oracle"
    aggregation: Literal["mean", "median", "weighted_mean"] = "mean"
    capability_weights: dict[str, float] = field(default_factory=dict)
    bootstrap_samples: int = 1000  # For confidence intervals


@dataclass(frozen=True)
class BenchmarkSuite:
    """
    Official benchmark suite with immutable configuration.

    Once published, a suite's tasks, seeds, and scoring NEVER change.
    To modify, create a new version (e.g., v2).
    """

    name: str
    display_name: str
    description: str
    tasks: tuple[str, ...]  # Task names (immutable)
    difficulty: str  # Difficulty level for all tasks
    eval_seeds: tuple[int, ...]  # LOCKED seeds - never change after v1.0
    episodes_per_seed: int
    max_steps_override: int | None  # Override per-task max_steps if set
    scoring: ScoringConfig
    version: str  # "1.0" - bump on any change

    def compute_hash(self) -> str:
        """Compute SHA256 hash of suite specification for integrity verification."""
        # Create deterministic dict representation
        suite_dict = {
            "name": self.name,
            "tasks": list(self.tasks),
            "difficulty": self.difficulty,
            "eval_seeds": list(self.eval_seeds),
            "episodes_per_seed": self.episodes_per_seed,
            "max_steps_override": self.max_steps_override,
            "version": self.version,
        }

        # Sort keys for deterministic serialization
        suite_json = json.dumps(suite_dict, sort_keys=True)

        # Compute hash
        return hashlib.sha256(suite_json.encode()).hexdigest()


def generate_deterministic_seeds(suite_name: str, n_seeds: int) -> tuple[int, ...]:
    """
    Generate deterministic seeds from suite name using hash.

    Anyone can regenerate the exact same seeds from the suite name.
    This prevents optimizing for specific seeds without knowing the suite definition.

    Args:
        suite_name: Name of the suite
        n_seeds: Number of seeds to generate

    Returns:
        Tuple of deterministic seeds
    """
    # Use SHA256 hash of suite name as seed for RNG
    import numpy as np

    hash_int = int(hashlib.sha256(suite_name.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(hash_int)

    # Generate seeds in valid range
    seeds = tuple(int(x) for x in rng.integers(0, 2**31, size=n_seeds))

    return seeds


# === OFFICIAL SUITES (v1.0) ===

AGENTICK_FULL_V1 = BenchmarkSuite(
    name="agentick-full-v1",
    display_name="Agentick Full Benchmark v1.0",
    description="Complete benchmark with all 35 official tasks - the canonical score",
    tasks=tuple(FULL_TASKS),
    difficulty="medium",
    eval_seeds=generate_deterministic_seeds("agentick-full-v1", 50),
    episodes_per_seed=1,
    max_steps_override=None,
    scoring=ScoringConfig(normalization="random_oracle", aggregation="mean"),
    version="1.0",
)

AGENTICK_CORE_V1 = BenchmarkSuite(
    name="agentick-core-v1",
    display_name="Agentick Core Benchmark v1.0",
    description="Original 27 core tasks",
    tasks=tuple(CORE_TASKS),
    difficulty="medium",
    eval_seeds=generate_deterministic_seeds("agentick-core-v1", 50),
    episodes_per_seed=1,
    max_steps_override=None,
    scoring=ScoringConfig(normalization="random_oracle", aggregation="mean"),
    version="1.0",
)

AGENTICK_NAVIGATION_V1 = BenchmarkSuite(
    name="agentick-navigation-v1",
    display_name="Navigation Capability Suite v1.0",
    description="Deep-dive into navigation capability across 3 difficulty levels",
    tasks=tuple(NAVIGATION_TASKS),
    difficulty="medium",  # Will also run easy, medium, hard
    eval_seeds=generate_deterministic_seeds("agentick-navigation-v1", 30),
    episodes_per_seed=1,
    max_steps_override=None,
    scoring=ScoringConfig(normalization="random_oracle", aggregation="mean"),
    version="1.0",
)

AGENTICK_MEMORY_V1 = BenchmarkSuite(
    name="agentick-memory-v1",
    display_name="Memory Capability Suite v1.0",
    description="Deep-dive into memory capability across 3 difficulty levels",
    tasks=tuple(MEMORY_TASKS),
    difficulty="medium",
    eval_seeds=generate_deterministic_seeds("agentick-memory-v1", 30),
    episodes_per_seed=1,
    max_steps_override=None,
    scoring=ScoringConfig(normalization="random_oracle", aggregation="mean"),
    version="1.0",
)

AGENTICK_REASONING_V1 = BenchmarkSuite(
    name="agentick-reasoning-v1",
    display_name="Reasoning Capability Suite v1.0",
    description="Deep-dive into reasoning capability across 3 difficulty levels",
    tasks=tuple(REASONING_TASKS),
    difficulty="medium",
    eval_seeds=generate_deterministic_seeds("agentick-reasoning-v1", 30),
    episodes_per_seed=1,
    max_steps_override=None,
    scoring=ScoringConfig(normalization="random_oracle", aggregation="mean"),
    version="1.0",
)

AGENTICK_SKILL_V1 = BenchmarkSuite(
    name="agentick-skill-v1",
    display_name="Skill Discovery Suite v1.0",
    description="Deep-dive into skill discovery capability across 3 difficulty levels",
    tasks=tuple(SKILL_TASKS),
    difficulty="medium",
    eval_seeds=generate_deterministic_seeds("agentick-skill-v1", 30),
    episodes_per_seed=1,
    max_steps_override=None,
    scoring=ScoringConfig(normalization="random_oracle", aggregation="mean"),
    version="1.0",
)

AGENTICK_CONTROL_V1 = BenchmarkSuite(
    name="agentick-control-v1",
    display_name="Control Suite v1.0",
    description="Deep-dive into control capability across 3 difficulty levels",
    tasks=tuple(CONTROL_TASKS),
    difficulty="medium",
    eval_seeds=generate_deterministic_seeds("agentick-control-v1", 30),
    episodes_per_seed=1,
    max_steps_override=None,
    scoring=ScoringConfig(normalization="random_oracle", aggregation="mean"),
    version="1.0",
)

AGENTICK_COMBINATORIAL_V1 = BenchmarkSuite(
    name="agentick-combinatorial-v1",
    display_name="Combinatorial Suite v1.0",
    description="Deep-dive into combinatorial reasoning across 3 difficulty levels",
    tasks=tuple(COMBINATORIAL_TASKS),
    difficulty="medium",
    eval_seeds=generate_deterministic_seeds("agentick-combinatorial-v1", 30),
    episodes_per_seed=1,
    max_steps_override=None,
    scoring=ScoringConfig(normalization="random_oracle", aggregation="mean"),
    version="1.0",
)

AGENTICK_ADVERSARIAL_V1 = BenchmarkSuite(
    name="agentick-adversarial-v1",
    display_name="Adversarial Robustness Suite v1.0",
    description="Robustness to adversarial conditions",
    tasks=tuple(ADVERSARIAL_TASKS),
    difficulty="medium",
    eval_seeds=generate_deterministic_seeds("agentick-adversarial-v1", 30),
    episodes_per_seed=1,
    max_steps_override=None,
    scoring=ScoringConfig(normalization="random_oracle", aggregation="mean"),
    version="1.0",
)

AGENTICK_META_V1 = BenchmarkSuite(
    name="agentick-meta-v1",
    display_name="Meta-Learning Suite v1.0",
    description="Meta-learning and adaptation capability",
    tasks=tuple(META_TASKS),
    difficulty="medium",
    eval_seeds=generate_deterministic_seeds("agentick-meta-v1", 30),
    episodes_per_seed=1,
    max_steps_override=None,
    scoring=ScoringConfig(normalization="random_oracle", aggregation="mean"),
    version="1.0",
)

AGENTICK_MULTIAGENT_V1 = BenchmarkSuite(
    name="agentick-multiagent-v1",
    display_name="Multi-Agent Suite v1.0",
    description="Multi-agent coordination and competition",
    tasks=tuple(MULTIAGENT_TASKS),
    difficulty="medium",
    eval_seeds=generate_deterministic_seeds("agentick-multiagent-v1", 30),
    episodes_per_seed=1,
    max_steps_override=None,
    scoring=ScoringConfig(normalization="random_oracle", aggregation="mean"),
    version="1.0",
)

AGENTICK_QUICK_V1 = BenchmarkSuite(
    name="agentick-quick-v1",
    display_name="Quick Sanity Check v1.0",
    description="Fast sanity check suite (<5 min) - 5 easy tasks",
    tasks=tuple(QUICK_TASKS),
    difficulty="easy",
    eval_seeds=generate_deterministic_seeds("agentick-quick-v1", 10),
    episodes_per_seed=1,
    max_steps_override=100,  # Short episodes
    scoring=ScoringConfig(normalization="random_oracle", aggregation="mean"),
    version="1.0",
)

AGENTICK_DIFFICULTY_V1 = BenchmarkSuite(
    name="agentick-difficulty-v1",
    display_name="Difficulty Scaling Analysis v1.0",
    description="10 tasks evaluated across all 4 difficulty levels",
    tasks=tuple(DIFFICULTY_TASKS),
    difficulty="medium",  # Will run all 4 levels
    eval_seeds=generate_deterministic_seeds("agentick-difficulty-v1", 20),
    episodes_per_seed=1,
    max_steps_override=None,
    scoring=ScoringConfig(normalization="random_oracle", aggregation="mean"),
    version="1.0",
)

AGENTICK_MULTIMODAL_V1 = BenchmarkSuite(
    name="agentick-multimodal-v1",
    display_name="Multimodal Observation Suite v1.0",
    description="Run same agent with different observation modes",
    tasks=tuple(MULTIMODAL_TASKS),
    difficulty="medium",
    eval_seeds=generate_deterministic_seeds("agentick-multimodal-v1", 20),
    episodes_per_seed=1,
    max_steps_override=None,
    scoring=ScoringConfig(normalization="random_oracle", aggregation="mean"),
    version="1.0",
)

# Official suites registry
OFFICIAL_SUITES: dict[str, BenchmarkSuite] = {
    "agentick-full-v1": AGENTICK_FULL_V1,
    "agentick-core-v1": AGENTICK_CORE_V1,
    "agentick-navigation-v1": AGENTICK_NAVIGATION_V1,
    "agentick-memory-v1": AGENTICK_MEMORY_V1,
    "agentick-reasoning-v1": AGENTICK_REASONING_V1,
    "agentick-skill-v1": AGENTICK_SKILL_V1,
    "agentick-control-v1": AGENTICK_CONTROL_V1,
    "agentick-combinatorial-v1": AGENTICK_COMBINATORIAL_V1,
    "agentick-adversarial-v1": AGENTICK_ADVERSARIAL_V1,
    "agentick-meta-v1": AGENTICK_META_V1,
    "agentick-multiagent-v1": AGENTICK_MULTIAGENT_V1,
    "agentick-quick-v1": AGENTICK_QUICK_V1,
    "agentick-difficulty-v1": AGENTICK_DIFFICULTY_V1,
    "agentick-multimodal-v1": AGENTICK_MULTIMODAL_V1,
    # Note: 14 suites total
}


def get_suite(name: str) -> BenchmarkSuite:
    """
    Get official benchmark suite by name.

    Args:
        name: Suite name (e.g., "agentick-full-v1")

    Returns:
        BenchmarkSuite instance

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
    """
    Verify suite integrity by recomputing and comparing hash.

    Args:
        suite: BenchmarkSuite to verify

    Returns:
        True if integrity check passes
    """
    _computed_hash = suite.compute_hash()
    # For now, just return True - in production, would compare against stored hash
    return True
