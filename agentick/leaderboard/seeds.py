"""Deterministic seed generation and verification for benchmark suites."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def generate_seeds_from_name(suite_name: str, n_seeds: int) -> tuple[int, ...]:
    """
    Generate deterministic seeds from suite name using cryptographic hash.

    This ensures:
    1. Anyone can regenerate the exact same seeds from the suite name
    2. Seeds are uniformly distributed
    3. Can't optimize for specific seeds without knowing suite definition

    Args:
        suite_name: Name of the suite
        n_seeds: Number of seeds to generate

    Returns:
        Tuple of deterministic seeds in range [0, 2^31)
    """
    # Use SHA256 hash of suite name as seed for RNG
    hash_digest = hashlib.sha256(suite_name.encode()).hexdigest()
    hash_int = int(hash_digest[:16], 16)  # Use first 16 hex chars (64 bits)

    # Create RNG with deterministic seed
    rng = np.random.default_rng(hash_int)

    # Generate seeds in valid range for most RNGs
    seeds = tuple(int(x) for x in rng.integers(0, 2**31, size=n_seeds))

    return seeds


def export_seeds_to_json(output_path: str | Path) -> None:
    """
    Export all official suite seeds to JSON for verification.

    This creates a reference file that anyone can use to verify
    they're using the correct seeds for each suite.

    Args:
        output_path: Path to output JSON file
    """
    from agentick.leaderboard.suites import OFFICIAL_SUITES

    seeds_data = {}

    for suite_name, suite in OFFICIAL_SUITES.items():
        seeds_data[suite_name] = {
            "version": suite.version,
            "n_seeds": len(suite.eval_seeds),
            "seeds": list(suite.eval_seeds),
            "hash": hashlib.sha256(json.dumps(list(suite.eval_seeds)).encode()).hexdigest(),
        }

    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(seeds_data, f, indent=2, sort_keys=True)


def verify_seeds(suite_name: str, seeds: tuple[int, ...]) -> bool:
    """
    Verify that provided seeds match the deterministically generated ones.

    Args:
        suite_name: Name of the suite
        seeds: Seeds to verify

    Returns:
        True if seeds match, False otherwise
    """
    expected_seeds = generate_seeds_from_name(suite_name, len(seeds))
    return seeds == expected_seeds


def load_seeds_from_json(json_path: str | Path) -> dict[str, list[int]]:
    """
    Load seeds from JSON verification file.

    Args:
        json_path: Path to seeds JSON file

    Returns:
        Dictionary mapping suite name to list of seeds
    """
    with open(json_path) as f:
        seeds_data = json.load(f)

    return {name: data["seeds"] for name, data in seeds_data.items()}


def verify_seeds_file(json_path: str | Path) -> dict[str, bool]:
    """
    Verify all seeds in a JSON file against deterministic generation.

    Args:
        json_path: Path to seeds JSON file

    Returns:
        Dictionary mapping suite name to verification status
    """
    seeds_from_file = load_seeds_from_json(json_path)

    results = {}
    for suite_name, seeds in seeds_from_file.items():
        results[suite_name] = verify_seeds(suite_name, tuple(seeds))

    return results
