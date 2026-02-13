#!/usr/bin/env python3
"""
07_generate_leaderboard.py - Compile all evaluation results into leaderboard data

Reads all JSON files from leaderboard_data/entries/ and generates:
- Aggregated statistics
- Rankings
- Capability scores
- Leaderboard metadata

Output: leaderboard_data/leaderboard.json (used by site generation)
"""

import json
import sys
from pathlib import Path
from typing import Any


def load_results(entries_dir: Path) -> list[dict[str, Any]]:
    """Load all result JSON files from entries directory."""
    results = []

    if not entries_dir.exists():
        print(f"Error: Entries directory not found: {entries_dir}")
        sys.exit(1)

    json_files = list(entries_dir.glob("*.json"))

    if not json_files:
        print(f"Warning: No JSON files found in {entries_dir}")
        print("Run evaluation scripts first (01_run_baselines.sh, etc.)")
        sys.exit(1)

    print(f"Loading {len(json_files)} result files...")

    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)
                results.append(data)
                print(f"  ✓ {json_file.name}")
        except Exception as e:
            print(f"  ✗ Error loading {json_file.name}: {e}")

    return results


def compute_rankings(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute rankings based on overall score."""
    # Sort by overall score (descending)
    ranked = sorted(results, key=lambda x: x.get("overall_score", 0), reverse=True)

    # Add rank field
    for i, result in enumerate(ranked):
        result["rank"] = i + 1

    return ranked


def generate_leaderboard_data(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate complete leaderboard data structure."""
    from datetime import datetime

    ranked_results = compute_rankings(results)

    leaderboard = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "num_submissions": len(results),
            "version": "1.0",
        },
        "entries": ranked_results,
        "statistics": {
            "mean_overall_score": sum(r.get("overall_score", 0) for r in results)
            / len(results),
            "best_overall_score": max(r.get("overall_score", 0) for r in results),
        },
    }

    return leaderboard


def main():
    """Main entry point."""
    print("=" * 60)
    print("Generating Leaderboard Data")
    print("=" * 60)
    print()

    # Paths
    entries_dir = Path("leaderboard_data/entries")
    output_file = Path("leaderboard_data/leaderboard.json")

    # Load results
    results = load_results(entries_dir)

    # Generate leaderboard
    print("\nComputing rankings...")
    leaderboard = generate_leaderboard_data(results)

    # Save output
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(leaderboard, f, indent=2)

    print(f"\n✓ Leaderboard data saved to: {output_file}")
    print(f"  {leaderboard['metadata']['num_submissions']} submissions")
    print(f"  Mean score: {leaderboard['statistics']['mean_overall_score']:.3f}")
    print(f"  Best score: {leaderboard['statistics']['best_overall_score']:.3f}")

    # Print top 5
    print("\nTop 5 submissions:")
    for entry in leaderboard["entries"][:5]:
        print(
            f"  {entry['rank']}. {entry.get('agent_name', 'Unknown')} "
            f"- {entry.get('overall_score', 0):.3f}"
        )

    print("\nNext step: bash scripts/populate_leaderboard/08_generate_site.sh")


if __name__ == "__main__":
    main()
