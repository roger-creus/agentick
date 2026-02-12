#!/usr/bin/env python3
"""Populate leaderboard with baseline entries."""

from pathlib import Path

from agentick.leaderboard.database import LeaderboardDatabase
from agentick.leaderboard.rankings import compute_rankings


def main():
    """Generate initial leaderboard from baseline results."""
    print("=== Populating Leaderboard ===\n")

    # Initialize database
    db = LeaderboardDatabase("leaderboard_data")

    # Get all entries
    entries = db.get_entries()

    if not entries:
        print("No entries found. Run evaluations first.")
        return

    print(f"Found {len(entries)} entries")

    # Compute rankings per suite
    suites = set(entry.suite_name for entry in entries)

    for suite_name in suites:
        print(f"\nComputing rankings for {suite_name}...")

        suite_entries = db.get_entries(suite_name)
        rankings = compute_rankings(suite_entries)

        # Save rankings
        db.save_rankings(suite_name, rankings)

        print(f"  ✓ {len(rankings)} agents ranked")

    # Create snapshot
    db.create_snapshot()

    print("\n✓ Leaderboard populated!")
    print("Run site generator next: python -m agentick.leaderboard.site.generate")


if __name__ == "__main__":
    main()
