#!/usr/bin/env python
"""Admin tool: publish agent results to the Agentick leaderboard.

This script is for the leaderboard admin only. Users should use
scripts/validate_submission.py to validate and package their results,
then email the zip to the admin for publication.

Supports publishing results for a single task+difficulty, a full category,
or a complete benchmark run. Partial results appear only in the per-task
breakdown. Full benchmark and category scores require complete data.

Usage:
    # Preview what will be published (dry-run)
    uv run python scripts/publish_results.py results.json

    # Publish (writes to leaderboard_data/entries.json)
    uv run python scripts/publish_results.py results.json --publish

    # Merge new task/difficulty data into an existing agent's entry
    uv run python scripts/publish_results.py results.json --publish --merge "Agent Name"

Input JSON format:
    {
        "agent_name": "My Agent",
        "author": "Your Name",
        "agent_type": "llm",           # llm, vlm, rl, hybrid, other
        "observation_mode": "ascii",    # ascii, language, rgb_array, etc.
        "harness": "MarkovianReasoner", # or "" if N/A
        "model": "gpt-4o",
        "open_weights": false,
        "results": {
            "GoToGoal-v0": {
                "easy": {"success_rate": 0.8, "mean_return": 0.75, "n_episodes": 25}
            }
        }
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

# Add repo root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentick.leaderboard.database import load_entries, save_entries
from agentick.leaderboard.scoring import TASK_CAPABILITY_MAP

ALL_TASKS = sorted(TASK_CAPABILITY_MAP.keys())
ALL_DIFFICULTIES = ["easy", "medium", "hard", "expert"]
ALL_CATEGORIES = sorted(set(TASK_CAPABILITY_MAP.values()))
CATEGORY_TASKS = {}
for t, c in TASK_CAPABILITY_MAP.items():
    CATEGORY_TASKS.setdefault(c, []).append(t)


def validate_results(data: dict) -> list[str]:
    """Validate input JSON. Returns list of error strings (empty = valid)."""
    errors = []

    for key in ["agent_name", "author", "agent_type", "results"]:
        if key not in data:
            errors.append(f"Missing required field: {key}")

    if data.get("agent_type") not in ("llm", "vlm", "rl", "hybrid", "other"):
        errors.append(
            f"Invalid agent_type: {data.get('agent_type')!r}. "
            "Must be one of: llm, vlm, rl, hybrid, other"
        )

    results = data.get("results", {})
    if not results:
        errors.append("No results provided (empty 'results' dict)")
        return errors

    for task_name, diffs in results.items():
        if task_name not in TASK_CAPABILITY_MAP:
            errors.append(f"Unknown task: {task_name}")
            continue
        if not isinstance(diffs, dict):
            errors.append(f"{task_name}: expected dict of difficulties, got {type(diffs).__name__}")
            continue
        for diff, scores in diffs.items():
            if diff not in ALL_DIFFICULTIES:
                errors.append(f"{task_name}.{diff}: unknown difficulty")
                continue
            if not isinstance(scores, dict):
                errors.append(f"{task_name}.{diff}: expected dict with success_rate/mean_return")
                continue
            if "success_rate" not in scores and "mean_return" not in scores:
                errors.append(f"{task_name}.{diff}: must have 'success_rate' or 'mean_return'")
            sr = scores.get("success_rate", 0)
            if not (0 <= sr <= 1):
                errors.append(f"{task_name}.{diff}: success_rate {sr} not in [0, 1]")

    return errors


def compute_coverage(results: dict) -> dict:
    """Compute what's covered: which tasks, categories, and overall."""
    tasks_complete = []  # tasks with all 4 difficulties
    tasks_partial = []   # tasks with some difficulties

    for task_name in ALL_TASKS:
        if task_name not in results:
            continue
        diffs_present = [d for d in ALL_DIFFICULTIES if d in results[task_name]]
        if len(diffs_present) == 4:
            tasks_complete.append(task_name)
        elif diffs_present:
            tasks_partial.append(task_name)

    categories_complete = []
    for cat in ALL_CATEGORIES:
        cat_tasks = CATEGORY_TASKS[cat]
        if all(t in tasks_complete for t in cat_tasks):
            categories_complete.append(cat)

    overall_complete = len(tasks_complete) == len(ALL_TASKS)

    return {
        "tasks_complete": tasks_complete,
        "tasks_partial": tasks_partial,
        "categories_complete": categories_complete,
        "overall_complete": overall_complete,
        "n_task_diffs": sum(
            len([d for d in ALL_DIFFICULTIES if d in results.get(t, {})])
            for t in ALL_TASKS
        ),
        "n_total": len(ALL_TASKS) * len(ALL_DIFFICULTIES),
    }


def compute_category_score(results: dict, category: str) -> float:
    """Mean success_rate across all tasks in a category (all difficulties)."""
    cat_tasks = CATEGORY_TASKS[category]
    scores = []
    for task in cat_tasks:
        for diff in ALL_DIFFICULTIES:
            d = results.get(task, {}).get(diff, {})
            scores.append(d.get("success_rate", 0.0))
    return sum(scores) / len(scores) if scores else 0.0


def compute_overall_score(results: dict) -> float:
    """Mean of category scores (equal category weighting)."""
    cat_scores = [compute_category_score(results, c) for c in ALL_CATEGORIES]
    return sum(cat_scores) / len(cat_scores) if cat_scores else 0.0


def build_entry(data: dict, merge_target: dict | None = None) -> dict:
    """Build a leaderboard entry from input data, optionally merging."""
    results = data["results"]

    # If merging, combine results
    if merge_target:
        existing_results = merge_target.get("scores", {}).get("per_task", {})
        merged = {}
        for task in ALL_TASKS:
            merged[task] = {}
            # Start with existing
            if task in existing_results:
                merged[task].update(existing_results[task])
            # Overlay new
            if task in results:
                merged[task].update(results[task])
            if not merged[task]:
                del merged[task]
        results = merged

    coverage = compute_coverage(results)

    # Build per_category (only for complete categories)
    per_category = {}
    for cat in ALL_CATEGORIES:
        if cat in coverage["categories_complete"]:
            per_category[cat] = round(compute_category_score(results, cat), 4)

    # Build overall score (only if all tasks present)
    agentick_score = 0.0
    agentick_score_ci = [0.0, 0.0]
    if coverage["overall_complete"]:
        agentick_score = round(compute_overall_score(results), 4)

    entry = {
        "agent_name": data.get("agent_name", merge_target.get("agent_name", "") if merge_target else ""),
        "author": data.get("author", merge_target.get("author", "") if merge_target else ""),
        "description": data.get("description", data.get("agent_name", "")),
        "agent_type": data.get("agent_type", merge_target.get("agent_type", "") if merge_target else ""),
        "observation_mode": data.get("observation_mode", merge_target.get("observation_mode", "") if merge_target else ""),
        "harness": data.get("harness", merge_target.get("harness", "") if merge_target else ""),
        "model": data.get("model", merge_target.get("model", "") if merge_target else ""),
        "open_weights": data.get("open_weights", merge_target.get("open_weights", False) if merge_target else False),
        "date": str(date.today()),
        "scores": {
            "agentick_score": agentick_score,
            "agentick_score_ci": agentick_score_ci,
            "per_category": per_category,
            "per_task": results,
        },
    }
    return entry


def print_verification(data: dict, coverage: dict, entry: dict) -> None:
    """Print what will be published for verification."""
    results = data["results"]

    print()
    print("=" * 60)
    print("  LEADERBOARD PUBLISH VERIFICATION")
    print("=" * 60)
    print()
    print(f"  Agent:     {data.get('agent_name', '?')}")
    print(f"  Author:    {data.get('author', '?')}")
    print(f"  Type:      {data.get('agent_type', '?')}")
    print(f"  Modality:  {data.get('observation_mode', '?')}")
    print(f"  Harness:   {data.get('harness', '') or '-'}")
    print(f"  Model:     {data.get('model', '?')}")
    print()
    print(f"  Coverage:  {coverage['n_task_diffs']}/{coverage['n_total']} task-difficulty pairs")
    print(f"  Tasks:     {len(coverage['tasks_complete'])} complete, "
          f"{len(coverage['tasks_partial'])} partial")
    print()

    # Show what will appear where
    print("  WILL APPEAR IN:")
    print("  ─────────────────────────────────────────────")

    if coverage["overall_complete"]:
        score = entry["scores"]["agentick_score"]
        print(f"  ✓ Full Benchmark    score={score:.4f}")
    else:
        print("  ✗ Full Benchmark    (need all 37 tasks × 4 difficulties)")

    for cat in ALL_CATEGORIES:
        cat_tasks = CATEGORY_TASKS[cat]
        if cat in coverage["categories_complete"]:
            score = entry["scores"]["per_category"][cat]
            print(f"  ✓ {cat:<18s} score={score:.4f}")
        else:
            present = sum(1 for t in cat_tasks if t in coverage["tasks_complete"])
            print(f"  ✗ {cat:<18s} ({present}/{len(cat_tasks)} tasks complete)")

    print()
    print("  PER-TASK RESULTS:")
    print("  ─────────────────────────────────────────────")
    for task_name in sorted(results.keys()):
        diffs = results[task_name]
        parts = []
        for diff in ALL_DIFFICULTIES:
            if diff in diffs:
                sr = diffs[diff].get("success_rate", 0)
                parts.append(f"{diff[0].upper()}={sr:.0%}")
            else:
                parts.append(f"{diff[0].upper()}=–")
        cat = TASK_CAPABILITY_MAP.get(task_name, "?")
        print(f"  {task_name:<30s} [{cat}]  {', '.join(parts)}")

    print()
    print("  DESTINATION:")
    entries_path = Path("leaderboard_data/entries.json").resolve()
    print(f"  {entries_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Publish agent results to the Agentick leaderboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("results_json", help="Path to results JSON file")
    parser.add_argument(
        "--publish", action="store_true",
        help="Actually publish (admin only). Without this flag, only verifies.",
    )
    parser.add_argument(
        "--merge", metavar="AGENT_NAME",
        help="Merge results into an existing agent's entry (add new task/difficulty data)",
    )
    parser.add_argument(
        "--entries", default="leaderboard_data/entries.json",
        help="Path to entries.json (default: leaderboard_data/entries.json)",
    )
    args = parser.parse_args()

    # Load input
    results_path = Path(args.results_json)
    if not results_path.exists():
        print(f"Error: {results_path} not found")
        sys.exit(1)

    with open(results_path) as f:
        data = json.load(f)

    # Validate
    errors = validate_results(data)
    if errors:
        print("Validation errors:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)

    # Find merge target if requested
    merge_target = None
    if args.merge:
        entries = load_entries(args.entries)
        for e in entries:
            if e.get("agent_name") == args.merge:
                merge_target = e
                break
        if merge_target is None:
            print(f"Error: no existing entry found for agent '{args.merge}'")
            sys.exit(1)

    # Build entry and compute coverage
    results = data["results"]
    if merge_target:
        # For coverage display, show merged results
        existing = merge_target.get("scores", {}).get("per_task", {})
        display_results = {}
        for t in ALL_TASKS:
            display_results[t] = {}
            if t in existing:
                display_results[t].update(existing[t])
            if t in results:
                display_results[t].update(results[t])
            if not display_results[t]:
                del display_results[t]
        coverage = compute_coverage(display_results)
    else:
        coverage = compute_coverage(results)

    entry = build_entry(data, merge_target)

    # Always show verification
    print_verification(data, coverage, entry)

    if not args.publish:
        print("  DRY RUN — no changes made.")
        print("  Add --publish to write to the leaderboard.")
        print()
        return

    # Publish
    entries = load_entries(args.entries)

    if merge_target:
        # Replace existing entry
        entries = [e for e in entries if e.get("agent_name") != args.merge]

    entries.append(entry)
    save_entries(entries, args.entries)

    print(f"  ✓ PUBLISHED to {Path(args.entries).resolve()}")
    print(f"  Entry count: {len(entries)}")
    print()


if __name__ == "__main__":
    main()
