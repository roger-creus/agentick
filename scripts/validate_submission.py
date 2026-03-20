#!/usr/bin/env python3
"""Validate an Agentick evaluation results directory and package for submission.

Usage:
    python scripts/validate_submission.py results/my_agent/
    python scripts/validate_submission.py results/my_agent/ --agent-name "GPT-4o ZeroShot"
    python scripts/validate_submission.py results/my_agent/ --skip-packaging

Auto-detects two directory formats:

  1. ExperimentRunner output (preferred — just point at the run directory):
        per_task/GoToGoal-v0/metrics.json   (nested per_difficulty with episodes)

  2. Flat submission format:
        per_task/GoToGoal-v0/easy.json      (with seeds, episode_returns, success_flags)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: make the agentick package importable regardless of cwd
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from agentick.leaderboard.scoring import (  # noqa: E402
    TASK_CAPABILITY_MAP,
    bootstrap_confidence_interval,
)
from agentick.leaderboard.seeds import generate_task_seeds  # noqa: E402
from agentick.tasks.registry import list_tasks  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DIFFICULTIES = ("easy", "medium", "hard", "expert")
EXPECTED_SEEDS_PER_DIFF = 25
SUBMISSION_EMAIL = "roger.creus-castanyer@mila.quebec"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

class ValidationReport:
    """Accumulates pass/warn/fail messages."""

    def __init__(self) -> None:
        self.passes: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def ok(self, msg: str) -> None:
        self.passes.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def print_report(self) -> None:
        width = 72
        print("\n" + "=" * width)
        print("  AGENTICK SUBMISSION VALIDATION REPORT")
        print("=" * width)

        if self.passes:
            print(f"\n  PASSED ({len(self.passes)})")
            for msg in self.passes:
                print(f"    [PASS] {msg}")

        if self.warnings:
            print(f"\n  WARNINGS ({len(self.warnings)})")
            for msg in self.warnings:
                print(f"    [WARN] {msg}")

        if self.errors:
            print(f"\n  ERRORS ({len(self.errors)})")
            for msg in self.errors:
                print(f"    [FAIL] {msg}")

        print("\n" + "-" * width)
        if self.is_valid:
            print("  RESULT: VALID -- ready to package")
        else:
            print("  RESULT: INVALID -- fix errors above before submitting")
        print("=" * width + "\n")


def _load_json(path: Path) -> dict | None:
    """Load and return a JSON file, or None on error."""
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _load_difficulty_data(
    task_dir: Path, task_name: str, diff: str
) -> dict | None:
    """Load difficulty data, auto-detecting format.

    Supports three layouts:
      1. Flat submission format:   {difficulty}.json with seeds/episode_returns/success_flags
      2. ExperimentRunner output:  metrics.json with nested per_difficulty.episodes[]
      3. TrainingBenchmarkRunner:  {difficulty}/metrics.json with eval_returns[]

    Returns a normalized dict with keys: seeds, episode_returns, success_flags
    or None if not found.
    """
    # --- Format 1: flat {diff}.json (submission format) ---
    diff_file = task_dir / f"{diff}.json"
    if diff_file.exists():
        return _load_json(diff_file)

    # --- Format 2: metrics.json from ExperimentRunner ---
    metrics_file = task_dir / "metrics.json"
    if metrics_file.exists():
        data = _load_json(metrics_file)
        if data is not None:
            per_diff = data.get("per_difficulty", {})
            diff_data = per_diff.get(diff)
            if diff_data is not None:
                episodes = diff_data.get("episodes", [])
                if episodes:
                    return {
                        "task_name": task_name,
                        "difficulty": diff,
                        "seeds": [ep["seed"] for ep in episodes],
                        "episode_returns": [ep["return"] for ep in episodes],
                        "success_flags": [ep["success"] for ep in episodes],
                        "episode_steps": [ep.get("length", 0) for ep in episodes],
                    }

    # --- Format 3: {difficulty}/metrics.json from TrainingBenchmarkRunner ---
    # PPO runner writes per_task/{Task}/{difficulty}/metrics.json with
    # eval_returns (list of per-seed returns) and success_rate (aggregate).
    diff_dir_metrics = task_dir / diff / "metrics.json"
    if diff_dir_metrics.exists():
        data = _load_json(diff_dir_metrics)
        if data is not None:
            eval_returns = data.get("eval_returns", [])
            if eval_returns:
                # Reconstruct per-seed data from the eval seeds
                eval_seeds = list(
                    generate_task_seeds(task_name, diff, "eval", len(eval_returns))
                )
                # success_flags: PPO runner stores aggregate success_rate but not
                # per-episode flags. Use return > 0 as proxy.
                success_flags = [r > 0 for r in eval_returns]
                return {
                    "task_name": task_name,
                    "difficulty": diff,
                    "seeds": eval_seeds,
                    "episode_returns": eval_returns,
                    "success_flags": success_flags,
                }

    return None


def validate_results_dir(
    results_dir: Path,
    report: ValidationReport,
) -> dict[str, dict[str, dict]]:
    """Validate the results directory structure and contents.

    Auto-detects format: ExperimentRunner (metrics.json), TrainingBenchmarkRunner
    ({difficulty}/metrics.json), or flat submission format ({difficulty}.json).

    Returns:
        Nested dict: task_name -> difficulty -> normalized data dict.
        Only includes entries that passed validation.
    """
    all_tasks = sorted(list_tasks())
    per_task_dir = results_dir / "per_task"

    # ------------------------------------------------------------------
    # 1. Check per_task/ directory exists
    # ------------------------------------------------------------------
    if not per_task_dir.is_dir():
        report.error(f"Missing directory: {per_task_dir}")
        return {}
    report.ok("per_task/ directory found")

    # ------------------------------------------------------------------
    # 2. Check all tasks present
    # ------------------------------------------------------------------
    found_tasks = sorted(
        d.name for d in per_task_dir.iterdir() if d.is_dir()
    )
    missing_tasks = sorted(set(all_tasks) - set(found_tasks))
    extra_tasks = sorted(set(found_tasks) - set(all_tasks))

    if missing_tasks:
        report.error(f"Missing tasks ({len(missing_tasks)}): {', '.join(missing_tasks)}")
    else:
        report.ok(f"All {len(all_tasks)} tasks present under per_task/")

    if extra_tasks:
        report.warn(f"Extra (unknown) task dirs will be ignored: {', '.join(extra_tasks)}")

    # ------------------------------------------------------------------
    # 3. For each task, check all 4 difficulties
    # ------------------------------------------------------------------
    valid_data: dict[str, dict[str, dict]] = {}
    total_episodes = 0
    seed_mismatches = 0

    for task_name in all_tasks:
        task_dir = per_task_dir / task_name
        if not task_dir.is_dir():
            continue  # already flagged above

        valid_data[task_name] = {}

        for diff in DIFFICULTIES:
            data = _load_difficulty_data(task_dir, task_name, diff)
            if data is None:
                report.error(f"{task_name}/{diff}: no data found")
                continue

            # --- Check required keys ---
            required = {"seeds", "episode_returns", "success_flags"}
            missing_keys = required - set(data.keys())
            if missing_keys:
                report.error(
                    f"{task_name}/{diff} missing keys: {missing_keys}"
                )
                continue

            # --- Check seed count ---
            seeds = data["seeds"]
            returns = data["episode_returns"]
            flags = data["success_flags"]

            if len(seeds) != EXPECTED_SEEDS_PER_DIFF:
                report.error(
                    f"{task_name}/{diff}: expected {EXPECTED_SEEDS_PER_DIFF} seeds, "
                    f"got {len(seeds)}"
                )

            if len(returns) != len(seeds):
                report.error(
                    f"{task_name}/{diff}: episode_returns length ({len(returns)}) "
                    f"!= seeds length ({len(seeds)})"
                )

            if len(flags) != len(seeds):
                report.error(
                    f"{task_name}/{diff}: success_flags length ({len(flags)}) "
                    f"!= seeds length ({len(seeds)})"
                )

            # --- Verify seeds match official eval seeds ---
            expected_seeds = list(
                generate_task_seeds(task_name, diff, "eval", EXPECTED_SEEDS_PER_DIFF)
            )
            if list(seeds) != expected_seeds:
                mismatch_idx = next(
                    (i for i, (a, b) in enumerate(zip(seeds, expected_seeds)) if a != b),
                    "?",
                )
                report.error(
                    f"{task_name}/{diff}: seeds do not match official eval seeds "
                    f"(first mismatch at index {mismatch_idx})"
                )
                seed_mismatches += 1
            else:
                total_episodes += len(returns)
                valid_data[task_name][diff] = data

    # Summary counts
    expected_total = len(all_tasks) * len(DIFFICULTIES) * EXPECTED_SEEDS_PER_DIFF
    if total_episodes == expected_total:
        report.ok(
            f"All {total_episodes} episodes present "
            f"({len(all_tasks)} tasks x {len(DIFFICULTIES)} difficulties x "
            f"{EXPECTED_SEEDS_PER_DIFF} seeds)"
        )
    else:
        report.warn(
            f"Valid episodes: {total_episodes}/{expected_total} "
            f"({expected_total - total_episodes} missing or invalid)"
        )

    if seed_mismatches == 0 and not missing_tasks:
        report.ok("All seeds match official eval seeds from leaderboard/seeds.py")

    return valid_data


def compute_scores(
    valid_data: dict[str, dict[str, dict]],
    report: ValidationReport,
) -> dict | None:
    """Compute aggregate scores from validated per-task data.

    Uses success_rate as the normalized score (no external baselines needed).

    Returns:
        Score summary dict, or None if too little data.
    """
    if not valid_data:
        report.error("No valid data to compute scores from")
        return None

    # For the leaderboard we use success_rate as the primary metric.
    # Each (task, difficulty) contributes equally.
    task_scores_map: dict[str, float] = {}  # "task::diff" -> success_rate
    per_task_summary: dict[str, dict] = {}
    per_category_summary: dict[str, list[float]] = {}

    for task_name, diffs in sorted(valid_data.items()):
        per_task_summary[task_name] = {}
        category = TASK_CAPABILITY_MAP.get(task_name, "other")
        if category not in per_category_summary:
            per_category_summary[category] = []

        for diff in DIFFICULTIES:
            if diff not in diffs:
                continue
            data = diffs[diff]
            flags = data["success_flags"]
            returns = data["episode_returns"]
            sr = sum(1 for f in flags if f) / len(flags) if flags else 0.0
            mean_ret = sum(returns) / len(returns) if returns else 0.0

            key = f"{task_name}::{diff}"
            task_scores_map[key] = sr
            per_task_summary[task_name][diff] = {
                "success_rate": round(sr, 4),
                "mean_return": round(mean_ret, 4),
                "n_episodes": len(returns),
            }
            per_category_summary[category].append(sr)

    # Per-category mean
    per_category_scores: dict[str, float] = {}
    for cat, scores in per_category_summary.items():
        per_category_scores[cat] = round(sum(scores) / len(scores), 4) if scores else 0.0

    # Agentick score = mean of per-category means (capability-weighted)
    if per_category_scores:
        cat_means = list(per_category_scores.values())
        agentick_score = sum(cat_means) / len(cat_means)
        ci = bootstrap_confidence_interval(cat_means)
    else:
        agentick_score = 0.0
        ci = (0.0, 0.0)

    score_summary = {
        "agentick_score": round(agentick_score, 4),
        "agentick_score_ci": [round(ci[0], 4), round(ci[1], 4)],
        "per_category": per_category_scores,
        "per_task": per_task_summary,
    }

    report.ok(f"Agentick score: {agentick_score:.4f} (95% CI: {ci[0]:.4f}-{ci[1]:.4f})")
    for cat, score in sorted(per_category_scores.items()):
        report.ok(f"  {cat}: {score:.4f}")

    return score_summary


def package_submission(
    results_dir: Path,
    score_summary: dict,
    agent_name: str,
    output_dir: Path | None = None,
) -> Path:
    """Package results into a submission zip file.

    Args:
        results_dir: The validated results directory.
        score_summary: Computed scores dict.
        agent_name: Name of the agent.
        output_dir: Where to write the zip. Defaults to results_dir parent.

    Returns:
        Path to the created zip file.
    """
    if output_dir is None:
        output_dir = results_dir.parent

    safe_name = agent_name.replace(" ", "_").replace("/", "_")
    date_str = datetime.now().strftime("%Y%m%d")
    zip_name = f"agentick_submission_{safe_name}_{date_str}.zip"
    zip_path = output_dir / zip_name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add per_task results
        per_task_dir = results_dir / "per_task"
        for root, _dirs, files in os.walk(per_task_dir):
            for fname in files:
                fpath = Path(root) / fname
                arcname = str(fpath.relative_to(results_dir))
                zf.write(fpath, arcname)

        # Add computed score summary
        score_json = json.dumps(score_summary, indent=2)
        zf.writestr("score_summary.json", score_json)

        # Add metadata
        meta = {
            "agent_name": agent_name,
            "date": datetime.now().isoformat(),
            "n_tasks": len(score_summary.get("per_task", {})),
            "agentick_score": score_summary.get("agentick_score"),
        }
        zf.writestr("submission_meta.json", json.dumps(meta, indent=2))

    return zip_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Agentick evaluation results and package for submission.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Path to the results directory (must contain per_task/ subdirectory)",
    )
    parser.add_argument(
        "--agent-name",
        type=str,
        default="MyAgent",
        help="Name of the agent (used in submission zip filename)",
    )
    parser.add_argument(
        "--skip-packaging",
        action="store_true",
        help="Only validate, do not create the submission zip",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the submission zip (default: parent of results_dir)",
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir.resolve()
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory")
        sys.exit(1)

    report = ValidationReport()

    # Step 1-4: Validate structure, tasks, difficulties, seeds, episode counts
    print(f"Validating results in: {results_dir}\n")
    valid_data = validate_results_dir(results_dir, report)

    # Step 5: Compute scores
    score_summary = compute_scores(valid_data, report)

    # Print detailed report
    report.print_report()

    if not report.is_valid:
        print("Fix the errors above and re-run this script.")
        sys.exit(1)

    if score_summary is None:
        print("Could not compute scores. Aborting.")
        sys.exit(1)

    # Step 8: Package into submission zip
    if not args.skip_packaging:
        zip_path = package_submission(
            results_dir,
            score_summary,
            args.agent_name,
            output_dir=args.output_dir,
        )
        print(f"Submission packaged: {zip_path}")
        print(f"  Size: {zip_path.stat().st_size / 1024:.1f} KB\n")

        # Step 9: Print email instructions
        print("=" * 72)
        print("  HOW TO SUBMIT")
        print("=" * 72)
        print()
        print("  1. Email your submission zip to:")
        print(f"     {SUBMISSION_EMAIL}")
        print()
        print("  2. Subject line:")
        print(f"     [Agentick Submission] {args.agent_name}")
        print()
        print("  3. In the email body, include:")
        print(f"     - Agent name: {args.agent_name}")
        print("     - Your name / affiliation")
        print("     - Brief description of the agent")
        print("     - Agent type (rl / llm / vlm / hybrid / human / other)")
        print("     - Observation mode used (ascii / language / rgb_array / state_dict)")
        print("     - Harness preset (if using agentick agent harness)")
        print("     - Model name (e.g. gpt-4o, claude-3.5-sonnet, custom-cnn)")
        print("     - Whether model weights are open (yes/no)")
        print()
        print("  4. Attach the zip file:")
        print(f"     {zip_path.name}")
        print()
        print("=" * 72)
    else:
        print("Packaging skipped (--skip-packaging). Results are valid.")


if __name__ == "__main__":
    main()
