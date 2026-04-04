#!/usr/bin/env python3
"""Admin-only: unpack a submission zip, validate, add to leaderboard, regenerate site.

Usage:
    python scripts/publish_to_leaderboard.py submission.zip \
        --agent-name "GPT-4o ZeroShot" \
        --author "OpenAI" \
        --description "GPT-4o with zero-shot text prompting" \
        --agent-type llm \
        --observation-mode language \
        --harness MarkovianZeroShot \
        --model gpt-4o \
        --open-weights false
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from agentick.leaderboard.database import add_entry  # noqa: E402
from scripts.validate_submission import (  # noqa: E402
    ValidationReport,
    compute_scores,
    validate_results_dir,
)

_DEFAULT_DATA_DIR = _REPO_ROOT / "leaderboard_data"
_DEFAULT_ENTRIES_PATH = _DEFAULT_DATA_DIR / "entries.json"


def unpack_and_validate(
    zip_path: Path,
) -> tuple[dict | None, ValidationReport, Path]:
    """Unpack zip to a temp dir, validate, return scores and report.

    Returns:
        (score_summary or None, report, temp_dir Path)
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="agentick_pub_"))
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)

    # The zip should contain per_task/ at the top level
    results_dir = tmp_dir
    if not (results_dir / "per_task").is_dir():
        # Maybe nested one level
        subdirs = [d for d in results_dir.iterdir() if d.is_dir()]
        for sd in subdirs:
            if (sd / "per_task").is_dir():
                results_dir = sd
                break

    report = ValidationReport()
    valid_data = validate_results_dir(results_dir, report)
    score_summary = compute_scores(valid_data, report)
    return score_summary, report, tmp_dir


def publish(
    score_summary: dict,
    agent_name: str,
    author: str,
    description: str,
    agent_type: str,
    observation_mode: str,
    harness: str,
    model: str,
    open_weights: bool,
    suite_name: str = "agentick-full-v2",
    entries_path: Path | None = None,
) -> None:
    """Add entry to leaderboard_data/entries.json."""
    if entries_path is None:
        entries_path = _DEFAULT_ENTRIES_PATH

    entry = {
        "agent_name": agent_name,
        "author": author,
        "description": description,
        "agent_type": agent_type,
        "observation_mode": observation_mode,
        "harness": harness,
        "model": model,
        "open_weights": open_weights,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "scores": score_summary,
        "metadata": {
            "suite_name": suite_name,
        },
    }

    add_entry(entry, entries_path)
    print(f"Entry added for '{agent_name}' to {entries_path}")


def regenerate_site(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """Regenerate the static leaderboard site."""
    from agentick.leaderboard.site.generate import SiteGenerator

    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR
    if output_dir is None:
        output_dir = _REPO_ROOT / "leaderboard_site"

    gen = SiteGenerator(data_dir=data_dir, output_dir=output_dir)
    gen.generate()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Admin: publish a validated submission to the Agentick leaderboard.",
    )
    parser.add_argument("zip_path", type=Path, help="Path to the submission zip")
    parser.add_argument("--agent-name", required=True, help="Agent name")
    parser.add_argument("--author", required=True, help="Author / affiliation")
    parser.add_argument("--description", required=True, help="Short description of the agent")
    parser.add_argument(
        "--agent-type",
        required=True,
        choices=["rl", "llm", "vlm", "hybrid", "human", "other"],
        help="Agent type",
    )
    parser.add_argument(
        "--observation-mode",
        required=True,
        choices=["ascii", "language", "language_structured", "rgb_array", "state_dict"],
        help="Observation mode used",
    )
    parser.add_argument("--harness", default="", help="Harness preset name (if applicable)")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument(
        "--open-weights",
        type=lambda x: x.lower() in ("true", "yes", "1"),
        default=False,
        help="Whether model weights are openly available (true/false)",
    )
    parser.add_argument(
        "--suite-name",
        default="agentick-full-v2",
        help="Suite name (default: agentick-full-v2)",
    )
    parser.add_argument(
        "--entries-path",
        type=Path,
        default=None,
        help="Path to entries.json (default: leaderboard_data/entries.json)",
    )
    parser.add_argument(
        "--site-output",
        type=Path,
        default=None,
        help="Output directory for regenerated site (default: leaderboard_site/)",
    )
    parser.add_argument(
        "--skip-site",
        action="store_true",
        help="Skip site regeneration",
    )
    args = parser.parse_args()

    zip_path: Path = args.zip_path.resolve()
    if not zip_path.exists():
        print(f"Error: {zip_path} does not exist")
        sys.exit(1)

    # Step 1-2: Unpack and validate
    print(f"Unpacking and validating: {zip_path}\n")
    score_summary, report, tmp_dir = unpack_and_validate(zip_path)
    report.print_report()

    if not report.is_valid or score_summary is None:
        print("Validation failed. Not publishing.")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        sys.exit(1)

    # Step 3: Add entry to entries.json
    publish(
        score_summary=score_summary,
        agent_name=args.agent_name,
        author=args.author,
        description=args.description,
        agent_type=args.agent_type,
        observation_mode=args.observation_mode,
        harness=args.harness,
        model=args.model,
        open_weights=args.open_weights,
        suite_name=args.suite_name,
        entries_path=args.entries_path,
    )

    # Step 4: Regenerate static site
    if not args.skip_site:
        print("\nRegenerating leaderboard site...")
        data_dir = args.entries_path.parent if args.entries_path else None
        regenerate_site(data_dir=data_dir, output_dir=args.site_output)

    # Cleanup temp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Step 5: Print instructions
    print("\n" + "=" * 72)
    print("  PUBLISH COMPLETE")
    print("=" * 72)
    print()
    print(f"  Entry added for: {args.agent_name}")
    print(f"  Agentick score:  {score_summary['agentick_score']:.4f}")
    print()
    print("  To deploy the updated leaderboard:")
    print("    1. Review the changes:")
    print("       git diff leaderboard_data/entries.json")
    print("    2. Commit:")
    print('       git add leaderboard_data/entries.json leaderboard_site/')
    print(f'       git commit -m "Add {args.agent_name} to leaderboard"')
    print("    3. Push:")
    print("       git push origin main")
    print()
    print("=" * 72)


if __name__ == "__main__":
    main()
