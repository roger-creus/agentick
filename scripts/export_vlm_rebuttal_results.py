#!/usr/bin/env python3
"""Export the controlled Qwen3.5-4B ASCII-versus-pixel rebuttal tables."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path
from typing import Any

from agentick.leaderboard.scoring import TASK_CAPABILITY_MAP

DIFFICULTIES = ("easy", "medium", "hard", "expert")
CATEGORIES = (
    "navigation",
    "planning",
    "reasoning",
    "memory",
    "generalization",
    "multi_agent",
)
ASCII_ENTRY_KEYS = {
    "agent_name": "Qwen3.5-4B",
    "agent_type": "llm",
    "observation_mode": "ascii",
    "harness": "markovian_zero_shot",
}


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def load_ascii_summary(entries_path: Path) -> dict[str, Any]:
    """Load the one baseline entry matching all controlled-comparison keys."""
    raw = _load_json(entries_path)
    entries = raw.get("entries", []) if isinstance(raw, dict) else raw
    matches = [
        entry
        for entry in entries
        if all(entry.get(key) == value for key, value in ASCII_ENTRY_KEYS.items())
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one ASCII baseline matching "
            f"{ASCII_ENTRY_KEYS}, found {len(matches)}"
        )
    summary = matches[0].get("scores")
    if not isinstance(summary, dict):
        raise ValueError("Matched ASCII entry has no score summary")
    return summary


def load_pixel_summary(path: Path) -> dict[str, Any]:
    """Load score_summary.json from a validated ZIP or a standalone JSON file."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            with archive.open("score_summary.json") as f:
                summary = json.load(f)
    else:
        summary = _load_json(path)
    if not isinstance(summary, dict):
        raise ValueError("Pixel score summary must be a JSON object")
    return summary


def difficulty_averages(summary: dict[str, Any]) -> dict[str, float]:
    """Compute category-balanced success for each difficulty."""
    per_task = summary.get("per_task", {})
    result: dict[str, float] = {}
    for difficulty in DIFFICULTIES:
        category_means = []
        for category in CATEGORIES:
            values = [
                float(task_scores[difficulty]["success_rate"])
                for task_name, task_scores in per_task.items()
                if TASK_CAPABILITY_MAP.get(task_name) == category
                and difficulty in task_scores
            ]
            if not values:
                raise ValueError(f"No {category}/{difficulty} task scores found")
            category_means.append(sum(values) / len(values))
        result[difficulty] = sum(category_means) / len(category_means)
    return result


def comparison_row(observation: str, summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "Agent": "Qwen3.5-4B, direct action",
        "Observation": observation,
    }
    row.update(difficulty_averages(summary))
    row["overall"] = float(summary["agentick_score"])
    return row


def category_row(observation: str, summary: dict[str, Any]) -> dict[str, Any]:
    per_category = summary.get("per_category", {})
    return {
        "Observation": observation,
        **{category: float(per_category[category]) for category in CATEGORIES},
        "overall": float(summary["agentick_score"]),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _display(value: float) -> str:
    return f"{value:.3f}"


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Agent | Observation | Easy | Medium | Hard | Expert | Overall |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {Agent} | {Observation} | {easy} | {medium} | {hard} | "
            "{expert} | {overall} |".format(
                Agent=row["Agent"],
                Observation=row["Observation"],
                **{key: _display(float(row[key])) for key in (*DIFFICULTIES, "overall")},
            )
        )
    return "\n".join(lines) + "\n"


def export_tables(
    entries_path: Path,
    pixel_summary_path: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    ascii_summary = load_ascii_summary(entries_path)
    pixel_summary = load_pixel_summary(pixel_summary_path)
    comparison_rows = [
        comparison_row("ASCII", ascii_summary),
        comparison_row("Pixels", pixel_summary),
    ]
    category_rows = [
        category_row("ASCII", ascii_summary),
        category_row("Pixels", pixel_summary),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "qwen35_4b_vlm_summary.csv"
    category_csv = output_dir / "qwen35_4b_vlm_category_summary.csv"
    markdown_path = output_dir / "qwen35_4b_vlm_summary.md"
    _write_csv(
        summary_csv,
        comparison_rows,
        ["Agent", "Observation", *DIFFICULTIES, "overall"],
    )
    _write_csv(
        category_csv,
        category_rows,
        ["Observation", *CATEGORIES, "overall"],
    )
    markdown_path.write_text(render_markdown(comparison_rows))
    return summary_csv, category_csv, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pixel_summary", type=Path)
    parser.add_argument(
        "--entries",
        type=Path,
        default=Path("leaderboard_data/entries.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = export_tables(args.entries, args.pixel_summary, args.output_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
