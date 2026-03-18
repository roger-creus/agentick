#!/usr/bin/env python
"""Generate task_descriptions.json from the live task registry.

Usage:
    python scripts/generate_task_descriptions.py          # writes to docs/showcase/
    python scripts/generate_task_descriptions.py -o out/  # writes to out/task_descriptions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path so the import works when invoked
# from any directory.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from agentick.tasks.descriptions import get_all_task_descriptions  # noqa: E402


def _find_video_file(task_name: str, videos_dir: Path) -> str | None:
    """Return the relative video path for a task (easy dense preferred)."""
    # Strip version suffix: "GoToGoal-v0" -> "GoToGoal"
    base = task_name.replace("-v0", "").replace("-v1", "")
    # Videos are named like: 14_GoToGoal_easy_dense.mp4
    for f in sorted(videos_dir.iterdir()):
        if f.suffix == ".mp4" and base in f.name and "easy_dense" in f.name:
            return f.name
    # Fallback: any video for this task
    for f in sorted(videos_dir.iterdir()):
        if f.suffix == ".mp4" and base in f.name:
            return f.name
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate task_descriptions.json")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=_project_root / "docs" / "showcase",
        help="Directory to write task_descriptions.json into (default: docs/showcase/)",
    )
    args = parser.parse_args()

    descriptions = get_all_task_descriptions()

    videos_dir = _project_root / "videos"

    output: list[dict] = []
    for name, desc in sorted(descriptions.items()):
        entry = desc.to_dict()
        # Attach video filename if available
        if videos_dir.is_dir():
            entry["video"] = _find_video_file(name, videos_dir)
        else:
            entry["video"] = None
        output.append(entry)

    out_path = args.output_dir / "task_descriptions.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n")

    print(f"Wrote {len(output)} task descriptions to {out_path}")


if __name__ == "__main__":
    main()
