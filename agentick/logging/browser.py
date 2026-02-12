"""Rich terminal UI for browsing experiment logs."""

from __future__ import annotations

from pathlib import Path


def browse_logs(results_dir: str | Path) -> None:
    """
    Browse experiment logs with rich terminal UI.

    Args:
        results_dir: Results directory
    """
    # Stub - would use rich for terminal UI
    print(f"Browsing logs in {results_dir}")
    print("(Terminal UI not yet implemented)")
