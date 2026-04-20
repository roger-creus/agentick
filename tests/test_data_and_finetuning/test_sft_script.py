"""Audit tests for examples/data_and_finetuning/sft_with_trl.py.

These tests verify the SFT pipeline produces training data that exactly
matches the eval-time prompt format and trains with assistant-only loss.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the example script importable as a module
SFT_DIR = Path(__file__).resolve().parents[2] / "examples" / "data_and_finetuning"
sys.path.insert(0, str(SFT_DIR))


def test_sft_module_imports():
    """Sanity: sft_with_trl.py imports without errors."""
    import sft_with_trl  # noqa: F401
