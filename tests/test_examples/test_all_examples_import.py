"""Test that every example file is valid Python (syntax check only)."""

import ast
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path("examples")


def get_all_example_files():
    """Get all Python example files."""
    return sorted(EXAMPLES_DIR.rglob("*.py"))


@pytest.mark.parametrize(
    "example_file",
    get_all_example_files(),
    ids=lambda p: str(p),
)
def test_example_syntax(example_file):
    """Verify example file has valid Python syntax."""
    source = example_file.read_text()
    try:
        ast.parse(source)
    except SyntaxError as e:
        pytest.fail(f"Syntax error in {example_file}: {e}")
