"""Test that every example file can be imported without errors."""

import subprocess
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
def test_example_imports(example_file):
    """Verify example can be parsed and its imports resolve."""
    # Test syntax parsing
    result = subprocess.run(
        [
            "python",
            "-c",
            f"import ast; ast.parse(open('{example_file}').read())",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"Syntax error in {example_file}: {result.stderr}"

    # Test that top-level imports work (run file but exit before main)
    result = subprocess.run(
        [
            "python",
            "-c",
            f"import importlib.util; "
            f"spec = importlib.util.spec_from_file_location('mod', '{example_file}'); "
            f"mod = importlib.util.module_from_spec(spec); "
            f"spec.loader.exec_module(mod)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Check for import errors specifically
    # Some files may error on missing API keys, which is OK
    # We just want to ensure ModuleNotFoundError doesn't occur
    if result.returncode != 0:
        stderr = result.stderr
        assert "ModuleNotFoundError" not in stderr, (
            f"Import error in {example_file}: {stderr}"
        )
        assert "cannot import name" not in stderr, (
            f"Import error in {example_file}: {stderr}"
        )
        # If it's just a missing API key or env var, that's fine for this test
