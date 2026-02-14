"""Test that basic examples run end-to-end."""

import subprocess
from pathlib import Path

import pytest

EXAMPLES_DIR = Path("examples/basics")

def get_basic_examples():
    """Get all basic example files."""
    return sorted(EXAMPLES_DIR.glob("*.py"))

@pytest.mark.parametrize(
    "example_file",
    get_basic_examples(),
    ids=lambda p: p.name,
)
def test_basic_example_runs(example_file):
    """Test that basic example runs without errors."""
    # Run the example with a short timeout
    result = subprocess.run(
        ["uv", "run", "python", str(example_file)],
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Check it didn't error
    assert result.returncode == 0, (
        f"{example_file} failed:\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    # Basic smoke test - check for common error patterns
    stderr_lower = result.stderr.lower()
    assert "traceback" not in stderr_lower, f"Traceback found in {example_file}"
    assert "error:" not in stderr_lower or "warning" in stderr_lower, f"Error found in {example_file}"
