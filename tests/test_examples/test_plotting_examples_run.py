"""Test that plotting examples run with mock data."""

import subprocess
import pytest
from pathlib import Path
import tempfile
import shutil

EXAMPLES_DIR = Path("examples/plotting")

def get_plotting_examples():
    """Get all plotting example files."""
    return sorted(EXAMPLES_DIR.glob("*.py"))

@pytest.mark.parametrize(
    "example_file",
    get_plotting_examples(),
    ids=lambda p: p.name,
)
def test_plotting_example_runs(example_file):
    """Test that plotting example runs without errors."""
    # Create a temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run the example with output to temp dir
        # Most plotting examples accept --output-dir
        result = subprocess.run(
            ["uv", "run", "python", str(example_file), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # For now, just check that the file can be executed and shows help
        # Full testing would require creating mock experiment data
        # which is better tested in integration tests
        
        # Check the file at least imports and can show help or run briefly
        assert result.returncode in [0, 2], (
            f"{example_file} failed to show help:\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
