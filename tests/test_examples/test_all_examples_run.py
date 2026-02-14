"""EVERY example must at least import without errors.
Examples that don't need API keys must run to completion."""

import subprocess
from pathlib import Path

import pytest

EXAMPLES_DIR = Path("examples")
# Examples that need API keys — test import only, not execution
API_EXAMPLES = {"openai", "anthropic", "compare_llms", "cot_agent"}
# Examples that need GPU — test import only
GPU_EXAMPLES = {"sft_with_trl", "huggingface_local"}
# Examples that need long training — test import only
TRAINING_EXAMPLES = {"ppo_cleanrl", "dqn_cleanrl", "sb3_ppo", "sb3_dqn", "curriculum", "ppo_pixels", "dqn_pixels"}


def get_all_examples():
    """Get all .py files in examples/ except __init__.py"""
    return sorted([p for p in EXAMPLES_DIR.rglob("*.py") if p.name != "__init__.py"])


def should_skip_execution(example: Path) -> bool:
    """Check if example needs API keys, GPU, or long training."""
    stem = example.stem
    return any(key in stem for key in API_EXAMPLES | GPU_EXAMPLES | TRAINING_EXAMPLES)


@pytest.mark.parametrize("example", get_all_examples(), ids=lambda p: str(p))
def test_example_imports(example):
    """Every example must import without ModuleNotFoundError."""
    result = subprocess.run(
        ["uv", "run", "python", "-c",
         f"import importlib.util; "
         f"spec = importlib.util.spec_from_file_location('test_mod', '{example}'); "
         f"import types; mod = types.ModuleType('test_mod'); "
         f"exec(compile(open('{example}').read(), '{example}', 'exec'), "
         f"{{'__name__': 'not___main__', '__file__': '{example}'}})"],
        capture_output=True, text=True, timeout=60,
        env={**dict(__import__('os').environ), "TESTING_IMPORTS_ONLY": "1"}
    )
    # Allow failures from missing API keys, but NOT from missing modules
    if result.returncode != 0:
        assert "ModuleNotFoundError" not in result.stderr, \
            f"Broken import in {example}:\n{result.stderr}"
        assert "ImportError" not in result.stderr, \
            f"Import error in {example}:\n{result.stderr}"


def get_basic_examples():
    """Examples that should run fully without any special deps."""
    return sorted((EXAMPLES_DIR / "basics").rglob("*.py"))


@pytest.mark.parametrize("example", get_basic_examples(), ids=lambda p: str(p))
def test_basic_examples_run(example):
    """Basic examples must run to completion."""
    result = subprocess.run(
        ["uv", "run", "python", str(example)],
        capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, \
        f"Failed to run {example}:\nSTDOUT: {result.stdout[-500:]}\nSTDERR: {result.stderr[-500:]}"
