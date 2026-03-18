"""Visual regression tests for rendering consistency.

Tests that compare current renders against saved reference renders to catch
any unintended visual changes.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from agentick import make

# Directory for reference renders
REFERENCE_DIR = Path(__file__).parent / "reference_renders"
REFERENCE_DIR.mkdir(exist_ok=True)

# Reference file for render hashes
REFERENCE_HASHES_FILE = REFERENCE_DIR / "reference_hashes.json"


def compute_render_hash(render_output: str | np.ndarray) -> str:
    """Compute hash of render output for comparison."""
    if isinstance(render_output, str):
        # Text render
        return hashlib.sha256(render_output.encode()).hexdigest()
    elif isinstance(render_output, np.ndarray):
        # Pixel render
        return hashlib.sha256(render_output.tobytes()).hexdigest()
    else:
        raise ValueError(f"Unsupported render type: {type(render_output)}")


def load_reference_hashes() -> dict:
    """Load reference hashes from file."""
    if REFERENCE_HASHES_FILE.exists():
        with open(REFERENCE_HASHES_FILE) as f:
            return json.load(f)
    return {}


def save_reference_hashes(hashes: dict):
    """Save reference hashes to file."""
    with open(REFERENCE_HASHES_FILE, "w") as f:
        json.dump(hashes, f, indent=2)


def save_reference_render(
    task_id: str, render_mode: str, render_output: str | np.ndarray, step: int = 0
):
    """Save a reference render for comparison."""
    # Create subdirectory for this task
    task_dir = REFERENCE_DIR / task_id.replace("-", "_")
    task_dir.mkdir(exist_ok=True)

    # Save render output
    if isinstance(render_output, str):
        # Text render
        filename = f"{render_mode}_step{step}.txt"
        filepath = task_dir / filename
        with open(filepath, "w") as f:
            f.write(render_output)
    elif isinstance(render_output, np.ndarray):
        # Pixel render
        filename = f"{render_mode}_step{step}.npy"
        filepath = task_dir / filename
        np.save(filepath, render_output)


def load_reference_render(task_id: str, render_mode: str, step: int = 0):
    """Load a reference render for comparison."""
    task_dir = REFERENCE_DIR / task_id.replace("-", "_")

    if render_mode in ["ascii", "language"]:
        filename = f"{render_mode}_step{step}.txt"
        filepath = task_dir / filename
        if filepath.exists():
            with open(filepath) as f:
                return f.read()
    else:
        filename = f"{render_mode}_step{step}.npy"
        filepath = task_dir / filename
        if filepath.exists():
            return np.load(filepath)

    return None


@pytest.mark.parametrize("render_mode", ["ascii", "language", "rgb_array"])
@pytest.mark.parametrize(
    "task_id",
    [
        "GoToGoal-v0",
        "KeyDoorPuzzle-v0",
        "MazeNavigation-v0",
        "SokobanPush-v0",
    ],
)
def test_visual_regression(task_id: str, render_mode: str, request):
    """Test that renders remain consistent with reference renders.

    This test can operate in two modes:
    1. --update-references flag: Save current renders as new references
    2. Normal mode: Compare current renders against saved references
    """
    # Check if we should update references
    update_references = request.config.getoption("--update-references", default=False)

    # Create environment — force 2D for visual regression (these reference renders are 2D)
    env = make(task_id, difficulty="easy", render_mode=render_mode)

    # Reset and get initial render
    env.reset(seed=42)
    initial_render = env.render()

    # Take a few deterministic steps and capture renders
    # Use fixed actions instead of random to ensure reproducibility
    renders = [initial_render]
    deterministic_actions = [1, 2, 1]  # down, right, down
    for action in deterministic_actions:
        obs, _, terminated, truncated, _ = env.step(action)
        if not (terminated or truncated):
            renders.append(env.render())

    # Process each render
    for step, render_output in enumerate(renders):
        if update_references:
            # Save new reference
            save_reference_render(task_id, render_mode, render_output, step)
        else:
            # Compare against reference
            reference = load_reference_render(task_id, render_mode, step)

            if reference is None:
                pytest.skip(
                    f"No reference render found for {task_id} {render_mode} step {step}. "
                    "Run with --update-references to create references."
                )

            # Compare
            current_hash = compute_render_hash(render_output)
            reference_hash = compute_render_hash(reference)

            if current_hash != reference_hash:
                # Try to provide helpful error message
                if isinstance(render_output, str):
                    # Show diff for text renders
                    current_lines = render_output.split("\n")
                    reference_lines = reference.split("\n")
                    diff_msg = (
                        f"Visual regression detected in {task_id} {render_mode} step {step}:\n"
                    )

                    # Find first differing line
                    for i, (curr, ref) in enumerate(zip(current_lines, reference_lines)):
                        if curr != ref:
                            diff_msg += f"Line {i}:\n  Expected: {ref!r}\n  Got: {curr!r}\n"
                            break

                    pytest.fail(diff_msg)
                else:
                    # Pixel render - show shape and some stats
                    diff = np.abs(render_output.astype(float) - reference.astype(float))
                    max_diff = diff.max()
                    mean_diff = diff.mean()
                    pytest.fail(
                        f"Visual regression detected in {task_id} {render_mode} step {step}:\n"
                        f"  Max pixel difference: {max_diff}\n"
                        f"  Mean pixel difference: {mean_diff}\n"
                    )

    env.close()


def test_ascii_render_consistency():
    """Test that ASCII renders are consistent across resets with same seed."""
    env = make("KeyDoorPuzzle-v0", difficulty="easy", render_mode="ascii")

    # Get two renders with same seed
    env.reset(seed=42)
    render1 = env.render()

    env.reset(seed=42)
    render2 = env.render()

    assert render1 == render2, "ASCII renders should be identical for same seed"

    env.close()


def test_rgb_render_consistency():
    """Test that RGB renders are consistent across resets with same seed."""
    env = make("KeyDoorPuzzle-v0", difficulty="easy", render_mode="rgb_array")

    # Get two renders with same seed
    env.reset(seed=42)
    render1 = env.render()

    env.reset(seed=42)
    render2 = env.render()

    assert np.array_equal(render1, render2), "RGB renders should be identical for same seed"

    env.close()


def test_language_render_consistency():
    """Test that language renders are consistent across resets with same seed."""
    env = make("KeyDoorPuzzle-v0", difficulty="easy", render_mode="language")

    # Get two renders with same seed
    env.reset(seed=42)
    render1 = env.render()

    env.reset(seed=42)
    render2 = env.render()

    assert render1 == render2, "Language renders should be identical for same seed"

    env.close()


def test_render_changes_after_action():
    """Test that renders change appropriately after actions."""
    env = make("GoToGoal-v0", difficulty="easy", render_mode="ascii")

    env.reset(seed=42)
    render1 = env.render()

    # Take multiple actions until we successfully move
    moved = False
    for action in [2, 3, 1, 0]:  # Try right, left, down, up
        obs, reward, terminated, truncated, info = env.step(action)
        render2 = env.render()
        if render1 != render2:
            moved = True
            break

    # At least one action should change the render
    assert moved, "At least one action should change the render"

    env.close()


def test_multiple_render_modes_same_state():
    """Test that different render modes can render the same state."""
    # Create two envs with same seed (force 2D for consistent comparison)
    env_ascii = make("KeyDoorPuzzle-v0", difficulty="easy", render_mode="ascii", seed=42)
    env_rgb = make("KeyDoorPuzzle-v0", difficulty="easy", render_mode="rgb_array", seed=42)

    env_ascii.reset(seed=42)
    env_rgb.reset(seed=42)

    # Both should successfully render
    ascii_render = env_ascii.render()
    rgb_render = env_rgb.render()

    assert ascii_render is not None
    assert rgb_render is not None
    assert isinstance(ascii_render, str)
    assert isinstance(rgb_render, np.ndarray)

    env_ascii.close()
    env_rgb.close()
