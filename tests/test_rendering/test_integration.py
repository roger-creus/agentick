"""Integration tests for 3D rendering through the environment."""

import os

import numpy as np
import pytest

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

try:
    import pyrender  # noqa: F401
    import trimesh  # noqa: F401

    HAS_3D = True
except ImportError:
    HAS_3D = False

import agentick


# ------------------------------------------------------------------
# 3D rendering through agentick.make()
# ------------------------------------------------------------------


@pytest.mark.skipif(not HAS_3D, reason="trimesh/pyrender not installed")
class TestRgbArray3D:
    """Tests for render_mode='rgb_array' with 3D renderer."""

    def test_make_with_3d_returns_image(self):
        env = agentick.make("GoToGoal-v0", render_mode="rgb_array", render_3d=True, seed=42)
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (512, 512, 3)
        assert obs.dtype == np.uint8
        env.close()

    def test_step_returns_valid_obs(self):
        env = agentick.make("GoToGoal-v0", render_mode="rgb_array", render_3d=True, seed=42)
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(1)
        assert obs.shape == (512, 512, 3)
        assert obs.dtype == np.uint8
        env.close()

    def test_observation_space_matches(self):
        env = agentick.make("GoToGoal-v0", render_mode="rgb_array", render_3d=True, seed=42)
        env.reset(seed=42)
        obs = env.render()
        assert env.observation_space.contains(obs)
        env.close()

    @pytest.mark.parametrize(
        "task_id",
        [
            "GoToGoal-v0",
            "KeyDoorPuzzle-v0",
            "MazeNavigation-v0",
        ],
    )
    def test_multiple_tasks_render_3d(self, task_id):
        env = agentick.make(task_id, difficulty="easy", render_mode="rgb_array", render_3d=True, seed=42)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (512, 512, 3)
        assert obs.mean() > 10.0  # Not all black
        env.close()


# ------------------------------------------------------------------
# 2D fallback rendering
# ------------------------------------------------------------------


class TestRgbArray2D:
    """Tests for render_mode='rgb_array_2d' (old 2D sprite renderer)."""

    def test_2d_mode_works(self):
        env = agentick.make("GoToGoal-v0", render_mode="rgb_array_2d", seed=42)
        obs, _ = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.ndim == 3
        assert obs.shape[2] == 3
        assert obs.dtype == np.uint8
        env.close()

    def test_2d_shape_differs_from_3d(self):
        env_2d = agentick.make("GoToGoal-v0", render_mode="rgb_array_2d", seed=42)
        obs_2d, _ = env_2d.reset(seed=42)
        env_2d.close()

        if HAS_3D:
            env_3d = agentick.make("GoToGoal-v0", render_mode="rgb_array", render_3d=True, seed=42)
            obs_3d, _ = env_3d.reset(seed=42)
            env_3d.close()
            # 3D is 512x512, 2D depends on grid size
            assert obs_2d.shape != obs_3d.shape

    def test_render_3d_false_forces_2d(self):
        env = agentick.make("GoToGoal-v0", render_mode="rgb_array", render_3d=False, seed=42)
        obs, _ = env.reset(seed=42)
        # Should be 2D tile-based size, not 512x512
        assert obs.shape[0] != 512 or obs.shape[1] != 512
        env.close()


# ------------------------------------------------------------------
# Graceful fallback
# ------------------------------------------------------------------


class TestFallback:
    """Tests for graceful degradation."""

    def test_auto_detection_defaults_to_2d(self):
        """render_3d=None should default to 2D (opt-in for 3D)."""
        env = agentick.make("GoToGoal-v0", render_mode="rgb_array", seed=42)
        obs, _ = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.ndim == 3
        assert obs.shape[2] == 3
        # Auto defaults to 2D — shape should match grid*32, not 512
        assert obs.shape[0] != 512
        env.close()

    def test_close_cleans_up(self):
        env = agentick.make("GoToGoal-v0", render_mode="rgb_array", render_3d=True, seed=42)
        env.reset(seed=42)
        env.render()
        env.close()
        assert env._renderer_3d is None


# ------------------------------------------------------------------
# Other render modes still work
# ------------------------------------------------------------------


class TestOtherModes:
    """Ensure non-pixel render modes are unaffected."""

    def test_ascii_mode(self):
        env = agentick.make("GoToGoal-v0", render_mode="ascii", seed=42)
        obs, _ = env.reset(seed=42)
        assert isinstance(obs, str)
        env.close()

    def test_language_mode(self):
        env = agentick.make("GoToGoal-v0", render_mode="language", seed=42)
        obs, _ = env.reset(seed=42)
        assert isinstance(obs, str)
        env.close()

    def test_state_dict_mode(self):
        env = agentick.make("GoToGoal-v0", render_mode="state_dict", seed=42)
        obs, _ = env.reset(seed=42)
        assert isinstance(obs, dict)
        env.close()
