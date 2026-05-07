"""Test that all tasks support all render modes."""

import numpy as np
import pytest

import agentick

RENDER_MODES = ["ascii", "language", "language_structured", "rgb_array", "state_dict"]


@pytest.mark.parametrize("task_name", agentick.list_tasks())
@pytest.mark.timeout(30)
def test_render_modes_work(task_name):
    """Each task must reset, step, and render in every public observation mode."""
    for render_mode in RENDER_MODES:
        env = agentick.make(task_name, difficulty="easy", render_mode=render_mode)
        try:
            obs, _info = env.reset(seed=42)

            if render_mode == "ascii":
                assert isinstance(obs, str)
                assert len(obs) > 0
                assert "\n" in obs
            elif render_mode == "language":
                assert isinstance(obs, str)
                assert len(obs) > 0
            elif render_mode == "language_structured":
                assert isinstance(obs, dict)
                assert "description" in obs
                assert "position" in obs
                assert "task_annotations" in obs
            elif render_mode == "rgb_array":
                assert isinstance(obs, np.ndarray)
                assert obs.ndim == 3
                assert obs.shape[2] == 3
                assert obs.dtype == np.uint8
            elif render_mode == "state_dict":
                assert isinstance(obs, dict)
                assert "grid" in obs
                assert "agent" in obs
                assert "annotations" in obs
                assert "metadata" in obs["grid"]

            action = env.action_space.sample()
            env.step(action)

            rendered = env.render()
            if render_mode in {"ascii", "language"}:
                assert isinstance(rendered, str)
            elif render_mode in {"language_structured", "state_dict"}:
                assert isinstance(rendered, dict)
            elif render_mode == "rgb_array":
                assert isinstance(rendered, np.ndarray)
        finally:
            env.close()


@pytest.mark.parametrize("task_name", agentick.list_tasks()[:5])
@pytest.mark.timeout(30)
def test_text_and_pixel_always_available(task_name):
    """Test that text and pixel observations are always available regardless of render mode."""
    env = agentick.make(task_name, difficulty="easy", render_mode="ascii")
    try:
        obs, info = env.reset(seed=42)

        # These should always work
        text_obs = env.get_text_observation()
        assert isinstance(text_obs, str)
        assert len(text_obs) > 0

        pixel_obs = env.get_pixel_observation()
        assert isinstance(pixel_obs, np.ndarray)
        assert pixel_obs.ndim == 3
        assert pixel_obs.shape[2] == 3
    finally:
        env.close()
