"""Tests for the 3D isometric offscreen renderer."""

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

pytestmark = pytest.mark.skipif(not HAS_3D, reason="trimesh/pyrender not installed")

from agentick.core.entity import Agent
from agentick.core.types import CellType, Direction, ObjectType


@pytest.fixture()
def renderer():
    from agentick.rendering.renderer_3d import IsometricRenderer

    r = IsometricRenderer(width=256, height=256)
    yield r
    r.close()


def _simple_terrain(size=5):
    terrain = np.zeros((size, size), dtype=np.int8)
    terrain[0, :] = CellType.WALL
    terrain[-1, :] = CellType.WALL
    terrain[:, 0] = CellType.WALL
    terrain[:, -1] = CellType.WALL
    return terrain


def _simple_objects(size=5):
    return np.zeros((size, size), dtype=np.int8)


def _make_agent(x=1, y=1):
    agent = Agent(id="a0", entity_type="agent", position=(x, y))
    agent.orientation = Direction.NORTH
    return agent


# ------------------------------------------------------------------
# Core rendering tests
# ------------------------------------------------------------------


def test_render_returns_correct_shape(renderer):
    """Render produces (H, W, 3) uint8."""
    terrain = _simple_terrain()
    objects = _simple_objects()
    agent = _make_agent()

    img = renderer.render(terrain, objects, [], agent)

    assert isinstance(img, np.ndarray)
    assert img.shape == (256, 256, 3)
    assert img.dtype == np.uint8


def test_render_non_black(renderer):
    """Rendered image should not be all-black."""
    terrain = _simple_terrain()
    objects = _simple_objects()
    agent = _make_agent()

    img = renderer.render(terrain, objects, [], agent)

    assert img.mean() > 10.0  # Not all black


def test_different_grids_produce_different_images(renderer):
    """Different grid states should produce different images."""
    terrain = _simple_terrain()
    objects = _simple_objects()
    agent = _make_agent()

    img1 = renderer.render(terrain, objects, [], agent)

    # Add a goal
    objects2 = objects.copy()
    objects2[3, 3] = ObjectType.GOAL
    img2 = renderer.render(terrain, objects2, [], agent)

    assert not np.array_equal(img1, img2)


@pytest.mark.parametrize("size", [3, 5, 8, 10])
def test_various_grid_sizes(renderer, size):
    """Different grid sizes all produce valid images."""
    terrain = _simple_terrain(size)
    objects = _simple_objects(size)
    agent = _make_agent()

    img = renderer.render(terrain, objects, [], agent)

    assert img.shape == (256, 256, 3)
    assert img.dtype == np.uint8
    assert img.mean() > 5.0


def test_hud_overlay(renderer):
    """HUD info adds text overlay to the image."""
    terrain = _simple_terrain()
    objects = _simple_objects()
    agent = _make_agent()

    img_no_hud = renderer.render(terrain, objects, [], agent)
    img_with_hud = renderer.render(
        terrain, objects, [], agent,
        hud_info={"step_count": 5, "max_steps": 100, "episode_reward": 1.5},
    )

    # Images should differ due to HUD text
    assert not np.array_equal(img_no_hud, img_with_hud)


def test_fog_mask_changes_output(renderer):
    """Fog mask should visually change the output."""
    terrain = _simple_terrain()
    objects = _simple_objects()
    agent = _make_agent()

    img_full = renderer.render(terrain, objects, [], agent)

    fog_mask = np.zeros((5, 5), dtype=bool)
    fog_mask[1:3, 1:3] = True  # Only reveal a 2×2 area
    img_fog = renderer.render(terrain, objects, [], agent, fog_mask=fog_mask)

    assert not np.array_equal(img_full, img_fog)


def test_close_and_rerender(renderer):
    """Renderer can close and re-render after lazy re-init."""
    terrain = _simple_terrain()
    objects = _simple_objects()
    agent = _make_agent()

    img1 = renderer.render(terrain, objects, [], agent)
    renderer.close()

    # Should lazy-reinit on next render
    img2 = renderer.render(terrain, objects, [], agent)
    assert img2.shape == (256, 256, 3)
