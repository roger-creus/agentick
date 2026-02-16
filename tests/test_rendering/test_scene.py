"""Tests for 3D scene composition."""

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
def assets():
    from agentick.rendering.assets import AssetLibrary

    return AssetLibrary()


@pytest.fixture()
def scene_builder(assets):
    from agentick.rendering.scene import IsometricScene

    return IsometricScene(assets)


def _make_simple_grid():
    """3×3 bordered grid with floor interior."""
    terrain = np.zeros((3, 3), dtype=np.int8)
    terrain[0, :] = CellType.WALL
    terrain[-1, :] = CellType.WALL
    terrain[:, 0] = CellType.WALL
    terrain[:, -1] = CellType.WALL
    objects = np.zeros((3, 3), dtype=np.int8)
    return terrain, objects


def _make_agent():
    agent = Agent(id="a0", entity_type="agent", position=(1, 1))
    agent.orientation = Direction.NORTH
    return agent


# ------------------------------------------------------------------
# Scene composition tests
# ------------------------------------------------------------------


def test_compose_returns_pyrender_scene(scene_builder):
    terrain, objects = _make_simple_grid()
    agent = _make_agent()
    scene = scene_builder.compose(terrain, objects, [], agent)
    assert isinstance(scene, pyrender.Scene)


def test_scene_has_mesh_nodes(scene_builder):
    """Scene should contain mesh nodes for terrain, agent, etc."""
    terrain, objects = _make_simple_grid()
    agent = _make_agent()
    scene = scene_builder.compose(terrain, objects, [], agent)
    # At least: 9 terrain tiles + 1 agent + camera + 2 lights = 13+
    assert len(scene.mesh_nodes) >= 9  # terrain + agent


def test_scene_has_camera(scene_builder):
    terrain, objects = _make_simple_grid()
    agent = _make_agent()
    scene = scene_builder.compose(terrain, objects, [], agent)
    cam_nodes = scene.camera_nodes
    assert len(cam_nodes) == 1


def test_scene_has_lights(scene_builder):
    terrain, objects = _make_simple_grid()
    agent = _make_agent()
    scene = scene_builder.compose(terrain, objects, [], agent)
    light_nodes = scene.light_nodes
    assert len(light_nodes) >= 2  # key + fill


def test_objects_add_extra_nodes(scene_builder):
    """Adding objects to the grid should add more mesh nodes."""
    terrain, objects = _make_simple_grid()
    agent = _make_agent()
    scene_no_obj = scene_builder.compose(terrain, objects, [], agent)
    n_no_obj = len(scene_no_obj.mesh_nodes)

    objects[1, 1] = ObjectType.GOAL
    scene_with_obj = scene_builder.compose(terrain, objects, [], agent)
    n_with_obj = len(scene_with_obj.mesh_nodes)

    assert n_with_obj > n_no_obj


def test_agent_direction_changes_pose(scene_builder):
    """Different agent orientations should produce different scenes."""
    terrain, objects = _make_simple_grid()
    agent = _make_agent()

    agent.orientation = Direction.NORTH
    s1 = scene_builder.compose(terrain, objects, [], agent)
    poses1 = [n.matrix.copy() for n in s1.mesh_nodes]

    agent.orientation = Direction.EAST
    s2 = scene_builder.compose(terrain, objects, [], agent)
    poses2 = [n.matrix.copy() for n in s2.mesh_nodes]

    # At least the agent node should differ
    any_different = any(
        not np.allclose(p1, p2) for p1, p2 in zip(poses1, poses2)
    )
    assert any_different


def test_grid_to_world_mapping(scene_builder):
    """Grid coordinates map to expected world positions."""
    pos = scene_builder._grid_to_world(0, 0)
    assert np.allclose(pos, [0.0, 0.0, 0.0])

    pos = scene_builder._grid_to_world(2, 3, height=0.5)
    assert np.allclose(pos, [3.0, 0.5, 2.0])


def test_fog_mask_hides_cells(scene_builder):
    """Fog mask should reduce the number of rendered tiles."""
    terrain, objects = _make_simple_grid()
    agent = _make_agent()

    scene_full = scene_builder.compose(terrain, objects, [], agent, fog_mask=None)
    n_full = len(scene_full.mesh_nodes)

    # Only reveal center cell
    fog_mask = np.zeros((3, 3), dtype=bool)
    fog_mask[1, 1] = True
    scene_fog = scene_builder.compose(terrain, objects, [], agent, fog_mask=fog_mask)
    n_fog = len(scene_fog.mesh_nodes)

    assert n_fog < n_full
