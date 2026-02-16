"""Tests for 3D asset loading, caching, fallbacks, and tinting."""

import os

import numpy as np
import pytest

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

try:
    import trimesh  # noqa: F401

    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

pytestmark = pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh not installed")


@pytest.fixture()
def asset_lib():
    from agentick.rendering.assets import AssetLibrary

    return AssetLibrary()


# ------------------------------------------------------------------
# Loading GLB files
# ------------------------------------------------------------------

ALL_SLUGS = [
    "player",
    "wall",
    "floor",
    "key",
    "door",
    "goal",
    "box",
    "enemy",
    "hazard",
    "heart",
    "switch",
    "npc",
    "gem",
    "lightning",
    "water",
    "target",
    "ice",
    "hole",
    "tool",
    "breadcrumb",
]


@pytest.mark.parametrize("slug", ALL_SLUGS)
def test_load_glb_model(asset_lib, slug):
    """Each GLB loads as a valid trimesh Scene."""
    scene = asset_lib.get_model(slug)
    assert scene is not None
    assert len(scene.geometry) > 0


@pytest.mark.parametrize("slug", ALL_SLUGS)
def test_get_mesh_returns_trimesh(asset_lib, slug):
    """get_mesh returns a Trimesh with face colors (texture baked)."""
    mesh = asset_lib.get_mesh(slug)
    assert hasattr(mesh, "vertices")
    assert hasattr(mesh, "faces")
    assert mesh.vertices.shape[0] > 0
    assert mesh.faces.shape[0] > 0
    # After simplification, visual should be face-based
    assert mesh.visual.kind == "face"


def test_caching(asset_lib):
    """Repeated loads return cached object."""
    scene1 = asset_lib.get_model("player")
    scene2 = asset_lib.get_model("player")
    assert scene1 is scene2


# ------------------------------------------------------------------
# Fallback primitives
# ------------------------------------------------------------------


def test_fallback_for_missing_slug(asset_lib):
    """get_or_fallback creates a primitive when GLB is missing."""
    scene = asset_lib.get_or_fallback("nonexistent_slug_xyz")
    assert scene is not None
    assert len(scene.geometry) > 0


def test_fallback_mesh_has_face_colors(asset_lib):
    """Fallback primitives have face colors set."""
    mesh = asset_lib.get_mesh("nonexistent_slug_xyz")
    assert mesh.visual.kind == "face"
    assert mesh.visual.face_colors.shape[0] == mesh.faces.shape[0]


# ------------------------------------------------------------------
# Color tinting
# ------------------------------------------------------------------


def test_tint_applied():
    """_apply_tint modifies face colors."""
    from agentick.rendering.scene import IsometricScene

    import trimesh

    mesh = trimesh.creation.box(extents=(1, 1, 1))
    mesh.visual.face_colors = np.full((len(mesh.faces), 4), [255, 255, 255, 255], dtype=np.uint8)

    IsometricScene._apply_tint(mesh, (1.0, 0.0, 0.0))

    fc = mesh.visual.face_colors
    # Red channel should stay 255, green and blue should be 0
    assert fc[:, 0].mean() == 255
    assert fc[:, 1].mean() == 0
    assert fc[:, 2].mean() == 0


# ------------------------------------------------------------------
# Preload
# ------------------------------------------------------------------


def test_preload_all(asset_lib):
    """preload_all fills the cache."""
    asset_lib.preload_all()
    assert len(asset_lib._cache) >= len(ALL_SLUGS)
