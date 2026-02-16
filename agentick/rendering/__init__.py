"""3D isometric rendering engine for Agentick environments.

This package provides a 3D isometric renderer that uses GLB models
to create visually rich observations from the same 2D grid state.

Requires optional dependencies: ``pip install agentick[render3d]``
"""

from __future__ import annotations

__all__ = [
    "AssetLibrary",
    "IsometricRenderer",
    "IsometricScene",
    "ENTITY_MODEL_MAP",
    "COLOR_TINTS",
]


def __getattr__(name: str):
    """Lazy imports to avoid loading heavy deps at package import time."""
    if name == "AssetLibrary":
        from agentick.rendering.assets import AssetLibrary

        return AssetLibrary
    if name == "IsometricRenderer":
        from agentick.rendering.renderer_3d import IsometricRenderer

        return IsometricRenderer
    if name == "IsometricScene":
        from agentick.rendering.scene import IsometricScene

        return IsometricScene
    if name == "ENTITY_MODEL_MAP":
        from agentick.rendering.entity_mapping import ENTITY_MODEL_MAP

        return ENTITY_MODEL_MAP
    if name == "COLOR_TINTS":
        from agentick.rendering.entity_mapping import COLOR_TINTS

        return COLOR_TINTS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
