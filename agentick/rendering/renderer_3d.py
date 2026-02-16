"""3D isometric offscreen renderer for Agentick environments."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from agentick.core.entity import Agent, Entity

logger = logging.getLogger(__name__)


class IsometricRenderer:
    """Renders game state as a 3D isometric image.

    Usage::

        renderer = IsometricRenderer(width=512, height=512)
        image = renderer.render(grid_terrain, grid_objects, entities, agent)
        # image is numpy array (512, 512, 3) uint8

    The renderer is re-entrant but **not** thread-safe.
    It rebuilds the pyrender scene each frame (models are lightweight cached
    references, so this is fast enough for step-by-step evaluation).

    For RL training speed, use ``render_mode='rgb_array_2d'`` instead.
    """

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        asset_dir: str | Path | None = None,
        tile_size: float = 1.0,
    ):
        self.width = width
        self.height = height
        self._tile_size = tile_size
        self._asset_dir = asset_dir

        # Lazy-initialised in _ensure_ready()
        self._assets = None
        self._scene_builder = None
        self._offscreen = None
        self._ready = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        grid_terrain: np.ndarray,
        grid_objects: np.ndarray,
        entities: list[Entity],
        agent: Agent,
        fog_mask: np.ndarray | None = None,
        step_count: int = 0,
        hud_info: dict | None = None,
    ) -> np.ndarray:
        """Render current game state as a 3D isometric image.

        Returns:
            numpy array ``(height, width, 3)`` uint8.
        """
        self._ensure_ready()

        # Build scene
        scene = self._scene_builder.compose(
            grid_terrain=grid_terrain,
            grid_objects=grid_objects,
            entities=entities,
            agent=agent,
            fog_mask=fog_mask,
            step_count=step_count,
        )

        # Render offscreen
        color, _depth = self._offscreen.render(scene)
        image = np.ascontiguousarray(color[:, :, :3])  # Drop alpha if present

        # Optional HUD overlay
        if hud_info:
            image = self._draw_hud(image, hud_info)

        return image.astype(np.uint8)

    def close(self) -> None:
        """Clean up renderer resources."""
        if self._offscreen is not None:
            try:
                self._offscreen.delete()
            except Exception:
                pass
            self._offscreen = None
        self._ready = False

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_ready(self) -> None:
        if self._ready:
            return

        # Set OpenGL platform for headless rendering before importing pyrender.
        # Prefer EGL (works on most modern Linux) over OSMesa.
        if "PYOPENGL_PLATFORM" not in os.environ:
            if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
                os.environ["PYOPENGL_PLATFORM"] = "egl"

        import pyrender

        from agentick.rendering.assets import AssetLibrary
        from agentick.rendering.scene import IsometricScene

        self._assets = AssetLibrary(asset_dir=self._asset_dir)
        self._scene_builder = IsometricScene(self._assets)
        self._scene_builder.TILE_SIZE = self._tile_size
        self._offscreen = pyrender.OffscreenRenderer(self.width, self.height)
        self._ready = True
        logger.info(
            "IsometricRenderer ready (%dx%d, tile=%.2f)",
            self.width,
            self.height,
            self._tile_size,
        )

    # ------------------------------------------------------------------
    # HUD overlay
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_hud(image: np.ndarray, hud_info: dict) -> np.ndarray:
        """Draw a simple HUD overlay on the rendered image."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            return image

        img = Image.fromarray(image)
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

        # Build text lines
        lines = []
        if "step_count" in hud_info:
            lines.append(f"Step: {hud_info['step_count']}/{hud_info.get('max_steps', '?')}")
        if "episode_reward" in hud_info:
            lines.append(f"Reward: {hud_info['episode_reward']:.2f}")
        if "inventory" in hud_info and hud_info["inventory"]:
            inv_str = ", ".join(str(item) for item in hud_info["inventory"])
            lines.append(f"Inv: {inv_str}")

        # Draw with shadow
        y_offset = 4
        for line in lines:
            draw.text((6, y_offset + 1), line, fill=(0, 0, 0), font=font)  # shadow
            draw.text((5, y_offset), line, fill=(255, 255, 255), font=font)
            y_offset += 18

        return np.array(img)
