"""3D scene composition from 2D grid state."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from agentick.core.types import CellType, Direction, ObjectType
from agentick.rendering.entity_mapping import (
    CELLTYPE_MODEL_MAP,
    COLOR_TINTS,
    FLOATING_OBJECTS,
    OBJECTTYPE_MODEL_MAP,
)

if TYPE_CHECKING:
    from agentick.core.entity import Agent, Entity
    from agentick.rendering.assets import AssetLibrary

logger = logging.getLogger(__name__)


# Direction → Y-axis rotation in radians (counter-clockwise from +Z)
DIRECTION_ROTATIONS: dict[Direction, float] = {
    Direction.NORTH: 0.0,
    Direction.EAST: -math.pi / 2,
    Direction.SOUTH: math.pi,
    Direction.WEST: math.pi / 2,
}


class IsometricScene:
    """Composes a 3D pyrender scene from a 2D grid state.

    Coordinate mapping (grid → 3D world):
        x = col * TILE_SIZE
        y = height (0 for ground)
        z = row * TILE_SIZE
    """

    TILE_SIZE: float = 1.0
    FLOAT_HEIGHT: float = 0.35
    WALL_HEIGHT: float = 1.0

    def __init__(self, assets: AssetLibrary):
        self.assets = assets

    def compose(
        self,
        grid_terrain: np.ndarray,
        grid_objects: np.ndarray,
        entities: list[Entity],
        agent: Agent,
        fog_mask: np.ndarray | None = None,
        step_count: int = 0,
    ):
        """Build a complete pyrender Scene from game state.

        Args:
            grid_terrain: 2D array of CellType values (H, W).
            grid_objects: 2D array of ObjectType values (H, W).
            entities: Additional entities (NPCs, enemies).
            agent: The player agent.
            fog_mask: Optional bool mask (True = visible).
            step_count: Current step (used for animation hints).

        Returns:
            pyrender.Scene ready for rendering.
        """
        import pyrender

        scene = pyrender.Scene(
            ambient_light=np.array([0.6, 0.6, 0.6, 1.0]),
            bg_color=np.array([0.15, 0.15, 0.22, 1.0]),
        )

        rows, cols = grid_terrain.shape

        # --- 1. Terrain (floor / wall / hazard / etc.) ---
        for r in range(rows):
            for c in range(cols):
                if fog_mask is not None and not fog_mask[r, c]:
                    continue

                cell = int(grid_terrain[r, c])
                slug = CELLTYPE_MODEL_MAP.get(cell)
                if slug is None:
                    continue

                pos = self._grid_to_world(r, c)

                if cell == CellType.WALL:
                    # Wall raised above ground
                    pos[1] = self.WALL_HEIGHT / 2
                    self._place_model(scene, slug, pos, scale=self.TILE_SIZE)
                elif cell == CellType.EMPTY:
                    # Flat floor tile
                    pos[1] = -0.05
                    self._place_model(scene, slug, pos, scale=self.TILE_SIZE)
                else:
                    # Other terrain (hazard, water, ice, hole) as flat tiles
                    pos[1] = -0.02
                    self._place_model(scene, slug, pos, scale=self.TILE_SIZE)

        # --- 2. Objects ---
        for r in range(rows):
            for c in range(cols):
                if fog_mask is not None and not fog_mask[r, c]:
                    continue

                obj = int(grid_objects[r, c])
                slug = OBJECTTYPE_MODEL_MAP.get(obj)
                if slug is None:
                    continue

                pos = self._grid_to_world(r, c)

                # Float items above ground
                if slug in FLOATING_OBJECTS:
                    # Slight height variation for visual interest
                    hash_offset = ((r * 31 + c * 17) % 7) * 0.03
                    pos[1] = self.FLOAT_HEIGHT + hash_offset
                else:
                    pos[1] = 0.0

                # Scale objects to be clearly visible — 0.8x tile size
                self._place_model(scene, slug, pos, scale=0.8)

        # --- 3. Extra entities (NPCs, enemies) ---
        for entity in entities:
            if fog_mask is not None:
                ex, ey = entity.position
                if not fog_mask[ey, ex]:
                    continue

            slug_name = entity.entity_type
            from agentick.rendering.entity_mapping import ENTITY_MODEL_MAP

            slug = ENTITY_MODEL_MAP.get(slug_name, slug_name)
            if slug is None:
                continue

            ex, ey = entity.position
            pos = self._grid_to_world(ey, ex)
            pos[1] = 0.0
            # Scale NPCs/enemies similar to player
            self._place_model(scene, slug, pos, scale=1.0)

        # --- 4. Agent ---
        ax, ay = agent.position
        agent_pos = self._grid_to_world(ay, ax)
        agent_pos[1] = 0.0
        rotation = DIRECTION_ROTATIONS.get(agent.orientation, 0.0)
        # Scale player to be prominent — 1.2x tile size so it stands out
        self._place_model(scene, "player", agent_pos, rotation=rotation, scale=1.2)

        # --- 5. Lighting ---
        self._add_lighting(scene, rows, cols)

        # --- 6. Camera ---
        self._add_camera(scene, rows, cols)

        return scene

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _grid_to_world(self, row: int, col: int, height: float = 0.0) -> np.ndarray:
        x = col * self.TILE_SIZE
        y = height
        z = row * self.TILE_SIZE
        return np.array([x, y, z], dtype=np.float64)

    # ------------------------------------------------------------------
    # Model placement
    # ------------------------------------------------------------------

    def _place_model(
        self,
        scene,
        slug: str,
        position: np.ndarray,
        rotation: float = 0.0,
        scale: float = 1.0,
        tint: tuple[float, float, float] | None = None,
    ) -> None:
        """Place a model instance into the pyrender scene."""
        import pyrender
        import trimesh

        try:
            mesh_data = self.assets.get_mesh(slug)
        except Exception:
            logger.debug("Could not load mesh for '%s', skipping", slug)
            return

        # Apply color tint if requested
        if tint is not None:
            self._apply_tint(mesh_data, tint)

        # Build 4×4 pose matrix
        pose = np.eye(4)

        # Scale
        pose[:3, :3] *= scale

        # Rotation around Y axis
        if rotation != 0.0:
            c, s = math.cos(rotation), math.sin(rotation)
            rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            pose[:3, :3] = rot @ pose[:3, :3]

        # Translation
        pose[:3, 3] = position

        # Convert trimesh mesh to pyrender mesh.
        # After _simplify_visual(), meshes should all be face-colored →
        # use smooth=False to avoid the pyrender "face colors + smooth" crash.
        smooth = False
        if hasattr(mesh_data, "visual") and mesh_data.visual is not None:
            kind = getattr(mesh_data.visual, "kind", None)
            if kind and kind != "face":
                smooth = True
        pr_mesh = pyrender.Mesh.from_trimesh(mesh_data, smooth=smooth)
        scene.add(pr_mesh, pose=pose)

    @staticmethod
    def _apply_tint(mesh, tint: tuple[float, float, float]) -> None:
        """Multiply vertex/face colours by a tint factor."""
        if hasattr(mesh.visual, "face_colors"):
            colors = mesh.visual.face_colors.astype(np.float64)
            colors[:, 0] *= tint[0]
            colors[:, 1] *= tint[1]
            colors[:, 2] *= tint[2]
            mesh.visual.face_colors = np.clip(colors, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Lighting
    # ------------------------------------------------------------------

    def _add_lighting(self, scene, rows: int, cols: int) -> None:
        import pyrender

        center = self._grid_to_world(rows / 2, cols / 2)

        # Key light — warm directional from upper-left
        key_light = pyrender.DirectionalLight(
            color=np.array([1.0, 0.95, 0.85]),
            intensity=6.0,
        )
        key_pose = np.eye(4)
        # Point light down and to the right
        key_pose[:3, 3] = center + np.array([-5.0, 10.0, -5.0])
        # Look-at rotation: light direction
        direction = center - key_pose[:3, 3]
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        key_pose[:3, :3] = _look_at_rotation(direction)
        scene.add(key_light, pose=key_pose)

        # Fill light — cool blue from lower-right
        fill_light = pyrender.DirectionalLight(
            color=np.array([0.7, 0.8, 1.0]),
            intensity=3.0,
        )
        fill_pose = np.eye(4)
        fill_pose[:3, 3] = center + np.array([5.0, 8.0, 5.0])
        direction2 = center - fill_pose[:3, 3]
        direction2 = direction2 / (np.linalg.norm(direction2) + 1e-8)
        fill_pose[:3, :3] = _look_at_rotation(direction2)
        scene.add(fill_light, pose=fill_pose)

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _add_camera(self, scene, rows: int, cols: int) -> None:
        import pyrender

        center = self._grid_to_world(rows / 2, cols / 2)

        # Orthographic camera — eliminates perspective distortion
        grid_extent = max(rows, cols) * self.TILE_SIZE
        ortho_scale = grid_extent * 0.75
        camera = pyrender.OrthographicCamera(
            xmag=ortho_scale / 2,
            ymag=ortho_scale / 2,
            znear=0.01,
            zfar=200.0,
        )

        camera_pose = _compute_isometric_pose(center, distance=grid_extent * 1.5)
        scene.add(camera, pose=camera_pose)


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------


def _compute_isometric_pose(
    target: np.ndarray,
    distance: float = 20.0,
    azimuth_deg: float = 45.0,
    elevation_deg: float = 35.264,
) -> np.ndarray:
    """Compute a 4×4 camera pose for an isometric view.

    Classic isometric: azimuth 45°, elevation ≈ 35.264° (arctan(1/√2)).
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)

    # Camera position in spherical coords around target
    eye = target + distance * np.array([
        math.cos(el) * math.sin(az),
        math.sin(el),
        math.cos(el) * math.cos(az),
    ])

    # Build look-at matrix
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)

    true_up = np.cross(right, forward)
    true_up = true_up / (np.linalg.norm(true_up) + 1e-8)

    # pyrender camera looks down -Z in its local frame
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def _look_at_rotation(direction: np.ndarray) -> np.ndarray:
    """Build a 3×3 rotation that orients -Z along *direction*."""
    forward = direction / (np.linalg.norm(direction) + 1e-8)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    norm = np.linalg.norm(right)
    if norm < 1e-6:
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        norm = np.linalg.norm(right)
    right = right / (norm + 1e-8)
    true_up = np.cross(right, forward)
    true_up = true_up / (np.linalg.norm(true_up) + 1e-8)
    R = np.eye(3)
    R[:, 0] = right
    R[:, 1] = true_up
    R[:, 2] = -forward
    return R
