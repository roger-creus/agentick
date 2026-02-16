"""3D model asset loading and caching."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Default model directory inside the package
_PACKAGE_MODELS_DIR = Path(__file__).parent / "models"


class AssetLibrary:
    """Loads and caches GLB models for the 3D renderer.

    Looks for models in this order:
      1. User-specified directory (*asset_dir* parameter)
      2. Package data: ``agentick/rendering/models/``
      3. Project root: ``assets/models/glb/`` (development layout)

    Falls back to primitive shapes (colored cubes/spheres) if a GLB file
    is not found for a given slug.
    """

    def __init__(self, asset_dir: str | Path | None = None):
        import trimesh

        self._trimesh = trimesh
        self._cache: dict[str, trimesh.Scene] = {}
        self._fallback_cache: dict[str, trimesh.Scene] = {}
        self._asset_dirs = self._build_search_dirs(asset_dir)
        logger.debug("AssetLibrary search dirs: %s", self._asset_dirs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_model(self, slug: str) -> "trimesh.Scene":
        """Load a model by slug name. Returns cached copy if already loaded.

        Raises ``FileNotFoundError`` if the GLB is not found anywhere.
        """
        if slug in self._cache:
            return self._cache[slug]

        path = self._find_glb(slug)
        if path is None:
            raise FileNotFoundError(
                f"GLB model '{slug}.glb' not found in any search directory: "
                f"{self._asset_dirs}"
            )

        scene = self._trimesh.load(str(path), force="scene")
        self._cache[slug] = scene
        logger.debug("Loaded GLB model: %s from %s", slug, path)
        return scene

    def get_or_fallback(self, slug: str) -> "trimesh.Scene":
        """Load model, or generate a primitive fallback shape."""
        try:
            return self.get_model(slug)
        except FileNotFoundError:
            return self._get_fallback(slug)

    def get_mesh(self, slug: str) -> "trimesh.Trimesh":
        """Load model and return a single merged Trimesh (for placement in pyrender).

        Textures from GLB files are baked down to per-face vertex colors so
        that pyrender can handle them reliably under headless backends (EGL).
        """
        scene = self.get_or_fallback(slug)
        # Dump all geometry into a single mesh
        if hasattr(scene, "geometry") and len(scene.geometry) > 0:
            meshes = [_simplify_visual(m.copy()) for m in scene.geometry.values()]
            if len(meshes) == 1:
                mesh = meshes[0]
            else:
                mesh = self._trimesh.util.concatenate(meshes)
        elif isinstance(scene, self._trimesh.Trimesh):
            mesh = _simplify_visual(scene.copy())
        else:
            # Last resort: empty tiny mesh
            mesh = self._trimesh.creation.box(extents=(0.01, 0.01, 0.01))
        return mesh

    def preload_all(self) -> None:
        """Pre-load every GLB found in the search directories."""
        for d in self._asset_dirs:
            if d.is_dir():
                for glb in d.glob("*.glb"):
                    slug = glb.stem
                    if slug not in self._cache:
                        try:
                            self.get_model(slug)
                        except Exception:
                            logger.warning("Failed to preload %s", glb)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_search_dirs(user_dir: str | Path | None) -> list[Path]:
        dirs: list[Path] = []
        if user_dir is not None:
            dirs.append(Path(user_dir))
        dirs.append(_PACKAGE_MODELS_DIR)
        # Dev layout fallback
        project_root = Path(__file__).resolve().parents[2]
        dirs.append(project_root / "assets" / "models" / "glb")
        dirs.append(project_root / "agentick" / "assets" / "3d_models" / "glb")
        return dirs

    def _find_glb(self, slug: str) -> Path | None:
        for d in self._asset_dirs:
            candidate = d / f"{slug}.glb"
            if candidate.is_file():
                return candidate
        return None

    def _get_fallback(self, slug: str) -> "trimesh.Scene":
        if slug in self._fallback_cache:
            return self._fallback_cache[slug]

        mesh = _create_fallback_mesh(self._trimesh, slug)
        scene = self._trimesh.Scene(geometry={slug: mesh})
        self._fallback_cache[slug] = scene
        logger.debug("Created fallback primitive for '%s'", slug)
        return scene


# ------------------------------------------------------------------
# Fallback primitive generation
# ------------------------------------------------------------------

# (shape_type, color_rgb_0_255, scale)
_FALLBACK_DEFS: dict[str, tuple[str, tuple[int, int, int], float]] = {
    "player": ("sphere", (77, 128, 255), 0.4),
    "wall": ("box", (128, 128, 128), 0.9),
    "floor": ("box_flat", (204, 204, 179), 0.95),
    "key": ("sphere_small", (255, 204, 0), 0.25),
    "door": ("box_tall", (153, 102, 51), 0.8),
    "goal": ("sphere", (51, 255, 51), 0.35),
    "box": ("box", (179, 128, 77), 0.6),
    "switch": ("box_flat", (128, 128, 255), 0.5),
    "target": ("box_flat", (0, 255, 0), 0.45),
    "hazard": ("box_flat", (255, 77, 0), 0.9),
    "water": ("box_flat", (0, 150, 255), 0.95),
    "ice": ("box_flat", (200, 230, 255), 0.95),
    "hole": ("box_flat", (40, 40, 40), 0.95),
    "enemy": ("sphere", (255, 51, 51), 0.4),
    "npc": ("sphere", (51, 204, 255), 0.4),
    "heart": ("sphere_small", (255, 51, 102), 0.25),
    "gem": ("sphere_small", (102, 0, 204), 0.25),
    "lightning": ("sphere_small", (255, 255, 51), 0.2),
    "breadcrumb": ("sphere_small", (255, 192, 203), 0.15),
    "tool": ("box", (200, 200, 200), 0.3),
}


def _simplify_visual(mesh):
    """Convert textured/PBR visuals to per-vertex colors for pyrender compatibility.

    GLB models carry PBR materials with texture maps. Pyrender struggles with
    texture uploads under headless backends (EGL/OSMesa). We sample the texture
    at each vertex's UV coordinate to bake colours directly onto the mesh, then
    convert to face colors so pyrender can use ``smooth=False``.
    """
    import trimesh

    if mesh.visual is None:
        return mesh

    kind = mesh.visual.kind
    if kind == "face":
        return mesh

    if kind == "texture":
        # Try sampling the UV-mapped texture into per-vertex colors
        sampled = _sample_texture_to_vertex_colors(mesh)
        if sampled:
            return mesh

        # Fallback: try trimesh's built-in conversion
        try:
            colors = mesh.visual.to_color()
            if hasattr(colors, "face_colors") and colors.face_colors is not None:
                mesh.visual = colors
                return mesh
        except Exception:
            pass

    if kind == "vertex":
        try:
            colors = mesh.visual.to_color()
            if hasattr(colors, "face_colors") and colors.face_colors is not None:
                mesh.visual = colors
                return mesh
        except Exception:
            pass

    # Ultimate fallback: light gray
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        face_colors=np.tile(
            np.array([180, 180, 180, 255], dtype=np.uint8),
            (len(mesh.faces), 1),
        ),
    )
    return mesh


def _sample_texture_to_vertex_colors(mesh) -> bool:
    """Sample the texture at UV coords and write per-face colors onto *mesh*.

    Returns ``True`` on success, ``False`` if sampling could not be done.
    """
    import trimesh

    try:
        uv = mesh.visual.uv
        mat = mesh.visual.material
        tex = getattr(mat, "baseColorTexture", None)
        if tex is None or uv is None:
            return False

        img_arr = np.array(tex)
        if img_arr.ndim < 3:
            return False

        h, w = img_arr.shape[:2]
        channels = img_arr.shape[2]

        # Map UV → pixel coordinates
        u = np.clip(uv[:, 0], 0.0, 1.0) * (w - 1)
        v = np.clip(1.0 - uv[:, 1], 0.0, 1.0) * (h - 1)
        u_int = u.astype(np.intp)
        v_int = v.astype(np.intp)

        vertex_rgb = img_arr[v_int, u_int]  # (n_verts, 3 or 4)

        # Ensure RGBA
        if channels == 3:
            alpha = np.full((vertex_rgb.shape[0], 1), 255, dtype=np.uint8)
            vertex_rgba = np.hstack([vertex_rgb, alpha])
        else:
            vertex_rgba = vertex_rgb

        # Convert vertex colors → face colors by averaging the 3 face vertices
        face_colors = vertex_rgba[mesh.faces].mean(axis=1).astype(np.uint8)

        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            face_colors=face_colors,
        )
        return True
    except Exception:
        return False


def _create_fallback_mesh(trimesh_mod, slug: str):
    """Create a simple colored primitive mesh."""
    shape_type, color, scale = _FALLBACK_DEFS.get(slug, ("box", (200, 200, 200), 0.5))

    if shape_type == "sphere":
        mesh = trimesh_mod.creation.icosphere(subdivisions=2, radius=scale / 2)
    elif shape_type == "sphere_small":
        mesh = trimesh_mod.creation.icosphere(subdivisions=1, radius=scale / 2)
    elif shape_type == "box":
        mesh = trimesh_mod.creation.box(extents=(scale, scale, scale))
    elif shape_type == "box_flat":
        mesh = trimesh_mod.creation.box(extents=(scale, 0.1, scale))
    elif shape_type == "box_tall":
        mesh = trimesh_mod.creation.box(extents=(scale, scale * 1.5, scale))
    else:
        mesh = trimesh_mod.creation.box(extents=(scale, scale, scale))

    # Apply colour
    rgba = (*color, 255)
    mesh.visual.face_colors = np.full((len(mesh.faces), 4), rgba, dtype=np.uint8)
    return mesh
