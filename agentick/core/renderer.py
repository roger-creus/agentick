"""Enhanced multi-modal rendering engine for gridworld environments.

This module provides production-quality renderers with:
- ASCII: ANSI color support, legend, multi-character cells
- Language: Full spatial reasoning, memory awareness, goals, threats
- Pixel: Distinct programmatic sprites, HUD overlay, multiple tile sizes
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pygame

from agentick.core.entity import Agent, Entity
from agentick.core.grid import Grid
from agentick.core.language import AdvancedLanguageRenderer, LanguageConfig
from agentick.core.sprites import SpriteGenerator, create_hud_overlay
from agentick.core.types import COLORS, CellType, Direction, ObjectType

# Set pygame to headless by default
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


class Renderer(Protocol):
    """Protocol for renderer implementations."""

    def render(
        self,
        grid: Grid,
        entities: list[Entity],
        agent: Agent,
        info: dict[str, Any],
    ) -> Any:
        """Render the current state."""
        ...


@dataclass
class RenderConfig:
    """Configuration for rendering."""

    tile_size: int = 32
    show_grid: bool = True
    show_fog: bool = False
    show_hud: bool = True
    use_ansi_colors: bool = True
    font_size: int = 12


# ANSI color codes
ANSI_COLORS = {
    "reset": "\033[0m",
    "agent": "\033[94m",  # Blue
    "wall": "\033[90m",  # Dark gray
    "goal": "\033[92m",  # Green
    "key": "\033[93m",  # Yellow
    "door": "\033[33m",  # Orange/brown
    "hazard": "\033[91m",  # Red
    "box": "\033[95m",  # Magenta
    "empty": "\033[37m",  # White
    "water": "\033[96m",  # Cyan
    "ice": "\033[97m",  # Bright white
}


class ASCIIRenderer:
    """Render grid as ASCII art with ANSI colors and legend."""

    def __init__(self, config: RenderConfig | None = None):
        self.config = config or RenderConfig()

    def render(
        self,
        grid: Grid,
        entities: list[Entity],
        agent: Agent,
        info: dict[str, Any],
    ) -> str:
        """
        Render grid as ASCII string with ANSI colors and legend.

        Returns:
            Multi-line ASCII representation with colors and legend
        """
        output = []

        # Build character grid
        char_grid = np.full((grid.height, grid.width), ".", dtype=object)
        color_grid = np.full((grid.height, grid.width), "empty", dtype=object)

        # Terrain layer
        for y in range(grid.height):
            for x in range(grid.width):
                terrain_val = int(grid.terrain[y, x])
                if terrain_val == CellType.EMPTY:
                    char_grid[y, x] = "."
                    color_grid[y, x] = "empty"
                elif terrain_val == CellType.WALL:
                    char_grid[y, x] = "#"
                    color_grid[y, x] = "wall"
                elif terrain_val == CellType.HAZARD:
                    char_grid[y, x] = "X"
                    color_grid[y, x] = "hazard"
                elif terrain_val == CellType.WATER:
                    char_grid[y, x] = "~"
                    color_grid[y, x] = "water"
                elif terrain_val == CellType.ICE:
                    char_grid[y, x] = "i"
                    color_grid[y, x] = "ice"
                elif terrain_val == CellType.HOLE:
                    char_grid[y, x] = "O"
                    color_grid[y, x] = "wall"

        # Objects layer
        for y in range(grid.height):
            for x in range(grid.width):
                obj_val = int(grid.objects[y, x])
                if obj_val == ObjectType.GOAL:
                    char_grid[y, x] = "G"
                    color_grid[y, x] = "goal"
                elif obj_val == ObjectType.KEY:
                    char_grid[y, x] = "K"
                    color_grid[y, x] = "key"
                elif obj_val == ObjectType.DOOR:
                    char_grid[y, x] = "D"
                    color_grid[y, x] = "door"
                elif obj_val == ObjectType.SWITCH:
                    char_grid[y, x] = "S"
                    color_grid[y, x] = "key"
                elif obj_val == ObjectType.BOX:
                    char_grid[y, x] = "B"
                    color_grid[y, x] = "box"
                elif obj_val == ObjectType.TARGET:
                    char_grid[y, x] = "T"
                    color_grid[y, x] = "goal"
                elif obj_val == ObjectType.TOOL:
                    char_grid[y, x] = "t"
                    color_grid[y, x] = "key"
                elif obj_val == ObjectType.RESOURCE:
                    char_grid[y, x] = "r"
                    color_grid[y, x] = "key"
                elif obj_val == ObjectType.BREADCRUMB:
                    char_grid[y, x] = "*"
                    color_grid[y, x] = "key"

        # Agent layer
        ax, ay = agent.position
        # Multi-character cell: show agent orientation
        direction_chars = {
            Direction.NORTH: "^",
            Direction.EAST: ">",
            Direction.SOUTH: "v",
            Direction.WEST: "<",
        }
        char_grid[ay, ax] = direction_chars[agent.orientation]
        color_grid[ay, ax] = "agent"

        # If agent is on an object, show both
        obj_val = int(grid.objects[ay, ax])
        if obj_val != ObjectType.NONE:
            # Multi-character representation
            obj_char = {
                ObjectType.GOAL: "G",
                ObjectType.KEY: "K",
                ObjectType.DOOR: "D",
            }.get(obj_val, "")
            if obj_char:
                char_grid[ay, ax] = f"{direction_chars[agent.orientation]}{obj_char}"

        # Add other agents/NPCs from entities
        for entity in entities:
            if entity.entity_type in ("npc", "enemy"):
                ex, ey = entity.position
                char_grid[ey, ex] = "N" if entity.entity_type == "npc" else "E"
                color_grid[ey, ex] = "hazard" if entity.entity_type == "enemy" else "agent"

        # Convert to string with ANSI colors
        for y in range(grid.height):
            row_parts = []
            for x in range(grid.width):
                char = char_grid[y, x]
                # Pad multi-character cells
                if len(str(char)) > 1:
                    char_str = str(char)[:2].ljust(2)
                else:
                    char_str = str(char).ljust(2)

                if self.config.use_ansi_colors:
                    color = ANSI_COLORS.get(color_grid[y, x], "")
                    reset = ANSI_COLORS["reset"]
                    row_parts.append(f"{color}{char_str}{reset}")
                else:
                    row_parts.append(char_str)

            output.append("".join(row_parts))

        # Add legend
        legend_parts = [
            "\n--- Legend ---",
            f"{'^ v < >' if self.config.use_ansi_colors else 'A'}: Agent (facing direction)",
            "#: Wall",
            "G: Goal",
            "K: Key",
            "D: Door",
            "B: Box",
            "X: Hazard",
            "~: Water",
        ]

        if self.config.use_ansi_colors:
            # Colorize legend
            colored_legend = []
            for line in legend_parts:
                if ":" in line and line != legend_parts[0]:
                    parts = line.split(":", 1)
                    symbol = parts[0].strip()
                    desc = parts[1].strip()

                    # Find color for this symbol
                    color_map = {
                        "^ v < >": "agent",
                        "A": "agent",
                        "#": "wall",
                        "G": "goal",
                        "K": "key",
                        "D": "door",
                        "X": "hazard",
                        "B": "box",
                        "~": "water",
                    }

                    if symbol in color_map:
                        color = ANSI_COLORS.get(color_map[symbol], "")
                        reset = ANSI_COLORS["reset"]
                        colored_legend.append(f"{color}{symbol}{reset}: {desc}")
                    else:
                        colored_legend.append(line)
                else:
                    colored_legend.append(line)
            legend_parts = colored_legend

        output.extend(legend_parts)

        return "\n".join(output)


class EnhancedLanguageRenderer:
    """Wrapper for AdvancedLanguageRenderer with configuration options."""

    def __init__(
        self,
        structured: bool = False,
        verbosity: str = "standard",
        perspective: str = "first_person",
    ):
        """
        Initialize language renderer.

        Args:
            structured: Return structured dict instead of text
            verbosity: "minimal", "standard", or "verbose"
            perspective: "first_person", "third_person", or "omniscient"
        """
        self.structured = structured  # Keep for backward compatibility
        config = LanguageConfig(
            verbosity=verbosity,
            perspective=perspective,
            structured=structured,
        )
        self.renderer = AdvancedLanguageRenderer(config)

    def render(
        self,
        grid: Grid,
        entities: list[Entity],
        agent: Agent,
        info: dict[str, Any],
    ) -> str | dict[str, Any]:
        """
        Render grid as language description.

        Returns:
            Natural language string or structured dict
        """
        return self.renderer.render(grid, entities, agent, info)


class EnhancedPixelRenderer:
    """Render grid as RGB pixel array with distinct sprites and HUD."""

    def __init__(self, config: RenderConfig | None = None):
        self.config = config or RenderConfig()
        pygame.init()
        self.tile_size = self.config.tile_size
        self.sprite_gen = SpriteGenerator(self.tile_size)
        self._cache = {}  # Cache for rendered frames
        self._last_state_hash = None

    def render(
        self,
        grid: Grid,
        entities: list[Entity],
        agent: Agent,
        info: dict[str, Any],
    ) -> np.ndarray:
        """
        Render grid as RGB array with sprites and HUD.

        Returns:
            Array of shape (H, W, 3) with uint8 values
        """
        # Compute state hash for caching
        state_hash = self._compute_state_hash(grid, agent, entities)

        # Check cache
        if state_hash == self._last_state_hash and state_hash in self._cache:
            return self._cache[state_hash].copy()

        h = grid.height * self.tile_size
        w = grid.width * self.tile_size

        surface = pygame.Surface((w, h))
        surface.fill(COLORS.get("empty", (240, 240, 240)))

        # Draw terrain
        for y in range(grid.height):
            for x in range(grid.width):
                cell_type = CellType(grid.terrain[y, x])

                if cell_type == CellType.WALL:
                    sprite = self.sprite_gen.get_wall_sprite()
                    surface.blit(sprite, (x * self.tile_size, y * self.tile_size))
                elif cell_type == CellType.HAZARD:
                    sprite = self.sprite_gen.get_hazard_sprite()
                    surface.blit(sprite, (x * self.tile_size, y * self.tile_size))
                elif cell_type == CellType.WATER:
                    self._draw_tile(surface, x, y, COLORS.get("water", (0, 100, 200)))
                elif cell_type == CellType.ICE:
                    self._draw_tile(surface, x, y, COLORS.get("ice", (200, 230, 255)))

        # Draw objects
        for y in range(grid.height):
            for x in range(grid.width):
                obj_type = ObjectType(grid.objects[y, x])

                if obj_type == ObjectType.GOAL:
                    sprite = self.sprite_gen.get_goal_sprite()
                    surface.blit(sprite, (x * self.tile_size, y * self.tile_size))
                elif obj_type == ObjectType.KEY:
                    sprite = self.sprite_gen.get_key_sprite()
                    surface.blit(sprite, (x * self.tile_size, y * self.tile_size))
                elif obj_type == ObjectType.DOOR:
                    sprite = self.sprite_gen.get_door_sprite()
                    surface.blit(sprite, (x * self.tile_size, y * self.tile_size))
                elif obj_type == ObjectType.BOX:
                    sprite = self.sprite_gen.get_box_sprite()
                    surface.blit(sprite, (x * self.tile_size, y * self.tile_size))
                elif obj_type == ObjectType.SWITCH:
                    sprite = self.sprite_gen.get_switch_sprite()
                    surface.blit(sprite, (x * self.tile_size, y * self.tile_size))
                elif obj_type == ObjectType.RESOURCE:
                    sprite = self.sprite_gen.get_resource_sprite("gem")
                    surface.blit(sprite, (x * self.tile_size, y * self.tile_size))

        # Draw other entities
        for entity in entities:
            ex, ey = entity.position
            if entity.entity_type == "enemy":
                # Enemy sprite (red agent)
                sprite = self.sprite_gen.get_agent_sprite(Direction.NORTH, color=(255, 0, 0))
                surface.blit(sprite, (ex * self.tile_size, ey * self.tile_size))

        # Draw agent with orientation
        ax, ay = agent.position
        agent_sprite = self.sprite_gen.get_agent_sprite(agent.orientation)
        surface.blit(agent_sprite, (ax * self.tile_size, ay * self.tile_size))

        # Add grid lines if enabled
        if self.config.show_grid:
            for y in range(grid.height + 1):
                pygame.draw.line(
                    surface,
                    (200, 200, 200),
                    (0, y * self.tile_size),
                    (w, y * self.tile_size),
                    1,
                )
            for x in range(grid.width + 1):
                pygame.draw.line(
                    surface,
                    (200, 200, 200),
                    (x * self.tile_size, 0),
                    (x * self.tile_size, h),
                    1,
                )

        # Add HUD overlay if enabled
        if self.config.show_hud:
            hud = create_hud_overlay(
                w,
                h,
                step_count=info.get("step_count", 0),
                max_steps=info.get("max_steps", 100),
                reward=info.get("episode_reward", 0.0),
                inventory=agent.inventory,
                energy=agent.energy,
                health=agent.health,
            )
            surface.blit(hud, (0, 0))

        # Convert to numpy array
        arr = pygame.surfarray.array3d(surface)
        # Pygame uses (W, H, 3), transpose to (H, W, 3)
        arr = np.transpose(arr, (1, 0, 2))
        result = arr.astype(np.uint8)

        # Cache the result
        self._cache[state_hash] = result.copy()
        self._last_state_hash = state_hash

        # Keep cache bounded
        if len(self._cache) > 10:
            # Remove oldest entries
            to_remove = list(self._cache.keys())[: len(self._cache) - 10]
            for key in to_remove:
                del self._cache[key]

        return result

    def _compute_state_hash(self, grid: Grid, agent: Agent, entities: list[Entity]) -> int:
        """Compute hash of current state for caching."""
        # Hash grid terrain and objects
        terrain_hash = hash(grid.terrain.tobytes())
        objects_hash = hash(grid.objects.tobytes())

        # Hash agent state
        agent_hash = hash((agent.position, agent.orientation.value))

        # Hash entities
        entity_hash = hash(tuple((e.position, e.entity_type) for e in entities))

        return hash((terrain_hash, objects_hash, agent_hash, entity_hash))

    def _draw_tile(self, surface: pygame.Surface, x: int, y: int, color: tuple):
        """Draw a colored tile."""
        rect = pygame.Rect(
            x * self.tile_size,
            y * self.tile_size,
            self.tile_size,
            self.tile_size,
        )
        pygame.draw.rect(surface, color, rect)


class StateDictRenderer:
    """Render grid as structured state dictionary."""

    def __init__(self, fast_mode: bool = False):
        """
        Initialize renderer.

        Args:
            fast_mode: If True, skip expensive conversions (tolist(), etc.)
        """
        self.fast_mode = fast_mode

    def render(
        self,
        grid: Grid,
        entities: list[Entity],
        agent: Agent,
        info: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Render grid as complete state dictionary.

        Returns:
            Dictionary with full grid and entity state
        """
        if self.fast_mode:
            # Fast mode: keep numpy arrays, minimal conversions
            return {
                "grid": {
                    "height": grid.height,
                    "width": grid.width,
                    "terrain": grid.terrain,  # Keep as numpy array
                    "objects": grid.objects,  # Keep as numpy array
                    "agents": grid.agents,  # Keep as numpy array
                },
                "agent": {
                    "position": agent.position,
                    "orientation": agent.orientation,  # Keep as enum
                    "inventory": agent.inventory,  # Keep as list
                    "energy": agent.energy,
                    "health": agent.health,
                },
                "entities": entities,  # Keep as list
                "info": info,
            }
        else:
            # Standard mode: full conversion for compatibility
            return {
                "grid": {
                    "height": grid.height,
                    "width": grid.width,
                    "terrain": grid.terrain.tolist(),
                    "objects": grid.objects.tolist(),
                    "agents": grid.agents.tolist(),
                },
                "agent": {
                    "position": agent.position,
                    "orientation": agent.orientation.name.lower(),
                    "inventory": [item.to_dict() for item in agent.inventory],
                    "energy": agent.energy,
                    "health": agent.health,
                },
                "entities": [entity.to_dict() for entity in entities],
                "info": info,
            }


def create_renderer(
    mode: str,
    tile_size: int = 32,
    show_hud: bool = True,
    verbosity: str = "standard",
    perspective: str = "first_person",
    fast_mode: bool = False,
    **kwargs,
) -> Renderer:
    """
    Factory function to create renderer for given mode.

    Args:
        mode: Render mode ("ascii", "language", "language_structured", "rgb_array", "state_dict")
        tile_size: Tile size for pixel rendering (8, 16, 32, or 64)
        show_hud: Show HUD overlay in pixel mode
        verbosity: Language verbosity ("minimal", "standard", "verbose")
        perspective: Language perspective ("first_person", "third_person", "omniscient")
        fast_mode: Enable fast mode for state_dict (skip expensive conversions)
        **kwargs: Additional arguments for renderer config

    Returns:
        Renderer instance
    """
    if mode == "ascii":
        config = RenderConfig(tile_size=tile_size, **kwargs)
        return ASCIIRenderer(config)
    elif mode == "language":
        return EnhancedLanguageRenderer(
            structured=False,
            verbosity=verbosity,
            perspective=perspective,
        )
    elif mode == "language_structured":
        return EnhancedLanguageRenderer(
            structured=True,
            verbosity=verbosity,
            perspective=perspective,
        )
    elif mode in ("rgb_array", "rgb_array_2d", "human"):
        config = RenderConfig(tile_size=tile_size, show_hud=show_hud, **kwargs)
        return EnhancedPixelRenderer(config)
    elif mode == "state_dict":
        return StateDictRenderer(fast_mode=fast_mode)
    else:
        raise ValueError(f"Unknown render mode: {mode}")


# Backward compatibility aliases
LanguageRenderer = EnhancedLanguageRenderer
PixelRenderer = EnhancedPixelRenderer
