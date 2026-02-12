"""Programmatic sprite generation for pixel rendering.

This module provides functions to generate distinct, recognizable sprites
for all entity types using pygame.draw primitives.
"""

from __future__ import annotations

import math

import pygame

from agentick.core.types import COLORS, Direction


class SpriteGenerator:
    """Generate programmatic sprites for entities."""

    def __init__(self, tile_size: int = 32):
        """
        Initialize sprite generator.

        Args:
            tile_size: Size of each tile in pixels (8, 16, 32, or 64)
        """
        self.tile_size = tile_size
        self.cache: dict[tuple, pygame.Surface] = {}

    def get_agent_sprite(
        self,
        orientation: Direction,
        color: tuple[int, int, int] = None,
        team: int = 0,
    ) -> pygame.Surface:
        """
        Generate agent sprite with orientation indicator.

        Args:
            orientation: Direction agent is facing
            color: RGB color (default uses agent color)
            team: Team number for multi-agent scenarios

        Returns:
            Surface with agent sprite
        """
        cache_key = ("agent", orientation, color, team, self.tile_size)
        if cache_key in self.cache:
            return self.cache[cache_key]

        surface = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
        surface.fill((0, 0, 0, 0))  # Transparent

        if color is None:
            color = COLORS.get("agent", (0, 120, 255))

        center = self.tile_size // 2
        radius = self.tile_size // 3

        # Draw circle for agent body
        pygame.draw.circle(surface, color, (center, center), radius)

        # Draw direction indicator (triangle pointing in orientation direction)
        triangle_size = radius // 2
        if orientation == Direction.NORTH:
            points = [
                (center, center - radius),
                (center - triangle_size, center - radius + triangle_size),
                (center + triangle_size, center - radius + triangle_size),
            ]
        elif orientation == Direction.EAST:
            points = [
                (center + radius, center),
                (center + radius - triangle_size, center - triangle_size),
                (center + radius - triangle_size, center + triangle_size),
            ]
        elif orientation == Direction.SOUTH:
            points = [
                (center, center + radius),
                (center - triangle_size, center + radius - triangle_size),
                (center + triangle_size, center + radius - triangle_size),
            ]
        else:  # WEST
            points = [
                (center - radius, center),
                (center - radius + triangle_size, center - triangle_size),
                (center - radius + triangle_size, center + triangle_size),
            ]

        pygame.draw.polygon(surface, (255, 255, 255), points)

        self.cache[cache_key] = surface
        return surface

    def get_wall_sprite(self, textured: bool = True) -> pygame.Surface:
        """Generate wall sprite with optional texture."""
        cache_key = ("wall", textured, self.tile_size)
        if cache_key in self.cache:
            return self.cache[cache_key]

        surface = pygame.Surface((self.tile_size, self.tile_size))
        color = COLORS.get("wall", (64, 64, 64))
        surface.fill(color)

        if textured and self.tile_size >= 16:
            # Cross-hatch pattern
            lighter = tuple(min(255, c + 20) for c in color)
            for i in range(0, self.tile_size, 4):
                pygame.draw.line(surface, lighter, (i, 0), (0, i), 1)
                pygame.draw.line(surface, lighter, (self.tile_size - i, 0), (self.tile_size, i), 1)

        self.cache[cache_key] = surface
        return surface

    def get_key_sprite(self, color: tuple[int, int, int] = None) -> pygame.Surface:
        """Generate key sprite in specified color."""
        if color is None:
            color = COLORS.get("key", (255, 215, 0))

        cache_key = ("key", color, self.tile_size)
        if cache_key in self.cache:
            return self.cache[cache_key]

        surface = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
        surface.fill((0, 0, 0, 0))

        center = self.tile_size // 2
        # Key head (circle)
        head_radius = self.tile_size // 6
        pygame.draw.circle(surface, color, (center - self.tile_size // 6, center), head_radius)

        # Key shaft
        shaft_width = self.tile_size // 12
        shaft_length = self.tile_size // 2
        shaft_rect = pygame.Rect(
            center - self.tile_size // 6 + head_radius,
            center - shaft_width // 2,
            shaft_length,
            shaft_width,
        )
        pygame.draw.rect(surface, color, shaft_rect)

        # Key teeth
        if self.tile_size >= 16:
            tooth_size = self.tile_size // 12
            for i in range(2):
                tooth_rect = pygame.Rect(
                    center + self.tile_size // 6 + i * tooth_size * 2,
                    center - shaft_width // 2,
                    tooth_size,
                    tooth_size,
                )
                pygame.draw.rect(surface, color, tooth_rect)

        self.cache[cache_key] = surface
        return surface

    def get_door_sprite(
        self, color: tuple[int, int, int] = None, locked: bool = True
    ) -> pygame.Surface:
        """Generate door sprite with keyhole."""
        if color is None:
            color = COLORS.get("door", (139, 69, 19))

        cache_key = ("door", color, locked, self.tile_size)
        if cache_key in self.cache:
            return self.cache[cache_key]

        surface = pygame.Surface((self.tile_size, self.tile_size))
        surface.fill(color)

        # Door frame
        frame_color = tuple(max(0, c - 30) for c in color)
        pygame.draw.rect(surface, frame_color, (0, 0, self.tile_size, self.tile_size), 2)

        if locked and self.tile_size >= 16:
            # Keyhole
            center = self.tile_size // 2
            keyhole_radius = self.tile_size // 10
            pygame.draw.circle(surface, (30, 30, 30), (center, center), keyhole_radius)
            # Keyhole slot
            slot_rect = pygame.Rect(
                center - keyhole_radius // 3,
                center,
                keyhole_radius * 2 // 3,
                keyhole_radius,
            )
            pygame.draw.rect(surface, (30, 30, 30), slot_rect)

        self.cache[cache_key] = surface
        return surface

    def get_goal_sprite(self, pulsing_phase: float = 0.0) -> pygame.Surface:
        """Generate goal sprite (star/flag)."""
        cache_key = ("goal", int(pulsing_phase * 10), self.tile_size)
        if cache_key in self.cache:
            return self.cache[cache_key]

        surface = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
        surface.fill((0, 0, 0, 0))

        color = COLORS.get("goal", (0, 255, 0))
        center = self.tile_size // 2

        # Draw star
        num_points = 5
        outer_radius = self.tile_size // 2 - 2
        inner_radius = outer_radius // 2

        points = []
        for i in range(num_points * 2):
            angle = math.pi / 2 + (i * math.pi / num_points)
            radius = outer_radius if i % 2 == 0 else inner_radius
            x = center + radius * math.cos(angle)
            y = center - radius * math.sin(angle)
            points.append((x, y))

        pygame.draw.polygon(surface, color, points)

        self.cache[cache_key] = surface
        return surface

    def get_hazard_sprite(self) -> pygame.Surface:
        """Generate hazard sprite with warning pattern."""
        cache_key = ("hazard", self.tile_size)
        if cache_key in self.cache:
            return self.cache[cache_key]

        surface = pygame.Surface((self.tile_size, self.tile_size))
        color1 = COLORS.get("hazard", (255, 0, 0))
        color2 = (255, 255, 0)

        # Diagonal stripes
        stripe_width = max(2, self.tile_size // 8)
        for i in range(-self.tile_size, self.tile_size * 2, stripe_width * 2):
            points = [
                (i, 0),
                (i + stripe_width, 0),
                (i + stripe_width - self.tile_size, self.tile_size),
                (i - self.tile_size, self.tile_size),
            ]
            pygame.draw.polygon(surface, color1 if (i // stripe_width) % 2 == 0 else color2, points)

        self.cache[cache_key] = surface
        return surface

    def get_switch_sprite(self, state: bool = False) -> pygame.Surface:
        """Generate switch sprite showing on/off state."""
        cache_key = ("switch", state, self.tile_size)
        if cache_key in self.cache:
            return self.cache[cache_key]

        surface = pygame.Surface((self.tile_size, self.tile_size))
        bg_color = (180, 180, 180)
        surface.fill(bg_color)

        center_x = self.tile_size // 2
        center_y = self.tile_size // 2

        # Switch base
        base_rect = pygame.Rect(
            center_x - self.tile_size // 3,
            center_y - self.tile_size // 6,
            self.tile_size * 2 // 3,
            self.tile_size // 3,
        )
        pygame.draw.rect(surface, (100, 100, 100), base_rect, border_radius=5)

        # Switch lever
        lever_color = (0, 255, 0) if state else (255, 0, 0)
        lever_width = self.tile_size // 6
        lever_height = self.tile_size // 3

        if state:  # ON position (right)
            lever_x = center_x + self.tile_size // 6
        else:  # OFF position (left)
            lever_x = center_x - self.tile_size // 3

        lever_rect = pygame.Rect(
            lever_x,
            center_y - lever_height // 2,
            lever_width,
            lever_height,
        )
        pygame.draw.rect(surface, lever_color, lever_rect, border_radius=3)

        self.cache[cache_key] = surface
        return surface

    def get_box_sprite(self) -> pygame.Surface:
        """Generate 3D-ish box sprite."""
        cache_key = ("box", self.tile_size)
        if cache_key in self.cache:
            return self.cache[cache_key]

        surface = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
        surface.fill((0, 0, 0, 0))

        color = COLORS.get("box", (139, 90, 43))

        # Main box face
        box_size = self.tile_size - 4
        box_rect = pygame.Rect(2, 2, box_size, box_size)
        pygame.draw.rect(surface, color, box_rect)

        # 3D effect - side panels
        darker = tuple(max(0, c - 40) for c in color)
        lightest = tuple(min(255, c + 20) for c in color)

        offset = max(2, self.tile_size // 8)

        # Top face (lighter)
        top_points = [
            (2, 2),
            (2 + offset, 2 - offset),
            (2 + box_size + offset, 2 - offset),
            (2 + box_size, 2),
        ]
        pygame.draw.polygon(surface, lightest, top_points)

        # Right face (darker)
        right_points = [
            (2 + box_size, 2),
            (2 + box_size + offset, 2 - offset),
            (2 + box_size + offset, 2 + box_size - offset),
            (2 + box_size, 2 + box_size),
        ]
        pygame.draw.polygon(surface, darker, right_points)

        # Draw border on main face
        pygame.draw.rect(surface, darker, box_rect, 2)

        self.cache[cache_key] = surface
        return surface

    def get_resource_sprite(self, resource_type: str = "heart") -> pygame.Surface:
        """
        Generate resource sprite.

        Args:
            resource_type: "heart" (health), "lightning" (energy), "gem" (score)
        """
        cache_key = ("resource", resource_type, self.tile_size)
        if cache_key in self.cache:
            return self.cache[cache_key]

        surface = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
        surface.fill((0, 0, 0, 0))

        center = self.tile_size // 2

        if resource_type == "heart":
            color = (255, 0, 100)
            # Draw heart shape
            pygame.draw.circle(
                surface,
                color,
                (center - self.tile_size // 6, center - self.tile_size // 8),
                self.tile_size // 4,
            )
            pygame.draw.circle(
                surface,
                color,
                (center + self.tile_size // 6, center - self.tile_size // 8),
                self.tile_size // 4,
            )
            points = [
                (center - self.tile_size // 3, center - self.tile_size // 8),
                (center, center + self.tile_size // 3),
                (center + self.tile_size // 3, center - self.tile_size // 8),
            ]
            pygame.draw.polygon(surface, color, points)

        elif resource_type == "lightning":
            color = (255, 255, 0)
            # Draw lightning bolt
            points = [
                (center + self.tile_size // 6, 4),
                (center - self.tile_size // 12, center),
                (center + self.tile_size // 12, center),
                (center - self.tile_size // 6, self.tile_size - 4),
                (center, center + self.tile_size // 8),
                (center - self.tile_size // 12, center - self.tile_size // 8),
            ]
            pygame.draw.polygon(surface, color, points)
            pygame.draw.polygon(surface, (200, 200, 0), points, 2)

        elif resource_type == "gem":
            color = (0, 200, 255)
            # Draw gem (diamond shape)
            points = [
                (center, 4),
                (center + self.tile_size // 3, center),
                (center, self.tile_size - 4),
                (center - self.tile_size // 3, center),
            ]
            pygame.draw.polygon(surface, color, points)
            # Inner facets
            inner_points = [
                (center, center - self.tile_size // 4),
                (center + self.tile_size // 6, center),
                (center, center + self.tile_size // 4),
                (center - self.tile_size // 6, center),
            ]
            lighter = tuple(min(255, c + 50) for c in color)
            pygame.draw.polygon(surface, lighter, inner_points)

        self.cache[cache_key] = surface
        return surface

    def get_fog_sprite(self, alpha: int = 180) -> pygame.Surface:
        """Generate fog of war overlay."""
        cache_key = ("fog", alpha, self.tile_size)
        if cache_key in self.cache:
            return self.cache[cache_key]

        surface = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
        surface.fill((0, 0, 0, alpha))

        self.cache[cache_key] = surface
        return surface

    def clear_cache(self):
        """Clear sprite cache (useful when changing tile size)."""
        self.cache.clear()


def create_hud_overlay(
    width: int,
    height: int,
    step_count: int,
    max_steps: int,
    reward: float,
    inventory: list,
    energy: float = 1.0,
    health: float = 1.0,
) -> pygame.Surface:
    """
    Create HUD overlay for pixel rendering.

    Args:
        width: Render width in pixels
        height: Render height in pixels
        step_count: Current step
        max_steps: Maximum steps
        reward: Cumulative reward
        inventory: List of inventory items
        energy: Energy level (0-1)
        health: Health level (0-1)

    Returns:
        Semi-transparent surface with HUD elements
    """
    hud = pygame.Surface((width, height), pygame.SRCALPHA)

    # HUD background bar at top
    hud_height = 30
    pygame.draw.rect(hud, (0, 0, 0, 180), (0, 0, width, hud_height))

    # Initialize font (if not already done)
    pygame.font.init()
    font = pygame.font.Font(None, 20)

    # Step counter
    step_text = font.render(f"Step: {step_count}/{max_steps}", True, (255, 255, 255))
    hud.blit(step_text, (10, 5))

    # Reward
    reward_text = font.render(f"Reward: {reward:.2f}", True, (255, 255, 255))
    hud.blit(reward_text, (150, 5))

    # Energy bar
    bar_width = 80
    bar_height = 10
    bar_x = width - bar_width - 90
    bar_y = 10

    pygame.draw.rect(hud, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height))
    pygame.draw.rect(hud, (255, 255, 0), (bar_x, bar_y, int(bar_width * energy), bar_height))
    pygame.draw.rect(hud, (255, 255, 255), (bar_x, bar_y, bar_width, bar_height), 1)

    energy_label = font.render("E", True, (255, 255, 255))
    hud.blit(energy_label, (bar_x - 15, bar_y - 5))

    # Health bar
    bar_x = width - bar_width - 10
    pygame.draw.rect(hud, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height))
    pygame.draw.rect(hud, (255, 0, 0), (bar_x, bar_y, int(bar_width * health), bar_height))
    pygame.draw.rect(hud, (255, 255, 255), (bar_x, bar_y, bar_width, bar_height), 1)

    health_label = font.render("H", True, (255, 255, 255))
    hud.blit(health_label, (bar_x - 15, bar_y - 5))

    # Inventory (show count)
    if inventory:
        inv_text = font.render(f"Inv: {len(inventory)}", True, (255, 255, 255))
        hud.blit(inv_text, (300, 5))

    return hud
