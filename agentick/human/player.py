"""Human player interface using pygame.

Provides enhanced pygame interface for human evaluation.
"""

from __future__ import annotations

import os
import time
from typing import Any

import pygame

# Set pygame to headless mode by default (can override when actually playing)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


class HumanPlayer:
    """
    Enhanced pygame interface for human play.

    Features:
    - Tutorial mode with overlays
    - Practice rounds before scored rounds
    - Timer display, step counter, score display
    - Pause/resume functionality
    - Undo last action (configurable)
    - End-of-episode summary
    """

    def __init__(
        self,
        env: Any,
        window_size: tuple[int, int] = (800, 600),
        fps: int = 10,
        show_tutorial: bool = True,
        allow_undo: bool = False,
        practice_rounds: int = 1,
    ):
        """
        Initialize human player interface.

        Args:
            env: Environment to play
            window_size: Window dimensions
            fps: Frames per second
            show_tutorial: Show tutorial overlay
            allow_undo: Allow undoing last action
            practice_rounds: Number of practice rounds before scoring
        """
        self.env = env
        self.window_size = window_size
        self.fps = fps
        self.show_tutorial = show_tutorial
        self.allow_undo = allow_undo
        self.practice_rounds = practice_rounds

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Agentick Human Evaluation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # Game state
        self.running = False
        self.paused = False
        self.round_number = 0
        self.history: list[tuple[Any, int, float, bool, bool, dict]] = []

    def play_episode(self, practice: bool = False) -> dict[str, Any]:
        """
        Play one episode with human control.

        Args:
            practice: Whether this is a practice round (not scored)

        Returns:
            Dict with episode statistics
        """
        obs, info = self.env.reset()
        self.history = [(obs, None, 0.0, False, False, info)]

        total_reward = 0.0
        step_count = 0
        done = False
        start_time = time.time()

        self.running = True
        self.paused = False

        # Show tutorial if first time
        if self.show_tutorial and self.round_number == 0:
            self._show_tutorial()

        while self.running and not done:
            # Handle events
            action = self._handle_input()

            if action is None:
                # No action (pause, undo, quit)
                continue

            # Execute action
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated

            # Record history
            self.history.append((obs, action, reward, terminated, truncated, info))

            # Render
            self._render(
                obs=obs,
                reward=total_reward,
                step_count=step_count,
                practice=practice,
            )

            self.clock.tick(self.fps)

        end_time = time.time()
        duration = end_time - start_time

        # Show summary
        episode_stats = {
            "total_reward": total_reward,
            "step_count": step_count,
            "duration": duration,
            "success": info.get("success", False),
            "practice": practice,
            "round_number": self.round_number,
        }

        self._show_episode_summary(episode_stats)

        return episode_stats

    def play_session(self, n_rounds: int = 3) -> list[dict[str, Any]]:
        """
        Play full session with practice + scored rounds.

        Args:
            n_rounds: Number of scored rounds to play

        Returns:
            List of episode statistics for each round
        """
        results = []

        # Practice rounds
        for i in range(self.practice_rounds):
            self.round_number = -(i + 1)  # Negative for practice
            stats = self.play_episode(practice=True)
            results.append(stats)

            if not self.running:
                break

        # Scored rounds
        for i in range(n_rounds):
            if not self.running:
                break

            self.round_number = i + 1
            stats = self.play_episode(practice=False)
            results.append(stats)

        pygame.quit()

        return results

    def _handle_input(self) -> int | None:
        """
        Handle keyboard input.

        Returns:
            Action index or None if no action
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return None

            if event.type == pygame.KEYDOWN:
                # Movement keys
                if event.key == pygame.K_UP or event.key == pygame.K_w:
                    return 0  # MOVE_UP
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    return 1  # MOVE_DOWN
                elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    return 2  # MOVE_LEFT
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    return 3  # MOVE_RIGHT

                # Action keys
                elif event.key == pygame.K_SPACE:
                    return 4  # PICKUP/USE
                elif event.key == pygame.K_e:
                    return 5  # DROP

                # Control keys
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                    return None
                elif event.key == pygame.K_u and self.allow_undo:
                    self._undo_last_action()
                    return None
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
                    return None

        return None

    def _undo_last_action(self) -> None:
        """Undo last action (if allowed and possible)."""
        if len(self.history) > 1:
            self.history.pop()
            # Note: Can't actually undo in environment, just remove from history
            # Would need environment to support state restoration

    def _render(
        self,
        obs: Any,
        reward: float,
        step_count: int,
        practice: bool,
    ) -> None:
        """
        Render game state with HUD.

        Args:
            obs: Current observation
            reward: Total reward
            step_count: Number of steps
            practice: Whether in practice mode
        """
        # Clear screen
        self.screen.fill((0, 0, 0))

        # Render environment observation
        if hasattr(self.env, "render"):
            frame = self.env.render()
            if isinstance(frame, pygame.Surface):
                # Scale to fit window
                scaled_frame = pygame.transform.scale(frame, self.window_size)
                self.screen.blit(scaled_frame, (0, 0))

        # Render HUD
        self._render_hud(reward, step_count, practice)

        # Show pause overlay if paused
        if self.paused:
            self._render_pause_overlay()

        pygame.display.flip()

    def _render_hud(self, reward: float, step_count: int, practice: bool) -> None:
        """Render HUD overlay."""
        # Round indicator
        mode_text = "PRACTICE" if practice else f"Round {self.round_number}"
        mode_color = (255, 255, 0) if practice else (255, 255, 255)
        mode_surface = self.font.render(mode_text, True, mode_color)
        self.screen.blit(mode_surface, (10, 10))

        # Score
        score_text = f"Score: {reward:.1f}"
        score_surface = self.small_font.render(score_text, True, (255, 255, 255))
        self.screen.blit(score_surface, (10, 50))

        # Steps
        steps_text = f"Steps: {step_count}"
        steps_surface = self.small_font.render(steps_text, True, (255, 255, 255))
        self.screen.blit(steps_surface, (10, 75))

        # Controls hint
        controls_text = "Arrow/WASD: Move | Space: Use | P: Pause | ESC: Quit"
        controls_surface = self.small_font.render(controls_text, True, (150, 150, 150))
        self.screen.blit(
            controls_surface,
            (10, self.window_size[1] - 30),
        )

    def _render_pause_overlay(self) -> None:
        """Render pause overlay."""
        overlay = pygame.Surface(self.window_size, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        self.screen.blit(overlay, (0, 0))

        pause_text = "PAUSED"
        pause_surface = self.font.render(pause_text, True, (255, 255, 255))
        text_rect = pause_surface.get_rect(
            center=(self.window_size[0] // 2, self.window_size[1] // 2)
        )
        self.screen.blit(pause_surface, text_rect)

    def _show_tutorial(self) -> None:
        """Show tutorial overlay explaining controls and objectives."""
        showing_tutorial = True

        tutorial_text = [
            "Welcome to Agentick Human Evaluation!",
            "",
            "Controls:",
            "  Arrow Keys / WASD - Move",
            "  SPACE - Pickup/Use",
            "  P - Pause",
            "  ESC - Quit",
            "",
            "Objective: Complete the task as efficiently as possible!",
            "",
            "Press SPACE to start...",
        ]

        while showing_tutorial:
            self.screen.fill((0, 0, 50))

            y_offset = 100
            for line in tutorial_text:
                text_surface = self.small_font.render(line, True, (255, 255, 255))
                text_rect = text_surface.get_rect(center=(self.window_size[0] // 2, y_offset))
                self.screen.blit(text_surface, text_rect)
                y_offset += 40

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    showing_tutorial = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        showing_tutorial = False

    def _show_episode_summary(self, stats: dict[str, Any]) -> None:
        """Show end-of-episode summary."""
        showing_summary = True

        summary_text = [
            "Episode Complete!" if not stats["practice"] else "Practice Round Complete!",
            "",
            f"Score: {stats['total_reward']:.1f}",
            f"Steps: {stats['step_count']}",
            f"Time: {stats['duration']:.1f}s",
            f"Success: {'Yes' if stats.get('success', False) else 'No'}",
            "",
            "Press SPACE to continue or ESC to quit...",
        ]

        while showing_summary:
            self.screen.fill((0, 50, 0) if stats.get("success") else (50, 0, 0))

            y_offset = 100
            for line in summary_text:
                text_surface = self.small_font.render(line, True, (255, 255, 255))
                text_rect = text_surface.get_rect(center=(self.window_size[0] // 2, y_offset))
                self.screen.blit(text_surface, text_rect)
                y_offset += 40

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    showing_summary = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        showing_summary = False
                    elif event.key == pygame.K_ESCAPE:
                        self.running = False
                        showing_summary = False
