"""Episode replay utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def replay_episode(log_file: str | Path, speed: float = 1.0) -> None:
    """
    Replay episode in pygame window.

    Args:
        log_file: Path to episode log
        speed: Playback speed multiplier
    """
    try:
        import pygame
    except ImportError:
        raise ImportError(
            "pygame not installed. Install with: uv sync (pygame is a core dependency)\n"
            "Or use replay_episode_terminal() for ASCII replay."
        )

    from agentick.logging.episode_logger import EpisodeLogger

    # Load steps
    steps = EpisodeLogger.load(log_file)

    if not steps:
        print("No steps to replay")
        return

    # Initialize pygame
    pygame.init()

    # Get window size from first frame if available
    window_size = (800, 600)
    if "observation" in steps[0] and "rgb_array" in steps[0]["observation"]:
        import numpy as np

        frame = steps[0]["observation"]["rgb_array"]
        if isinstance(frame, (list, np.ndarray)):
            if hasattr(frame, "shape"):
                h, w = frame.shape[:2]
                window_size = (w, h)

    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Episode Replay")
    clock = pygame.time.Clock()

    # Fonts for overlay
    try:
        font = pygame.font.SysFont("monospace", 16)
    except Exception:
        font = pygame.font.Font(None, 16)

    # Playback state
    step_idx = 0
    paused = False
    running = True

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_LEFT and step_idx > 0:
                    step_idx -= 1
                elif event.key == pygame.K_RIGHT and step_idx < len(steps) - 1:
                    step_idx += 1
                elif event.key == pygame.K_q:
                    running = False

        # Update step
        if not paused:
            step_idx += 1
            if step_idx >= len(steps):
                step_idx = 0  # Loop

        step_data = steps[step_idx]

        # Clear screen
        screen.fill((0, 0, 0))

        # Render frame
        if "observation" in step_data:
            obs = step_data["observation"]

            # Try RGB array first
            if "rgb_array" in obs:
                _render_rgb_frame(screen, obs["rgb_array"])
            # Fallback to ASCII
            elif "ascii" in obs:
                _render_ascii_frame(screen, obs["ascii"], font)

        # Render HUD overlay
        _render_hud(
            screen,
            font,
            step=step_idx,
            total_steps=len(steps),
            action=step_data.get("action", {}).get("name", "N/A"),
            reward=step_data.get("reward", {}).get("total", 0),
            paused=paused,
        )

        pygame.display.flip()

        # Control playback speed
        clock.tick(10 * speed)

    pygame.quit()


def _render_rgb_frame(screen: Any, frame: Any) -> None:
    """Render RGB array frame."""
    import numpy as np
    import pygame

    if isinstance(frame, list):
        frame = np.array(frame)

    # Convert to pygame surface
    if len(frame.shape) == 3:  # H x W x C
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    else:
        # Grayscale
        surface = pygame.surfarray.make_surface(frame)

    # Scale to fit screen
    screen_size = screen.get_size()
    surface = pygame.transform.scale(surface, screen_size)

    screen.blit(surface, (0, 0))


def _render_ascii_frame(screen: Any, ascii_str: str, font: Any) -> None:
    """Render ASCII frame."""
    lines = ascii_str.split("\n")
    y = 10
    for line in lines:
        text_surface = font.render(line, True, (255, 255, 255))
        screen.blit(text_surface, (10, y))
        y += 20


def _render_hud(
    screen: Any,
    font: Any,
    step: int,
    total_steps: int,
    action: str,
    reward: float,
    paused: bool,
) -> None:
    """Render HUD overlay."""
    import pygame

    # Create semi-transparent overlay
    overlay = pygame.Surface((screen.get_width(), 80))
    overlay.set_alpha(200)
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, screen.get_height() - 80))

    # Render text
    y = screen.get_height() - 75
    texts = [
        f"Step: {step}/{total_steps}",
        f"Action: {action}",
        f"Reward: {reward:.2f}",
        "SPACE: pause | LEFT/RIGHT: step | Q: quit" + (" | PAUSED" if paused else ""),
    ]

    for text in texts:
        text_surface = font.render(text, True, (255, 255, 255))
        screen.blit(text_surface, (10, y))
        y += 18


def replay_episode_terminal(log_file: str | Path) -> None:
    """
    Replay episode in terminal (ASCII frames).

    Args:
        log_file: Path to episode log
    """
    import time

    from agentick.logging.episode_logger import EpisodeLogger

    steps = EpisodeLogger.load(log_file)

    for step_data in steps:
        print(f"\n=== Step {step_data['step']} ===")

        if "observation" in step_data and "ascii" in step_data["observation"]:
            print(step_data["observation"]["ascii"])

        print(f"Action: {step_data.get('action', {}).get('name', 'N/A')}")
        print(f"Reward: {step_data.get('reward', {}).get('total', 0)}")

        time.sleep(0.5)


def replay_step_by_step(log_file: str | Path) -> None:
    """
    Interactive step-by-step replay.

    Args:
        log_file: Path to episode log
    """
    from agentick.logging.episode_logger import EpisodeLogger

    steps = EpisodeLogger.load(log_file)

    for step_data in steps:
        print(f"\n=== Step {step_data['step']} ===")

        if "observation" in step_data:
            obs = step_data["observation"]
            print("\nObservation:")
            for mode, content in obs.items():
                print(f"  {mode}: {content[:100] if isinstance(content, str) else content}")

        print(f"\nAction: {step_data.get('action')}")
        print(f"Reward: {step_data.get('reward')}")
        print(f"Info: {step_data.get('info')}")

        input("\nPress Enter for next step...")


def export_replay_video(log_file: str | Path, output_path: str | Path) -> None:
    """
    Convert log to video.

    Args:
        log_file: Path to episode log
        output_path: Output video path
    """
    # Stub - would convert to video
    pass
