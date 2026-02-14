"""Video recording utilities for Agentick environments."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Try importing video backends
try:
    import imageio

    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    warnings.warn("imageio not available. Install with: uv sync --extra rl")

try:
    from gymnasium.wrappers import RecordVideo as GymRecordVideo

    GYM_RECORD_VIDEO_AVAILABLE = True
except ImportError:
    GYM_RECORD_VIDEO_AVAILABLE = False


def record_episode(
    env: Any,
    agent: Callable | None = None,
    output_path: str | Path = "episode.mp4",
    max_steps: int | None = None,
    fps: int = 10,
    fallback_format: str = "gif",
) -> Path:
    """
    Record a single episode to video.

    Args:
        env: Environment instance (must support rgb_array render mode)
        agent: Agent function that takes (obs, info) and returns action.
               If None, uses random actions.
        output_path: Path to save video (MP4 or GIF)
        max_steps: Maximum steps to record (uses env default if None)
        fps: Frames per second
        fallback_format: Format to use if MP4 fails ("gif" or "png")

    Returns:
        Path to saved video file

    Example:
        >>> env = agentick.make("GoToGoal-v0", render_mode="rgb_array")
        >>> def random_agent(obs, info):
        ...     return env.action_space.sample()
        >>> video_path = record_episode(env, random_agent, "episode.mp4")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect frames
    frames = []
    obs, info = env.reset()

    # Initial frame
    if hasattr(env, "render"):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

    # Run episode
    done = False
    steps = 0
    max_steps = max_steps or getattr(env, "max_steps", 1000)

    while not done and steps < max_steps:
        # Get action
        if agent is not None:
            action = agent(obs, info)
        else:
            action = env.action_space.sample()

        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

        # Capture frame
        if hasattr(env, "render"):
            frame = env.render()
            if frame is not None:
                frames.append(frame)

    if not frames:
        raise ValueError("No frames captured. Ensure env has render_mode='rgb_array'")

    # Save video
    try:
        if output_path.suffix.lower() == ".mp4":
            _save_mp4(frames, output_path, fps)
        elif output_path.suffix.lower() == ".gif":
            _save_gif(frames, output_path, fps)
        else:
            # Default to MP4
            _save_mp4(frames, output_path, fps)
    except Exception as e:
        warnings.warn(f"Failed to save as {output_path.suffix}: {e}. Trying fallback format.")

        # Fallback
        if fallback_format == "gif":
            fallback_path = output_path.with_suffix(".gif")
            _save_gif(frames, fallback_path, fps)
            output_path = fallback_path
        else:
            # Save as PNG sequence
            fallback_dir = output_path.parent / output_path.stem
            output_path = _save_png_sequence(frames, fallback_dir)

    return output_path


def _save_mp4(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    """Save frames as MP4 video."""
    if not IMAGEIO_AVAILABLE:
        raise ImportError("imageio required for MP4. Install with: uv sync --extra rl")

    # Ensure frames are uint8
    frames_uint8 = []
    for frame in frames:
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        frames_uint8.append(frame)

    # Write video
    imageio.mimsave(str(output_path), frames_uint8, fps=fps, codec="libx264", quality=8)


def _save_gif(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    """Save frames as GIF."""
    # Convert frames to PIL Images
    pil_frames = []
    for frame in frames:
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

        pil_frames.append(Image.fromarray(frame))

    # Save as GIF
    duration = int(1000 / fps)  # milliseconds per frame
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
        optimize=False,
    )


def _save_png_sequence(frames: list[np.ndarray], output_dir: Path) -> Path:
    """Save frames as PNG sequence."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

        img = Image.fromarray(frame)
        img.save(output_dir / f"frame_{i:04d}.png")

    # Return path to directory
    return output_dir


def record_episodes_to_video(
    env: Any,
    agent: Callable,
    num_episodes: int = 1,
    output_dir: str | Path = "videos",
    fps: int = 10,
) -> list[Path]:
    """
    Record multiple episodes to separate video files.

    Args:
        env: Environment instance
        agent: Agent function
        num_episodes: Number of episodes to record
        output_dir: Directory to save videos
        fps: Frames per second

    Returns:
        List of paths to saved video files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_paths = []
    for episode in range(num_episodes):
        output_path = output_dir / f"episode_{episode:03d}.mp4"
        try:
            path = record_episode(env, agent, output_path, fps=fps)
            video_paths.append(path)
        except Exception as e:
            warnings.warn(f"Failed to record episode {episode}: {e}")

    return video_paths


def wrap_env_with_video_recording(
    env: Any,
    output_dir: str | Path = "videos",
    episode_trigger: Callable[[int], bool] | None = None,
    fps: int = 30,
) -> Any:
    """
    Wrap environment with automatic video recording using gymnasium's RecordVideo.

    Args:
        env: Environment to wrap
        output_dir: Directory to save videos
        episode_trigger: Function that takes episode number and returns whether to record.
                        If None, records all episodes.
        fps: Frames per second

    Returns:
        Wrapped environment

    Example:
        >>> env = agentick.make("GoToGoal-v0", render_mode="rgb_array")
        >>> # Record every 10th episode
        >>> env = wrap_env_with_video_recording(env, episode_trigger=lambda ep: ep % 10 == 0)
    """
    if not GYM_RECORD_VIDEO_AVAILABLE:
        warnings.warn("gymnasium.wrappers.RecordVideo not available. Skipping video recording.")
        return env

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if episode_trigger is None:
        # Record all episodes
        def episode_trigger(_):
            return True

    wrapped_env = GymRecordVideo(
        env,
        video_folder=str(output_dir),
        episode_trigger=episode_trigger,
        fps=fps,
    )

    return wrapped_env
