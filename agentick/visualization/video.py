"""Video generation from episode trajectories."""

from __future__ import annotations

import gzip
import json
import subprocess
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render_episode_video(
    trajectory: dict[str, Any] | str | Path,
    task: str,
    output_path: str | Path,
    fps: int = 10,
    resolution: tuple[int, int] = (640, 480),
    overlay: bool = True,
    codec: str = "auto",
) -> None:
    """
    Render episode video with overlays.

    Args:
        trajectory: Episode trajectory data or path to trajectory file
        task: Task name
        output_path: Output video path (.mp4 or .gif)
        fps: Frames per second
        resolution: Video resolution
        overlay: Whether to add HUD overlay
        codec: Video codec ("h264", "gif", or "auto" to try h264 then gif)
    """
    output_path = Path(output_path)

    # Load trajectory if path
    if isinstance(trajectory, (str, Path)):
        trajectory = _load_trajectory(trajectory)

    # Create environment to render frames
    env = gym.make(task, render_mode="rgb_array")

    # Collect frames
    frames = []
    obs = env.reset(seed=trajectory.get("seed", 0))[0]

    for step_data in trajectory["steps"]:
        # Render frame
        frame = env.render()
        frame = _resize_frame(frame, resolution)

        # Add overlay
        if overlay:
            frame = _add_overlay(
                frame,
                step=step_data.get("step", 0),
                action=step_data.get("action", 0),
                reward=step_data.get("reward", 0.0),
                info=step_data.get("info", {}),
            )

        frames.append(frame)

        # Step environment
        action = step_data["action"]
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    env.close()

    # Determine codec
    if codec == "auto":
        # Try H.264 MP4 first, fallback to GIF
        if output_path.suffix == ".gif" or not _has_ffmpeg():
            codec = "gif"
        else:
            codec = "h264"
    elif codec == "h264" and not _has_ffmpeg():
        raise RuntimeError("ffmpeg not found. Install ffmpeg or use codec='gif'")

    # Save video
    if codec == "h264":
        _save_mp4(frames, output_path, fps)
    elif codec == "gif":
        _save_gif(frames, output_path, fps)
    else:
        raise ValueError(f"Unknown codec: {codec}")


def render_comparison_video(
    trajectories_dict: dict[str, dict[str, Any]],
    task: str,
    seed: int,
    output_path: str | Path,
    fps: int = 10,
    resolution: tuple[int, int] = (640, 480),
    codec: str = "auto",
) -> None:
    """
    Side-by-side comparison video.

    Args:
        trajectories_dict: Agent name -> trajectory
        task: Task name
        seed: Seed number
        output_path: Output path
        fps: Frames per second
        resolution: Resolution per agent
        codec: Video codec
    """
    output_path = Path(output_path)

    # Render each agent's video separately
    agent_frames = {}
    for agent_name, trajectory in trajectories_dict.items():
        env = gym.make(task, render_mode="rgb_array")
        env.reset(seed=seed)

        frames = []
        for step_data in trajectory["steps"]:
            frame = env.render()
            frame = _resize_frame(frame, resolution)
            frame = _add_overlay(frame, step=step_data.get("step", 0), info={"agent": agent_name})
            frames.append(frame)
            env.step(step_data["action"])

        env.close()
        agent_frames[agent_name] = frames

    # Combine frames side-by-side
    max_steps = max(len(frames) for frames in agent_frames.values())
    combined_frames = []

    for step in range(max_steps):
        row_frames = []
        for agent_name in trajectories_dict.keys():
            if step < len(agent_frames[agent_name]):
                row_frames.append(agent_frames[agent_name][step])
            else:
                # Use last frame if this agent finished early
                row_frames.append(agent_frames[agent_name][-1])

        # Concatenate horizontally
        combined = np.concatenate(row_frames, axis=1)
        combined_frames.append(combined)

    # Save
    if codec == "auto":
        codec = "gif" if output_path.suffix == ".gif" or not _has_ffmpeg() else "h264"

    if codec == "h264":
        _save_mp4(combined_frames, output_path, fps)
    else:
        _save_gif(combined_frames, output_path, fps)


def render_capability_montage(
    results_dict: dict[str, Any],
    agent: str,
    output_path: str | Path,
    fps: int = 10,
    resolution: tuple[int, int] = (320, 240),
    codec: str = "auto",
) -> None:
    """
    Grid montage of agent across different tasks.

    Args:
        results_dict: Results dictionary
        agent: Agent name
        output_path: Output path
        fps: Frames per second
        resolution: Resolution per task
        codec: Video codec
    """
    # Similar implementation - render multiple tasks in grid layout
    pass


def render_learning_progression(
    checkpoints: list[dict[str, Any]],
    task: str,
    seed: int,
    output_path: str | Path,
    fps: int = 10,
    resolution: tuple[int, int] = (640, 480),
    codec: str = "auto",
) -> None:
    """
    Show behavior evolution during training.

    Args:
        checkpoints: List of checkpoint trajectories
        task: Task name
        seed: Seed
        output_path: Output path
        fps: Frames per second
        resolution: Video resolution
        codec: Video codec
    """
    # Similar to comparison video but show checkpoints over time
    pass


# Helper functions


def _load_trajectory(path: str | Path) -> dict[str, Any]:
    """Load trajectory from file."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            lines = f.readlines()
    else:
        with open(path) as f:
            lines = f.readlines()

    steps = [json.loads(line) for line in lines]
    return {"steps": steps}


def _resize_frame(frame: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    """Resize frame to target resolution."""
    img = Image.fromarray(frame)
    img = img.resize(resolution, Image.Resampling.LANCZOS)
    return np.array(img)


def _add_overlay(
    frame: np.ndarray,
    step: int = 0,
    action: int | None = None,
    reward: float = 0.0,
    info: dict[str, Any] | None = None,
) -> np.ndarray:
    """Add HUD overlay to frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Try to load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    # Draw overlay
    text_lines = [f"Step: {step}"]
    if action is not None:
        text_lines.append(f"Action: {action}")
    if reward != 0.0:
        text_lines.append(f"Reward: {reward:.2f}")
    if info and "agent" in info:
        text_lines.append(f"Agent: {info['agent']}")

    y = 10
    for line in text_lines:
        # Draw shadow
        draw.text((11, y + 1), line, font=font, fill=(0, 0, 0))
        # Draw text
        draw.text((10, y), line, font=font, fill=(255, 255, 255))
        y += 20

    return np.array(img)


def _has_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _save_mp4(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    """Save frames as H.264 MP4 video."""
    if not frames:
        raise ValueError("No frames to save")

    height, width = frames[0].shape[:2]

    # Ensure output has .mp4 extension
    if output_path.suffix != ".mp4":
        output_path = output_path.with_suffix(".mp4")

    # Save frames as temporary images
    temp_dir = output_path.parent / ".temp_frames"
    temp_dir.mkdir(exist_ok=True)

    try:
        # Save frames
        for i, frame in enumerate(frames):
            img = Image.fromarray(frame)
            img.save(temp_dir / f"frame_{i:06d}.png")

        # Use ffmpeg to create video
        subprocess.run(
            [
                "ffmpeg",
                "-y",  # Overwrite output
                "-framerate",
                str(fps),
                "-i",
                str(temp_dir / "frame_%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "23",  # Quality (lower = better, 18-28 recommended)
                str(output_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        # Clean up temp files
        import shutil

        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def _save_gif(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    """Save frames as GIF."""
    if not frames:
        raise ValueError("No frames to save")

    # Ensure output has .gif extension
    if output_path.suffix != ".gif":
        output_path = output_path.with_suffix(".gif")

    # Convert frames to PIL Images
    pil_frames = [Image.fromarray(frame) for frame in frames]

    # Calculate duration per frame (in milliseconds)
    duration = int(1000 / fps)

    # Save as GIF
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,  # Loop forever
        optimize=True,
    )
