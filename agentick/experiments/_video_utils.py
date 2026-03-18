"""Video encoding utilities for experiment runners."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


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

    # Ensure output has .mp4 extension
    if output_path.suffix != ".mp4":
        output_path = output_path.with_suffix(".mp4")

    # Save frames as temporary images
    temp_dir = output_path.parent / ".temp_frames"
    temp_dir.mkdir(exist_ok=True)

    try:
        for i, frame in enumerate(frames):
            img = Image.fromarray(frame)
            img.save(temp_dir / f"frame_{i:06d}.png")

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(temp_dir / "frame_%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "23",
                str(output_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        import shutil

        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def _save_gif(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    """Save frames as GIF."""
    if not frames:
        raise ValueError("No frames to save")

    if output_path.suffix != ".gif":
        output_path = output_path.with_suffix(".gif")

    pil_frames = [Image.fromarray(frame) for frame in frames]
    duration = int(1000 / fps)

    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )
