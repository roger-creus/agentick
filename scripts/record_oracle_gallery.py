#!/usr/bin/env python
"""Record oracle trajectory GIFs for all tasks at all difficulties.

Concatenates episodes from multiple eval seeds into a single GIF per
(task, difficulty), with black separator frames between episodes.

Output naming convention: ``{output_dir}/{task_name}_{difficulty}.gif``

Default output locations:
  Isometric (rgb_array):    docs/showcase/videos/iso/
  Flat 2D   (rgb_array_flat): docs/showcase/videos/flat/

Usage:
    uv run python scripts/record_oracle_gallery.py
    uv run python scripts/record_oracle_gallery.py --render-mode rgb_array_flat \\
        --output docs/showcase/videos/flat
    uv run python scripts/record_oracle_gallery.py --difficulties easy medium
    uv run python scripts/record_oracle_gallery.py --tasks GoToGoal-v0 MazeNavigation-v0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import agentick
from agentick.leaderboard.seeds import generate_task_seeds
from agentick.oracles import get_oracle, list_oracles

ALL_DIFFICULTIES = ["easy", "medium", "hard", "expert"]

_DEFAULT_OUTPUT = {
    "rgb_array": "docs/showcase/videos/iso",
    "rgb_array_flat": "docs/showcase/videos/flat",
}


def save_gif(
    frames: list[np.ndarray],
    path: Path,
    fps: int = 5,
    size: int | None = None,
    colors: int = 64,
) -> None:
    """Save a list of numpy frames as an animated GIF."""
    from PIL import Image

    pil_frames = []
    for frame in frames:
        if frame.dtype != np.uint8:
            frame = (
                (frame * 255).astype(np.uint8)
                if frame.max() <= 1.0
                else frame.astype(np.uint8)
            )
        img = Image.fromarray(frame)
        if size is not None and (img.width != size or img.height != size):
            img = img.resize((size, size), Image.LANCZOS)
        if colors < 256:
            img = img.quantize(
                colors=colors, method=Image.Quantize.MEDIANCUT,
            ).convert("RGB")
        pil_frames.append(img)

    pil_frames[0].save(
        str(path),
        save_all=True,
        append_images=pil_frames[1:],
        duration=1000 // fps,
        loop=0,
        optimize=True,
    )


def _make_separator_frames(
    height: int, width: int, n_frames: int = 3
) -> list[np.ndarray]:
    """Create black separator frames between episodes."""
    return [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(n_frames)]


def record_task(
    task_name: str,
    difficulty: str,
    output_dir: Path,
    render_mode: str = "rgb_array",
    n_concat: int = 5,
    fps: int = 5,
    size: int | None = None,
    colors: int = 64,
) -> bool:
    """Record oracle GIF with concatenated eval-seed episodes. Returns True on success."""
    # Use the first n_concat eval seeds
    seeds = generate_task_seeds(task_name, difficulty, "eval", n_concat)

    all_frames: list[np.ndarray] = []
    frame_shape = None

    for seed in seeds:
        try:
            env = agentick.make(
                task_name, difficulty=difficulty, render_mode=render_mode,
            )
            oracle = get_oracle(task_name, env)

            obs, info = env.reset(seed=seed)
            oracle.reset(obs, info)

            frames = []
            frame = env.render()
            if isinstance(frame, np.ndarray):
                frames.append(frame)
                if frame_shape is None:
                    frame_shape = frame.shape

            done = False
            steps = 0
            while not done and steps < 500:
                action = oracle.act(obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                oracle.update(obs, info)

                frame = env.render()
                if isinstance(frame, np.ndarray):
                    frames.append(frame)

                done = terminated or truncated
                steps += 1

            env.close()

            # Add separator between episodes (not after last)
            if all_frames and frames and frame_shape is not None:
                all_frames.extend(_make_separator_frames(frame_shape[0], frame_shape[1]))

            all_frames.extend(frames)

        except Exception as e:
            print(f"    seed {seed} failed: {e}")
            continue

    if all_frames:
        gif_path = output_dir / f"{task_name}_{difficulty}.gif"
        save_gif(all_frames, gif_path, fps=fps, size=size, colors=colors)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Record oracle GIF gallery")
    parser.add_argument(
        "--render-mode",
        default="rgb_array",
        choices=["rgb_array", "rgb_array_flat"],
        help="Render mode: rgb_array (isometric 512x512) or rgb_array_flat (2D grid). "
             "Default: rgb_array (isometric).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: docs/showcase/videos/iso for rgb_array, "
             "docs/showcase/videos/flat for rgb_array_flat)",
    )
    parser.add_argument(
        "--n-concat", type=int, default=5,
        help="Number of eval seeds to concatenate per GIF (default: 5)",
    )
    parser.add_argument("--fps", type=int, default=5, help="GIF framerate")
    parser.add_argument(
        "--size", type=int, default=256,
        help="Output GIF size in pixels (default: 256). Use 0 for native resolution.",
    )
    parser.add_argument(
        "--colors", type=int, default=256,
        help="GIF palette colors, 2-256 (default: 256). Lower = smaller file.",
    )
    parser.add_argument("--tasks", nargs="*", help="Specific tasks (default: all)")
    parser.add_argument(
        "--difficulties", nargs="*", default=ALL_DIFFICULTIES,
        help="Difficulty levels (default: all four)",
    )
    args = parser.parse_args()

    render_mode = args.render_mode
    output_path = args.output or _DEFAULT_OUTPUT[render_mode]
    root = Path(output_path)
    root.mkdir(parents=True, exist_ok=True)
    gif_size = args.size if args.size > 0 else None
    gif_colors = min(max(args.colors, 2), 256)

    task_names = args.tasks if args.tasks else list_oracles()
    difficulties = args.difficulties

    total_jobs = len(task_names) * len(difficulties)
    print(f"Recording oracle gallery: {len(task_names)} tasks x "
          f"{len(difficulties)} difficulties = {total_jobs} GIFs")
    print(f"  Render mode: {render_mode}  |  GIF size: {gif_size or 'native'}px"
          f"  |  Colors: {gif_colors}")
    print(f"  Output: {root.resolve()}/  |  Seeds: {args.n_concat} concat  |  FPS: {args.fps}")
    print()

    start = time.time()
    successes = 0
    failures: list[str] = []
    job = 0

    for diff in difficulties:
        print(f"── {diff.upper()} ─{'─' * 50}")
        for task_name in task_names:
            job += 1
            label = f"[{job}/{total_jobs}] {task_name} ({diff})"
            print(f"{label}...", end=" ", flush=True)
            try:
                ok = record_task(
                    task_name, diff, root,
                    render_mode=render_mode,
                    n_concat=args.n_concat,
                    fps=args.fps,
                    size=gif_size,
                    colors=gif_colors,
                )
                if ok:
                    gif_path = root / f"{task_name}_{diff}.gif"
                    size_kb = gif_path.stat().st_size / 1024
                    print(f"done ({size_kb:.0f} KB)")
                    successes += 1
                else:
                    print("FAILED (no frames)")
                    failures.append(f"{task_name}@{diff}")
            except Exception as e:
                print(f"FAILED ({e})")
                failures.append(f"{task_name}@{diff}")
        print()

    elapsed = time.time() - start
    print("=" * 60)
    print(f"Gallery complete: {successes}/{total_jobs} GIFs in {elapsed:.1f}s")
    print(f"Output: {root.resolve()}/")
    if failures:
        print(f"Failed ({len(failures)}): {', '.join(failures)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
