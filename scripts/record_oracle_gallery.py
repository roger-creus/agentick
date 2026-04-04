#!/usr/bin/env python
"""Record oracle trajectory GIFs for all tasks at all difficulties.

Records a single oracle episode per (task, difficulty) as a looping GIF.

Output naming convention: ``{output_dir}/{task_name}_{difficulty}.gif``

Default output: docs/showcase/videos/iso/

Usage:
    uv run python scripts/record_oracle_gallery.py
    uv run python scripts/record_oracle_gallery.py --difficulties easy medium
    uv run python scripts/record_oracle_gallery.py --tasks GoToGoal-v0 MazeNavigation-v0
    uv run python scripts/record_oracle_gallery.py --n-episodes 3
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


def record_task(
    task_name: str,
    difficulty: str,
    output_dir: Path,
    render_mode: str = "rgb_array",
    n_episodes: int = 1,
    fps: int = 5,
    size: int | None = None,
    colors: int = 64,
) -> bool:
    """Record oracle GIF from eval-seed episodes. Returns True on success.

    Tries ``n_episodes`` eval seeds and picks the first successful one.
    If no episode succeeds, falls back to the longest recorded episode.
    """
    seeds = generate_task_seeds(task_name, difficulty, "eval", max(n_episodes, 5))

    best_frames: list[np.ndarray] = []

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

            done = False
            steps = 0
            success = False
            while not done and steps < 500:
                action = oracle.act(obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                oracle.update(obs, info)

                frame = env.render()
                if isinstance(frame, np.ndarray):
                    frames.append(frame)

                done = terminated or truncated
                success = info.get("success", False)
                steps += 1

            env.close()

            # Prefer successful episodes
            if success and frames:
                best_frames = frames
                break

            # Keep longest as fallback
            if len(frames) > len(best_frames):
                best_frames = frames

        except Exception as e:
            print(f"    seed {seed} failed: {e}")
            continue

    if best_frames:
        gif_path = output_dir / f"{task_name}_{difficulty}.gif"
        save_gif(best_frames, gif_path, fps=fps, size=size, colors=colors)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Record oracle GIF gallery")
    parser.add_argument(
        "--render-mode",
        default="rgb_array",
        choices=["rgb_array"],
        help="Render mode: rgb_array (isometric 512x512). Default: rgb_array.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: docs/showcase/videos/iso)",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=1,
        help="Number of eval seeds to try per GIF; picks first success (default: 1)",
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
    print(f"  Output: {root.resolve()}/  |  Episodes: {args.n_episodes}  |  FPS: {args.fps}")
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
                    n_episodes=args.n_episodes,
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
