#!/usr/bin/env python
"""Record oracle trajectory GIFs for all tasks at all difficulties.

Iterates over all registered oracles, runs the oracle at each difficulty
level for multiple seeds, picks the best (successful) episode, and saves
an animated GIF to ``gallery/{difficulty}/{task_name}.gif``.

Usage:
    uv run python scripts/record_oracle_gallery.py
    uv run python scripts/record_oracle_gallery.py --output gallery --seeds 3 --fps 5
    uv run python scripts/record_oracle_gallery.py --difficulties easy hard
    uv run python scripts/record_oracle_gallery.py --tasks GoToGoal-v0 MazeNavigation-v0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import agentick
from agentick.oracles import get_oracle, list_oracles

ALL_DIFFICULTIES = ["easy", "medium", "hard", "expert"]


def save_gif(frames: list[np.ndarray], path: Path, fps: int = 5) -> None:
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
        pil_frames.append(Image.fromarray(frame))

    pil_frames[0].save(
        str(path),
        save_all=True,
        append_images=pil_frames[1:],
        duration=1000 // fps,
        loop=0,
    )


def record_task(
    task_name: str,
    difficulty: str,
    output_dir: Path,
    num_seeds: int = 3,
    fps: int = 5,
) -> bool:
    """Record oracle GIF for one task at one difficulty. Returns True on success."""
    best_frames: list[np.ndarray] | None = None
    best_reward = -float("inf")
    best_success = False

    for seed in range(num_seeds):
        try:
            env = agentick.make(
                task_name, difficulty=difficulty, render_mode="rgb_array",
            )
            oracle = get_oracle(task_name, env)

            obs, info = env.reset(seed=seed)
            oracle.reset(obs, info)

            frames = []
            frame = env.render()
            if isinstance(frame, np.ndarray):
                frames.append(frame)

            total_reward = 0.0
            done = False
            steps = 0
            while not done and steps < 500:
                action = oracle.act(obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                oracle.update(obs, info)

                frame = env.render()
                if isinstance(frame, np.ndarray):
                    frames.append(frame)

                total_reward += reward
                done = terminated or truncated
                steps += 1

            env.close()

            success = info.get("success", False)

            if frames and (
                (success and not best_success)
                or (success == best_success and total_reward > best_reward)
            ):
                best_frames = frames
                best_reward = total_reward
                best_success = success

        except Exception as e:
            print(f"    seed {seed} failed: {e}")
            continue

    if best_frames:
        gif_path = output_dir / f"{task_name}.gif"
        save_gif(best_frames, gif_path, fps=fps)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Record oracle GIF gallery")
    parser.add_argument("--output", default="gallery", help="Root output directory")
    parser.add_argument("--seeds", type=int, default=3, help="Seeds per task")
    parser.add_argument("--fps", type=int, default=5, help="GIF framerate")
    parser.add_argument("--tasks", nargs="*", help="Specific tasks (default: all)")
    parser.add_argument(
        "--difficulties", nargs="*", default=ALL_DIFFICULTIES,
        help="Difficulty levels (default: all four)",
    )
    args = parser.parse_args()

    root = Path(args.output)
    task_names = args.tasks if args.tasks else list_oracles()
    difficulties = args.difficulties

    total_jobs = len(task_names) * len(difficulties)
    print(f"Recording oracle gallery: {len(task_names)} tasks x "
          f"{len(difficulties)} difficulties = {total_jobs} GIFs")
    print(f"  Output: {root}/  |  Seeds: {args.seeds}  |  FPS: {args.fps}")
    print()

    start = time.time()
    successes = 0
    failures: list[str] = []
    job = 0

    for diff in difficulties:
        diff_dir = root / diff
        diff_dir.mkdir(parents=True, exist_ok=True)

        print(f"── {diff.upper()} ─{'─' * 50}")
        for i, task_name in enumerate(task_names):
            job += 1
            label = f"[{job}/{total_jobs}] {task_name} ({diff})"
            print(f"{label}...", end=" ", flush=True)
            try:
                ok = record_task(
                    task_name, diff, diff_dir,
                    num_seeds=args.seeds, fps=args.fps,
                )
                if ok:
                    gif_path = diff_dir / f"{task_name}.gif"
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
