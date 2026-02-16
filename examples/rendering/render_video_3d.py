#!/usr/bin/env python3
"""Record a 3D isometric video of a random agent playing a task.

Usage:
    python examples/rendering/render_video_3d.py
    python examples/rendering/render_video_3d.py --task MazeNavigation-v0 --steps 50
"""

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse


def main():
    parser = argparse.ArgumentParser(description="Record 3D video of an episode")
    parser.add_argument("--task", default="GoToGoal-v0", help="Task name")
    parser.add_argument("--difficulty", default="easy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30, help="Max steps to record")
    parser.add_argument("--output", default="episode_3d.mp4")
    parser.add_argument("--fps", type=int, default=4)
    args = parser.parse_args()

    import numpy as np

    import agentick

    env = agentick.make(
        args.task,
        difficulty=args.difficulty,
        render_mode="rgb_array",
        render_3d=True,
        seed=args.seed,
    )

    obs, info = env.reset(seed=args.seed)
    frames = [obs]

    print(f"Recording {args.task} for up to {args.steps} steps...")
    for step in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(obs)
        if terminated or truncated:
            print(f"  Episode ended at step {step + 1} (terminated={terminated})")
            break

    env.close()

    # Save video
    try:
        import imageio

        writer = imageio.get_writer(args.output, fps=args.fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"Saved {len(frames)} frames to: {args.output}")
    except ImportError:
        print("imageio not available. Saving frames as PNGs instead.")
        from pathlib import Path

        from PIL import Image

        frames_dir = Path("frames_3d")
        frames_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(frames_dir / f"frame_{i:04d}.png")
        print(f"Saved {len(frames)} frames to: {frames_dir}/")


if __name__ == "__main__":
    main()
