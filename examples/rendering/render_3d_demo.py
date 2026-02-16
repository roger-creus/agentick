#!/usr/bin/env python3
"""Render a single Agentick task in 3D isometric view and save as PNG.

Usage:
    python examples/rendering/render_3d_demo.py
    python examples/rendering/render_3d_demo.py --task MazeNavigation-v0 --seed 123
"""

import argparse
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def main():
    parser = argparse.ArgumentParser(description="Render a task in 3D isometric view")
    parser.add_argument("--task", default="GoToGoal-v0", help="Task name")
    parser.add_argument("--difficulty", default="easy", help="Difficulty level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="render_3d_demo.png", help="Output PNG path")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    args = parser.parse_args()

    import agentick

    print(f"Creating environment: {args.task} (difficulty={args.difficulty}, seed={args.seed})")
    env = agentick.make(
        args.task,
        difficulty=args.difficulty,
        render_mode="rgb_array",
        render_3d=True,
        seed=args.seed,
    )

    obs, info = env.reset(seed=args.seed)
    print(f"Observation shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"Grid size: {env.grid.height}x{env.grid.width}")

    from PIL import Image

    img = Image.fromarray(obs)
    img.save(args.output)
    print(f"Saved 3D render to: {args.output}")

    env.close()


if __name__ == "__main__":
    main()
