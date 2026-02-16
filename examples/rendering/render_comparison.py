#!/usr/bin/env python3
"""Render the same task state in 2D and 3D side by side.

Usage:
    python examples/rendering/render_comparison.py
"""

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse


def main():
    parser = argparse.ArgumentParser(description="Compare 2D vs 3D rendering")
    parser.add_argument("--task", default="KeyDoorPuzzle-v0", help="Task name")
    parser.add_argument("--difficulty", default="easy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="render_comparison.png")
    args = parser.parse_args()

    import numpy as np
    from PIL import Image

    import agentick

    # 2D render
    env_2d = agentick.make(
        args.task, difficulty=args.difficulty, render_mode="rgb_array_2d", seed=args.seed,
    )
    obs_2d, _ = env_2d.reset(seed=args.seed)
    env_2d.close()

    # 3D render
    env_3d = agentick.make(
        args.task, difficulty=args.difficulty, render_mode="rgb_array", render_3d=True, seed=args.seed,
    )
    obs_3d, _ = env_3d.reset(seed=args.seed)
    env_3d.close()

    # Resize 2D to match 3D height for side-by-side
    img_2d = Image.fromarray(obs_2d)
    img_3d = Image.fromarray(obs_3d)

    target_h = img_3d.size[1]
    scale = target_h / img_2d.size[1]
    new_w = int(img_2d.size[0] * scale)
    img_2d_resized = img_2d.resize((new_w, target_h), Image.Resampling.NEAREST)

    # Combine side by side
    gap = 10
    combined_w = img_2d_resized.size[0] + gap + img_3d.size[0]
    combined = Image.new("RGB", (combined_w, target_h), (30, 30, 45))
    combined.paste(img_2d_resized, (0, 0))
    combined.paste(img_3d, (img_2d_resized.size[0] + gap, 0))

    combined.save(args.output)
    print(f"2D shape: {obs_2d.shape}, 3D shape: {obs_3d.shape}")
    print(f"Saved comparison to: {args.output}")


if __name__ == "__main__":
    main()
