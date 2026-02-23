"""Generate isometric gallery of all tasks using oracle agents.

For each task: run oracle agent, capture key frames, compose into gallery.
Produces the default showcase image for the library.

Usage:
    uv run python examples/rendering/iso_gallery.py
    uv run python examples/rendering/iso_gallery.py --difficulty hard --output iso_gallery_hard.png
    uv run python examples/rendering/iso_gallery.py --save-individual --output-dir gallery_iso
"""

from __future__ import annotations

import argparse
import math
import os

import agentick
from agentick.oracles import get_oracle
from agentick.tasks.registry import list_tasks


def main():
    parser = argparse.ArgumentParser(description="Render all tasks as isometric gallery")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--difficulty", default="medium", help="Difficulty level")
    parser.add_argument("--output", default="iso_gallery.png", help="Output PNG path")
    parser.add_argument("--tile-size", type=int, default=256, help="Size of each task thumbnail")
    parser.add_argument("--save-individual", action="store_true", help="Also save individual PNGs")
    parser.add_argument("--output-dir", default="gallery_iso", help="Dir for individual PNGs")
    parser.add_argument("--mid-episode", action="store_true",
                        help="Use a mid-episode frame instead of initial state")
    args = parser.parse_args()

    from PIL import Image, ImageDraw, ImageFont

    tasks = list_tasks()
    n = len(tasks)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    thumb = args.tile_size
    gallery = Image.new("RGB", (cols * thumb, rows * thumb), (40, 44, 52))
    draw = ImageDraw.Draw(gallery)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 10
        )
    except (OSError, IOError):
        font = ImageFont.load_default()

    if args.save_individual:
        os.makedirs(args.output_dir, exist_ok=True)

    for i, task_name in enumerate(tasks):
        row, col = divmod(i, cols)
        try:
            env = agentick.make(
                task_name,
                render_mode="rgb_iso",
                difficulty=args.difficulty,
                seed=args.seed,
            )
            obs, info = env.reset(seed=args.seed)

            if args.mid_episode:
                # Run oracle to ~1/3 of episode for a mid-action frame
                oracle = get_oracle(task_name, env)
                oracle.reset(obs, info)
                target_step = max(1, info.get("max_steps", 30) // 3)
                for _ in range(target_step):
                    action = oracle.act(obs, info)
                    obs, _, term, trunc, info = env.step(action)
                    if term or trunc:
                        break

            frame = env.render()
            img = Image.fromarray(frame).resize((thumb, thumb), Image.LANCZOS)
            gallery.paste(img, (col * thumb, row * thumb))

            if args.save_individual:
                safe = task_name.replace("/", "_")
                Image.fromarray(frame).save(f"{args.output_dir}/{safe}.png")

            env.close()
            print(f"  [{i + 1}/{n}] {task_name}: OK")

        except Exception as e:
            draw.text(
                (col * thumb + 4, row * thumb + thumb // 2),
                f"{task_name}\nERROR: {e}",
                fill=(255, 80, 80),
                font=font,
            )
            print(f"  [{i + 1}/{n}] {task_name}: ERROR {e}")

        # Draw task name label
        draw.text(
            (col * thumb + 4, row * thumb + thumb - 14),
            task_name,
            fill=(200, 200, 200),
            font=font,
        )

    gallery.save(args.output)
    print(f"\nSaved gallery ({n} tasks, {cols}x{rows}) to {args.output}")


if __name__ == "__main__":
    main()
