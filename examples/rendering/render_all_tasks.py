#!/usr/bin/env python3
"""Generate a gallery of all registered tasks rendered in 3D.

Usage:
    python examples/rendering/render_all_tasks.py
    python examples/rendering/render_all_tasks.py --output-dir gallery/
"""

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Render all tasks in 3D gallery")
    parser.add_argument("--output-dir", default="gallery_3d", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--difficulty", default="easy")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()

    from PIL import Image

    import agentick

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = agentick.list_tasks()
    print(f"Rendering {len(tasks)} tasks in 3D...")

    rendered = []
    for task_name in tasks:
        try:
            env = agentick.make(
                task_name,
                difficulty=args.difficulty,
                render_mode="rgb_array",
                render_3d=True,
                seed=args.seed,
            )
            obs, _ = env.reset(seed=args.seed)
            env.close()

            safe_name = task_name.replace("-", "_").lower()
            path = output_dir / f"{safe_name}.png"
            Image.fromarray(obs).save(path)
            rendered.append((task_name, path))
            print(f"  {task_name}: saved to {path}")
        except Exception as e:
            print(f"  {task_name}: FAILED - {e}")

    # Build gallery mosaic
    if rendered:
        _build_mosaic(rendered, output_dir / "gallery.png", args.width, args.height)

    print(f"\nRendered {len(rendered)}/{len(tasks)} tasks")
    print(f"Gallery saved to: {output_dir / 'gallery.png'}")


def _build_mosaic(
    rendered: list[tuple[str, Path]], output_path: Path, cell_w: int, cell_h: int,
):
    """Arrange renders into a grid mosaic."""
    import math

    from PIL import Image, ImageDraw, ImageFont

    n = len(rendered)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    label_h = 24
    mosaic_w = cols * cell_w
    mosaic_h = rows * (cell_h + label_h)
    mosaic = Image.new("RGB", (mosaic_w, mosaic_h), (30, 30, 45))
    draw = ImageDraw.Draw(mosaic)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for idx, (name, path) in enumerate(rendered):
        row = idx // cols
        col = idx % cols
        x = col * cell_w
        y = row * (cell_h + label_h)

        img = Image.open(path).resize((cell_w, cell_h))
        mosaic.paste(img, (x, y))
        draw.text((x + 5, y + cell_h + 3), name, fill=(220, 220, 220), font=font)

    mosaic.save(output_path)


if __name__ == "__main__":
    main()
