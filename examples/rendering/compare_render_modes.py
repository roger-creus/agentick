"""Compare rgb_array vs rgb_iso side by side for the same task state."""

from __future__ import annotations

import argparse

import agentick


def main():
    parser = argparse.ArgumentParser(description="Compare render modes side by side")
    parser.add_argument("--task", default="GoToGoal-v0", help="Task name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--difficulty", default="medium", help="Difficulty level")
    parser.add_argument("--output", default="compare_modes.png", help="Output PNG path")
    args = parser.parse_args()

    from PIL import Image, ImageDraw, ImageFont

    # Create environment in rgb_array mode (flat 2D)
    env_flat = agentick.make(
        args.task,
        render_mode="rgb_array",
        difficulty=args.difficulty,
        seed=args.seed,
    )
    obs_flat, _ = env_flat.reset(seed=args.seed)
    img_flat = Image.fromarray(obs_flat)

    # Create environment in rgb_iso mode (isometric)
    env_iso = agentick.make(
        args.task,
        render_mode="rgb_iso",
        difficulty=args.difficulty,
        seed=args.seed,
    )
    obs_iso, _ = env_iso.reset(seed=args.seed)
    img_iso = Image.fromarray(obs_iso)

    # Resize both to same height for comparison
    target_h = 512
    img_flat = img_flat.resize(
        (int(img_flat.width * target_h / img_flat.height), target_h),
        Image.LANCZOS,
    )
    img_iso = img_iso.resize((target_h, target_h), Image.LANCZOS)

    # Create side-by-side comparison
    gap = 20
    label_h = 30
    total_w = img_flat.width + gap + img_iso.width
    total_h = target_h + label_h

    canvas = Image.new("RGB", (total_w, total_h), (40, 44, 52))
    canvas.paste(img_flat, (0, label_h))
    canvas.paste(img_iso, (img_flat.width + gap, label_h))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 16
        )
    except (OSError, IOError):
        font = ImageFont.load_default()

    draw.text((10, 5), "rgb_array (flat 2D)", fill=(200, 200, 200), font=font)
    draw.text(
        (img_flat.width + gap + 10, 5),
        "rgb_iso (isometric)",
        fill=(200, 200, 200),
        font=font,
    )

    canvas.save(args.output)
    print(f"Saved comparison to {args.output}")
    print(f"  Task: {args.task}, Seed: {args.seed}")
    print(f"  Flat: {obs_flat.shape}, Iso: {obs_iso.shape}")

    env_flat.close()
    env_iso.close()


if __name__ == "__main__":
    main()
