"""Demo: Render a single task isometrically and save as PNG."""

from __future__ import annotations

import argparse

import agentick


def main():
    parser = argparse.ArgumentParser(description="Render a task with isometric view")
    parser.add_argument("--task", default="GoToGoal-v0", help="Task name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--difficulty", default="medium", help="Difficulty level")
    parser.add_argument("--output", default="iso_demo.png", help="Output PNG path")
    parser.add_argument("--steps", type=int, default=5, help="Steps to take before rendering")
    args = parser.parse_args()

    env = agentick.make(
        args.task,
        render_mode="rgb_iso",
        difficulty=args.difficulty,
        seed=args.seed,
    )
    obs, info = env.reset(seed=args.seed)

    # Take a few random steps
    for _ in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    # Save the final frame
    from PIL import Image

    img = Image.fromarray(obs)
    img.save(args.output)
    print(f"Saved isometric render to {args.output}")
    print(f"  Task: {args.task}, Seed: {args.seed}, Shape: {obs.shape}")
    env.close()


if __name__ == "__main__":
    main()
