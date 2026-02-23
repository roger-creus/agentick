"""Generate isometric video/GIF of oracle agent solving each task.

Produces GIFs for all tasks across all difficulties, saved to showcase/videos/iso/.
These become the default videos shown in the showcase webapp.

Usage:
    # All tasks, all difficulties:
    uv run python examples/rendering/iso_video.py

    # Single task:
    uv run python examples/rendering/iso_video.py --task GoToGoal-v0

    # Specific difficulty:
    uv run python examples/rendering/iso_video.py --task GoToGoal-v0 --difficulty hard

    # All tasks, single difficulty:
    uv run python examples/rendering/iso_video.py --difficulty medium
"""

from __future__ import annotations

import argparse
import os
import traceback

import agentick
from agentick.oracles import get_oracle
from agentick.tasks.registry import list_tasks


DIFFICULTIES = ["easy", "medium", "hard", "expert"]


def record_oracle_gif(
    task_name: str,
    difficulty: str,
    seed: int,
    output_path: str,
    duration_ms: int = 150,
) -> int:
    """Record oracle solving a task and save as GIF.

    Returns:
        Number of frames captured.
    """
    from PIL import Image

    env = agentick.make(
        task_name,
        render_mode="rgb_iso",
        difficulty=difficulty,
        seed=seed,
    )
    obs, info = env.reset(seed=seed)
    oracle = get_oracle(task_name, env)
    oracle.reset(obs, info)

    frames_pil = [Image.fromarray(env.render())]

    done = False
    max_steps = info.get("max_steps", 200)
    for _ in range(max_steps):
        action = oracle.act(obs, info)
        obs, reward, term, trunc, info = env.step(action)
        frames_pil.append(Image.fromarray(env.render()))
        if term or trunc:
            break

    env.close()

    # Save as GIF
    if len(frames_pil) > 1:
        frames_pil[0].save(
            output_path,
            save_all=True,
            append_images=frames_pil[1:],
            duration=duration_ms,
            loop=0,
        )
    else:
        frames_pil[0].save(output_path)

    return len(frames_pil)


def main():
    parser = argparse.ArgumentParser(
        description="Generate isometric GIFs of oracle agents solving tasks"
    )
    parser.add_argument("--task", default=None, help="Single task name (default: all tasks)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--difficulty", default=None,
                        help="Single difficulty (default: all difficulties)")
    parser.add_argument("--output-dir", default="showcase/videos/iso",
                        help="Output directory for GIFs")
    parser.add_argument("--duration", type=int, default=150, help="Frame duration in ms")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tasks = [args.task] if args.task else list_tasks()
    difficulties = [args.difficulty] if args.difficulty else DIFFICULTIES

    total = len(tasks) * len(difficulties)
    done = 0
    errors = []

    for task_name in tasks:
        for diff in difficulties:
            done += 1
            safe_name = task_name.replace("/", "_")
            output_path = os.path.join(args.output_dir, f"{safe_name}_{diff}.gif")

            try:
                n_frames = record_oracle_gif(
                    task_name, diff, args.seed, output_path, args.duration
                )
                print(f"  [{done}/{total}] {task_name} ({diff}): "
                      f"{n_frames} frames -> {output_path}")
            except Exception as e:
                errors.append((task_name, diff, str(e)))
                print(f"  [{done}/{total}] {task_name} ({diff}): ERROR {e}")
                traceback.print_exc()

    print(f"\nDone: {total - len(errors)}/{total} GIFs generated in {args.output_dir}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for name, diff, err in errors:
            print(f"  {name} ({diff}): {err}")


if __name__ == "__main__":
    main()
