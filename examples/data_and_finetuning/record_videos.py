"""Record oracle agent video/GIF episodes.

Records oracle agents solving tasks as GIF animations across multiple tasks
and difficulties.

Requires: Pillow (uv sync --extra viz)

Usage:
    # Default (GoToGoal-v0, 3 episodes)
    uv run python examples/data_and_finetuning/record_videos.py

    # Multi-task recording
    uv run python examples/data_and_finetuning/record_videos.py \
        --tasks GoToGoal-v0 MazeNavigation-v0 --difficulties easy medium

    # Custom FPS and output
    uv run python examples/data_and_finetuning/record_videos.py \
        --fps 10 --n-episodes 5 --output-dir my_videos
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agentick
from agentick.oracles import get_oracle


def save_gif(frames: list, path: Path, fps: int = 5) -> None:
    """Save a list of numpy frames as an animated GIF."""
    from PIL import Image

    images = [Image.fromarray(f) for f in frames]
    images[0].save(
        str(path),
        save_all=True,
        append_images=images[1:],
        duration=1000 // fps,
        loop=0,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Record oracle agent episodes as GIF animations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["GoToGoal-v0"],
        help="Task names to record",
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=["easy"],
        choices=["easy", "medium", "hard", "expert"],
        help="Difficulty levels",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=3,
        help="Episodes per task/difficulty",
    )
    parser.add_argument(
        "--output-dir",
        default="videos",
        help="Output directory for GIFs",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames per second in GIF",
    )

    args = parser.parse_args()
    tasks = args.tasks
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Oracle Video Recording")
    print("=" * 80)
    print(f"Tasks: {tasks}")
    print(f"Difficulties: {args.difficulties}")
    print(f"Episodes per combo: {args.n_episodes}")
    print(f"FPS: {args.fps}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    total_gifs = 0

    for task_id in tasks:
        for difficulty in args.difficulties:
            print(f"\nRecording {task_id} @ {difficulty}...")

            try:
                env = agentick.make(task_id, difficulty=difficulty, render_mode="rgb_array")
            except Exception as e:
                print(f"  Skipping: {e}")
                continue

            try:
                oracle = get_oracle(task_id, env)
            except Exception as e:
                print(f"  No oracle: {e}")
                env.close()
                continue

            for episode in range(args.n_episodes):
                frames = []
                obs, info = env.reset(seed=42 + episode)
                oracle.reset(obs, info)

                frame = env.render()
                frames.append(frame)

                total_reward = 0.0
                done = False
                while not done:
                    action = oracle.act(obs, info)
                    obs, reward, terminated, truncated, info = env.step(action)
                    oracle.update(obs, info)

                    frame = env.render()
                    frames.append(frame)

                    total_reward += reward
                    done = terminated or truncated

                success = info.get("success", False)

                gif_path = output_dir / f"{task_id}_{difficulty}_ep{episode:02d}.gif"
                try:
                    save_gif(frames, gif_path, fps=args.fps)
                    print(
                        f"  Episode {episode + 1}: {len(frames)} frames, "
                        f"reward={total_reward:.2f}, success={success} -> {gif_path.name}"
                    )
                    total_gifs += 1
                except ImportError:
                    print(
                        f"  Episode {episode + 1}: {len(frames)} frames, "
                        f"reward={total_reward:.2f}, success={success} "
                        "(Pillow not installed, skipped save)"
                    )

            env.close()

    print(f"\nSaved {total_gifs} GIFs to {output_dir}/")


if __name__ == "__main__":
    main()
