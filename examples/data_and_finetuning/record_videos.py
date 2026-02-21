"""
Record oracle agent video/GIF episodes.

Demonstrates recording oracle agents solving tasks as GIF animations.
Runtime: ~30 seconds
Requires: Pillow (uv sync --extra viz)

Usage:
    uv run python examples/data_and_finetuning/record_videos.py
"""

from pathlib import Path

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
    print("Oracle Video Recording Example")
    print("=" * 80)

    # Create output directory
    output_dir = Path("videos")
    output_dir.mkdir(exist_ok=True)

    task_id = "GoToGoal-v0"

    # Create environment with pixel rendering
    env = agentick.make(task_id, difficulty="easy", render_mode="rgb_array")

    # Create oracle agent
    oracle = get_oracle(task_id, env)

    # Record 3 episodes
    print(f"\nRecording 3 oracle episodes for {task_id}...")

    for episode in range(3):
        frames = []
        obs, info = env.reset(seed=42 + episode)
        oracle.reset(obs, info)

        # Capture initial frame
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

        # Save as GIF
        gif_path = output_dir / f"{task_id}_episode_{episode:02d}.gif"
        try:
            save_gif(frames, gif_path, fps=5)
            print(
                f"  Episode {episode + 1}: {len(frames)} frames, "
                f"reward={total_reward:.2f}, success={success}, saved to {gif_path.name}"
            )
        except ImportError:
            print(
                f"  Episode {episode + 1}: {len(frames)} frames, "
                f"reward={total_reward:.2f}, success={success} (Pillow not installed, skipped save)"
            )

    env.close()
    print(f"\nGIFs saved to {output_dir}/")


if __name__ == "__main__":
    main()
