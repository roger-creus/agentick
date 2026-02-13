"""
Record video episodes of agent playing tasks.

Demonstrates video recording functionality.
Runtime: ~30 seconds
Requires: imageio-ffmpeg (uv sync --extra viz)
"""

from pathlib import Path

import agentick


def main():
    print("Video Recording Example")
    print("=" * 80)

    # Check for ffmpeg
    try:
        import imageio_ffmpeg
    except ImportError:
        print("WARNING: imageio-ffmpeg not installed. Will save as frames instead.")
        print("Install with: uv sync --extra viz")

    # Create environment
    env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="rgb_array")

    # Create output directory
    output_dir = Path("videos")
    output_dir.mkdir(exist_ok=True)

    # Record 3 episodes
    print(f"\nRecording 3 episodes to {output_dir}/")

    for episode in range(3):
        frames = []
        obs, info = env.reset(seed=42 + episode)

        total_reward = 0
        for step in range(50):
            # Render frame
            frame = env.render()
            frames.append(frame)

            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            if terminated or truncated:
                break

        # Save video
        try:
            import imageio

            video_path = output_dir / f"episode_{episode:02d}.mp4"
            imageio.mimsave(str(video_path), frames, fps=10)
            print(
                f"  Episode {episode + 1}: {len(frames)} frames, "
                f"reward={total_reward:.2f}, saved to {video_path.name}"
            )
        except ImportError:
            # Fallback: save first frame as image
            from PIL import Image

            img_path = output_dir / f"episode_{episode:02d}_frame0.png"
            Image.fromarray(frames[0]).save(img_path)
            print(
                f"  Episode {episode + 1}: {len(frames)} frames, "
                f"reward={total_reward:.2f}, saved first frame to {img_path.name}"
            )

    env.close()
    print(f"\nVideos saved to {output_dir}/")


if __name__ == "__main__":
    main()
