# Utils

Shared utility functions for the Agentick package.

## Video Recording (`video.py`)

Functions for recording agent episodes as video files.

- `record_episode(env, agent=None, output_path="episode.mp4", max_steps=None, fps=10, fallback_format="gif")` -- records a single episode to MP4 or GIF. If `agent` is None, uses random actions. Falls back to GIF or PNG sequence if MP4 encoding fails.
- `record_episodes_to_video(env, agent, num_episodes=1, output_dir="videos", fps=10)` -- records multiple episodes to separate video files in a directory.
- `wrap_env_with_video_recording(env, output_dir="videos", episode_trigger=None, fps=30)` -- wraps an environment with Gymnasium's `RecordVideo` for automatic recording during training. Accepts an `episode_trigger` function to selectively record (e.g., every 10th episode).

Internal helpers:
- `_save_mp4(frames, path, fps)` -- writes frames using imageio with libx264 codec
- `_save_gif(frames, path, fps)` -- writes frames as animated GIF using PIL
- `_save_png_sequence(frames, output_dir)` -- saves frames as numbered PNGs

Requires `imageio` for MP4 output (`uv sync --extra rl`). GIF output uses PIL only.
