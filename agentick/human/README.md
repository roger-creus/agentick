# Human Evaluation Toolkit

Tools for collecting, recording, and analyzing human performance on Agentick tasks, plus pre-computed human baselines for all 38 tasks.

## Modules

### `webapp.py` -- ShowcaseWebApp

Flask-based web UI for browsing and playing tasks interactively. Manages play sessions via an in-memory `_sessions` dict keyed by session UUID.

Key API endpoints:
- `/api/task_descriptions` -- list all registered tasks with metadata
- `/api/start_task` -- create a new play session for a given task/difficulty
- `/api/step` -- submit an action and receive updated multi-modal observations
- `/api/reset` -- reset the current session environment
- `/gallery/` -- oracle/solution gallery

Helper functions:
- `_render_multimodal(env)` -- renders ASCII, language, and base64-encoded pixel observations simultaneously
- `_strip_ansi(text)` -- strips ANSI escape codes for web display

### `player.py` -- HumanPlayer

Pygame-based terminal interface for human evaluation with:
- Tutorial mode with overlays
- Practice rounds before scored rounds
- Timer display, step counter, score display
- Pause/resume and optional undo
- End-of-episode summary

Constructor params: `env`, `window_size`, `fps`, `show_tutorial`, `allow_undo`, `practice_rounds`.

### `recorder.py` -- HumanDataRecorder

Records human play sessions to disk for later analysis. Captures actions, timing, episode outcomes, and optional demographic data. Each session gets an auto-generated `participant_id` and `session_id`. Data is saved to a configurable `save_dir` (default `human_data/`).

Also provides `load_session_data(data_dir)` for bulk loading.

### `analysis.py` -- HumanBaselineAnalyzer

Loads all recorded sessions from disk and computes per-task baseline statistics. Primary method: `compute_task_baseline(task_name, difficulty=None)` returns a dict of aggregate metrics (success rate, mean steps, etc.).

### `baselines.py` -- Pre-computed Human Baselines

`HUMAN_BASELINES` is a dict mapping every task name to estimated human performance:
- `success_rate` -- probability of completion
- `avg_steps` -- average steps when successful
- `optimal_ratio` -- steps taken / optimal steps (efficiency)
- `learning_curve` -- performance improvement over attempts
- `difficulty` and `notes`

Utility functions:
- `get_human_baseline(task_name)` -- single task lookup
- `get_all_baselines()` / `get_baselines_by_difficulty(difficulty)`
- `compare_to_human(agent_results)` -- compare agent metrics against human baselines
- `estimate_human_performance(task_name)` / `get_summary_statistics()`
