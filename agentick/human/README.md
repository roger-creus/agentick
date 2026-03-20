# Human Evaluation

Web interface for humans to play Agentick tasks and record session data.

## Modules

### `webapp.py` -- ShowcaseWebApp

Flask-based web UI for browsing and playing tasks interactively. Serves the showcase page from `docs/showcase/index.html`.

Key API endpoints:
- `/api/task_descriptions` -- list all registered tasks with metadata
- `/api/start_task` -- create a new play session for a given task/difficulty
- `/api/step` -- submit an action and receive multi-modal observations (ASCII + language + pixels)
- `/api/reset` -- reset the current session
- `/api/quit` -- close the session

### `recorder.py` -- HumanDataRecorder

Records human play sessions to disk (JSON). Captures actions, timing, episode outcomes, and optional demographics. Each session gets an auto-generated `participant_id` and `session_id`.
