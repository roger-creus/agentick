# Logging

Structured logging system for Agentick experiments, episodes, and LLM agent interactions.

## Modules

### `episode_logger.py`
- **`EpisodeLogger`** -- Logs per-step data (action, observation, reward, info) for a single episode. Supports four verbosity levels (`minimal`, `standard`, `full`, `debug`), optional gzip compression, and append mode. Saves to JSON. Class method `EpisodeLogger.load()` reads logs back.

### `experiment_logger.py`
- **`ExperimentLogger`** -- Tracks experiment-level metadata: per-task timing (`log_task_start`/`log_task_end`), errors with context, warnings, and total wall-clock time. Writes a summary JSON via `finalize()`.

### `agent_logger.py`
- **`LoggableAgent`** -- Protocol for agents that expose internals via `get_log_data()`.
- **`LLMAgentLogger`** -- Records prompts, responses, parsed actions, token counts, latencies, and costs across an episode for LLM-based agents.

### `llm_logger.py`
- **`LLMLogger`** -- Comprehensive LLM call logger. Captures prompt, system prompt, response, parse success, token counts, latency, and cost per call. Tracks running totals for tokens and cost.

### `browser.py`
- **`browse_logs()`** -- Terminal UI for exploring experiment result directories. (Stub; intended for rich-based interactive browsing.)

### `replay.py`
- **`replay_episode()`** -- Replays a logged episode in a pygame window at configurable speed. Loads steps via `EpisodeLogger.load()` and renders frame-by-frame. Falls back to terminal replay if pygame is unavailable.

## Usage

```python
from agentick.logging.episode_logger import EpisodeLogger

logger = EpisodeLogger("episode_001.json", verbosity="standard")
logger.log_step(step=0, action=..., observation=..., reward=..., info=...)
logger.save()

# Replay later
from agentick.logging.replay import replay_episode
replay_episode("episode_001.json", speed=2.0)
```
