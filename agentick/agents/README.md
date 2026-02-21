# agents/ -- LLM/VLM Agent Harness System

Composable agent framework for running LLM and VLM models on Agentick tasks.

## Architecture

`BaseAgent` composes two pluggable components:

- **ModelBackend** -- how to call the model (API or local inference)
- **HarnessPreset** -- how to construct prompts and manage conversation context

The factory function `create_agent(AgentConfig)` wires these together from
a YAML experiment config, returning a `BaseAgent` that conforms to `AgentProtocol`.

## Files

### `base.py` -- `BaseAgent`
Core agent class. `act(observation, info)` delegates to the harness for message
construction, the backend for generation, and `parse_action_from_text` for
response parsing. Tracks token counts, latency, and per-step call logs.
Returns `get_stats()` with aggregate metrics.

### `harness.py` -- Prompt Presets
Abstract `HarnessPreset` base class with three concrete implementations:

- **`MarkovianZeroShot`** -- Memoryless; sends system prompt + current
  observation only. No history retained between steps.
- **`NonMarkovianZeroShot`** -- History-aware; maintains full conversation
  history (user/assistant turns). Supports `max_history` truncation.
- **`MarkovianReasoner`** -- Memoryless chain-of-thought; appends CoT
  instructions to the system prompt requesting step-by-step reasoning
  before `ACTION: <number>`.

All presets are registered in `HARNESS_REGISTRY` keyed by snake_case names
(`markovian_zero_shot`, `non_markovian_zero_shot`, `markovian_reasoner`).

### `factory.py` -- `create_agent(AgentConfig)`
Builds a `BaseAgent` from an `AgentConfig` (parsed from experiment YAML).
Returns `None` for non-LLM/VLM agent types (e.g. `"random"`, `"ppo"`).
Resolves the backend class via `get_backend_class(name)` and the harness
class via `HARNESS_REGISTRY`.

### `observation.py` -- Observation Utilities
- `format_text_observation(obs, info, mode)` -- formats any observation
  modality as text for LLM consumption, using pre-rendered secondary
  observations from `info` when available.
- `numpy_to_base64(image_array)` -- converts numpy RGB arrays to
  base64-encoded PNG for multimodal API messages.
- `numpy_to_pil(image_array)` -- converts numpy RGB arrays to PIL Image
  for local VLM backends.

### `backends/` (subpackage)
Lazy-loaded model backends: `OpenAI`, `Anthropic`, `HuggingFaceLLM`,
`HuggingFaceVLM`. Each implements the `ModelBackend` protocol and returns
a `BackendResponse` with `text`, `input_tokens`, and `output_tokens`.

## Usage

```python
from agentick.agents import create_agent
from agentick.experiments.config import AgentConfig

config = AgentConfig(type="llm", hyperparameters={
    "backend": "openai",
    "model": "gpt-4o",
    "harness": "markovian_reasoner",
    "observation_modes": ["language"],
})
agent = create_agent(config)

obs, info = env.reset()
action = agent.act(obs, info)
```
