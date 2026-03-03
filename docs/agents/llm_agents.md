# LLM / VLM Agents

Evaluate language and vision-language models on Agentick using OpenAI, Anthropic, or HuggingFace.

## Quick Start

```python
import agentick
from agentick.leaderboard.adapters.api_adapter import APIAgent

env = agentick.make("GoToGoal-v0", render_mode="language")
agent = APIAgent(
    provider="openai",
    model="gpt-4o-mini",
    observation_mode="language",
    api_key_env="OPENAI_API_KEY",
)

obs, info = env.reset()
for step in range(100):
    action = agent.act(obs, info)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## Providers

**OpenAI**: `examples/llm/openai_text_agent.py`
```python
agent = APIAgent(provider="openai", model="gpt-4o", observation_mode="language", temperature=0.0)
```

**Anthropic**: `examples/llm/anthropic_text_agent.py`
```python
agent = APIAgent(provider="anthropic", model="claude-sonnet-4-20250514", observation_mode="language")
```

**HuggingFace (local)**: See `examples/llm/huggingface_local_agent.py` for custom inference with Qwen, Llama, etc.

## VLM Agents

For vision-language models, use `render_mode="rgb_array"` (isometric 512x512) with a vision-capable backend:

```python
env = agentick.make("GoToGoal-v0", render_mode="rgb_array")
agent = APIAgent(provider="openai", model="gpt-4o", observation_mode="rgb_array")
```

Vision-capable backends: `openai`, `anthropic`, `huggingface_vlm`, `vllm_vlm`.

## Agent Harness System

For structured evaluation, Agentick provides a composable harness system separating **model inference** (backend) from **prompt management** (harness preset).

### Backends

| Backend | Class | Vision |
|---|---|---|
| `openai` | `OpenAIBackend` | Yes |
| `anthropic` | `AnthropicBackend` | Yes |
| `huggingface_llm` | `HuggingFaceLLMBackend` | No |
| `huggingface_vlm` | `HuggingFaceVLMBackend` | Yes |
| `vllm_llm` / `vllm_vlm` | vLLM backends | No / Yes |

Backends are lazy-loaded — you only need the SDK for the backend you use. When `vllm` is installed, HuggingFace backends auto-upgrade for faster inference.

### Harness Presets

| Preset | History | Reasoning |
|---|---|---|
| `markovian_zero_shot` | No | No |
| `non_markovian_zero_shot` | Sliding window | No |
| `markovian_reasoner` | No | Chain-of-thought |
| `non_markovian_reasoner` | Sliding window | Chain-of-thought |

### Factory

```python
from agentick.agents import create_agent
from agentick.experiments.config import AgentConfig

agent = create_agent(AgentConfig(
    type="llm",
    hyperparameters={
        "backend": "openai",
        "model": "gpt-4o",
        "harness": "markovian_zero_shot",
        "observation_modes": ["language"],
        "api_key_env": "OPENAI_API_KEY",
        "max_tokens": 100,
        "temperature": 0.0,
    },
))
```

### YAML Config

```yaml
name: claude-ascii
agent:
  type: llm
  hyperparameters:
    backend: anthropic
    model: claude-sonnet-4-20250514
    harness: markovian_zero_shot
    observation_modes: [ascii]
    api_key_env: ANTHROPIC_API_KEY
    max_tokens: 100
tasks: "full"
difficulties: [easy, medium]
n_episodes: 3
n_seeds: 3
render_modes: [ascii]
output_dir: results/claude
```

### Direct Usage

```python
from agentick.agents import BaseAgent
from agentick.agents.backends.openai_backend import OpenAIBackend
from agentick.agents.harness import NonMarkovianZeroShot

backend = OpenAIBackend(model="gpt-4o", temperature=0.0, max_tokens=100)
harness = NonMarkovianZeroShot(max_context_tokens=16384, diff_mode=True)
agent = BaseAgent(backend=backend, harness=harness, observation_modes=["language"])

env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="language")
obs, info = env.reset(seed=42)
agent.reset()

done = False
while not done:
    action = agent.act(obs, info)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

stats = agent.get_stats()
print(f"API calls: {stats['total_calls']}, tokens: {stats['total_tokens']}")
env.close()
```

## Complete Examples

See `examples/llm/`:
- `openai_text_agent.py`, `anthropic_text_agent.py` — API agents
- `huggingface_local_agent.py` — Local model inference
- `compare_llms.py` — Multi-provider comparison
