# LLM / VLM Agents

Evaluate language and vision-language models on Agentick using OpenAI, Google Gemini, HuggingFace, or vLLM.

## Quick Start

```python
import agentick
from agentick.agents import BaseAgent, create_agent
from agentick.experiments.config import AgentConfig

# Create agent from config
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

env = agentick.make("GoToGoal-v0", render_mode="language")
obs, info = env.reset()
agent.reset()

for step in range(100):
    action = agent.act(obs, info)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

## Providers

**OpenAI**: `examples/llm/openai_text_agent.py`

**Google Gemini**: Use `backend: gemini` with `api_key_env: GOOGLE_API_KEY`

**HuggingFace (local)**: See `examples/llm/huggingface_local_agent.py` for custom inference with Qwen, Llama, etc.

**vLLM**: When `vllm` is installed, HuggingFace backends auto-upgrade for faster inference. Use `backend: vllm_llm` or `backend: vllm_vlm` directly.

## VLM Agents

For vision-language models, use `render_mode="rgb_array"` (isometric 512x512) with a vision-capable backend:

```python
env = agentick.make("GoToGoal-v0", render_mode="rgb_array")
```

Vision-capable backends: `openai`, `huggingface_vlm`, `vllm_vlm`, `gemini`.

## Agent Harness System

Agentick provides a composable harness system separating **model inference** (backend) from **prompt management** (harness preset).

### Backends

| Backend | Class | Vision |
|---|---|---|
| `openai` | `OpenAIBackend` | Yes |
| `gemini` | `GeminiBackend` | Yes |
| `huggingface_llm` | `HuggingFaceLLMBackend` | No |
| `huggingface_vlm` | `HuggingFaceVLMBackend` | Yes |
| `vllm_llm` / `vllm_vlm` | vLLM backends | No / Yes |

Backends are lazy-loaded — you only need the SDK for the backend you use.

### Harness Presets

| Preset | History | Reasoning |
|---|---|---|
| `markovian_zero_shot` | No | No |
| `markovian_reasoner` | No | Chain-of-thought |

### YAML Config

```yaml
name: gpt4o-ascii
agent:
  type: llm
  hyperparameters:
    backend: openai
    model: gpt-4o
    harness: markovian_zero_shot
    observation_modes: [ascii]
    api_key_env: OPENAI_API_KEY
    max_tokens: 100
tasks: "full"
difficulties: [easy, medium]
n_episodes: 3
n_seeds: 3
render_modes: [ascii]
output_dir: results/gpt4o
```

### Direct Usage

```python
from agentick.agents import BaseAgent
from agentick.agents.backends.openai_backend import OpenAIBackend
from agentick.agents.harness import MarkovianZeroShot

backend = OpenAIBackend(model="gpt-4o", temperature=0.0, max_tokens=100)
harness = MarkovianZeroShot()
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
- `openai_text_agent.py` — OpenAI API agent
- `huggingface_local_agent.py` — Local model inference
