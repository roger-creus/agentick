# LLM Agents

Evaluate LLMs on Agentick using OpenAI, Anthropic, and HuggingFace.

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
action = agent.act(obs, info)
obs, reward, terminated, truncated, info = env.step(action)
```

## OpenAI

```python
import agentick
from agentick.leaderboard.adapters.api_adapter import APIAgent

env = agentick.make("GoToGoal-v0", render_mode="language")
agent = APIAgent(
    provider="openai",
    model="gpt-4o-mini",
    observation_mode="language",
    api_key_env="OPENAI_API_KEY",
    temperature=0.0,
)

obs, info = env.reset()
for step in range(100):
    action = agent.act(obs, info)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break
```

**Example**: `examples/llm/openai_text_agent.py`

## Anthropic

```python
import agentick
from agentick.leaderboard.adapters.api_adapter import APIAgent

env = agentick.make("GoToGoal-v0", render_mode="language")
agent = APIAgent(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    observation_mode="language",
    api_key_env="ANTHROPIC_API_KEY",
    max_tokens=100,
)

obs, info = env.reset()
for step in range(100):
    action = agent.act(obs, info)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break
```

**Example**: `examples/llm/anthropic_text_agent.py`

## HuggingFace

For local HuggingFace models, see the complete example in `examples/llm/huggingface_local_agent.py` which includes:
- Custom agent class for model inference
- Prompt engineering with few-shot examples
- Action parsing from model outputs
- Works with Qwen, Llama, and other instruction-tuned models

## Advanced Usage

The `APIAgent` class handles prompt formatting and action parsing automatically. For custom prompt engineering or action parsing, see `examples/llm/huggingface_local_agent.py` which demonstrates:
- Custom prompt templates with few-shot examples
- Flexible action parsing from free-form text
- Working with different model response formats

## Complete Examples

See `examples/llm/`:
- `openai_text_agent.py` - OpenAI GPT-4
- `anthropic_text_agent.py` - Anthropic Claude
- `huggingface_local_agent.py` - Local HF models
- `compare_llms.py` - Multi-provider comparison
