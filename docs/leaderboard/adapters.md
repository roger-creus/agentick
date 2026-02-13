# Leaderboard Adapters

Adapters define how to load and run agent types.

## Supported Adapters

| Type | Use Case |
|------|----------|
| **API** | OpenAI, Anthropic LLMs |
| **HuggingFace** | HF Hub models |
| **Code** | Custom Python agents |
| **Docker** | Containerized agents |

## API Adapter

### OpenAI

```yaml
agent:
  type: api
  provider: openai
  model: gpt-4o
  api_key_env: OPENAI_API_KEY
  observation_mode: language
```

### Anthropic

```yaml
agent:
  type: api
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key_env: ANTHROPIC_API_KEY
  observation_mode: language
```

### Config Options

```yaml
agent:
  type: api
  provider: string      # openai, anthropic
  model: string
  api_key_env: string
  observation_mode: string  # language, rgb_array
  temperature: float    # Default: 0.0
  max_tokens: int       # Default: 100
  timeout: int          # Default: 30
```

## HuggingFace Adapter

```yaml
agent:
  type: huggingface
  model_id: meta-llama/Llama-3.1-8B-Instruct
  observation_mode: language
  device: cuda
  torch_dtype: float16
```

### Config Options

```yaml
agent:
  type: huggingface
  model_id: string
  observation_mode: string
  device: string        # cpu, cuda
  torch_dtype: string   # float32, float16, bfloat16
  max_new_tokens: int   # Default: 100
  temperature: float    # Default: 0.1
```

**Supported models**: Llama, Mistral, Qwen, Gemma (text); LLaVA, Qwen-VL (vision)

## Code Adapter

```yaml
agent:
  type: code
  module: my_agent.agent
  class: MyAgent
  observation_mode: state_dict
  config:
    model_path: ./model.pth
```

### Agent Protocol

```python
from agentick.leaderboard.adapters import AgentProtocol

class MyAgent(AgentProtocol):
    def __init__(self, config: dict):
        self.model = load_model(config["model_path"])

    def act(self, observation, valid_actions: list[str]) -> str:
        return "move_up"

    def reset(self):
        pass
```

### Config Options

```yaml
agent:
  type: code
  module: string
  class: string
  observation_mode: string
  config: dict
```

## Docker Adapter

```yaml
agent:
  type: docker
  image: myregistry/agent:latest
  port: 8000
  health_check: /health
```

### HTTP API Spec

**POST /act**
```json
Request: {"observation": "...", "valid_actions": [...]}
Response: {"action": "move_up"}
```

**POST /reset**
```json
Request: {}
Response: {"status": "ok"}
```

**GET /health**
```json
Response: {"status": "healthy"}
```

### Config Options

```yaml
agent:
  type: docker
  image: string
  port: int             # Default: 8000
  health_check: string  # Default: /health
  environment: dict
  timeout: int          # Default: 30
```

## Adapter Comparison

| Adapter | Latency | GPU | Use Case |
|---------|---------|-----|----------|
| API | High | No | Cloud LLMs |
| HuggingFace | Medium | Yes | Open-source |
| Code | Low | Optional | Custom |
| Docker | Medium | Optional | Containers |

## Examples

See `examples/leaderboard/`:
- `create_submission.py`
- `validate_submission.py`

## Resources

- [Submission Guide](submitting.md)
- [Overview](overview.md)
- [CLI](../cli.md)
