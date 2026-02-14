# Leaderboard Submission Guide

Submit an agent to the Agentick Leaderboard.

## Quick Start

```bash
uv run agentick submit init
uv run agentick submit validate submission.yaml
uv run agentick evaluate --submission submission.yaml --suite agentick-core-v1 --output results/
uv run agentick verify results/evaluation_result.json
```

## Submission File

### API Agent

```yaml
submission:
  name: "GPT-4o Agent"
  author: "Your Name"

agent:
  type: api
  provider: openai
  model: gpt-4o
  api_key_env: OPENAI_API_KEY

benchmark:
  suite: quick
  seeds: [42, 43, 44]
```

### HuggingFace Agent

```yaml
submission:
  name: "Llama 3.1 8B"

agent:
  type: huggingface
  model_id: meta-llama/Llama-3.1-8B-Instruct

benchmark:
  suite: quick
  seeds: [42, 43, 44]
```

### Code Agent

```yaml
submission:
  name: "Custom Agent"

agent:
  type: code
  module: my_agent.agent
  class: MyAgent

benchmark:
  suite: quick
  seeds: [42, 43, 44]
```

### Docker Agent

```yaml
submission:
  name: "Containerized"

agent:
  type: docker
  image: myregistry/agent:latest
  port: 8000

benchmark:
  suite: quick
```

See [Adapters](adapters.md).

## Workflow

```bash
# 1. Initialize submission template
uv run agentick submit init

# 2. Edit submission.yaml with your agent details

# 3. Validate submission format
uv run agentick submit validate submission.yaml

# 4. Test evaluation (quick suite: 3 tasks, ~2 min)
uv run agentick evaluate --submission submission.yaml --suite agentick-quick-v1 --output results/

# 5. View and verify results
uv run agentick verify results/evaluation_result.json

# 6. Full evaluation (35 tasks, ~4 hours)
uv run agentick evaluate --submission submission.yaml --suite agentick-core-v1 --output results/

# 7. Submit
uv run agentick submit upload results/submission_results.json
```

## Benchmark Suites

| Suite | Tasks | Seeds | Time |
|-------|-------|-------|------|
| `quick` | 3 | 3 | ~2 min |
| `smoke` | 10 | 5 | ~10 min |
| `full` | 35 | 10 | ~4 hours |

## CLI Commands

```bash
uv run agentick list-suites
uv run agentick submit init
uv run agentick submit validate submission.yaml
uv run agentick submit run submission.yaml --output results/
uv run agentick submit results results.json
uv run agentick submit compare results1.json results2.json
uv run agentick submit report results.json --format html
```

## Required Fields

```yaml
submission:
  name: string
  author: string

agent:
  type: string

benchmark:
  suite: string
  seeds: list[int]
```

## Examples

See `examples/leaderboard/`:
- `create_submission.py`
- `validate_submission.py`
- `run_evaluation.py`
- `view_results.py`

## Resources

- [Overview](overview.md)
- [Adapters](adapters.md)
- [CLI](../cli.md)
