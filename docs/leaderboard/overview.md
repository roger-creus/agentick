# Agentick Leaderboard Overview

The Agentick Leaderboard is a benchmarking platform for evaluating and comparing agent performance across standardized test suites.

## How It Works

### Evaluation Pipeline

1. **Submit Agent**: Define your agent using supported adapters (API, HuggingFace, Docker, code)

2. **Choose Benchmark Suites**:
   - **Quick Sanity Check** (~2 minutes): 3 tasks, 3 seeds
   - **Category Suites**: Navigation, reasoning, memory, etc.
   - **Full Benchmark** (~4 hours): All 38 tasks

3. **Run Evaluation**: Automated evaluation with:
   - Deterministic seeds for reproducibility
   - Multiple difficulty levels
   - Different observation modes
   - Cost and compute tracking

4. **Get Scored**: Normalized scores (0-1) against random and optimal baselines

5. **Compare**: View results on public leaderboard

## Hosted Leaderboard

**Coming soon**: Official site at `https://agentick-leaderboard.com`

Currently available:
- Local evaluations via CLI
- Programmatic comparisons
- Submission validation
- Visualization generation

## Scoring Methodology

### Normalized Scoring

```
normalized_score = (agent_return - random_baseline) / (optimal_return - random_baseline)
```

- **0.0**: Random performance
- **0.5**: Halfway between random and optimal
- **1.0**: Optimal performance

### Aggregate Scores

**Agentick Score**: Mean normalized score across all tasks in a suite

**Capability Scores**: Aggregate by capability tag (navigation, reasoning, memory, control)

## Benchmark Suites

| Suite | Tasks | Seeds | Time | Description |
|-------|-------|-------|------|-------------|
| `quick` | 3 | 3 | ~2 min | Sanity check during development |
| `smoke` | 10 | 5 | ~10 min | Broader coverage for quick validation |
| `navigation` | 5 | 10 | ~30 min | Navigation and pathfinding |
| `reasoning` | 5 | 10 | ~30 min | Planning and logical reasoning |
| `memory` | 4 | 10 | ~25 min | Memory-dependent tasks |
| `control` | 5 | 10 | ~30 min | Precise control and manipulation |
| `generalization` | 5 | 10 | ~30 min | Zero-shot generalization |
| `full` | 38 | 10 | ~4 hours | Complete benchmark |

## Submission Types

### API Adapter

For HTTP endpoint agents:

```yaml
agent:
  type: api
  endpoint: https://your-api.com/agent
  api_key: ${API_KEY}
  timeout: 30
```

### Code Adapter

For local Python agents:

```yaml
agent:
  type: code
  module: my_agent.agent
  class: MyAgent
  config:
    model_path: ./model.pth
```

### HuggingFace Adapter

For models on HuggingFace Hub:

```yaml
agent:
  type: huggingface
  model_id: username/model-name
  adapter: lora
```

### Docker Adapter

For containerized agents:

```yaml
agent:
  type: docker
  image: myregistry/agent:latest
  port: 8000
```

See [Adapters](adapters.md) for detailed configuration.

## Submitting to Leaderboard

### 1. Create Submission

```bash
uv run agentick submit init
```

Creates `submission.yaml`:

```yaml
submission:
  name: "My Agent"
  author: "Your Name"
  organization: "Your Org"
  contact: "email@example.com"

agent:
  type: api
  endpoint: https://your-api.com/agent

benchmark:
  suite: full
  seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
```

### 2. Validate Submission

```bash
uv run agentick submit validate submission.yaml
```

### 3. Run Evaluation

```bash
uv run agentick submit run submission.yaml --output results/
```

### 4. View Results

```bash
uv run agentick submit results results/submission_results.json
```

See [Submitting](submitting.md) for complete guide.

## Leaderboard CLI

```bash
# List available suites
uv run agentick list-suites

# Run evaluation locally
uv run agentick evaluate --agent-config agent.yaml --suite quick --output results.json

# Compare two agents
uv run agentick compare results1.json results2.json

# Generate report
uv run agentick report results.json --format html --output report.html
```

See [CLI Documentation](../cli.md) for all commands.

## Example Submissions

See `examples/leaderboard/` for complete examples:

- `create_submission.py` - Create submission file
- `validate_submission.py` - Validate before running
- `run_evaluation.py` - Run local evaluation
- `view_results.py` - Display results
- `compare_agents.py` - Compare multiple submissions

## Resources

- [Submitting Guide](submitting.md) - Complete submission instructions
- [Adapters](adapters.md) - Supported agent types
- [Scoring](../concepts/scoring.md) - Detailed scoring methodology
- [CLI](../cli.md) - Command reference
