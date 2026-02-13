# CLI Reference

Command-line interface for Agentick.

## Installation

```bash
uv sync --extra all
```

## Quick Start

```bash
uv run agentick --version
uv run agentick list-tasks
uv run agentick evaluate --task GoToGoal-v0 --agent random --episodes 10
```

## Commands

### agentick --version

```bash
uv run agentick --version
```

### agentick list-tasks

```bash
uv run agentick list-tasks
uv run agentick list-tasks --capability navigation
uv run agentick list-tasks --difficulty easy
```

### agentick list-suites

```bash
uv run agentick list-suites
uv run agentick list-suites --suite quick --verbose
```

### agentick evaluate

```bash
# Basic
uv run agentick evaluate --task GoToGoal-v0 --agent random --episodes 10

# With options
uv run agentick evaluate --task GoToGoal-v0 --difficulty hard --agent oracle --output results.json
uv run agentick evaluate --task GoToGoal-v0 --agent random --seeds 42,43,44
```

**Options**: `--task`, `--agent`, `--difficulty`, `--episodes`, `--seeds`, `--output`, `--render`

### agentick experiment run

```bash
uv run agentick experiment run configs/quick/sanity_check.yaml --output results/
uv run agentick experiment run configs/baselines/random.yaml --seeds 42,43,44
```

**Options**: `--config`, `--output`, `--seeds`, `--parallel`

### agentick submit init

```bash
uv run agentick submit init
```

### agentick submit validate

```bash
uv run agentick submit validate submission.yaml
```

### agentick submit run

```bash
uv run agentick submit run submission.yaml --suite quick --output results/
uv run agentick submit run submission.yaml --suite full --output results/
```

**Options**: `--suite`, `--output`, `--parallel`

### agentick submit results

```bash
uv run agentick submit results results/submission_results.json
uv run agentick submit results results.json --format csv --output report.csv
uv run agentick submit results results.json --format html --output report.html
```

### agentick submit compare

```bash
uv run agentick submit compare results1.json results2.json
```

## Global Options

```bash
uv run agentick --help
uv run agentick --verbose list-tasks
uv run agentick --quiet evaluate --task GoToGoal-v0 --agent random
```

## Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export AGENTICK_DATA_DIR="./data"
```

## Config Files

### Experiment Config

```yaml
experiment:
  name: "My Experiment"
  seeds: [42, 43, 44]

tasks:
  - name: GoToGoal-v0
    difficulty: medium

agent:
  type: random
```

### Submission Config

```yaml
submission:
  name: "My Agent"
  author: "Your Name"

agent:
  type: api
  provider: openai
  model: gpt-4o

benchmark:
  suite: quick
  seeds: [42, 43, 44]
```

## Examples

```bash
# Evaluate
uv run agentick evaluate --task GoToGoal-v0 --agent oracle --episodes 10

# Run experiment
uv run agentick experiment run configs/quick/sanity_check.yaml --output results/

# Submit
uv run agentick submit init
uv run agentick submit validate submission.yaml
uv run agentick submit run submission.yaml --suite quick --output results/
```

## See Also

- [Getting Started](getting_started/quickstart.md)
- [Leaderboard Submission](leaderboard/submitting.md)
