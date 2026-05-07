# CLI Reference

## Core Commands

```bash
uv run agentick --version
uv run agentick list-tasks
uv run agentick list-tasks --capability navigation
uv run agentick list-suites
uv run agentick info GoToGoal-v0
```

## Evaluate

```bash
uv run agentick evaluate --config examples/experiments/configs/random_agent.yaml
uv run python -m agentick.experiments.run --config examples/experiments/configs/random_agent.yaml
```

## Run Experiments

```bash
uv run agentick experiment run --config examples/experiments/configs/random_agent.yaml
uv run python -m agentick.experiments.run --config examples/experiments/configs/random_agent.yaml --output-dir results/
```

## Webapp

```bash
uv sync --extra webapp
uv run python -m agentick.human.webapp    # Interactive UI at http://127.0.0.1:8080
```

## Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```
