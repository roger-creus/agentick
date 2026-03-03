# CLI Reference

## Core Commands

```bash
uv run agentick --version
uv run agentick list-tasks
uv run agentick list-tasks --capability navigation
uv run agentick list-suites
```

## Evaluate

```bash
uv run agentick evaluate --task GoToGoal-v0 --agent random --episodes 10
uv run agentick evaluate --task GoToGoal-v0 --difficulty hard --agent oracle --output results.json
uv run agentick evaluate --task GoToGoal-v0 --agent random --seeds 42,43,44
```

## Run Experiments

```bash
uv run agentick experiment run configs/my_experiment.yaml --output results/
```

## Webapp

```bash
uv run agentick webapp    # Interactive UI at http://localhost:5000
```

## Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```
