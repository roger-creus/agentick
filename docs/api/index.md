# API Reference

API documentation for Agentick.

## Core Modules

- `agentick.core.env` - Environment implementation
- `agentick.core.grid` - Grid representation
- `agentick.core.agent` - Agent types
- `agentick.core.types` - Core type definitions

## Tasks

- `agentick.tasks.base` - Base task specification
- `agentick.tasks.registry` - Task registration and loading
- `agentick.tasks.configs` - Task configuration

## Experiments

- `agentick.experiments.config` - Experiment configuration
- `agentick.experiments.runner` - Experiment execution
- `agentick.experiments.registry` - Results registry
- `agentick.experiments.reproduce` - Reproducibility tools

## Analysis

- `agentick.analysis.statistics` - Statistical tests
- `agentick.analysis.metrics` - Performance metrics
- `agentick.analysis.comparisons` - Agent comparisons
- `agentick.analysis.learning_curves` - Learning analysis

## Visualization

- `agentick.visualization.plots` - Static plots
- `agentick.visualization.tables` - LaTeX tables
- `agentick.visualization.style` - Plot styling
- `agentick.visualization.video` - Video generation
- `agentick.visualization.interactive` - Interactive dashboards
- `agentick.visualization.report` - Report generation

## Logging

- `agentick.logging.episode_logger` - Episode logging
- `agentick.logging.agent_logger` - Agent internals logging
- `agentick.logging.experiment_logger` - Experiment metadata
- `agentick.logging.llm_logger` - LLM API logging
- `agentick.logging.replay` - Episode replay
- `agentick.logging.browser` - Log browsing

## Benchmark

- `agentick.benchmark.baselines` - Baseline agents

## Data

- `agentick.data.demonstrations` - Demonstration collection
- `agentick.data.formats` - Data format conversion

## Generation

- `agentick.generation.room` - Room generation
- `agentick.generation.objects` - Object placement
- `agentick.generation.maze` - Maze generation
- `agentick.generation.validation` - Task validation

## Human

- `agentick.human.recorder` - Human data recording

---

For detailed API documentation, use Python's built-in help:

```python
import agentick
help(agentick.experiments.runner)
```
