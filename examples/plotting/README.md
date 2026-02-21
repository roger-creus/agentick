# Plotting

Visualization scripts for experiment results, training logs, and agent
comparison. All scripts that produce figures require matplotlib.

## Prerequisites

```bash
uv sync --extra viz        # matplotlib, numpy
uv sync --extra plotting   # plot_experiment_results.py uses ExperimentPlotter
```

## Scripts

- **plot_experiment_results.py** -- Load results from an experiment directory
  and generate learning curves, final performance bars, difficulty scaling, and
  episode length distributions using `ExperimentPlotter`.
- **capability_radar.py** -- Generate a radar (spider) chart showing agent
  scores across capability dimensions (navigation, memory, reasoning, skill,
  control, combinatorial). Uses example data; replace with your own results.
- **bar_comparison.py** -- Create a bar chart comparing average rewards across
  multiple agents. Takes one or more result JSON files as positional arguments.
- **learning_curves.py** -- Plot training learning curves (raw + rolling
  average) and per-metric panels from a log directory containing JSON or JSONL
  files.
- **difficulty_scaling.py** -- Plot success rate vs difficulty level (easy,
  medium, hard, expert) from a result JSON file.
- **heatmap.py** -- Create a task-by-agent performance heatmap (success rate,
  colored green-to-red) from multiple result JSON files.
- **latex_tables.py** -- Generate a LaTeX table (`booktabs` style) of per-task
  reward and success rate from a result JSON file. Include the output `.tex`
  file directly in a paper.

## Running

```bash
# Full experiment plots (needs a results directory)
uv run python examples/plotting/plot_experiment_results.py --results-dir results/my_exp/

# Standalone radar chart (uses built-in example data)
uv run python examples/plotting/capability_radar.py

# Bar comparison from result files
uv run python examples/plotting/bar_comparison.py results1.json results2.json

# Learning curves from training logs
uv run python examples/plotting/learning_curves.py logs/

# Difficulty scaling
uv run python examples/plotting/difficulty_scaling.py results.json

# Heatmap across agents
uv run python examples/plotting/heatmap.py agent1.json agent2.json

# LaTeX table for a paper
uv run python examples/plotting/latex_tables.py results.json --output table.tex
```
