# Experiments

Scripts for configuring, running, comparing, and visualizing benchmark
experiments.

## Prerequisites

```bash
uv sync --extra all   # experiment runner, plotting, yaml support
```

## Scripts

- **run_predefined.py** -- Load an experiment configuration from a YAML file
  and run it through `ExperimentRunner`. Accepts `--config` and `--output-dir`
  arguments. Prints reward, step, and success-rate statistics on completion.
- **run_custom.py** -- Build an `ExperimentConfig` programmatically in Python
  (choosing tasks, agent type, seeds, episodes) and run it. Useful for quick
  one-off evaluations without writing YAML.
- **compare_experiments.py** -- Load two or more result directories, compute
  summary metrics (mean reward, success rate), and display a side-by-side
  comparison table with the best performer highlighted.
- **generate_plots.py** -- Generate four standard plots from an experiment
  result directory: reward distribution histogram, success rate by task,
  steps-vs-reward scatter, and learning curve. Requires matplotlib.
- **generate_paper_figures.py** -- Create publication-quality PDF and PNG
  figures with high DPI, serif fonts, and error bars. Produces a performance
  comparison bar chart and a capability radar chart.

Also includes a `configs/` directory with example experiment YAML files.

## Running

```bash
# Run a predefined config
uv run python examples/experiments/run_predefined.py \
    --config examples/experiments/configs/quick/sanity_check.yaml

# Run a custom experiment
uv run python examples/experiments/run_custom.py

# Compare two result directories
uv run python examples/experiments/compare_experiments.py results/exp1 results/exp2

# Generate plots from results
uv run python examples/experiments/generate_plots.py results/my_experiment

# Generate publication figures
uv run python examples/experiments/generate_paper_figures.py results/my_experiment
```

## Notes

- `results/`: Generated experiment outputs (not committed to git).

## Typical Workflow

1. Run an experiment with `run_predefined.py` or `run_custom.py`.
2. Visualize results with `generate_plots.py`.
3. Compare runs with `compare_experiments.py`.
4. Export publication figures with `generate_paper_figures.py`.
