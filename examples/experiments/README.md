# examples/experiments/

This directory contains **example scripts** demonstrating how to use the agentick experiment system.

## What's here

- `run_predefined.py`: Run a predefined experiment config
- `run_custom.py`: Create and run a custom experiment
- `compare_experiments.py`: Compare results from multiple experiments
- `generate_plots.py`: Generate visualizations from experiment results
- `generate_paper_figures.py`: Generate publication-ready figures
- `configs/`: Example experiment configuration files

These are **example scripts** showing how to use the library, and should be version-controlled.

## Notes

- `results/`: Generated experiment outputs (not committed to git)

## Usage

```bash
# Run a predefined experiment
uv run python examples/experiments/run_predefined.py

# Run a custom experiment
uv run python examples/experiments/run_custom.py

# Compare multiple experiments
uv run python examples/experiments/compare_experiments.py
```
