# Visualization

Matplotlib and Plotly visualization toolkit for generating publication-quality figures, tables, and reports from experiment results.

## Modules

### `style.py`
Plot styling and color configuration.
- `COLORBLIND_PALETTE` -- Wong 2011 colorblind-safe 8-color palette
- `AGENT_COLORS`, `AGENT_MARKERS`, `AGENT_LINESTYLES` -- consistent per-agent visual mappings across all figures
- `PAPER_STYLES` -- matplotlib rcParams presets for single-column and double-column paper figures
- `set_style(style)`, `save_figure(fig, path)`, `get_agent_color(name)`, `get_agent_marker(name)`, `get_agent_linestyle(name)`, `get_palette(n)`

### `plots.py`
Core plotting functions.
- `plot_bar_comparison(results_dict, metric, output_path, style)` -- grouped bar chart comparing agents across tasks

### `training_plots.py`
Training benchmark diagnostics.
- `TrainingBenchmarkPlotter` -- generates success rate heatmaps (tasks x difficulties), learning curves by category, difficulty scaling plots, capability radar charts, per-task final scores, and reward distributions

### `experiment_plots.py`
Per-experiment result visualization.
- `ExperimentPlotter(result_dir)` -- loads results from a directory and generates all diagnostic plots via `plot_all()`

### `comparison_plots.py`
Cross-agent comparison.
- `AgentComparisonPlotter(result_dirs, output_dir)` -- loads multiple experiment result directories and generates comparative plots via `plot_all()`

### `tables.py`
Publication-ready table generation.
- `generate_main_results_table(results_dict, output_path, formats)` -- produces results tables in LaTeX (with booktabs), Markdown, and CSV formats

### `report.py`
Automated report generation.
- `generate_report(results_dict, output_dir, format)` -- generates a complete results section in LaTeX or Markdown
- `generate_supplementary(results_dict, output_dir)` -- generates supplementary material

### `interactive.py`
Interactive HTML dashboards.
- `generate_dashboard(results_dict, output_dir, include_videos, include_trajectories)` -- creates a Plotly-based HTML dashboard with embedded videos and trajectory replay

### `video.py`
Video generation from saved trajectories.
- `render_episode_video(trajectory, task, output_path, fps, resolution, overlay, codec)` -- re-renders a saved trajectory as a video with optional HUD overlay showing step count, reward, and action info

## Usage

```python
from agentick.visualization import AgentComparisonPlotter
from pathlib import Path

plotter = AgentComparisonPlotter([
    Path("results/ppo_20260101/"),
    Path("results/claude_20260101/"),
])
plotter.plot_all()
```
