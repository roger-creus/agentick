# Visualization

Generate publication-ready figures and tables from experiment results.

## Quick Start with ExperimentPlotter

The `ExperimentPlotter` class provides a simple interface for visualizing experiment results:

```python
from pathlib import Path
import matplotlib.pyplot as plt
from agentick.visualization.experiment_plots import ExperimentPlotter

# Load experiment results
plotter = ExperimentPlotter(Path("results/my_experiment/"))

# 1. Learning curves
fig = plotter.plot_learning_curves(
    metrics=["reward", "success_rate"],
    smoothing=10,
)
fig.savefig("figures/learning_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 2. Final performance by task
fig = plotter.plot_final_performance(metric="success_rate")
fig.savefig("figures/performance.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 3. Difficulty scaling
fig = plotter.plot_difficulty_scaling(
    tasks=["GoToGoal-v0", "MazeNavigation-v0"],
    metric="reward",
)
fig.savefig("figures/difficulty.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 4. Episode length distribution
fig = plotter.plot_episode_lengths()
fig.savefig("figures/lengths.png", dpi=150, bbox_inches="tight")
plt.close(fig)
```

**Complete Example**: See `examples/plotting/plot_experiment_results.py` for a full working example.

## Individual Plot Examples

For custom visualizations beyond ExperimentPlotter, see the examples below.

## Plotting Styles

### Available Styles

```python
from agentick.visualization.style import set_style

# Academic publication (default)
set_style("paper_double_column")  # Two-column paper format
set_style("paper_single_column")   # Single-column paper format

# Presentations
set_style("poster")        # Large poster format
set_style("presentation")  # Slide presentation format

# Interactive
set_style("notebook")      # Jupyter notebook style
```

Each style automatically sets:
- Font sizes
- Line widths
- Figure dimensions
- Color palettes

## Bar Charts

### Basic Agent Comparison

```python
from agentick.visualization.plots import plot_bar_comparison

results_dict = {
    "PPO": {
        "GoToGoal-v0": {"mean_return": 0.95, "ci_lower": 0.90, "ci_upper": 1.00},
        "Maze-v0": {"mean_return": 0.85, "ci_lower": 0.80, "ci_upper": 0.90},
        "MemoryPath-v0": {"mean_return": 0.75, "ci_lower": 0.68, "ci_upper": 0.82},
    },
    "DQN": {
        "GoToGoal-v0": {"mean_return": 0.88, "ci_lower": 0.82, "ci_upper": 0.94},
        "Maze-v0": {"mean_return": 0.92, "ci_lower": 0.88, "ci_upper": 0.96},
        "MemoryPath-v0": {"mean_return": 0.82, "ci_lower": 0.75, "ci_upper": 0.89},
    },
}

fig = plot_bar_comparison(
    results_dict,
    metric="mean_return",
    output_path="figures/agent_comparison",
    style="paper_double_column",
)
```

## Learning Curves

### Single Agent Learning Curve

```python
from agentick.visualization.plots import plot_learning_curves

episode_returns = [
    [0.1, 0.2, 0.3, 0.45, 0.55, 0.65, 0.75, 0.80, 0.85, 0.88],  # Seed 0
    [0.05, 0.15, 0.25, 0.40, 0.50, 0.60, 0.70, 0.78, 0.82, 0.85],  # Seed 1
    [0.12, 0.22, 0.35, 0.50, 0.58, 0.68, 0.76, 0.82, 0.86, 0.90],  # Seed 2
]

fig = plot_learning_curves(
    {"PPO": episode_returns},
    output_path="figures/learning_curve",
    window_size=2,  # Smoothing
)
```

### Multiple Agent Comparison

```python
ppo_curves = [...] # Episode returns for PPO
dqn_curves = [...] # Episode returns for DQN

fig = plot_learning_curves(
    {
        "PPO": ppo_curves,
        "DQN": dqn_curves,
        "Random": random_curves,
    },
    output_path="figures/learning_curve_comparison",
    window_size=10,
    ci=0.95,
)
```

### Learning Curve Options

```python
fig = plot_learning_curves(
    results_dict,
    metric="mean_return",  # Metric to plot
    window_size=50,        # Smoothing window
    ci=0.95,              # Confidence interval
    x_label="Episode",    # X-axis label
    y_label="Return",     # Y-axis label
    output_path=None,     # Path to save
    style="paper_double_column",
)
```

## Radar Charts (Capability Profiles)

### Radar Chart for Capability Profiles

```python
from agentick.visualization.plots import plot_capability_radar

results_dict = {
    "PPO": {
        "navigation": 0.88,
        "memory": 0.75,
        "reasoning": 0.82,
        "control": 0.90,
        "skill": 0.80,
    },
    "DQN": {
        "navigation": 0.92,
        "memory": 0.68,
        "reasoning": 0.75,
        "control": 0.88,
        "skill": 0.78,
    },
}

fig = plot_capability_radar(
    results_dict,
    output_path="figures/capability_profiles",
    style="paper_double_column",
)
```

## Heatmaps

### Success Rate Heatmap

```python
from agentick.visualization.plots import plot_success_heatmap

# Organize by agent and task
data = {
    "agent": ["PPO", "PPO", "PPO", "DQN", "DQN", "DQN"],
    "task": ["GoToGoal", "Maze", "Memory", "GoToGoal", "Maze", "Memory"],
    "easy": [0.95, 0.85, 0.75, 0.88, 0.92, 0.82],
    "medium": [0.85, 0.75, 0.60, 0.82, 0.88, 0.70],
    "hard": [0.70, 0.60, 0.45, 0.75, 0.80, 0.55],
}

fig = plot_success_heatmap(data, output_path="figures/success_heatmap")
```

## Distribution Plots

### Return Distribution

```python
from agentick.visualization.plots import plot_return_distribution

distributions = {
    "PPO": [0.95, 0.92, 0.98, 0.91, 0.93],  # Returns per seed
    "DQN": [0.88, 0.90, 0.92, 0.85, 0.89],
    "Random": [0.10, 0.12, 0.08, 0.15, 0.11],
}

fig = plot_return_distribution(
    distributions,
    output_path="figures/return_distribution",
)
```

### Success Rate Distribution

```python
from agentick.visualization.plots import plot_success_distribution

success_rates = {
    "PPO": [0.95, 0.98, 0.92, 0.96],
    "DQN": [0.88, 0.92, 0.85, 0.90],
}

fig = plot_success_distribution(
    success_rates,
    output_path="figures/success_distribution",
)
```

## Tables

### Main Results Table (LaTeX)

```python
from agentick.visualization.tables import generate_main_results_table

results_dict = {
    "PPO": {
        "navigation": {"mean": 0.88, "ci_lower": 0.82, "ci_upper": 0.94},
        "memory": {"mean": 0.75, "ci_lower": 0.68, "ci_upper": 0.82},
        "reasoning": {"mean": 0.82, "ci_lower": 0.76, "ci_upper": 0.88},
    },
    "DQN": {
        "navigation": {"mean": 0.92, "ci_lower": 0.88, "ci_upper": 0.96},
        "memory": {"mean": 0.68, "ci_lower": 0.60, "ci_upper": 0.76},
        "reasoning": {"mean": 0.75, "ci_lower": 0.68, "ci_upper": 0.82},
    },
}

latex_table = generate_main_results_table(
    results_dict,
    output_path="tables/main_results",
    formats=["latex", "markdown", "csv"],
)

print(latex_table)
# Output: LaTeX table with booktabs formatting, best results bolded
```

### Results Table Markdown

```python
# Tables automatically exported in multiple formats
# Check generated files:
# - tables/main_results.tex  (LaTeX)
# - tables/main_results.md   (Markdown)
# - tables/main_results.csv  (CSV)

# Load and display markdown
with open("tables/main_results.md") as f:
    print(f.read())
```

### Comparison Table

```python
from agentick.visualization.tables import generate_comparison_table

# Create detailed comparison
comparison_data = {
    "Metric": ["Mean Return", "Success Rate", "Convergence (ep)", "Sample Efficiency"],
    "PPO": [0.82, 0.88, 150, 0.92],
    "DQN": [0.79, 0.85, 200, 0.88],
    "Random": [0.15, 0.10, None, 0.20],
}

table = generate_comparison_table(
    comparison_data,
    output_path="tables/comparison",
    formats=["latex", "html"],
)
```

## Video Generation

### Trajectory Videos

```python
from agentick.visualization.video import generate_trajectory_video

# Load episode data
episode_data = {
    "observations": [...],  # Rendered observations
    "actions": [...],
    "rewards": [...],
}

# Generate video
generate_trajectory_video(
    episode_data,
    output_path="videos/episode.mp4",
    fps=10,
)
```

### Episode Montage

```python
from agentick.visualization.video import create_episode_montage

# Create grid of multiple episodes
create_episode_montage(
    episodes=[episode_1, episode_2, episode_3, episode_4],
    output_path="videos/montage.mp4",
    grid_size=(2, 2),
    fps=10,
)
```

## Interactive Dashboards

### Generate Interactive Report

```python
from agentick.visualization.report import generate_html_report

# Create interactive dashboard
generate_html_report(
    results={
        "ppo": ppo_results,
        "dqn": dqn_results,
    },
    output_path="report.html",
    include_plots=True,
    include_tables=True,
    interactive=True,
)
```

### Plotly Interactive Plots

```python
from agentick.visualization.interactive import plot_interactive_learning_curves

fig = plot_interactive_learning_curves(
    {
        "PPO": episode_returns_ppo,
        "DQN": episode_returns_dqn,
    },
    output_path="plots/learning_curves.html",
)
```

## Customization

### Color Schemes

```python
from agentick.visualization.style import get_agent_color, get_color_palette

# Get color for agent
ppo_color = get_agent_color("PPO")
dqn_color = get_agent_color("DQN")

# Get full palette
palette = get_color_palette("agents")
print(palette)
```

### Font and Size Customization

```python
import matplotlib.pyplot as plt

# Customize for your needs
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['figure.figsize'] = (8, 6)

fig = plot_learning_curves(...)
```

### Custom Markers and Styles

```python
from agentick.visualization.style import get_agent_marker, get_agent_linestyle

# Per-agent styling
marker = get_agent_marker("PPO")  # Circle, square, etc.
linestyle = get_agent_linestyle("PPO")  # Solid, dashed, etc.
```

## Complete Visualization Example

```python
import numpy as np
from agentick.experiments.registry import ExperimentRegistry
from agentick.visualization.plots import (
    plot_bar_comparison,
    plot_learning_curves,
    plot_capability_radar,
)
from agentick.visualization.tables import generate_main_results_table
from agentick.visualization.style import set_style

# Load experiments
registry = ExperimentRegistry("results")
ppo_exp = registry.load_latest("ppo_benchmark")
dqn_exp = registry.load_latest("dqn_benchmark")

# Set style for publications
set_style("paper_double_column")

# 1. Bar chart comparison
results_for_bars = {}
for agent_name, exp in {"PPO": ppo_exp, "DQN": dqn_exp}.items():
    results_for_bars[agent_name] = {}
    for task, task_results in exp.per_task_results.items():
        agg = task_results['aggregate_metrics']
        results_for_bars[agent_name][task] = {
            "mean_return": agg.get('mean_return', 0),
            "ci_lower": agg.get('mean_return', 0) - agg.get('std_return', 0) / 2,
            "ci_upper": agg.get('mean_return', 0) + agg.get('std_return', 0) / 2,
        }

fig = plot_bar_comparison(results_for_bars, output_path="figures/main_results")

# 2. Learning curves
ppo_curves = []
for task, task_results in ppo_exp.per_task_results.items():
    episodes = []
    for diff, diff_results in task_results['per_difficulty'].items():
        episodes.extend(diff_results['episodes'])
    returns = [ep['return'] for ep in episodes]
    ppo_curves.append(returns)

fig = plot_learning_curves(
    {"PPO": ppo_curves},
    output_path="figures/learning_curve",
)

# 3. Capability profile radar
capability_scores = {
    "PPO": {
        "navigation": 0.88,
        "memory": 0.75,
        "reasoning": 0.82,
    },
    "DQN": {
        "navigation": 0.92,
        "memory": 0.68,
        "reasoning": 0.75,
    },
}

fig = plot_capability_radar(capability_scores, output_path="figures/profiles")

# 4. Results table
table_results = {}
for agent_name, exp in {"PPO": ppo_exp, "DQN": dqn_exp}.items():
    table_results[agent_name] = {}
    for task, task_results in exp.per_task_results.items():
        agg = task_results['aggregate_metrics']
        table_results[agent_name][task] = {
            "mean": agg.get('mean_return', 0),
            "ci_lower": agg.get('mean_return', 0) - 0.05,
            "ci_upper": agg.get('mean_return', 0) + 0.05,
        }

latex = generate_main_results_table(
    table_results,
    output_path="tables/main_results",
)

print("Visualization complete!")
print("  Figures: figures/")
print("  Tables: tables/")
```

## Saving Figures

### High-Quality PDF Export

```python
import matplotlib.pyplot as plt

fig = plot_learning_curves(...)

# Save as PDF (publication quality)
fig.savefig("figures/learning_curves.pdf", dpi=300, bbox_inches="tight")

# Also save as PNG for web
fig.savefig("figures/learning_curves.png", dpi=150, bbox_inches="tight")
```

### Figure Format Options

```python
fig.savefig("output.pdf", dpi=300)      # PDF (best for print)
fig.savefig("output.png", dpi=150)      # PNG (good for web)
fig.savefig("output.svg", format="svg") # SVG (scalable)
fig.savefig("output.eps", format="eps") # EPS (for some journals)
```

## Full Benchmark Pipeline Example

For a comprehensive example of generating all plots and reports, see the full benchmark pipeline:

**Scripts:**
- `examples/experiments/full_benchmark/plot_all_results.py` - Generate all plot types from multiple experiments
- `examples/experiments/full_benchmark/generate_report.py` - Create HTML report with embedded plots

**Notebooks:**
- `examples/notebooks/02_analyze_experiment.ipynb` - Interactive analysis of experiment results

**Plot Types Included:**
- **Comparison Bar Chart**: Compare average performance across agents
- **Per-Task Comparison**: Performance breakdown by task
- **Success Rate Comparison**: Success rates across agents with color-coded bars
- **Learning Curves**: Training progress over time with confidence intervals
- **Capability Radar Charts**: Multi-dimensional agent capability profiles
- **Heatmaps**: Task difficulty × agent performance matrices
- **Distribution Plots**: Violin/box plots showing performance variance

## Common Issues and Solutions

### Overlapping X-axis Labels
```python
fig.autofmt_xdate(rotation=45, ha='right')
plt.tight_layout()
```

### Legend Placement
```python
# Move legend outside plot
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
```

### Font Issues in PDF
```python
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42  # Use TrueType fonts
```

### Memory Issues with Large Figures
```python
plt.rcParams['figure.max_open_warning'] = 50
# Close figures after saving
plt.close('all')
```

## Additional Resources

**Example Scripts:**
- `examples/plotting/` - Individual plot type examples
  - `capability_radar.py` - Capability radar charts
  - `learning_curves.py` - Training progress plots
  - `bar_comparison.py` - Agent comparison bars
  - `difficulty_scaling.py` - Difficulty scaling curves
  - `heatmap.py` - Performance heatmaps
  - `latex_tables.py` - LaTeX table generation

**Jupyter Notebooks:**
- `examples/notebooks/02_analyze_experiment.ipynb` - Interactive experiment analysis
- `examples/notebooks/03_compare_agents.ipynb` - Multi-agent comparison
- `examples/notebooks/04_leaderboard_analysis.ipynb` - Leaderboard visualization

**Full Benchmark Pipeline:**
- `examples/experiments/full_benchmark/README.md` - Complete end-to-end benchmark with all plot types
