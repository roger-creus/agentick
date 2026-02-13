# Agentick Examples

Complete collection of runnable examples demonstrating all features of Agentick.

## Quick Start

All examples can be run with:
```bash
uv run python examples/path/to/example.py
```

## Directory Structure

### `basics/` - Getting Started (5 examples)
Simple introductory examples for new users.

- **01_make_and_step.py**: Simplest possible - make, reset, step, render
- **02_render_modes.py**: Same task in all observation modes
- **03_reward_modes.py**: Dense vs sparse rewards
- **04_list_tasks.py**: List all tasks, filter by capability
- **05_difficulty_levels.py**: Same task at easy/medium/hard/expert

### `rl/` - Reinforcement Learning (5 examples)
Complete RL training examples with logging and evaluation.

- **ppo_cleanrl.py**: Full PPO with wandb, eval, video, checkpoints
- **dqn_cleanrl.py**: Full DQN with same features
- **sb3_ppo.py**: Stable Baselines3 PPO integration
- **sb3_dqn.py**: Stable Baselines3 DQN integration
- **curriculum_training.py**: RL with adaptive curriculum

### `llm/` - LLM Agents (6 examples)
Using LLM/VLM agents with Agentick.

- **openai_text_agent.py**: GPT-4o with text observations
- **openai_vision_agent.py**: GPT-4o with pixel observations
- **anthropic_text_agent.py**: Claude with text observations
- **anthropic_vision_agent.py**: Claude with pixel observations
- **huggingface_local_agent.py**: Local HF model (e.g., Llama-3)
- **compare_llms.py**: Run multiple LLMs, compare results

### `experiments/` - Experiment System (5 examples)
Using the built-in experiment runner.

- **run_predefined.py**: Run a predefined experiment config
- **run_custom.py**: Create and run custom experiment
- **compare_experiments.py**: Load two experiments, compare
- **generate_plots.py**: Generate all plots from results
- **generate_paper_figures.py**: Publication-ready figures

### `data_and_finetuning/` - Data Collection & Fine-tuning (6 examples)
Collecting trajectories and fine-tuning models.

- **collect_oracle_trajectories.py**: Collect oracle demonstrations
- **collect_random_trajectories.py**: Collect random trajectories
- **collect_trajectories_finetune.py**: Collect trajectories for fine-tuning
- **export_to_huggingface.py**: Export to HF Datasets format
- **sft_with_trl.py**: Full SFT pipeline (collect→export→train→eval)
- **record_videos.py**: Record MP4 episode videos

### `leaderboard/` - Leaderboard Usage (5 examples)
Creating and evaluating leaderboard submissions.

- **create_submission.py**: Create submission.yaml
- **validate_submission.py**: Validate submission
- **run_evaluation.py**: Run leaderboard eval locally
- **view_results.py**: Display evaluation results
- **compare_agents.py**: Compare multiple entries

### `plotting/` - Visualization (6 examples)
Standalone plotting examples.

- **capability_radar.py**: Generate capability radar chart
- **learning_curves.py**: Plot learning curves from logs
- **bar_comparison.py**: Agent comparison bar chart
- **difficulty_scaling.py**: Performance vs difficulty
- **heatmap.py**: Task × Agent heatmap
- **latex_tables.py**: Generate LaTeX tables

### `notebooks/` - Jupyter Notebooks (5 notebooks)
Interactive analysis and exploration.

- **01_getting_started.ipynb**: Interactive intro
- **02_analyze_experiment.ipynb**: Explore experiment data
- **03_compare_agents.ipynb**: Multi-agent comparison
- **04_leaderboard_analysis.ipynb**: Leaderboard data analysis
- **05_custom_task.ipynb**: Build custom task interactively

### `advanced/` - Advanced Topics (4 examples)
Advanced customization and usage patterns.

- **custom_task.py**: Create custom task from scratch
- **custom_reward.py**: Create custom reward function
- **human_play.py**: Launch human play interface
- **parallel_eval.py**: Evaluate agent on many tasks in parallel

## Requirements

Different examples require different dependencies:

```bash
# For RL examples
uv sync --extra rl

# For LLM examples (requires API keys)
uv sync --extra llm
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# For all examples
uv sync --extra all
```

## Contributing

When adding new examples:
1. Add module docstring explaining what it does
2. Add `if __name__ == "__main__"` guard
3. Make it self-contained (no dependencies on other examples)
4. Add runtime estimate in docstring if >1 minute
5. Check for missing API keys and print clear error message
6. Update this README
