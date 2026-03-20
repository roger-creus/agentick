# Experiments

YAML-driven experiment orchestration: configure agents, tasks, seeds, and metrics in one file, then run reproducible evaluations or RL training loops.

## CLI Entry Point

```bash
python -m agentick.experiments.run --config path/to/config.yaml
python -m agentick.experiments.run --config config.yaml --resume results/prev_run/
python -m agentick.experiments.run --config config.yaml --n-parallel 4
```

`run.py` parses `--config`, `--resume`, `--n-parallel`, and `--output-dir`, then delegates to `ExperimentRunner`.

## Key Classes and Functions

### `config.py`

- **`ExperimentConfig`** -- Pydantic model for the full experiment spec: `name`, `agent` (AgentConfig), `tasks` (list or suite name), `difficulties`, `n_episodes`, `n_seeds`, `seeds`, `render_modes`, `reward_mode`, `record_trajectories`, `record_videos`, `metrics`, `output_dir`, `tags`, and optional `training` config. Supports `base_config` inheritance.
- **`AgentConfig`** -- Agent type string plus freeform `hyperparameters` dict.
- **`TrainingConfig`** -- PPO/RL training parameters: `total_timesteps`, `n_envs`, `eval_frequency`, `n_eval_episodes`, `checkpoint_frequency`, `device`.
- **`load_config(path)`** -- Load and validate a YAML config file.

### `runner.py`

- **`ExperimentRunner`** -- Main evaluation driver. Creates environments via `agentick.make()`, runs episodes per task/difficulty/seed, records trajectories, and computes metrics. Supports parallel task execution through `_run_task_worker()` multiprocessing. Uses `rich.progress` for progress display.
- **`ExperimentResults`** -- Container for per-task results, metadata, and output paths.
- **`run_experiment(config)`** -- Convenience function wrapping `ExperimentRunner`.

### `training_runner.py`

- **`TrainingBenchmarkRunner`** -- Runs PPO pixel-based RL training across all tasks. Contains a static `TASK_CATEGORIES` mapping from task names to category strings (navigation, memory, reasoning, etc.). Uses `rich.progress` for training progress.

### `reproduce.py`

- **`reproduce(results_dir)`** -- Reload `config.yaml` from a saved results directory and re-run the experiment with identical settings.
- **`diff_experiments(dir_a, dir_b)`** -- Compare metrics between two experiment runs.

### `registry.py`

- **`ExperimentRegistry`** -- Manages saved experiment results under a `base_dir`. Methods: `list_experiments(tag=None)` to enumerate runs filtered by tag, plus load/compare utilities that read `metadata.json` and `config.yaml` from each run directory.

## Lazy Import Pattern

`__init__.py` uses `__getattr__` to defer imports of `BaseAgent` and `create_agent` from `agentick.agents`. This breaks a circular dependency: `agents/` imports experiment configs, and `experiments/` needs agent classes.

```python
def __getattr__(name):
    if name in ("BaseAgent", "create_agent"):
        from agentick.agents import BaseAgent, create_agent
        ...
```

## Output Structure

```
results/<experiment_name>/
  config.yaml          # Frozen experiment config
  metadata.json        # Timestamps, git hash, etc.
  per_task/
    <TaskName>/
      episodes/        # Recorded trajectories (JSON)
      metrics.json     # Per-difficulty aggregate metrics
```
