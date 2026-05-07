# Experiments

Scripts and configs for reproducible benchmark runs.

## Prerequisites

```bash
uv sync
```

Install extras for the agent type you plan to run, for example
`uv sync --extra llm`, `uv sync --extra rl`, or `uv sync --extra all`.

## Scripts

- `run_predefined.py` loads a YAML config and runs it with `ExperimentRunner`.
- `run_single_benchmark.py` is a small CLI wrapper around a single config.
- `train_and_eval_ppo.py` trains and evaluates a PPO baseline.

## Configs

`examples/experiments/configs/` contains ready-made configs for:

- random and oracle baselines
- PPO pixel baselines
- OpenAI, Gemini, and open-weight model evaluations
- Qwen SFT model evaluation and training recipes

## Running

```bash
uv run python -m agentick.experiments.run \
    --config examples/experiments/configs/random_agent.yaml

uv run python examples/experiments/run_predefined.py \
    --config examples/experiments/configs/random_agent.yaml
```

## SLURM

The `slurm/` directory contains a generic job launcher and resource profile
template. Adjust `slurm/profiles.yaml` for your cluster partitions before
submitting jobs.

```bash
python examples/experiments/slurm/launch.py --dry-run
python examples/experiments/slurm/launch.py --configs "oracle_agent" --partition cpu
```

Generated `results/` and `slurm_logs/` directories are ignored by git.
