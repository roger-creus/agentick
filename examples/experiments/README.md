# Experiments

Scripts and configs for running reproducible benchmark experiments.

## Prerequisites

```bash
uv sync --extra all   # experiment runner, yaml support
```

## Scripts

- **run_predefined.py** -- Load an experiment configuration from a YAML file
  and run it through `ExperimentRunner`. Accepts `--config` and `--output-dir`
  arguments. Prints reward, step, and success-rate statistics on completion.
- **run_single_benchmark.py** -- CLI wrapper that runs a single benchmark
  config and saves results to a JSON file. Accepts `--config`, `--output`,
  and `--suite` arguments.

## configs/

46 pre-built YAML experiment configs covering:

| Group | Configs |
|-------|---------|
| Random baseline | `random_agent.yaml` |
| Oracle baseline | `oracle_agent.yaml` |
| PPO (pixel RL) | `ppo_pixels_dense.yaml`, `ppo_pixels_sparse.yaml` |
| OpenAI GPT-4o | `gpt4o_ascii.yaml`, `gpt4o_language.yaml`, `gpt4o_vision.yaml` |
| Anthropic Claude | `claude_sonnet_ascii.yaml`, `claude_sonnet_language.yaml`, `claude_sonnet_vision.yaml` |
| Qwen3 (30B-A3B) | 8 configs (ascii/lang × markov/nonmarkov/reasoner variants) |
| Qwen3 (4B) | 8 configs |
| Qwen3-VL (4B, 8B) | 16 configs |

## slurm/

SLURM job launcher for running experiments on a compute cluster:

- **launch.py** -- Submits one SLURM job per task in a config. Supports `--dry-run`,
  `--configs` glob filter, and `--partition` selection.
- **profiles.yaml** -- Cluster resource profiles (CPU/GPU, memory, time limits).
- **job_template.sh** -- SBATCH script template used by `launch.py`.

## Running

```bash
# Run a quick baseline
uv run python examples/experiments/run_predefined.py \
    --config examples/experiments/configs/random_agent.yaml

# Run a single benchmark with output
uv run python examples/experiments/run_single_benchmark.py \
    examples/experiments/configs/oracle_agent.yaml

# Run on SLURM cluster (dry run first)
python examples/experiments/slurm/launch.py --dry-run
python examples/experiments/slurm/launch.py --configs "oracle_agent" --partition gpu
```

## Notes

- `results/`: Generated experiment outputs (not committed to git).
- Analyze results with `examples/debug_results.ipynb`.
