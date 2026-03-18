# Examples

Runnable examples for Agentick. Start with `basics/` and explore from there.

## Directory Overview

```
examples/
├── basics/               # Quick-start scripts (start here)
├── rl/                   # RL training (PPO, DQN, SB3)
├── llm/                  # LLM/VLM agents (OpenAI, Anthropic, HuggingFace)
├── data_and_finetuning/  # Data collection, SFT, behavior cloning
├── experiments/          # Benchmark experiment configs and SLURM launcher
├── leaderboard/          # Leaderboard submission and evaluation
└── debug_results.ipynb   # Analyze benchmark results (Jupyter)
```

## basics/

Five intro scripts covering the full API:

| Script | What it shows |
|--------|---------------|
| `01_make_and_step.py` | Create env, reset, step loop |
| `02_render_modes.py` | All observation modes (ascii, language, pixels) |
| `03_reward_modes.py` | Sparse vs dense rewards |
| `04_list_tasks.py` | Browse tasks by capability tag |
| `05_difficulty_levels.py` | Easy → expert difficulty scaling |

```bash
uv run python examples/basics/01_make_and_step.py
```

## rl/

RL training examples using CleanRL and Stable Baselines3:

| Script | Description |
|--------|-------------|
| `ppo_cleanrl.py` | CleanRL-style PPO with Nature CNN, TensorBoard logging |
| `dqn_cleanrl.py` | CleanRL-style DQN with replay buffer, target network |
| `sb3_ppo.py` | SB3 PPO with CnnPolicy, wandb, checkpointing |
| `sb3_dqn.py` | SB3 DQN with CnnPolicy |

All pixel-based RL uses `render_mode="rgb_array"` (isometric 512x512).

```bash
uv sync --extra rl
uv run python examples/rl/sb3_ppo.py
```

## llm/

LLM and VLM agent examples:

| Script | Description |
|--------|-------------|
| `openai_text_agent.py` | GPT-4o with text observations |
| `openai_vision_agent.py` | GPT-4o with pixel observations |
| `openai_cot_agent.py` | GPT-4o with chain-of-thought |
| `anthropic_text_agent.py` | Claude with text observations |
| `anthropic_vision_agent.py` | Claude with pixel observations |
| `huggingface_local_agent.py` | Local Qwen/Llama model |
| `compare_llms.py` | Side-by-side multi-provider comparison |

```bash
uv sync --extra llm
export OPENAI_API_KEY="your-key"
uv run python examples/llm/openai_text_agent.py
```

## data_and_finetuning/

Data collection and fine-tuning:

| Script | Description |
|--------|-------------|
| `collect_oracle_trajectories.py` | Oracle demonstrations across all tasks |
| `collect_random_trajectories.py` | Random-policy baseline data |
| `export_to_huggingface.py` | Export to HF Datasets format |
| `sft_with_trl.py` | Full SFT pipeline (TRL + LoRA) |
| `behavior_cloning_training.py` | Nature CNN from pixels |
| `tinker_sft_training.py` | Remote LoRA SFT via Tinker |
| `tinker_rl_training.py` | Remote RL via Tinker |

```bash
uv sync --extra finetune
uv run python examples/data_and_finetuning/collect_oracle_trajectories.py
```

## experiments/

Run and manage benchmark experiments:

| File | Description |
|------|-------------|
| `run_predefined.py` | Run a single YAML config with ExperimentRunner |
| `run_single_benchmark.py` | CLI wrapper for benchmark runs |
| `configs/` | 48 pre-built YAML configs (random, oracle, PPO, GPT-4o, Claude, Qwen) |
| `slurm/launch.py` | SLURM job launcher (one job per task) |
| `slurm/profiles.yaml` | Cluster resource profiles |
| `slurm/job_template.sh` | SBATCH script template |

```bash
# Run a quick baseline
uv run python examples/experiments/run_single_benchmark.py \
    examples/experiments/configs/random_agent.yaml

# Run on SLURM cluster
python examples/experiments/slurm/launch.py --dry-run
python examples/experiments/slurm/launch.py --configs "oracle_agent" --partition gpu
```

## leaderboard/

Tools for submitting to the public leaderboard:

| Script | Description |
|--------|-------------|
| `create_submission.py` | Generate submission YAML template |
| `validate_submission.py` | Validate YAML before submitting |
| `run_evaluation.py` | Run full leaderboard eval locally |
| `view_results.py` | Display formatted results |
| `compare_agents.py` | Compare two or more result files |

## debug_results.ipynb

Jupyter notebook for analyzing benchmark results. Loads results from
`results/` subdirectories and produces:

- Global leaderboard table
- Task × Agent heatmap
- Difficulty scaling charts
- Per-config deep-dives

```bash
uv sync --extra viz
jupyter notebook examples/debug_results.ipynb
```
