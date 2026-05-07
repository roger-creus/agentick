# Examples

Runnable examples for Agentick. Start with `basics/`, then move to the agent
type or workflow you want to test.

## Directory Overview

```
examples/
├── basics/               # Quick-start scripts
├── rl/                   # RL training with CleanRL-style and SB3 scripts
├── llm/                  # OpenAI and local HuggingFace text agents
├── data_and_finetuning/  # Oracle data collection, SFT, and adapter merge
├── experiments/          # Benchmark experiment configs and runners
└── leaderboard/          # Leaderboard submission helpers
```

## Basics

| Script | What it shows |
|---|---|
| `01_make_and_step.py` | Create env, reset, step loop |
| `02_render_modes.py` | Observation modes: ASCII, language, pixels, state |
| `03_reward_modes.py` | Sparse vs dense rewards |
| `04_list_tasks.py` | Browse tasks by capability tag |
| `05_difficulty_levels.py` | Easy through expert difficulty scaling |

```bash
uv run python examples/basics/01_make_and_step.py
```

## RL

```bash
uv sync --extra rl
uv run python examples/rl/sb3_ppo.py
uv run python examples/rl/ppo_cleanrl.py --task-id MazeNavigation-v0 --difficulty medium
```

Available scripts: `ppo_cleanrl.py`, `dqn_cleanrl.py`, `sb3_ppo.py`, and
`sb3_dqn.py`.

## LLM

```bash
uv sync --extra llm
export OPENAI_API_KEY="your-openai-api-key"
uv run python examples/llm/openai_text_agent.py
uv run python examples/llm/huggingface_local_agent.py
```

For additional providers, use the YAML config system documented in
`docs/agents/llm_agents.md`.

## Data And Fine-Tuning

```bash
uv sync --extra finetune
uv run python examples/data_and_finetuning/collect_oracle_trajectories.py --output-dir trajectories/oracle
uv run python examples/data_and_finetuning/sft_with_trl.py --dataset rogercc/agentick-oracle-trajectories-120k
```

Available scripts: `collect_oracle_trajectories.py`,
`collect_random_trajectories.py`, `sft_with_trl.py`, `merge_and_push.py`, and
`run_sft_from_config.sh`.

## Experiments

```bash
uv run python -m agentick.experiments.run --config examples/experiments/configs/random_agent.yaml
uv run python examples/experiments/run_predefined.py --config examples/experiments/configs/random_agent.yaml
```

The `examples/experiments/configs/` directory contains ready-made random,
oracle, RL, API-model, open-weight model, and SFT evaluation configs.

## Leaderboard

```bash
uv run python examples/leaderboard/create_submission.py
uv run python scripts/validate_submission.py results/<your_run>/
```
