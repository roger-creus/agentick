# Evaluation Seeds

Agentick uses deterministic, per-task-difficulty seed generation for reproducible benchmarking.

## Architecture

Seeds are generated per `(task_name, difficulty, split)` triple using SHA-256 hashing:

```python
from agentick.leaderboard.seeds import generate_task_seeds, get_train_seeds, get_eval_seeds

# 25 eval seeds for GoToGoal-v0 at medium difficulty
eval_seeds = get_eval_seeds("GoToGoal-v0", "medium")

# 2000 training seeds
train_seeds = get_train_seeds("GoToGoal-v0", "medium")

# Custom count
seeds = generate_task_seeds("GoToGoal-v0", "medium", "eval", 10)
```

Each `(task, difficulty, split)` triple produces a unique, deterministic seed sequence. The same call always returns the same seeds.

## Train / Eval Split

| Split | Seeds | Use |
|---|---|---|
| **train** | 2000 per (task, difficulty) | RL training, behavior cloning, SFT |
| **eval** | 25 per (task, difficulty) | Benchmark evaluation, leaderboard |

**Never train on eval seeds.** Use `split="train"` in training configs and `split="eval"` in evaluation configs.

## YAML Configs

```yaml
# Training config
split: train
n_seeds: 25

# Evaluation config
split: eval
n_seeds: 25
n_episodes: 1
```

Seeds are auto-generated per task/difficulty from the split. No need to list explicit seeds.

## Benchmark Suites

7 official suites, all using per-task eval seeds (25 per task):

| Suite | Tasks | Description |
|---|---|---|
| `agentick-full-v2` | 38 | Complete benchmark |
| `agentick-navigation-v2` | 8 | Navigation capability |
| `agentick-planning-v2` | 9 | Planning capability |
| `agentick-reasoning-v2` | 9 | Reasoning capability |
| `agentick-memory-v2` | 4 | Memory capability |
| `agentick-generalization-v2` | 3 | Generalization capability |
| `agentick-multiagent-v2` | 5 | Multi-agent coordination |

```python
from agentick.leaderboard.suites import get_suite, list_suites

suite = get_suite("agentick-full-v2")
seeds = suite.get_eval_seeds("GoToGoal-v0")  # Per-task seeds
```

## Verification

```python
from agentick.leaderboard.seeds import verify_seeds

ok = verify_seeds("GoToGoal-v0", "medium", "eval", eval_seeds)
assert ok, "Seeds don't match official set"
```
