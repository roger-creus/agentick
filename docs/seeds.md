# Evaluation Seeds

Agentick provides standardized, locked seed sets for reproducible and comparable evaluation across different agents.

## Why Locked Seeds?

Reporting results on arbitrary seeds makes comparison unreliable. Official eval seeds are derived deterministically from suite names via SHA-256 — anyone can regenerate them, and no one can optimize for specific layouts.

## Train / Eval Split

**Never train on eval seeds.** The convention:

| Split | Seeds | Source |
|---|---|---|
| **Train** | Any seed not in the eval set | User-defined (e.g., `range(1000)`) |
| **Eval** | Generated from suite name | `generate_seeds_from_name()` |

## Using Official Seeds

```python
from agentick.leaderboard.seeds import generate_seeds_from_name

# Get deterministic eval seeds
eval_seeds = generate_seeds_from_name("full", n_seeds=10)
print(eval_seeds)  # Tuple of 10 ints, same every time
```

Use in an experiment config:

```yaml
name: my-eval
seeds: [...]  # Paste output of generate_seeds_from_name()
tasks: "full"
```

Or programmatically:

```python
from agentick.experiments.config import ExperimentConfig, AgentConfig
from agentick.leaderboard.seeds import generate_seeds_from_name

config = ExperimentConfig(
    name="eval-run",
    agent=AgentConfig(type="random"),
    tasks="full",
    seeds=list(generate_seeds_from_name("full", n_seeds=10)),
    n_episodes=10,
)
```

## Verifying Seeds

```python
from agentick.leaderboard.seeds import verify_seeds

ok = verify_seeds("full", eval_seeds)
assert ok, "Seeds don't match official set"
```

## Benchmark Suites

| Suite | Tasks | Default Seeds | Use Case |
|---|---|---|---|
| `quick` | 5 representative | 10 | Sanity checks |
| `core` | 25 core tasks | 30 | Standard evaluation |
| `full` | All 38 tasks | 50 | Complete benchmark |

```python
from agentick.leaderboard.suites import get_suite, list_suites

suite = get_suite("full")
print(f"Tasks: {len(suite.tasks)}, Seeds: {len(suite.eval_seeds)}")
```

Capability-specific suites are also available: `navigation`, `memory`, `reasoning`, `control`, `skill`, `combinatorial`, `adversarial`, `meta`, `multi_agent`, `compositional`.
