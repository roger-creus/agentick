# Oracles

Programmatic solvers for every Agentick task, powered by the privileged Coding API.

## Overview

Each oracle is a hand-coded bot that solves a single task optimally (or near-optimally) using full access to the environment's internal state via `AgentickAPI` from `coding_api.py`. Oracles serve as upper-bound baselines and as reward-signal validators.

## Usage

```python
from agentick.oracles import get_oracle
import agentick

env = agentick.make("GoToGoal-v0", difficulty="hard")
oracle = get_oracle("GoToGoal-v0", env)
obs, info = env.reset(seed=42)
oracle.reset(obs, info)

done = False
while not done:
    action = oracle.act(obs, info)
    obs, reward, done, trunc, info = env.step(action)
    oracle.update(obs, info)
    done = done or trunc
```

## Architecture

### base.py -- `OracleAgent`

Abstract base class. Wraps an `AgentickAPI` instance for grid queries, pathfinding, and entity lookups. Subclasses override `plan()` to fill `self.action_queue` with a sequence of action integers. The `act()` method pops from the queue and re-plans when it runs empty.

Key lifecycle: `reset(obs, info)` -> `plan()` -> `act(obs, info)` (repeats) -> `update(obs, info)` after each step.

### registry.py -- `@register_oracle`, `get_oracle`, `list_oracles`

- `@register_oracle(task_name)` -- class decorator that maps a task name to an oracle class.
- `get_oracle(task_name, env)` -- factory that returns an instantiated oracle. Triggers lazy auto-import of all oracle modules on first call.
- `list_oracles()` -- returns sorted list of task names that have registered oracles.
- `_import_all_oracles()` -- lazily imports all 10 category modules to populate the registry.

### Oracle modules (one per task category)

| Module                        | Category       |
|-------------------------------|----------------|
| `navigation_oracles.py`       | Navigation     |
| `memory_oracles.py`           | Memory         |
| `reasoning_oracles.py`        | Reasoning      |
| `skill_oracles.py`            | Skill          |
| `control_oracles.py`          | Control        |
| `adversarial_oracles.py`      | Adversarial    |
| `combinatorial_oracles.py`    | Combinatorial  |
| `compositional_oracles.py`    | Compositional  |
| `meta_oracles.py`             | Meta           |
| `multi_agent_oracles.py`      | Multi-agent    |

Each module contains one oracle class per task in that category (~38 oracles total). The standard pattern is:

```python
@register_oracle("TaskName-v0")
class TaskNameOracle(OracleAgent):
    def plan(self) -> None:
        # Use self.api for queries and pathfinding
        goal = self.api.get_nearest("goal")
        self.action_queue = self.api.move_to(*goal.position)
```

Navigation oracles use shared helpers (`_get_hazard_cells`, `_get_npc_cells`, `_navigate_with_fallback`) for BFS with multi-level avoidance fallback against hazards and NPCs.
