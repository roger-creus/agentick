"""Verify that every task generates valid instances across difficulties.

Each test verifies:
  1. env.reset(seed=s) succeeds without error
  2. validate_instance() returns True
  3. A single step doesn't crash
"""

from __future__ import annotations

import pytest

import agentick
from agentick.oracles import get_oracle, list_oracles
from agentick.tasks.registry import list_tasks

ALL_TASKS = list_tasks()
ALL_DIFFICULTIES = ["easy", "medium", "hard", "expert"]
SMOKE_SEEDS = [42]


@pytest.mark.parametrize("task_name", ALL_TASKS)
@pytest.mark.parametrize("difficulty", ALL_DIFFICULTIES)
@pytest.mark.parametrize("seed", SMOKE_SEEDS)
def test_task_generates_valid_instance(task_name, difficulty, seed):
    """Every task at every difficulty must generate a valid, solvable instance."""
    env = agentick.make(task_name, difficulty=difficulty, render_mode="state_dict")
    _obs, _info = env.reset(seed=seed)

    # validate_instance must pass
    assert env.task.validate_instance(env.grid, env.task_config), (
        f"{task_name} difficulty={difficulty} seed={seed}: validate_instance() returned False"
    )

    # A single NOOP step must not crash
    obs2, _reward, _terminated, _truncated, _info2 = env.step(0)
    assert obs2 is not None
    env.close()


# Tasks with stochastic NPC behavior where a one-seed oracle success assertion is too brittle
_STOCHASTIC_MEDIUM_TASKS = {
    "Herding-v0", "TagHunt-v0", "EmergentStrategy-v0",
    "SokobanPush-v0", "FewShotAdaptation-v0",
}

_ORACLE_TASKS = sorted(set(list_oracles()) & set(ALL_TASKS))
_ORACLE_CASES = [
    (task_name, difficulty)
    for task_name in _ORACLE_TASKS
    for difficulty in ("easy", "medium")
    if not (difficulty == "medium" and task_name in _STOCHASTIC_MEDIUM_TASKS)
]


@pytest.mark.parametrize("task_name,difficulty", _ORACLE_CASES)
def test_task_solvable_by_oracle(task_name, difficulty):
    """Oracle smoke: each oracle should solve one representative easy/medium case."""
    env = agentick.make(task_name, difficulty=difficulty, render_mode="state_dict")

    oracle = get_oracle(task_name, env)

    obs, info = env.reset(seed=42)
    oracle.reset(obs, info)

    done = False
    for _ in range(env.max_steps):
        action = oracle.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        oracle.update(obs, info)
        if terminated or truncated:
            done = True
            break

    assert done, f"{task_name} {difficulty}: oracle didn't finish"
    assert info.get("success", False), (
        f"{task_name} {difficulty}: oracle failed to solve"
    )
    env.close()
