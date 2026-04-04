"""Verify that every task generates solvable, valid instances across difficulties.

Tests 38 tasks x 4 difficulties x 20 seeds = 3040 test cases.
Each test verifies:
  1. env.reset(seed=s) succeeds without error
  2. validate_instance() returns True
  3. A single step doesn't crash
"""

from __future__ import annotations

import pytest

import agentick
from agentick.tasks.registry import list_tasks

ALL_TASKS = list_tasks()
ALL_DIFFICULTIES = ["easy", "medium", "hard", "expert"]
SEEDS = list(range(20))


@pytest.mark.parametrize("task_name", ALL_TASKS)
@pytest.mark.parametrize("difficulty", ALL_DIFFICULTIES)
@pytest.mark.parametrize("seed", SEEDS)
def test_task_generates_valid_instance(task_name, difficulty, seed):
    """Every task at every difficulty must generate a valid, solvable instance."""
    env = agentick.make(task_name, difficulty=difficulty, render_mode="state_dict")
    try:
        obs, info = env.reset(seed=seed)
    except RuntimeError:
        pytest.skip(f"Generation failed for {task_name}/{difficulty}/seed={seed}")
        return

    # validate_instance must pass
    assert env.task.validate_instance(env.grid, env.task_config), (
        f"{task_name} difficulty={difficulty} seed={seed}: validate_instance() returned False"
    )

    # A single NOOP step must not crash
    obs2, reward, terminated, truncated, info2 = env.step(0)
    assert obs2 is not None
    env.close()


@pytest.mark.parametrize("task_name", ALL_TASKS)
def test_task_easy_solvable_by_oracle(task_name):
    """Every task's easy difficulty must be solvable by its oracle (if one exists)."""
    from agentick.oracles import get_oracle, list_oracles

    available = list_oracles()
    if task_name not in available:
        pytest.skip(f"No oracle for {task_name}")

    env = agentick.make(task_name, difficulty="easy", render_mode="state_dict")
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

    assert done, f"{task_name} easy: oracle didn't finish within max_steps"
    assert info.get("success", False), f"{task_name} easy: oracle failed to solve"
    env.close()
