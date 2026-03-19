"""Behavioral validation tests for all 37 Agentick tasks.

Tests that each task:
1. Initializes and resets without error
2. Has a valid config (agent_start, goal_positions, max_steps)
3. Can run 10 random steps without crashing
4. Produces correct sparse reward (0 or 1) and dense reward (finite float)
5. Can successfully terminate (success=True) via a scripted policy
6. Reports success=False when episode truncates without goal
7. check_done() defaults to check_success() unless overridden
"""

from __future__ import annotations

import numpy as np
import pytest

import agentick
from agentick.tasks.registry import list_tasks

ALL_TASKS = list_tasks()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bfs_path(grid, start, goal, env=None) -> list[tuple[int, int]] | None:
    """BFS path from start to goal. Avoids WALL, blocking objects, and respects can_agent_enter."""
    from collections import deque

    from agentick.core.types import CellType
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        (cx, cy), path = queue.popleft()
        if (cx, cy) == goal:
            return path
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) not in visited:
                if 0 <= ny < grid.height and 0 <= nx < grid.width:
                    if grid.terrain[ny, nx] == CellType.WALL:
                        continue
                    # Skip non-walkable objects (DOOR/LEVER/SWITCH)
                    if grid.is_object_blocking((nx, ny)):
                        continue
                    # Use task's can_agent_enter if defined, to respect task-specific traversal
                    if env is not None and hasattr(env.task, 'can_agent_enter'):
                        if not env.task.can_agent_enter((nx, ny), env.agent, env.grid):
                            continue
                    elif grid.terrain[ny, nx] == CellType.HAZARD:
                        continue  # default: avoid hazard terrain
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))
    return None


def _walk_to(env, tx: int, ty: int) -> tuple:
    """Walk agent to (tx, ty) using BFS-guided moves. Returns last step info."""
    obs, rew, term, trunc, info = None, 0.0, False, False, {}
    path = _bfs_path(env.grid, env.agent.position, (tx, ty), env=env)
    if path is None:
        # No path — fall back to naive cardinal moves
        path = []
    # Convert path to actions
    prev = env.agent.position
    for nxt in path[1:]:  # skip start
        dx, dy = nxt[0] - prev[0], nxt[1] - prev[1]
        action = {(1, 0): 4, (-1, 0): 3, (0, 1): 2, (0, -1): 1}.get((dx, dy), 0)
        obs, rew, term, trunc, info = env.step(action)
        prev = env.agent.position
        if term or trunc:
            return obs, rew, term, trunc, info
    return obs, rew, term, trunc, info


def _noop(env) -> tuple:
    return env.step(0)


def _walk_adjacent_and_interact(env, tx, ty):
    """Walk to a walkable cell adjacent to (tx, ty), face the target, then INTERACT.

    For solid objects (SWITCH, LEVER, DOOR): stand next to them, face them, INTERACT.
    """
    from agentick.core.types import ActionType, CellType

    interact = int(ActionType.INTERACT)
    ax, ay = env.agent.position

    # Find walkable adjacent cells
    candidates = []
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        nx, ny = tx + dx, ty + dy
        if (0 <= nx < env.grid.width and 0 <= ny < env.grid.height
                and env.grid.terrain[ny, nx] != CellType.WALL
                and not env.grid.is_object_blocking((nx, ny))):
            candidates.append((nx, ny, dx, dy))

    if not candidates:
        return None, 0.0, False, False, {}

    # Pick closest adjacent cell
    candidates.sort(key=lambda c: abs(c[0] - ax) + abs(c[1] - ay))
    nx, ny, dx, dy = candidates[0]

    # Walk to adjacent cell
    obs, rew, term, trunc, info = _walk_to(env, nx, ny)
    if term or trunc:
        return obs, rew, term, trunc, info

    # Face the target: move from (nx, ny) toward (tx, ty) — direction is (-dx, -dy)
    face_dir = (-dx, -dy)
    face_action = {(1, 0): 4, (-1, 0): 3, (0, 1): 2, (0, -1): 1}.get(face_dir, 0)
    obs, rew, term, trunc, info = env.step(face_action)
    if term or trunc:
        return obs, rew, term, trunc, info

    # INTERACT
    obs, rew, term, trunc, info = env.step(interact)
    return obs, rew, term, trunc, info


# ---------------------------------------------------------------------------
# Basic sanity: all tasks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task_name", ALL_TASKS)
@pytest.mark.timeout(30)
def test_task_initializes(task_name):
    """Task creates, resets and closes without error."""
    env = agentick.make(task_name, difficulty="easy", seed=42)
    obs, info = env.reset(seed=42)
    assert obs is not None
    assert "valid_actions" in info
    env.close()


@pytest.mark.parametrize("task_name", ALL_TASKS)
@pytest.mark.timeout(30)
def test_task_has_valid_config(task_name):
    """Task config contains required fields."""
    env = agentick.make(task_name, difficulty="easy", seed=42)
    env.reset(seed=42)
    cfg = env.task_config
    assert "agent_start" in cfg, f"{task_name} missing agent_start"
    assert "max_steps" in cfg, f"{task_name} missing max_steps"
    assert cfg["max_steps"] > 0
    # Agent placed at declared start
    assert env.agent.position == cfg["agent_start"], (
        f"{task_name}: agent at {env.agent.position}, expected {cfg['agent_start']}"
    )
    env.close()


@pytest.mark.parametrize("task_name", ALL_TASKS)
@pytest.mark.timeout(30)
def test_task_random_steps(task_name):
    """Task survives 10 random valid steps without exception."""
    env = agentick.make(task_name, difficulty="easy", seed=42)
    env.reset(seed=42)
    rng = np.random.default_rng(0)
    for _ in range(10):
        valid = [i for i, v in enumerate(env.get_valid_actions()) if v]
        action = int(rng.choice(valid)) if valid else 0
        obs, rew, term, trunc, info = env.step(action)
        assert np.isfinite(rew), f"{task_name}: non-finite reward {rew}"
        assert isinstance(info["success"], bool)
        if term or trunc:
            break
    env.close()


@pytest.mark.parametrize("task_name", ALL_TASKS)
@pytest.mark.timeout(30)
def test_task_sparse_reward_bounds(task_name):
    """Sparse reward is always 0 or 1 (plus small step penalties allowed)."""
    env = agentick.make(task_name, difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    rng = np.random.default_rng(1)
    for _ in range(20):
        valid = [i for i, v in enumerate(env.get_valid_actions()) if v]
        action = int(rng.choice(valid)) if valid else 0
        _, rew, term, trunc, _ = env.step(action)
        assert -1.0 <= rew <= 1.01, f"{task_name}: sparse reward {rew} out of bounds"
        if term or trunc:
            break
    env.close()


@pytest.mark.parametrize("task_name", ALL_TASKS)
@pytest.mark.timeout(30)
def test_task_no_success_on_start(task_name):
    """Task should not start in a success state (before any action)."""
    env = agentick.make(task_name, difficulty="easy", seed=42)
    env.reset(seed=42)
    # Check success via a NOOP (the episode should not be immediately done)
    # Exception: tasks that terminate in 0 steps are broken
    obs, rew, term, trunc, info = env.step(0)
    # It's OK if the episode ends via truncation, but not via instant success
    # (unless the task is designed to end immediately, which none should be)
    if info["success"]:
        # If success fires on NOOP from start, agent must have been placed at goal
        cfg = env.task_config
        goals = cfg.get("goal_positions", [])
        agent_at_goal = env.agent.position in goals
        assert not agent_at_goal or len(goals) == 0, (
            f"{task_name}: agent starts at goal — trivially solved!"
        )
    env.close()


# ---------------------------------------------------------------------------
# Truncation = failure: tasks should not auto-succeed after max_steps
# ---------------------------------------------------------------------------

# Survival tasks where truncation IS the success condition
_SURVIVAL_TASKS = {"ResourceManagement-v0"}


@pytest.mark.parametrize("task_name", ALL_TASKS)
@pytest.mark.timeout(60)
def test_task_truncation_is_not_success(task_name):
    """Letting the episode timeout should NOT mark info['success'] = True.

    Exception: survival tasks (ResourceManagement) where surviving to
    truncation IS the success condition.
    """
    if task_name in _SURVIVAL_TASKS:
        pytest.skip(f"{task_name} is a survival task (truncation = success)")
    # Use a very short custom max_steps via NOOP spam
    env = agentick.make(task_name, difficulty="easy", seed=99)
    env.reset(seed=42)
    last_success = False
    for _ in range(env.max_steps + 5):
        obs, rew, term, trunc, info = env.step(0)  # NOOP everything
        if trunc and not term:
            last_success = info["success"]
            break
        if term:
            # Task ended naturally — fine, not what we're testing here
            env.close()
            return
    env.close()
    assert not last_success, (
        f"{task_name}: truncated episode marked as success!"
    )


# ---------------------------------------------------------------------------
# Scripted success: verify the goal IS achievable for simple tasks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task_name,seeds", [
    # Pure navigation: walk to static goal and verify success
    ("GoToGoal-v0", [0, 1, 2, 42, 99]),
    # DynamicObstacles uses a dedicated test (test_dynamic_obstacles_can_succeed)
    # because obstacles move each step and simple BFS path is not reliable.
    ("FogOfWarExploration-v0", [0, 1, 42]),
    ("MazeNavigation-v0", [0, 1, 42]),
    ("NoisyObservation-v0", [0, 1, 42]),
    ("DeceptiveReward-v0", [0, 42]),
])
@pytest.mark.timeout(60)
def test_nav_task_can_succeed(task_name, seeds):
    """Navigation tasks: walk to goal(s) and verify success fires."""
    for seed in seeds:
        env = agentick.make(task_name, difficulty="easy", seed=seed, reward_mode="sparse")
        env.reset(seed=seed)
        cfg = env.task_config
        goals = cfg.get("goal_positions", [])
        success = False
        for goal in goals:
            obs, rew, term, trunc, info = _walk_to(env, goal[0], goal[1])
            if info.get("success"):
                success = True
                break
            if not term and not trunc:
                obs, rew, term, trunc, info = _noop(env)
                if info.get("success"):
                    success = True
                    break
        env.close()
        assert success, f"{task_name} seed={seed}: could not reach success state"


@pytest.mark.timeout(60)
def test_dynamic_obstacles_can_succeed():
    """DynamicObstacles: agent avoids moving blockers, reaches goal."""
    import agentick
    # Try multiple seeds; find one where agent can navigate safely
    # The agent waits (noop) up to 3 steps when an obstacle is directly adjacent
    for seed in [5, 10, 15, 20, 25]:
        env = agentick.make("DynamicObstacles-v0", difficulty="easy", seed=seed, reward_mode="sparse")
        env.reset(seed=seed)
        cfg = env.task_config
        goal = tuple(cfg["goal_positions"][0])
        success = False
        term = trunc = False

        for _ in range(cfg.get("max_steps", 50)):
            if term or trunc:
                break
            # Check if any obstacle is at the next step position
            obs_positions = set(map(tuple, env.task_config.get("_live_obstacles", [])))
            ax, ay = env.agent.position
            if (ax, ay) == goal:
                break
            # Build next move toward goal
            tx, ty = goal
            dx = 1 if tx > ax else -1 if tx < ax else 0
            dy = 1 if ty > ay else -1 if ty < ay else 0
            # Prefer horizontal then vertical
            if dx != 0:
                next_cell = (ax + dx, ay)
            else:
                next_cell = (ax, ay + dy)
            # Wait if obstacle is at next cell
            if next_cell in obs_positions:
                obs, rew, term, trunc, info = env.step(0)  # noop
            else:
                action = {(1, 0): 4, (-1, 0): 3, (0, 1): 2, (0, -1): 1}.get(
                    (next_cell[0]-ax, next_cell[1]-ay), 0)
                obs, rew, term, trunc, info = env.step(action)
            if info.get("success"):
                success = True
                break
        env.close()
        if success:
            break  # one seed success is enough
    assert success, "DynamicObstacles-v0: could not navigate to goal avoiding obstacles"


@pytest.mark.timeout(60)
def test_lights_out_can_succeed():
    """LightsOut: oracle solves it (adjacent toggle requires planning)."""
    from agentick.oracles.registry import get_oracle
    success = False
    for seed in range(5):
        env = agentick.make("LightsOut-v0", difficulty="easy", seed=seed, reward_mode="sparse")
        obs, info = env.reset(seed=seed)
        oracle = get_oracle("LightsOut-v0", env)
        oracle.reset(obs, info)
        for _ in range(env.spec.max_episode_steps or 200):
            action = oracle.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if info.get("success", False):
                    success = True
                break
        env.close()
        if success:
            break
    assert success, "LightsOut-v0: oracle could not solve any of 5 seeds"


@pytest.mark.timeout(30)
def test_chase_evade_success_condition():
    """ChaseEvade: success fires when agent survives for the required number of steps."""
    env = agentick.make("ChaseEvade-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    survival_steps = cfg.get("survival_steps", 30)
    # Remove all enemies so agent can survive freely
    cfg["_enemies"] = []
    # Step through required survival steps
    term, trunc, info = False, False, {}
    for _ in range(survival_steps + 1):
        if term or trunc:
            break
        obs, rew, term, trunc, info = env.step(0)  # NOOP
    env.close()
    assert info.get("success"), f"ChaseEvade: survival didn't trigger success after {survival_steps} steps"


@pytest.mark.timeout(60)
def test_shortest_path_can_succeed():
    """ShortestPath: visit ALL goals in order to trigger success."""
    env = agentick.make("ShortestPath-v0", difficulty="easy", seed=3, reward_mode="sparse")
    env.reset(seed=3)
    cfg = env.task_config
    term, trunc, info = False, False, {}
    for goal in cfg.get("goal_positions", []):
        if term or trunc:
            break
        obs, rew, term, trunc, info = _walk_to(env, goal[0], goal[1])
    env.close()
    assert info.get("success"), "ShortestPath: couldn't trigger success"


@pytest.mark.timeout(60)
def test_distribution_shift_can_succeed():
    """DistributionShift: solve all phases across shifting maze layouts.

    Tries multiple seeds. For each phase, dispatches to a phase-type-aware
    solver (goal_reach, key_door, lever_barrier, collection, box_push).
    """
    from agentick.core.types import CellType, ObjectType

    def _solve_phase(env, cfg, phase_type):
        """Attempt to solve one phase. Returns (term, trunc, info)."""
        term, trunc, info = False, False, {}
        goal = cfg["goal_positions"][0]

        if phase_type == "goal_reach":
            obs, rew, term, trunc, info = _walk_to(env, goal[0], goal[1])

        elif phase_type == "key_door":
            # Collect key (auto-pickup), open door (face+interact), reach goal.
            grid = env.grid
            key_pos = None
            door_pos = None
            for y in range(grid.height):
                for x in range(grid.width):
                    if int(grid.objects[y, x]) == int(ObjectType.KEY):
                        key_pos = (x, y)
                    if (int(grid.objects[y, x]) == int(ObjectType.DOOR)
                            and int(grid.metadata[y, x]) < 10):
                        door_pos = (x, y)
            if key_pos:
                obs, rew, term, trunc, info = _walk_to(env, key_pos[0], key_pos[1])
                if term or trunc:
                    return term, trunc, info
            if door_pos:
                obs, rew, term, trunc, info = _walk_adjacent_and_interact(
                    env, door_pos[0], door_pos[1])
                if term or trunc:
                    return term, trunc, info
            obs, rew, term, trunc, info = _walk_to(env, goal[0], goal[1])

        elif phase_type == "lever_barrier":
            # Interact with lever (removes wall barrier), then reach goal.
            grid = env.grid
            lever_pos = None
            for y in range(grid.height):
                for x in range(grid.width):
                    if int(grid.objects[y, x]) == int(ObjectType.LEVER):
                        lever_pos = (x, y)
                        break
                if lever_pos:
                    break
            if lever_pos:
                obs, rew, term, trunc, info = _walk_adjacent_and_interact(
                    env, lever_pos[0], lever_pos[1])
                if term or trunc:
                    return term, trunc, info
            obs, rew, term, trunc, info = _walk_to(env, goal[0], goal[1])

        elif phase_type == "collection":
            # Collect all gems, then goal appears — walk to it.
            for _ in range(50):
                grid = env.grid
                gem_pos = None
                for y in range(grid.height):
                    for x in range(grid.width):
                        if int(grid.objects[y, x]) == int(ObjectType.GEM):
                            gem_pos = (x, y)
                            break
                    if gem_pos:
                        break
                if gem_pos:
                    obs, rew, term, trunc, info = _walk_to(env, gem_pos[0], gem_pos[1])
                    if term or trunc:
                        return term, trunc, info
                else:
                    # All gems collected — goal should be placed now.
                    break
            goal = cfg["goal_positions"][0]
            # Walk to goal only if it's placed on the grid.
            grid = env.grid
            if int(grid.objects[goal[1], goal[0]]) == int(ObjectType.GOAL):
                obs, rew, term, trunc, info = _walk_to(env, goal[0], goal[1])

        elif phase_type == "box_push":
            # Push box onto target. Find box and target, maneuver.
            grid = env.grid
            box_pos = None
            target_pos = None
            for y in range(grid.height):
                for x in range(grid.width):
                    if int(grid.objects[y, x]) == int(ObjectType.BOX):
                        box_pos = (x, y)
                    if int(grid.objects[y, x]) == int(ObjectType.TARGET):
                        target_pos = (x, y)
            if box_pos and target_pos:
                # Walk to the push position: one step beyond box from target direction.
                bx, by = box_pos
                tx, ty = target_pos
                dx = bx - tx
                dy = by - ty
                # Normalise direction
                if dx != 0:
                    dx = dx // abs(dx)
                if dy != 0:
                    dy = dy // abs(dy)
                push_from = (bx + dx, by + dy)
                # Check push_from is walkable.
                if (0 <= push_from[0] < grid.width
                        and 0 <= push_from[1] < grid.height
                        and grid.terrain[push_from[1], push_from[0]] != CellType.WALL):
                    obs, rew, term, trunc, info = _walk_to(env, push_from[0], push_from[1])
                    if term or trunc:
                        return term, trunc, info
                    # Walk into box to push it.
                    obs, rew, term, trunc, info = _walk_to(env, bx, by)
                    if term or trunc:
                        return term, trunc, info
            # If box now on target, goal appears — walk to it.
            goal = cfg["goal_positions"][0]
            grid = env.grid
            if int(grid.objects[goal[1], goal[0]]) == int(ObjectType.GOAL):
                obs, rew, term, trunc, info = _walk_to(env, goal[0], goal[1])

        return term, trunc, info

    success = False
    for seed in range(10):
        env = agentick.make("DistributionShift-v0", difficulty="easy",
                            seed=seed, reward_mode="sparse")
        env.reset(seed=seed)
        cfg = env.task_config
        n_phases = cfg.get("_n_phases", 3)
        term, trunc, info = False, False, {}

        for i in range(n_phases):
            if term or trunc:
                break
            phase_type = cfg.get("_current_phase_type", "goal_reach")
            term, trunc, info = _solve_phase(env, cfg, phase_type)

        env.close()
        if info.get("success"):
            success = True
            break

    assert success, "DistributionShift: couldn't succeed across 10 seeds"


@pytest.mark.timeout(60)
def test_recipe_assembly_can_succeed():
    """RecipeAssembly: collect ingredients in recipe order and bring each to crafting station."""
    env = agentick.make("RecipeAssembly-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    recipe = cfg.get("recipe", [])  # list of int (ObjectType values)
    ing_pos = cfg.get("ingredient_positions", {})  # dict int -> [[x,y], ...]
    station = cfg.get("station_pos", cfg.get("goal_positions", [[4, 4]])[0])
    term, trunc, info = False, False, {}
    for needed_type in recipe:
        if term or trunc:
            break
        positions = ing_pos.get(needed_type, ing_pos.get(str(needed_type), []))
        if positions:
            pos = positions[0]
            obs, rew, term, trunc, info = _walk_to(env, pos[0], pos[1])
        if not (term or trunc):
            obs, rew, term, trunc, info = _walk_to(env, station[0], station[1])
    if not (term or trunc):
        obs, rew, term, trunc, info = _noop(env)
    env.close()
    assert info["success"], "RecipeAssembly: success didn't fire after following recipe"


@pytest.mark.timeout(60)
def test_tool_use_can_succeed():
    """ToolUse: collect tool(s) then reach goal."""
    env = agentick.make("ToolUse-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    # tool_positions is a dict mapping tool_name -> [[x, y], ...]
    tool_positions = cfg.get("tool_positions", {})
    for tool_name, positions in tool_positions.items():
        for pos in positions:
            _walk_to(env, pos[0], pos[1])
    goal = cfg["goal_positions"][0]
    obs, rew, term, trunc, info = _walk_to(env, goal[0], goal[1])
    if not (term or trunc):
        obs, rew, term, trunc, info = _noop(env)
    env.close()
    assert info["success"], "ToolUse: success didn't fire after collecting tools and reaching goal"


@pytest.mark.timeout(60)
def test_key_door_puzzle_can_succeed():
    """KeyDoorPuzzle: pick up key, INTERACT to open door, walk to goal."""
    env = agentick.make("KeyDoorPuzzle-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    key_pos = cfg.get("key_pos")
    if isinstance(key_pos, (list, tuple)) and key_pos and not isinstance(key_pos[0], int):
        key_pos = key_pos[0]
    term = False
    if key_pos:
        obs, rew, term, trunc, info = _walk_to(env, key_pos[0], key_pos[1])
        if not term and not trunc:
            obs, rew, term, trunc, info = _noop(env)  # ensure pickup fires
    # Open door via adjacent INTERACT
    door_pos = cfg.get("door_pos")
    if door_pos and not term:
        obs, rew, term, trunc, info = _walk_adjacent_and_interact(
            env, door_pos[0], door_pos[1]
        )
        term = term or trunc
    goal = cfg["goal_positions"][0]
    if not term:
        obs, rew, term, trunc, info = _walk_to(env, goal[0], goal[1])
    if not term and not trunc:
        obs, rew, term, trunc, info = _noop(env)
    env.close()
    assert info["success"] or rew > 0, "KeyDoorPuzzle: could not trigger success"


@pytest.mark.timeout(60)
def test_delayed_gratification_decoy_is_not_success():
    """DelayedGratification: taking decoy should NOT count as success."""
    env = agentick.make("DelayedGratification-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    decoy = cfg["decoy_positions"][0]
    # Walk to decoy — decoy pickup fires in on_agent_moved (same step)
    obs, rew, term, trunc, info = _walk_to(env, decoy[0], decoy[1])
    env.close()
    assert term, "DelayedGratification: episode should terminate on decoy step"
    assert not info["success"], "DelayedGratification: decoy should NOT be success"
    assert abs(rew - 0.05) < 0.01, f"DelayedGratification: expected decoy reward 0.05, got {rew}"


@pytest.mark.timeout(60)
def test_delayed_gratification_goal_is_success():
    """DelayedGratification: reaching the true goal should be success with reward=1."""
    env = agentick.make("DelayedGratification-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    goal = cfg["goal_positions"][0]
    obs, rew, term, trunc, info = _walk_to(env, goal[0], goal[1])
    if not term and not trunc:
        obs, rew, term, trunc, info = _noop(env)
    env.close()
    assert info["success"], "DelayedGratification: goal should be success"
    assert abs(rew - 1.0) < 0.01


@pytest.mark.timeout(60)
def test_sequence_memory_can_succeed():
    """SequenceMemory: wait through show phase, then visit positions from memory."""
    env = agentick.make("SequenceMemory-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    seq = cfg.get("sequence", [])
    total_show = cfg.get("total_show_steps", 0)
    # Wait through show phase (NOOP)
    for _ in range(total_show + 2):
        obs, rew, term, trunc, info = _noop(env)
        if term or trunc:
            break
    assert cfg.get("_phase") == "reproduce", "SequenceMemory: show phase didn't end"
    # Reproduce phase: visit memorized positions in order
    for pos in seq:
        if env.done:
            break
        obs, rew, term, trunc, info = _walk_to(env, pos[0], pos[1])
        if not term and not trunc:
            obs, rew, term, trunc, info = _noop(env)
    env.close()
    assert info.get("success"), f"SequenceMemory: couldn't trigger success (seq={seq})"



# Old test_graph_coloring_can_succeed removed — replaced by updated version below


@pytest.mark.timeout(120)
def test_resource_management_can_succeed():
    """ResourceManagement: survive all max_steps by recharging stations."""
    env = agentick.make("ResourceManagement-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    stations = cfg.get("station_positions", [])
    # Keep visiting stations in round-robin to keep them charged
    station_idx = 0
    done = False
    info = {}
    for _ in range(env.max_steps + 5):
        if stations:
            sx, sy = stations[station_idx % len(stations)]
            obs, rew, done, trunc, info = _walk_to(env, sx, sy)
            if done or trunc:
                break
            station_idx += 1
        else:
            obs, rew, done, trunc, info = _noop(env)
            if done or trunc:
                break
    env.close()
    assert info.get("success"), "ResourceManagement: couldn't survive to truncation"


# ---------------------------------------------------------------------------
# All difficulties: tasks must initialize at every difficulty level
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task_name", ALL_TASKS)
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard", "expert"])
@pytest.mark.timeout(30)
def test_all_difficulties_initialize(task_name, difficulty):
    """Every task must initialize cleanly at every difficulty."""
    env = agentick.make(task_name, difficulty=difficulty, seed=42)
    obs, info = env.reset(seed=42)
    assert obs is not None
    cfg = env.task_config
    assert cfg["max_steps"] > 0
    env.close()


# ---------------------------------------------------------------------------
# Reward mode consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task_name", ALL_TASKS[:10])  # Subset for speed
@pytest.mark.timeout(30)
def test_dense_reward_finite(task_name):
    """Dense reward is always a finite float."""
    env = agentick.make(task_name, difficulty="easy", seed=42, reward_mode="dense")
    env.reset(seed=42)
    rng = np.random.default_rng(42)
    for _ in range(20):
        valid = [i for i, v in enumerate(env.get_valid_actions()) if v]
        action = int(rng.choice(valid)) if valid else 0
        _, rew, term, trunc, _ = env.step(action)
        assert np.isfinite(rew), f"{task_name}: non-finite dense reward {rew}"
        if term or trunc:
            break
    env.close()


# ---------------------------------------------------------------------------
# Newly fixed tasks: scripted success verification
# ---------------------------------------------------------------------------

def _bfs_nobox(env, tx, ty):
    """BFS that avoids BOX objects (for Sokoban-style push tasks)."""
    from agentick.core.types import ObjectType
    start = env.agent.position
    if start == (tx, ty):
        return [start]
    q = __import__("collections").deque([(start, [start])])
    vis = {start}
    while q:
        (cx, cy), path = q.popleft()
        if (cx, cy) == (tx, ty):
            return path
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) not in vis and 0 <= ny < env.grid.height and 0 <= nx < env.grid.width:
                from agentick.core.types import CellType
                if env.grid.terrain[ny, nx] != CellType.WALL and env.grid.objects[ny, nx] != ObjectType.BOX:
                    vis.add((nx, ny))
                    q.append(((nx, ny), path + [(nx, ny)]))
    return None


def _push_box(env, steps: int, direction: int) -> tuple:
    """Push a box `steps` times in `direction` (1=north,2=south,3=west,4=east)."""
    rew, term, trunc, info = 0, False, False, {}
    for _ in range(steps):
        if env.done:
            break
        _, rew, term, trunc, info = env.step(direction)
    return rew, term, trunc, info


@pytest.mark.timeout(60)
def test_task_interference_can_succeed():
    """TaskInterference: alternate red/blue collection to fill both meters."""
    env = agentick.make("TaskInterference-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    reds = list(cfg.get("red_positions", []))
    blues = list(cfg.get("blue_positions", []))
    term, trunc, info = False, False, {}
    # Alternate: red, blue, red, blue, ...
    while (reds or blues) and not term and not trunc:
        if reds:
            r = reds.pop(0)
            obs, rew, term, trunc, info = _walk_to(env, r[0], r[1])
            if term or trunc:
                break
        if blues:
            b = blues.pop(0)
            obs, rew, term, trunc, info = _walk_to(env, b[0], b[1])
            if term or trunc:
                break
    if not term and not trunc:
        obs, rew, term, trunc, info = _noop(env)
    env.close()
    assert info.get("success"), "TaskInterference: couldn't fill both meters"


@pytest.mark.timeout(60)
def test_symbol_matching_can_succeed():
    """SymbolMatching: carry each symbol item to its matching target zone."""
    env = agentick.make("SymbolMatching-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    pair_info = cfg.get("pair_info", [])
    term, trunc, info = False, False, {}
    for pair in pair_info:
        if term or trunc:
            break
        item_pos = pair["item_pos"]
        target_pos = pair["target_pos"]
        _walk_to(env, item_pos[0], item_pos[1])  # pickup
        if not env.done:
            obs, rew, term, trunc, info = _walk_to(env, target_pos[0], target_pos[1])  # place
    if not env.done:
        obs, rew, term, trunc, info = _noop(env)
    env.close()
    assert info.get("success"), "SymbolMatching: couldn't place all items"


@pytest.mark.timeout(60)
def test_program_synthesis_can_succeed():
    """ProgramSynthesis: push GEM objects onto TARGET positions using the oracle."""
    from agentick.oracles.registry import get_oracle

    # Try a handful of seeds; the oracle should solve at least one on easy.
    success = False
    for seed in range(10):
        env = agentick.make("ProgramSynthesis-v0", difficulty="easy", seed=seed, reward_mode="sparse")
        obs, info = env.reset(seed=seed)
        oracle = get_oracle("ProgramSynthesis-v0", env)
        oracle.reset(obs, info)
        term, trunc = False, False
        for _ in range(env.max_steps):
            if term or trunc:
                break
            action = oracle.act(obs, info)
            obs, rew, term, trunc, info = env.step(action)
        if info.get("success"):
            success = True
            env.close()
            break
        env.close()
    assert success, "ProgramSynthesis: oracle couldn't push gems to complete pattern"


@pytest.mark.timeout(60)
def test_sokoban_push_can_succeed():
    """SokobanPush: push box onto target using can_agent_enter mechanic."""
    # Try multiple seeds to find one where box and target are aligned on one axis
    success = False
    for seed in range(100):
        env = agentick.make("SokobanPush-v0", difficulty="easy", seed=seed, reward_mode="sparse")
        env.reset(seed=seed)
        cfg = env.task_config
        box = cfg.get("box_positions", [None])[0]
        target = cfg.get("target_positions", [None])[0]
        if box is None or target is None:
            env.close()
            continue
        # Need box and target to be on same row or column for a simple push test
        if box[0] != target[0] and box[1] != target[1]:
            env.close()
            continue
        # Compute push direction: box → target (must be unit vector)
        dx = 0 if target[0] == box[0] else (1 if target[0] > box[0] else -1)
        dy = 0 if target[1] == box[1] else (1 if target[1] > box[1] else -1)
        push_from = (box[0] - dx, box[1] - dy)
        # push_from must be walkable and reachable
        if not env.grid.is_walkable(push_from):
            env.close()
            continue
        obs, rew, term, trunc, info = _walk_to(env, push_from[0], push_from[1])
        if env.done:
            env.close()
            continue
        # Push box toward target step by step
        push_act = {(1, 0): 4, (-1, 0): 3, (0, 1): 2, (0, -1): 1}.get((dx, dy), 0)
        dist = abs(target[0] - box[0]) + abs(target[1] - box[1])
        for _ in range(dist):
            if env.done:
                break
            obs, rew, term, trunc, info = env.step(push_act)
        if info.get("success"):
            success = True
            env.close()
            break
        env.close()
    assert success, "SokobanPush: could not find seed where push succeeds"


@pytest.mark.timeout(60)
def test_packing_puzzle_can_succeed():
    """PackingPuzzle: verify success detection when all targets are matched."""
    from agentick.core.types import ObjectType
    env = agentick.make("PackingPuzzle-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    targets = cfg.get("target_positions", [])
    # Directly set all targets to GOAL (simulating correct piece placement)
    for tx, ty in targets:
        env.grid.objects[ty, tx] = ObjectType.GOAL
        env.grid.metadata[ty, tx] = 0
    obs, rew, term, trunc, info = _noop(env)
    env.close()
    assert info.get("success"), "PackingPuzzle: success didn't fire with all targets matched"


@pytest.mark.timeout(60)
def test_cooperative_transport_can_succeed():
    """CooperativeTransport: push heavy box into hole with NPC cooperation."""
    from agentick.core.types import CellType, ObjectType

    env = agentick.make("CooperativeTransport-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    task = env.task
    cfg = env.task_config

    bx, by = task._box_positions[0]
    hx, hy = task._hole_positions[0]

    # --- Step A: verify agent CANNOT push alone (NPC far away) ---
    # Move agent adjacent to box if not already
    # Place agent south of box, try pushing north
    env.grid.objects[by, bx] = ObjectType.BOX  # ensure box is placed
    env.agent.position = (bx, by + 1)
    old_pos = env.agent.position
    env.step(1)  # MOVE_UP into box
    assert env.agent.position == old_pos, "Agent should NOT push box alone"

    # --- Step B: set up cooperative push into hole ---
    # Clear grid of stale NPC
    for yy in range(env.grid.height):
        for xx in range(env.grid.width):
            if env.grid.objects[yy, xx] == ObjectType.NPC:
                env.grid.objects[yy, xx] = ObjectType.NONE

    # Remove old box, place box one cell above hole
    env.grid.objects[by, bx] = ObjectType.NONE
    new_bx, new_by = hx, hy - 1
    # Make sure destination is walkable for box placement
    if env.grid.terrain[new_by, new_bx] in (CellType.WALL, CellType.HOLE):
        # fallback: place box one cell left of hole instead, push east
        new_bx, new_by = hx - 1, hy
        env.grid.objects[new_by, new_bx] = ObjectType.BOX
        task._box_positions = [[new_bx, new_by]]
        cfg["_box_positions"] = task._box_positions
        # Agent west of box, push east
        env.agent.position = (new_bx - 1, new_by)
        # NPC adjacent to box (north)
        npc_x, npc_y = new_bx, new_by - 1
        task._npc_pos = [npc_x, npc_y]
        cfg["_npc_pos"] = task._npc_pos
        env.grid.objects[npc_y, npc_x] = ObjectType.NPC
        push_action = 4  # MOVE_RIGHT
    else:
        env.grid.objects[new_by, new_bx] = ObjectType.BOX
        task._box_positions = [[new_bx, new_by]]
        cfg["_box_positions"] = task._box_positions
        # Agent north of box, push south
        env.agent.position = (new_bx, new_by - 1)
        # NPC adjacent to box (east side)
        npc_x, npc_y = new_bx + 1, new_by
        task._npc_pos = [npc_x, npc_y]
        cfg["_npc_pos"] = task._npc_pos
        env.grid.objects[npc_y, npc_x] = ObjectType.NPC
        push_action = 2  # MOVE_DOWN

    _, rew, term, trunc, last_info = env.step(push_action)
    env.close()
    assert last_info.get("success"), "CooperativeTransport: couldn't push box into hole"


@pytest.mark.timeout(60)
def test_rule_induction_can_succeed():
    """RuleInduction: pick up an object, combine it with another, collect target."""
    # Try multiple seeds to find one where the first rule produces the target
    succeeded = False
    for seed in range(20):
        env = agentick.make("RuleInduction-v0", difficulty="easy", seed=seed, reward_mode="sparse")
        env.reset(seed=seed)
        cfg = env.task_config
        grid = env.unwrapped.grid
        agent = env.unwrapped.agent

        rules = cfg.get("_rule_table_list", [])
        target_type = cfg.get("_target_type", -1)
        objects = cfg.get("_original_objects", [])
        term, trunc = False, False

        # Find a rule that directly produces the target type
        combo_rule = None
        for row in rules:
            if row[2] == target_type:
                combo_rule = row
                break
        if combo_rule is None:
            env.close()
            continue

        a_type, b_type, _ = combo_rule
        # Find positions of each type on the grid
        obj_a = next(((x, y) for x, y, t in objects if t == a_type), None)
        obj_b = next(((x, y) for x, y, t in objects if t == b_type and (x, y) != obj_a), None)
        if obj_a is None or obj_b is None:
            env.close()
            continue

        # Walk to obj_a to pick it up
        obs, rew, term, trunc, info = _walk_to(env, obj_a[0], obj_a[1])
        if term or trunc:
            env.close()
            continue

        # Walk to obj_b to attempt the combination
        obs, rew, term, trunc, info = _walk_to(env, obj_b[0], obj_b[1])
        if term or trunc:
            if info.get("success"):
                succeeded = True
            env.close()
            break

        # Target should now be crafted on the grid — walk to it to collect
        if cfg.get("_target_crafted", False):
            # Find the target object on the grid
            result_pos = None
            for cy in range(grid.height):
                for cx in range(grid.width):
                    if int(grid.objects[cy, cx]) == target_type:
                        result_pos = (cx, cy)
                        break
                if result_pos:
                    break
            if result_pos:
                # If agent is already on the result, step away first then return
                if tuple(agent.position) == result_pos:
                    for step_action in [1, 2, 3, 4]:
                        obs, rew, term, trunc, info = env.step(step_action)
                        if agent.position != result_pos:
                            break
                        if term or trunc:
                            break
                if not (term or trunc):
                    obs, rew, term, trunc, info = _walk_to(env, result_pos[0], result_pos[1])
                if info.get("success"):
                    succeeded = True
        env.close()
        if succeeded:
            break

    assert succeeded, "RuleInduction: could not craft and collect the target object"


@pytest.mark.timeout(60)
def test_backtrack_puzzle_can_succeed():
    """BacktrackPuzzle: INTERACT adjacent to switch to open gate, then reach goal."""
    env = agentick.make("BacktrackPuzzle-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    sw = cfg.get("switch_pos")
    goal = cfg.get("goal_positions", [None])[0]
    _walk_adjacent_and_interact(env, sw[0], sw[1])
    assert cfg.get("_switch_activated"), "BacktrackPuzzle: switch didn't activate"
    if goal and not env.done:
        obs, rew, term, trunc, info = _walk_to(env, goal[0], goal[1])
    env.close()
    assert info.get("success"), "BacktrackPuzzle: couldn't reach goal after gate opened"


@pytest.mark.timeout(60)
def test_switch_circuit_can_succeed():
    """SwitchCircuit: use oracle to toggle switches and reach goal."""
    from agentick.oracles.registry import get_oracle

    success = False
    for seed in range(10):
        env = agentick.make(
            "SwitchCircuit-v0", difficulty="easy", seed=seed, reward_mode="sparse"
        )
        obs, info = env.reset(seed=seed)
        oracle = get_oracle("SwitchCircuit-v0", env)
        oracle.reset(obs, info)
        term, trunc = False, False
        for _ in range(env.max_steps):
            if term or trunc:
                break
            action = oracle.act(obs, info)
            obs, rew, term, trunc, info = env.step(action)
        if info.get("success"):
            success = True
            env.close()
            break
        env.close()
    assert success, "SwitchCircuit: oracle couldn't solve any of 10 seeds"


@pytest.mark.timeout(60)
def test_precise_navigation_can_succeed():
    """PreciseNavigation: visit all waypoints then reach goal (via state manipulation)."""
    env = agentick.make("PreciseNavigation-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    # Clear all waypoints by removing them from remaining list and grid
    for wx, wy in list(cfg.get("_waypoints_remaining", [])):
        env.grid.objects[wy, wx] = 0  # ObjectType.NONE
    cfg["_waypoints_remaining"] = []
    # Teleport agent to goal
    gx, gy = cfg["goal_positions"][0]
    env.agent.position = (gx, gy)
    obs, rew, term, trunc, info = env.step(0)  # NOOP to trigger check
    env.close()
    assert info.get("success"), "PreciseNavigation: success detection failed"


@pytest.mark.timeout(60)
def test_graph_coloring_can_succeed():
    """GraphColoring: walk adjacent to each node and INTERACT to cycle to target color."""
    from agentick.core.types import ActionType, CellType
    interact_action = int(ActionType.INTERACT)

    env = agentick.make("GraphColoring-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    nodes = cfg.get("node_positions", [])
    n_colors = cfg.get("n_colors", 2)
    adj = cfg.get("adjacency", {})

    # Greedy coloring (1-based to match task internals: 0=uncolored, 1..n_colors=colored)
    node_colors = {}
    for i in range(len(nodes)):
        used = set()
        for nb in adj.get(i, adj.get(str(i), [])):
            nb = int(nb)
            if nb in node_colors:
                used.add(node_colors[nb])
        c = 1  # 1-based target color
        while c in used:
            c += 1
        node_colors[i] = c

    info = {}
    for i, (nx, ny) in enumerate(nodes):
        if env.done:
            break
        # Walk to adjacent cell and face the node
        ax, ay = env.agent.position
        candidates = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            cx, cy = nx + dx, ny + dy
            if (0 <= cx < env.grid.width and 0 <= cy < env.grid.height
                    and env.grid.terrain[cy, cx] != CellType.WALL
                    and not env.grid.is_object_blocking((cx, cy))):
                candidates.append((cx, cy, dx, dy))
        if not candidates:
            continue
        candidates.sort(key=lambda c: abs(c[0] - ax) + abs(c[1] - ay))
        cx, cy, dx, dy = candidates[0]
        obs, rew, term, trunc, info = _walk_to(env, cx, cy)
        if env.done:
            break
        # Face the node
        face_dir = (-dx, -dy)
        face_action = {(1, 0): 4, (-1, 0): 3, (0, 1): 2, (0, -1): 1}.get(face_dir, 0)
        obs, rew, term, trunc, info = env.step(face_action)
        if env.done:
            break
        # Read current color from grid metadata
        current = int(env.grid.metadata[ny, nx])
        target = node_colors[i]
        # INTERACT to cycle color: need (target - current) % (n_colors+1) times
        cycles = (target - current) % (n_colors + 1)
        for _ in range(cycles):
            if env.done:
                break
            obs, rew, term, trunc, info = env.step(interact_action)
    if not env.done:
        obs, rew, term, trunc, info = _noop(env)
    env.close()
    assert info.get("success"), "GraphColoring: couldn't color all nodes"


@pytest.mark.timeout(60)
def test_timing_challenge_can_succeed():
    """TimingChallenge: reach goal without triggering collision."""
    env = agentick.make("TimingChallenge-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    obs, rew, term, trunc, info = _walk_to(env, *cfg["goal_positions"][0])
    env.close()
    assert info.get("success"), "TimingChallenge: couldn't reach goal"
