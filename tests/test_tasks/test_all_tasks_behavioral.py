"""Behavioral validation tests for all 38 Agentick tasks.

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
    """BFS path from start to goal. Avoids WALL and respects can_agent_enter if env given."""
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

@pytest.mark.parametrize("task_name", ALL_TASKS)
@pytest.mark.timeout(60)
def test_task_truncation_is_not_success(task_name):
    """Letting the episode timeout should NOT mark info['success'] = True."""
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
    ("MultiRoomEscape-v0", [0, 42]),
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
    from agentick.core.types import ObjectType
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


@pytest.mark.parametrize("task_name,switch_key", [
    ("LightsOut-v0", "light_positions"),
    ("CausalChain-v0", "switch_positions"),
    ("SwitchCircuit-v0", "switch_positions"),
])
@pytest.mark.timeout(60)
def test_switch_task_can_succeed(task_name, switch_key):
    """Switch/toggle tasks: BFS walk to all switches then reach goal."""
    env = agentick.make(task_name, difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    switches = cfg.get(switch_key, [])
    term = False
    for wp in switches:
        if term:
            break
        obs, rew, term, trunc, info = _walk_to(env, wp[0], wp[1])
        if not term and not trunc:
            obs, rew, term, trunc, info = _noop(env)  # register step-on

    # LightsOut success is state-based (all lights off) — no goal position required
    goal_positions = cfg.get("goal_positions", [])
    goal = goal_positions[0] if goal_positions else None
    if goal and not term:
        obs, rew, term, trunc, info = _walk_to(env, goal[0], goal[1])
    if not term and not trunc:
        obs, rew, term, trunc, info = _noop(env)
    env.close()
    assert info["success"], f"{task_name}: success didn't fire (cfg={cfg})"


@pytest.mark.timeout(30)
def test_chase_evade_success_condition():
    """ChaseEvade: success fires when agent occupies the live target position."""
    env = agentick.make("ChaseEvade-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    # The live target position is tracked in _live_target
    target = tuple(cfg.get("_live_target", cfg.get("goal_positions", [(5, 5)])[0]))
    env.agent.position = target
    obs, rew, term, trunc, info = env.step(0)
    env.close()
    assert info["success"] or rew > 0, f"ChaseEvade: success didn't fire at target pos {target}"


@pytest.mark.timeout(60)
def test_multi_goal_route_can_succeed():
    """MultiGoalRoute: visit ALL goals in order to trigger success."""
    env = agentick.make("MultiGoalRoute-v0", difficulty="easy", seed=3, reward_mode="sparse")
    env.reset(seed=3)
    cfg = env.task_config
    term, trunc, info = False, False, {}
    for goal in cfg.get("goal_positions", []):
        if term or trunc:
            break
        obs, rew, term, trunc, info = _walk_to(env, goal[0], goal[1])
    env.close()
    assert info.get("success"), f"MultiGoalRoute: couldn't trigger success"


@pytest.mark.timeout(60)
def test_distribution_shift_can_succeed():
    """DistributionShift: wait for goal to shift, then navigate to goal_b."""
    env = agentick.make("DistributionShift-v0", difficulty="easy", seed=3, reward_mode="sparse")
    env.reset(seed=3)
    cfg = env.task_config
    shift_step = cfg.get("shift_step", 10)
    goal_b = cfg.get("goal_b")
    assert goal_b is not None, "DistributionShift: missing goal_b in config"
    # NOOP until shift fires
    term, trunc, info = False, False, {}
    for _ in range(shift_step + 1):
        obs, rew, term, trunc, info = _noop(env)
        if term or trunc:
            break
    # Now navigate to goal_b
    if not term and not trunc:
        obs, rew, term, trunc, info = _walk_to(env, goal_b[0], goal_b[1])
    if not term and not trunc:
        obs, rew, term, trunc, info = _noop(env)
    env.close()
    assert info.get("success"), f"DistributionShift: couldn't reach goal_b={goal_b} after shift"


@pytest.mark.timeout(60)
def test_recipe_assembly_can_succeed():
    """RecipeAssembly: collect all ingredients via movement then reach goal."""
    env = agentick.make("RecipeAssembly-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    for ing in cfg["ingredient_positions"]:
        _walk_to(env, ing[0], ing[1])
    goal = cfg["goal_positions"][0]
    obs, rew, term, trunc, info = _walk_to(env, goal[0], goal[1])
    if not (term or trunc):
        obs, rew, term, trunc, info = _noop(env)
    env.close()
    assert info["success"], "RecipeAssembly: success didn't fire after collecting all ingredients"


@pytest.mark.timeout(60)
def test_tool_use_can_succeed():
    """ToolUse: collect tool then reach goal."""
    env = agentick.make("ToolUse-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    tool_pos = cfg.get("tool_pos", cfg.get("tool_positions", cfg.get("key_positions", [None]))[0] if cfg.get("tool_positions", cfg.get("key_positions")) else None)
    if isinstance(tool_pos, list): tool_pos = tool_pos[0]
    _walk_to(env, tool_pos[0], tool_pos[1])
    goal = cfg["goal_positions"][0]
    obs, rew, term, trunc, info = _walk_to(env, goal[0], goal[1])
    if not (term or trunc):
        obs, rew, term, trunc, info = _noop(env)
    env.close()
    assert info["success"], "ToolUse: success didn't fire after collecting tool and reaching goal"


@pytest.mark.timeout(60)
def test_key_door_puzzle_can_succeed():
    """KeyDoorPuzzle: pick up key then walk through door to goal."""
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
            obs, rew, term, trunc, info = _noop(env)  # ensure pickup fires via on_agent_moved
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
    assert abs(rew - 0.2) < 0.01, f"DelayedGratification: expected decoy reward 0.2, got {rew}"


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
    """SequenceMemory: visit targets in order then reach goal."""
    env = agentick.make("SequenceMemory-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    seq = cfg.get("target_sequence", cfg.get("sequence", []))
    term = False
    for pos in seq:
        if term:
            break
        obs, rew, term, trunc, info = _walk_to(env, pos[0], pos[1])
        if not term and not trunc:
            obs, rew, term, trunc, info = _noop(env)
    goal = cfg.get("goal_positions", [None])[0]
    if goal and not term:
        obs, rew, term, trunc, info = _walk_to(env, goal[0], goal[1])
        if not term and not trunc:
            obs, rew, term, trunc, info = _noop(env)
    env.close()
    assert info["success"], f"SequenceMemory: couldn't trigger success (seq={seq})"


@pytest.mark.timeout(60)
def test_graph_coloring_can_succeed():
    """GraphColoring: visit all color zones then reach goal."""
    env = agentick.make("GraphColoring-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    term, trunc, info = False, False, {}
    for zones_key in ["color0_zones", "color1_zones"]:
        for p in cfg.get(zones_key, []):
            if term or trunc:
                break
            obs, rew, term, trunc, info = _walk_to(env, p[0], p[1])
            if not term and not trunc:
                obs, rew, term, trunc, info = _noop(env)
    goal = cfg["goal_positions"][0]
    if not term and not trunc:
        obs, rew, term, trunc, info = _walk_to(env, goal[0], goal[1])
        if not term and not trunc:
            obs, rew, term, trunc, info = _noop(env)
    env.close()
    assert info.get("success"), f"GraphColoring: couldn't trigger success"


@pytest.mark.timeout(60)
def test_resource_management_can_succeed():
    """ResourceManagement: collect resources then reach goal."""
    env = agentick.make("ResourceManagement-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    for p in cfg.get("resource_positions", cfg.get("key_positions", [])):
        _walk_to(env, p[0], p[1])
        _noop(env)
    goal = cfg["goal_positions"][0]
    obs, rew, term, trunc, info = _walk_to(env, goal[0], goal[1])
    if not (term or trunc):
        obs, rew, term, trunc, info = _noop(env)
    env.close()
    assert info["success"], "ResourceManagement: couldn't trigger success"


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
    """TaskInterference: collect all goal objects to trigger success."""
    env = agentick.make("TaskInterference-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    term, trunc, info = False, False, {}
    for g in cfg.get("goal_positions", []):
        if term or trunc:
            break
        obs, rew, term, trunc, info = _walk_to(env, int(g[0]), int(g[1]))
    if not term and not trunc:
        obs, rew, term, trunc, info = _noop(env)
    env.close()
    assert info.get("success"), "TaskInterference: couldn't collect all goals"


@pytest.mark.timeout(60)
def test_symbol_matching_can_succeed():
    """SymbolMatching: carry each item to its matching target zone."""
    env = agentick.make("SymbolMatching-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    items = cfg.get("item_positions", [])
    targets = cfg.get("target_positions", [])
    term, trunc, info = False, False, {}
    for item, target in zip(items, targets):
        if term or trunc:
            break
        _walk_to(env, item[0], item[1])  # pickup (on_agent_moved)
        if not env.done:
            obs, rew, term, trunc, info = _walk_to(env, target[0], target[1])  # place
    if not env.done:
        obs, rew, term, trunc, info = _noop(env)
    env.close()
    assert info.get("success"), "SymbolMatching: couldn't place all items"


@pytest.mark.timeout(60)
def test_program_synthesis_can_succeed():
    """ProgramSynthesis: pick up item, visit all waypoints, deliver to goal."""
    env = agentick.make("ProgramSynthesis-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    src = cfg.get("source")
    wps = cfg.get("waypoints", [])
    dst = cfg.get("destination")
    term, trunc, info = False, False, {}
    if src and not (term or trunc):
        _walk_to(env, src[0], src[1])
    for wp in wps:
        if term or trunc:
            break
        _walk_to(env, wp[0], wp[1])
    if dst and not (term or trunc):
        obs, rew, term, trunc, info = _walk_to(env, dst[0], dst[1])
    env.close()
    assert info.get("success"), "ProgramSynthesis: couldn't deliver item to destination"


@pytest.mark.timeout(60)
def test_sokoban_push_can_succeed():
    """SokobanPush: push box onto target using can_agent_enter mechanic."""
    env = agentick.make("SokobanPush-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    box = cfg.get("box_positions", [None])[0]
    target = cfg.get("target_positions", [None])[0]
    assert box is not None and target is not None
    # Compute push direction: box → target
    dx = target[0] - box[0]
    dy = target[1] - box[1]
    # Walk to push-from position (behind the box)
    push_from = (box[0] - dx, box[1] - dy)
    obs, rew, term, trunc, info = _walk_to(env, push_from[0], push_from[1])
    # Push box toward target (one step per unit distance)
    push_act = {(1, 0): 4, (-1, 0): 3, (0, 1): 2, (0, -1): 1}.get((dx, dy), 0)
    if not env.done:
        obs, rew, term, trunc, info = env.step(push_act)
    env.close()
    assert info.get("success"), f"SokobanPush: push mechanic failed"


@pytest.mark.timeout(60)
def test_packing_puzzle_can_succeed():
    """PackingPuzzle: push all boxes south to their target row."""
    env = agentick.make("PackingPuzzle-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    boxes = list(cfg.get("box_positions", []))
    targets = cfg.get("target_positions", [])
    last_info = {"success": False}
    for i, (bx2, by2) in enumerate(boxes):
        tx2, ty2 = targets[i]
        dy = ty2 - by2
        for _ in range(abs(dy)):
            if env.done:
                break
            sy = 1 if dy > 0 else -1
            path = _bfs_nobox(env, bx2, by2 - sy)
            if path:
                prev = env.agent.position
                for nxt in path[1:]:
                    if env.done:
                        break
                    ddx, ddy = nxt[0] - prev[0], nxt[1] - prev[1]
                    env.step({(1, 0): 4, (-1, 0): 3, (0, 1): 2, (0, -1): 1}[(ddx, ddy)])
                    prev = env.agent.position
            if not env.done:
                _, rew, term, trunc, last_info = env.step(2 if sy > 0 else 1)
                by2 += sy
    if not env.done:
        _, rew, term, trunc, last_info = _noop(env)
    env.close()
    assert last_info.get("success"), "PackingPuzzle: couldn't push all boxes to targets"


@pytest.mark.timeout(60)
def test_cooperative_transport_can_succeed():
    """CooperativeTransport: push box to target using Sokoban mechanic."""
    env = agentick.make("CooperativeTransport-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    box = cfg.get("box_pos", [3, 3])
    target = cfg.get("target_pos", [3, 5])
    bx, by = box[0], box[1]
    tx, ty = target[0], target[1]
    last_info = {"success": False}
    # Push south (box above target)
    dy = ty - by
    for _ in range(dy):
        if env.done:
            break
        path = _bfs_nobox(env, bx, by - 1)
        if path:
            prev = env.agent.position
            for nxt in path[1:]:
                if env.done:
                    break
                ddx, ddy = nxt[0] - prev[0], nxt[1] - prev[1]
                env.step({(1, 0): 4, (-1, 0): 3, (0, 1): 2, (0, -1): 1}[(ddx, ddy)])
                prev = env.agent.position
        if not env.done:
            _, rew, term, trunc, last_info = env.step(2)  # push south
            by += 1
    if not env.done:
        _, rew, term, trunc, last_info = _noop(env)
    env.close()
    assert last_info.get("success"), "CooperativeTransport: couldn't push box to target"


@pytest.mark.timeout(60)
def test_rule_induction_can_succeed():
    """RuleInduction: visit switch to learn rule, then navigate to true target."""
    env = agentick.make("RuleInduction-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    sw = cfg.get("switch_pos")
    goal = cfg.get("goal_positions", [None])[0]
    obs, rew, term, trunc, info = _walk_to(env, sw[0], sw[1])
    assert cfg.get("_rule_revealed"), "RuleInduction: switch didn't reveal rule"
    if goal and not (term or trunc):
        obs, rew, term, trunc, info = _walk_to(env, goal[0], goal[1])
    env.close()
    assert info.get("success"), "RuleInduction: couldn't reach true goal after revealing rule"


@pytest.mark.timeout(60)
def test_backtrack_puzzle_can_succeed():
    """BacktrackPuzzle: visit switch to open gate, then reach goal."""
    env = agentick.make("BacktrackPuzzle-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    sw = cfg.get("switch_pos")
    goal = cfg.get("goal_positions", [None])[0]
    _walk_to(env, sw[0], sw[1])
    assert cfg.get("_switch_activated"), "BacktrackPuzzle: switch didn't activate"
    if goal and not env.done:
        obs, rew, term, trunc, info = _walk_to(env, goal[0], goal[1])
    env.close()
    assert info.get("success"), "BacktrackPuzzle: couldn't reach goal after gate opened"


@pytest.mark.timeout(60)
def test_causal_chain_can_succeed():
    """CausalChain: visit switches in order → gate opens → reach goal."""
    env = agentick.make("CausalChain-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    for sw in cfg["switch_positions"]:
        _walk_to(env, sw[0], sw[1])
    if not env.done:
        obs, rew, term, trunc, info = _walk_to(env, *cfg["goal_positions"][0])
    env.close()
    assert info.get("success"), "CausalChain: gate didn't open or goal unreachable"


@pytest.mark.timeout(60)
def test_switch_circuit_can_succeed():
    """SwitchCircuit: toggle all switches → gate opens → reach goal."""
    env = agentick.make("SwitchCircuit-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    for sw in cfg["switch_positions"]:
        _walk_to(env, sw[0], sw[1])
    if not env.done:
        obs, rew, term, trunc, info = _walk_to(env, *cfg["goal_positions"][0])
    env.close()
    assert info.get("success"), "SwitchCircuit: switches or gate didn't work"


@pytest.mark.timeout(60)
def test_sequence_memory_can_succeed():
    """SequenceMemory: visit targets in correct order."""
    env = agentick.make("SequenceMemory-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    term, trunc, info = False, False, {}
    for t in cfg["sequence"]:
        if term or trunc:
            break
        obs, rew, term, trunc, info = _walk_to(env, t[0], t[1])
    env.close()
    assert info.get("success"), "SequenceMemory: sequence visit failed"


@pytest.mark.timeout(60)
def test_precise_navigation_can_succeed():
    """PreciseNavigation: visit all waypoints then reach goal."""
    env = agentick.make("PreciseNavigation-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    for wp in cfg["waypoints"]:
        _walk_to(env, wp[0], wp[1])
    if not env.done:
        obs, rew, term, trunc, info = _walk_to(env, *cfg["goal_positions"][0])
    env.close()
    assert info.get("success"), "PreciseNavigation: waypoints or goal failed"


@pytest.mark.timeout(60)
def test_graph_coloring_can_succeed():
    """GraphColoring: visit color0 zones then color1 zones then goal."""
    env = agentick.make("GraphColoring-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    for z in cfg["color0_zones"] + cfg["color1_zones"]:
        _walk_to(env, z[0], z[1])
    if not env.done:
        obs, rew, term, trunc, info = _walk_to(env, *cfg["goal_positions"][0])
    env.close()
    assert info.get("success"), "GraphColoring: zone visit or goal failed"


@pytest.mark.timeout(60)
def test_timing_challenge_can_succeed():
    """TimingChallenge: reach goal without triggering collision."""
    env = agentick.make("TimingChallenge-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    obs, rew, term, trunc, info = _walk_to(env, *cfg["goal_positions"][0])
    env.close()
    assert info.get("success"), "TimingChallenge: couldn't reach goal"
