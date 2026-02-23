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

    if task_name == "LightsOut-v0":
        # LightsOut toggles cells when stepped on — teleport next to each light
        # then step onto it to avoid toggling other lights along the path
        from agentick.core.types import CellType
        for wp in switches:
            lx, ly = wp
            # Place agent adjacent to light (try left first, then other directions)
            for dx, dy, action in [(-1, 0, 4), (1, 0, 3), (0, -1, 2), (0, 1, 1)]:
                nx, ny = lx + dx, ly + dy
                if (0 < nx < env.grid.width - 1 and 0 < ny < env.grid.height - 1
                        and env.grid.terrain[ny, nx] != CellType.WALL):
                    env.agent.position = (nx, ny)
                    obs, rew, term, trunc, info = env.step(action)
                    break
            if term or trunc:
                break
    else:
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
    assert info.get("success"), "MultiGoalRoute: couldn't trigger success"


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
    # tool_positions is a dict mapping tool_name -> [x, y]
    tool_positions = cfg.get("tool_positions", {})
    for tool_name, pos in tool_positions.items():
        _walk_to(env, pos[0], pos[1])
    goal = cfg["goal_positions"][0]
    obs, rew, term, trunc, info = _walk_to(env, goal[0], goal[1])
    if not (term or trunc):
        obs, rew, term, trunc, info = _noop(env)
    env.close()
    assert info["success"], "ToolUse: success didn't fire after collecting tools and reaching goal"


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
    """TaskInterference: collect coins and gems, deliver to matching goals."""
    env = agentick.make("TaskInterference-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    coins = cfg.get("coin_positions", [])
    gems = cfg.get("gem_positions", [])
    coin_goal = cfg.get("coin_goal")
    gem_goal = cfg.get("gem_goal")
    term, trunc, info = False, False, {}
    # Collect all coins first
    for cp in coins:
        if term or trunc:
            break
        _walk_to(env, cp[0], cp[1])
    # Deliver coins
    if coin_goal and not term and not trunc:
        _walk_to(env, coin_goal[0], coin_goal[1])
    # Collect all gems
    for gp in gems:
        if term or trunc:
            break
        _walk_to(env, gp[0], gp[1])
    # Deliver gems
    if gem_goal and not term and not trunc:
        obs, rew, term, trunc, info = _walk_to(env, gem_goal[0], gem_goal[1])
    if not term and not trunc:
        obs, rew, term, trunc, info = _noop(env)
    env.close()
    assert info.get("success"), "TaskInterference: couldn't complete all objectives"


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
    from agentick.core.types import ObjectType
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
    """CooperativeTransport: push box to target using Sokoban mechanic (any direction)."""
    # Direction maps: (push_action, pre_push_offset_from_box, box_delta)
    # To push east: stand west of box, take east action
    # To push west: stand east of box, take west action
    # To push south: stand north of box, take south action
    # To push north: stand south of box, take north action
    ACTION_FOR_DELTA = {(1, 0): 4, (-1, 0): 3, (0, 1): 2, (0, -1): 1}

    env = agentick.make("CooperativeTransport-v0", difficulty="easy", seed=42, reward_mode="sparse")
    env.reset(seed=42)
    cfg = env.task_config
    box = cfg.get("box_pos", [3, 3])
    target = cfg.get("target_pos", [3, 5])
    bx, by = box[0], box[1]
    tx, ty = target[0], target[1]
    last_info = {"success": False}

    # Determine push direction(s) needed
    push_steps = []
    if tx > bx:  # push east: stand west (bx-1, by), step east
        for _ in range(tx - bx):
            push_steps.append(((1, 0), (-1, 0)))  # (box_delta, pre_offset)
    elif tx < bx:  # push west: stand east (bx+1, by), step west
        for _ in range(bx - tx):
            push_steps.append(((-1, 0), (1, 0)))
    if ty > by:  # push south: stand north (bx, by-1), step south
        for _ in range(ty - by):
            push_steps.append(((0, 1), (0, -1)))
    elif ty < by:  # push north: stand south (bx, by+1), step north
        for _ in range(by - ty):
            push_steps.append(((0, -1), (0, 1)))

    for box_delta, pre_offset in push_steps:
        if env.done:
            break
        # Navigate to position opposite the push direction (pre_offset from current box)
        pre_x = bx + pre_offset[0]
        pre_y = by + pre_offset[1]
        path = _bfs_nobox(env, pre_x, pre_y)
        if path:
            prev = env.agent.position
            for nxt in path[1:]:
                if env.done:
                    break
                ddx, ddy = nxt[0] - prev[0], nxt[1] - prev[1]
                env.step(ACTION_FOR_DELTA[(ddx, ddy)])
                prev = env.agent.position
        if not env.done:
            push_action = ACTION_FOR_DELTA[box_delta]
            _, rew, term, trunc, last_info = env.step(push_action)
            bx += box_delta[0]
            by += box_delta[1]

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
    """GraphColoring: walk to each node and INTERACT to cycle to target color."""
    from agentick.core.types import ActionType
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
        # Walk to node
        obs, rew, term, trunc, info = _walk_to(env, nx, ny)
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
