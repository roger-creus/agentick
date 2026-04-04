# Public Launch Readiness: Audit & Fix Design

**Date**: 2026-04-03
**Status**: Draft
**Goal**: Ensure all 38 tasks are logically correct across all difficulties, all 5 render modes expose identical information, and core system bugs are fixed before public release.

---

## Problem Statement

Agentick is a benchmark for evaluating AI agents across 38 procedurally generated gridworld tasks. Before public launch, three categories of issues must be resolved:

1. **Task generation correctness**: Several tasks can produce logically nonsensical levels (doors not blocking paths, unsolvable layouts, degenerate edge cases from random generation without proper validation).

2. **Multimodal rendering disparity**: The 5 observation modes (ascii, language, language_structured, rgb_array, state_dict) expose unequal information. An agent using pixel observations sees key colors, door states, and NPC types that are invisible in language_structured or state_dict mode. This makes benchmark comparisons unfair.

3. **Core system bugs**: `Grid.bfs()` ignores blocking objects, `compute_action_mask()` reports invalid moves as valid, and `AgentickEnv._move_agent()` allows walking through closed doors.

## Design

### Three Sequential PRs

#### PR 1: Core System Bugs + Cleanup (~150 lines, 5-7 files)

| Fix | File | Change |
|-----|------|--------|
| `Grid.bfs()` blocking objects | `core/grid.py:199-230` | Add `check_objects: bool = False` parameter |
| `AgentickEnv._move_agent()` | `core/env.py` | Add `is_object_blocking()` check |
| `compute_action_mask()` | `core/env.py:~306` | Include object blocking in walkable mask |
| Dead `rgb_array_flat` refs | 3 script/template files | Remove stale references |

**Key decision**: `Grid.bfs()` and `flood_fill()` default to `check_objects=False` because generation code relies on treating doors as passable during level building (doors will be opened later by the agent). Callers that need runtime pathfinding opt in with `check_objects=True`.

#### PR 2: Task Logical Correctness (~500-800 lines, 15-20 files)

Every task's `generate()` must produce levels that are logically sound. For each task:

1. **Audit generation**: Read `generate()`, identify placement logic, check for edge cases
2. **Add/strengthen validation**: `validate_instance()` must verify the level makes sense
3. **Add retry on failure**: If validation fails, regenerate with a different seed (up to N attempts)
4. **Fix specific bugs**: Per-task issues identified during audit

The full per-task breakdown is in the implementation plan. Key systemic fixes:

- Tasks with doors/keys must verify doors are actual chokepoints (blocking the only path to goal)
- Tasks with random object placement must verify objects serve their intended purpose
- Tasks with NPCs must verify NPC behavior doesn't trivially prevent success
- All tasks must have a working `validate_instance()` override (not just the base class flood-fill)

#### PR 3: Multimodal Rendering Parity (~800-1200 lines, 5-8 files)

**Architecture**: Create a shared `TaskAnnotations` dataclass in `core/annotations.py` that all 5 renderers consume. This centralizes metadata interpretation (currently duplicated in ascii and isometric, missing in language_structured and state_dict).

**What `TaskAnnotations` contains**:
- Per-cell metadata interpretation: key/door colors, door open/closed states, switch on/off states, NPC behavior types, resource energy levels, tile numbers, scroll directions
- Task-level annotations: target objects (InstructionFollowing), meters (TaskInterference), trial info (RuleInduction), phase info (DistributionShift), clues (TreasureHunt)

**Per-renderer changes**:

| Renderer | Current State | Changes Needed |
|----------|--------------|----------------|
| **ascii** | Most complete (reference) | Refactor to use shared annotations |
| **language** | Missing scroll directions, switch colors, tile numbers | Add from annotations |
| **language_structured** | ~60% of info missing | Major overhaul: add metadata interpretation, entity properties, task annotations |
| **rgb_array (iso)** | Mostly complete | Verify TreasureHunt clues, refactor to shared annotations |
| **state_dict** | Raw arrays only, no interpretation | Add metadata array, annotations dict, task info |

**language_structured new output format** (additions in bold):

```json
{
    "description": "A 10x10 gridworld environment",
    "position": {"x": 3, "y": 5},
    "orientation": "north",
    "visible_entities": [
        {
            "type": "key",
            "position": [2, 3],
            "distance": 3,
            "color": "red",           # NEW
        },
        {
            "type": "door",
            "position": [5, 7],
            "distance": 4,
            "color": "red",           # NEW
            "state": "closed",        # NEW
        },
        {
            "type": "npc",
            "position": [4, 2],
            "distance": 5,
            "behavior": "follower",   # NEW
        },
    ],
    "task_annotations": { ... },      # NEW
    "terrain_map": { ... },           # NEW: ICE/WATER/HAZARD cell positions
    ...
}
```

**state_dict new output format** (additions in bold):

```json
{
    "grid": {
        "height": 10, "width": 10,
        "terrain": [...], "objects": [...], "agents": [...],
        "metadata": [...],            # NEW: was missing!
    },
    "agent": { ... },
    "entities": [...],
    "annotations": {                  # NEW
        "key_colors": {"2,3": "red"},
        "door_states": {"5,7": "closed"},
        "npc_types": {"4,2": "follower"},
        "task_info": { ... },
    },
    "info": { ... },
}
```

### Testing Strategy

1. **PR 1**: Unit tests for blocking objects in `test_core/test_blocking_objects.py`
2. **PR 2**: `test_tasks/test_generation_correctness.py` — 38 tasks x 4 difficulties x 20 seeds = 3040 test cases
3. **PR 3**: `test_integration/test_rendering_parity.py` — for each task, render all 5 modes, assert identical semantic info
4. **Baseline**: 16 pre-existing failures recorded. Any new failure = regression.

### Non-Goals

- No task removals or major redesigns
- No changes to the Gymnasium API surface
- No new render modes
- No changes to oracle implementations (they'll be validated but not modified unless a task fix requires it)

## Risks

| Risk | Mitigation |
|------|------------|
| Task generation fix breaks oracle | Run oracle test suite after each task fix |
| Rendering changes break downstream consumers | Only add new keys/fields, never remove existing ones |
| Generation stress test reveals many broken seeds | Retry loop with configurable max attempts |
| language_structured overhaul is large | Test incrementally per entity type |
