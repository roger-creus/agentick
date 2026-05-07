"""Verify that all render modes expose identical semantic information.

For each task, generates an instance and checks that key facts (door colors,
switch states, NPC types, task annotations, etc.) are present in ALL modes
that support them.
"""

from __future__ import annotations

import pytest

import agentick
from agentick.core.annotations import extract_annotations
from agentick.core.types import CellType, ObjectType
from agentick.tasks.registry import list_tasks

ALL_TASKS = list_tasks()


@pytest.mark.parametrize("task_name", ALL_TASKS)
def test_state_dict_has_metadata_layer(task_name):
    """state_dict must include the metadata grid layer."""
    env = agentick.make(task_name, difficulty="easy", render_mode="state_dict")
    obs, info = env.reset(seed=42)
    assert "metadata" in obs["grid"], f"{task_name}: state_dict missing metadata layer"
    env.close()


@pytest.mark.parametrize("task_name", ALL_TASKS)
def test_state_dict_has_annotations(task_name):
    """state_dict must include an annotations dict."""
    env = agentick.make(task_name, difficulty="easy", render_mode="state_dict")
    obs, info = env.reset(seed=42)
    assert "annotations" in obs, f"{task_name}: state_dict missing annotations"
    env.close()


@pytest.mark.parametrize("task_name", ALL_TASKS)
def test_structured_has_task_annotations_key(task_name):
    """language_structured must include task_annotations dict."""
    env = agentick.make(task_name, difficulty="easy", render_mode="language_structured")
    obs, info = env.reset(seed=42)
    assert "task_annotations" in obs, f"{task_name}: language_structured missing task_annotations"
    env.close()


@pytest.mark.parametrize("task_name", ALL_TASKS)
def test_key_door_colors_in_structured(task_name):
    """If keys/doors exist, their colors must appear in language_structured entities."""
    env = agentick.make(task_name, difficulty="easy", render_mode="language_structured")
    obs, info = env.reset(seed=42)

    ann = extract_annotations(env.grid, env._get_info())
    if not ann.key_colors and not ann.door_colors:
        pytest.skip("No keys or doors in this task")

    entities = obs["visible_entities"]

    # Check keys
    for pos, color in ann.key_colors.items():
        matching = [
            e for e in entities
            if e["type"] == "key" and e["position"] == [pos[0], pos[1]]
        ]
        assert matching, f"Key at {pos} not found in structured entities"
        assert matching[0].get("color") == color, (
            f"Key at {pos}: expected color={color}, got {matching[0].get('color')}"
        )

    # Check doors
    for pos, color in ann.door_colors.items():
        matching = [
            e for e in entities
            if e["type"] == "door" and e["position"] == [pos[0], pos[1]]
        ]
        assert matching, f"Door at {pos} not found in structured entities"
        assert matching[0].get("color") == color, (
            f"Door at {pos}: expected color={color}, got {matching[0].get('color')}"
        )
    env.close()


@pytest.mark.parametrize("task_name", ALL_TASKS)
def test_door_states_in_state_dict(task_name):
    """If doors exist, their open/closed state must appear in state_dict annotations."""
    env = agentick.make(task_name, difficulty="easy", render_mode="state_dict")
    obs, info = env.reset(seed=42)

    ann = extract_annotations(env.grid, env._get_info())
    if not ann.door_states:
        pytest.skip("No doors in this task")

    sd_doors = obs["annotations"].get("door_states", {})
    for pos, state in ann.door_states.items():
        key = f"{pos[0]},{pos[1]}"
        assert key in sd_doors, f"Door at {pos} not in state_dict annotations"
        assert sd_doors[key] == state, (
            f"Door at {pos}: expected {state}, got {sd_doors[key]}"
        )
    env.close()


@pytest.mark.parametrize("task_name", ALL_TASKS)
def test_switch_states_in_state_dict(task_name):
    """If switches exist, their on/off state must appear in state_dict annotations."""
    env = agentick.make(task_name, difficulty="easy", render_mode="state_dict")
    obs, info = env.reset(seed=42)

    ann = extract_annotations(env.grid, env._get_info())
    if not ann.switch_states:
        pytest.skip("No switches in this task")

    sd_switches = obs["annotations"].get("switch_states", {})
    for pos, state in ann.switch_states.items():
        key = f"{pos[0]},{pos[1]}"
        assert key in sd_switches, f"Switch at {pos} not in state_dict annotations"
        assert sd_switches[key] == state
    env.close()


@pytest.mark.parametrize("task_name", ALL_TASKS)
def test_structured_entity_count_matches_grid(task_name):
    """language_structured must show the same number of objects as the grid has."""
    env = agentick.make(task_name, difficulty="easy", render_mode="language_structured")
    obs, info = env.reset(seed=42)

    ann = extract_annotations(env.grid, env._get_info())

    # Count non-NONE objects on grid (excluding fogged cells)
    from agentick.core.types import ObjectType
    grid_objects = 0
    for y in range(env.grid.height):
        for x in range(env.grid.width):
            if (x, y) in ann.fog_cells:
                continue
            if int(env.grid.objects[y, x]) != ObjectType.NONE:
                grid_objects += 1

    # Count object-type entities in structured output (exclude Entity-based ones)
    struct_objects = [
        e for e in obs["visible_entities"]
        if e["type"] not in ("agent", "npc_entity", "enemy_entity")
    ]

    # The structured output may also include Entity objects; count grid objects only
    grid_type_names = {ot.name.lower() for ot in ObjectType if ot != ObjectType.NONE}
    struct_grid_objects = [e for e in struct_objects if e["type"] in grid_type_names]

    assert len(struct_grid_objects) == grid_objects, (
        f"{task_name}: structured has {len(struct_grid_objects)} grid objects "
        f"but grid has {grid_objects}"
    )
    env.close()


def test_fogged_state_dict_cells_do_not_leak_hidden_grid_contents():
    """state_dict observations must mask objects/terrain under fog."""
    env = agentick.make("FogOfWarExploration-v0", difficulty="easy", render_mode="state_dict")
    obs, info = env.reset(seed=42)

    fog_cells = obs["annotations"].get("fog_cells", [])
    assert fog_cells, "FogOfWarExploration should expose explicit fog cell annotations"

    for key in fog_cells:
        x, y = map(int, key.split(","))
        assert obs["grid"]["objects"][y][x] == int(ObjectType.NONE)
        assert obs["grid"]["terrain"][y][x] == int(CellType.EMPTY)

    assert "goal_positions" not in info["task_config"]
    env.close()


@pytest.mark.parametrize(
    "task_name,hidden_keys",
    [
        ("FewShotAdaptation-v0", {"goal_positions", "true_goal", "trials", "rule_name"}),
        ("SequenceMemory-v0", {"goal_positions", "sequence", "distractors"}),
        ("TreasureHunt-v0", {"goal_positions", "_treasure_positions", "_clue_info"}),
        ("RuleInduction-v0", {"_rule_table_list", "_original_objects"}),
        ("DistributionShift-v0", {"_phase_configs", "_action_remap"}),
    ],
)
def test_public_info_omits_hidden_task_solution_state(task_name, hidden_keys):
    """Public Gymnasium info must not expose hidden solutions or future phases."""
    env = agentick.make(task_name, difficulty="easy", render_mode="state_dict")
    _obs, info = env.reset(seed=42)
    public_keys = set(info["task_config"])
    assert public_keys.isdisjoint(hidden_keys)
    env.close()


@pytest.mark.parametrize("task_name", ["PackingPuzzle-v0", "RecipeAssembly-v0"])
def test_typed_target_requirements_are_exposed_across_structured_modes(task_name):
    """Typed target slots need explicit semantics, not just raw metadata values."""
    env = agentick.make(task_name, difficulty="easy", render_mode="state_dict")
    obs, _info = env.reset(seed=42)
    typed_targets = obs["annotations"].get("typed_target_objects", {})
    assert typed_targets, f"{task_name}: state_dict missing typed target annotations"

    structured = env.render_in_mode("language_structured")
    structured_targets = [
        e for e in structured["visible_entities"]
        if e["type"] == "target" and "accepts" in e
    ]
    assert len(structured_targets) == len(typed_targets)
    env.close()


@pytest.mark.parametrize(
    "task_name",
    [
        "DynamicObstacles-v0",
        "TimingChallenge-v0",
        "ChaseEvade-v0",
        "NoisyObservation-v0",
        "TagHunt-v0",
        "TaskInterference-v0",
    ],
)
def test_dynamic_step_observation_matches_post_step_world(task_name):
    """Returned observations must include task dynamics applied during the step."""
    env = agentick.make(task_name, difficulty="easy", seed=42, render_mode="state_dict")
    env.reset(seed=42)
    obs, _reward, _terminated, _truncated, _info = env.step(0)
    assert obs["grid"]["objects"] == env.grid.objects.tolist()
    assert obs["grid"]["metadata"] == env.grid.metadata.tolist()
    env.close()
