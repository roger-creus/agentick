"""Verify that all render modes expose identical semantic information.

For each task, generates an instance and checks that key facts (door colors,
switch states, NPC types, task annotations, etc.) are present in ALL modes
that support them.
"""

from __future__ import annotations

import pytest

import agentick
from agentick.core.annotations import extract_annotations
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
