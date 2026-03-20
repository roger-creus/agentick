"""Prompt templates and formatting for LLM/VLM agents."""

from __future__ import annotations

import re
from typing import Any

# Regex to strip ANSI escape sequences
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# System prompt template
SYSTEM_PROMPT = """You are an AI agent playing grid-world tasks in the Agentick benchmark.

Your goal is to navigate the grid and complete the task objective by selecting the best action at each step.

## Action Space
The available actions are:
0: NOOP (no operation, stay in place)
1: MOVE_UP
2: MOVE_DOWN
3: MOVE_LEFT
4: MOVE_RIGHT
5: INTERACT (interact with objects like switches, levers — walk onto them first, then INTERACT)

## Task Objective
{task_description}

## Instructions
1. Observe the current grid state
2. Reason about the best action to take
3. Output your selected action as a single integer (0-5)

Respond with ONLY the action number, nothing else."""

# Direct action prompt (no reasoning)
DIRECT_ACTION_PROMPT = """Current observation:
{observation}

Select the best action (respond with just the number):"""

# Chain-of-thought prompt
COT_PROMPT = """Current observation:
{observation}

Think step-by-step:
1. What do I see?
2. What is my goal?
3. What is the best action?

Output your reasoning, then on the last line output: ACTION: <number>"""

# Few-shot examples (can be customized per task)
FEW_SHOT_EXAMPLES = """
Example 1:
Observation: Grid 5x5. Agent at (1,1). Goal at (3,3).
Valid actions: [1, 2, 3, 4]
Best action: 4 (MOVE_RIGHT - move toward goal)

Example 2:
Observation: Grid 5x5. Agent at (2,2). Wall at (2,3). Goal at (2,4).
Valid actions: [1, 2, 3, 4]
Best action: 1 (MOVE_UP - go around the wall)
"""


def format_observation_to_text(
    observation: Any,
    info: dict[str, Any],
    observation_mode: str,
) -> str:
    """
    Format observation into text prompt for LLM.

    Args:
        observation: Raw observation (format depends on obs_mode)
        info: Info dict
        observation_mode: Observation format

    Returns:
        Formatted text prompt
    """
    if observation_mode == "ascii":
        # ASCII is already text; strip ANSI color codes for LLM readability
        obs_text = _ANSI_RE.sub("", str(observation))

    elif observation_mode == "language":
        # Natural language description
        obs_text = str(observation)

    elif observation_mode == "language_structured":
        # Structured dict/JSON
        if isinstance(observation, dict):
            obs_text = "\n".join(f"{k}: {v}" for k, v in observation.items())
        else:
            obs_text = str(observation)

    elif observation_mode == "rgb_array":
        # Vision models would get image, but for text-only models describe the grid
        obs_text = "[Image observation - grid rendering]"

    elif observation_mode == "state_dict":
        # Full state dict - extract key info
        if isinstance(observation, dict):
            # Extract agent position, goal, objects
            agent_pos = observation.get("agent", {}).get("position", "unknown")
            grid_size = (
                observation.get("grid", {}).get("height", "?"),
                observation.get("grid", {}).get("width", "?"),
            )
            obs_text = f"Grid size: {grid_size}, Agent position: {agent_pos}"
        else:
            obs_text = str(observation)

    else:
        obs_text = str(observation)

    # Add task context
    task_name = info.get("task_name", "unknown")
    step = info.get("step", 0)

    prompt = f"""Task: {task_name}
Step: {step}

{obs_text}

Select the best action (respond with just the action number):"""

    return prompt


def parse_action_from_text(
    response_text: str,
    valid_actions: list[int] | list[str] | list[Any],
) -> int:
    """
    Parse action from LLM response text.

    Handles both integer actions and string action names.
    Tries multiple fallback strategies:
    1. If valid_actions are strings: map action name to index
    2. If valid_actions are integers: extract number from response
    3. Look for "ACTION: N" pattern
    4. Look for bare number
    5. Look for action name (e.g., "MOVE_UP")
    6. Return random valid action

    Args:
        response_text: LLM response
        valid_actions: List of valid actions (can be ints or strings)

    Returns:
        Action index (integer)
    """
    import numpy as np

    # Convert numpy string types to regular strings
    valid_actions_normalized = []
    for action in valid_actions:
        if isinstance(action, (np.str_, np.bytes_)):
            valid_actions_normalized.append(str(action))
        else:
            valid_actions_normalized.append(action)

    valid_actions = valid_actions_normalized

    # Determine if valid_actions are integers or strings
    if valid_actions and isinstance(valid_actions[0], str):
        # valid_actions are action names (strings)
        # Build mapping from action names to indices
        action_name_to_idx = {}
        for idx, action_name in enumerate(valid_actions):
            action_name_to_idx[action_name.lower().strip()] = idx

        # Try to match response text to action names
        response_lower = response_text.lower().strip()

        # Direct match
        if response_lower in action_name_to_idx:
            return action_name_to_idx[response_lower]

        # Partial match (e.g., "move_up" contains "up")
        for action_name, idx in action_name_to_idx.items():
            if action_name in response_lower or response_lower in action_name:
                return idx

        # Try common action name patterns
        for action_name, idx in action_name_to_idx.items():
            # Handle variations like "move up" vs "move_up"
            normalized_name = action_name.replace("_", " ")
            if (
                normalized_name in response_lower
                or response_lower.replace("_", " ") == normalized_name
            ):
                return idx

        # Fallback: random valid action index
        rng = np.random.default_rng()
        return int(rng.integers(0, len(valid_actions)))

    else:
        # valid_actions are integers (traditional case)
        valid_action_ints = [int(a) for a in valid_actions]

        # Strategy 1: Look for "ACTION: N" pattern
        match = re.search(r"ACTION:\s*(\d+)", response_text, re.IGNORECASE)
        if match:
            action = int(match.group(1))
            if action in valid_action_ints:
                return action

        # Strategy 2: Look for bare number at end
        match = re.search(r"(\d+)\s*$", response_text.strip())
        if match:
            action = int(match.group(1))
            if action in valid_action_ints:
                return action

        # Strategy 3: Look for any number in the text
        numbers = re.findall(r"\b(\d+)\b", response_text)
        for num_str in reversed(numbers):  # Try from end first
            action = int(num_str)
            if action in valid_action_ints:
                return action

        # Strategy 4: Look for action names and map to integers
        action_names = {
            "NOOP": 0,
            "UP": 1,
            "MOVE_UP": 1,
            "DOWN": 2,
            "MOVE_DOWN": 2,
            "LEFT": 3,
            "MOVE_LEFT": 3,
            "RIGHT": 4,
            "MOVE_RIGHT": 4,
            "INTERACT": 5,
            "TOGGLE": 5,
        }

        for name, action_idx in action_names.items():
            if name in response_text.upper() and action_idx in valid_action_ints:
                return action_idx

        # Fallback: Random valid action
        rng = np.random.default_rng()
        return int(rng.choice(valid_action_ints))


def get_task_description(task_name: str) -> str:
    """
    Get natural language description of task objective.

    Dynamically looks up the task in the registry so that all tasks are
    covered (not just a hardcoded subset).

    Args:
        task_name: Name of the task

    Returns:
        Task description
    """
    from agentick.tasks.descriptions import (
        get_task_description as _get_task_description,
    )

    return _get_task_description(task_name)


def create_system_prompt(task_name: str) -> str:
    """
    Create system prompt for a task.

    Args:
        task_name: Name of the task

    Returns:
        System prompt text
    """
    task_description = get_task_description(task_name)
    return SYSTEM_PROMPT.format(task_description=task_description)


def create_few_shot_prompt(task_name: str, observation: Any, info: dict[str, Any]) -> str:
    """
    Create few-shot prompt with examples.

    Args:
        task_name: Name of the task
        observation: Current observation
        info: Info dict

    Returns:
        Few-shot prompt
    """
    # System prompt
    system = create_system_prompt(task_name)

    # Examples
    examples = FEW_SHOT_EXAMPLES

    # Current observation
    obs_text = format_observation_to_text(observation, info, "language")

    return f"{system}\n\n{examples}\n\nNow your turn:\n{obs_text}"


def create_cot_prompt(observation: Any, info: dict[str, Any], observation_mode: str) -> str:
    """
    Create chain-of-thought prompt.

    Args:
        observation: Current observation
        info: Info dict
        observation_mode: Observation format

    Returns:
        CoT prompt
    """
    obs_text = str(observation)

    return COT_PROMPT.format(observation=obs_text)
