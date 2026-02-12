"""Prompt templates and formatting for LLM/VLM agents."""

from __future__ import annotations

import re
from typing import Any

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
5: TOGGLE (interact with objects like doors, switches)
6: PICKUP (pick up items)
7: DROP (drop items from inventory)

## Task Objective
{task_description}

## Instructions
1. Observe the current grid state
2. Reason about the best action to take
3. Output your selected action as a single integer (0-7)

Respond with ONLY the action number, nothing else."""

# Direct action prompt (no reasoning)
DIRECT_ACTION_PROMPT = """Current observation:
{observation}

Valid actions: {valid_actions}

Select the best action (respond with just the number):"""

# Chain-of-thought prompt
COT_PROMPT = """Current observation:
{observation}

Valid actions: {valid_actions}

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
        # ASCII is already text
        obs_text = str(observation)

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
    valid_actions = info.get("valid_actions", [])

    prompt = f"""Task: {task_name}
Step: {step}

{obs_text}

Valid actions: {valid_actions}

Select the best action (respond with just the action number):"""

    return prompt


def parse_action_from_text(
    response_text: str,
    valid_actions: list[int],
) -> int:
    """
    Parse action from LLM response text.

    Tries multiple fallback strategies:
    1. Look for "ACTION: N"
    2. Look for bare number
    3. Look for action name (e.g., "MOVE_UP")
    4. Return random valid action

    Args:
        response_text: LLM response
        valid_actions: List of valid action indices

    Returns:
        Action index
    """
    import numpy as np

    # Strategy 1: Look for "ACTION: N" pattern
    match = re.search(r"ACTION:\s*(\d+)", response_text, re.IGNORECASE)
    if match:
        action = int(match.group(1))
        if action in valid_actions:
            return action

    # Strategy 2: Look for bare number at end
    match = re.search(r"(\d+)\s*$", response_text.strip())
    if match:
        action = int(match.group(1))
        if action in valid_actions:
            return action

    # Strategy 3: Look for any number in the text
    numbers = re.findall(r"\b(\d+)\b", response_text)
    for num_str in reversed(numbers):  # Try from end first
        action = int(num_str)
        if action in valid_actions:
            return action

    # Strategy 4: Look for action names
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
        "TOGGLE": 5,
        "PICKUP": 6,
        "DROP": 7,
    }

    for name, action_idx in action_names.items():
        if name in response_text.upper() and action_idx in valid_actions:
            return action_idx

    # Fallback: Random valid action
    rng = np.random.default_rng()
    return int(rng.choice(valid_actions))


def get_task_description(task_name: str) -> str:
    """
    Get natural language description of task objective.

    Args:
        task_name: Name of the task

    Returns:
        Task description
    """
    # Task descriptions
    descriptions = {
        "GoToGoal-v0": "Navigate to the goal position (marked as G) on the grid.",
        "MazeNavigation-v0": "Navigate through a maze to reach the goal.",
        "KeyDoorPuzzle-v0": "Find the key, then use it to unlock the door, then reach the goal.",
        "SokobanPush-v0": "Push boxes onto target positions.",
        "ToolUse-v0": "Use tools to complete the task.",
        "ChaseEvade-v0": "Chase the target or evade the chaser.",
    }

    return descriptions.get(task_name, "Complete the task objective.")


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
    valid_actions = info.get("valid_actions", [])

    return COT_PROMPT.format(
        observation=obs_text,
        valid_actions=valid_actions,
    )
