# Quickstart

Get up and running with Agentick in 5 minutes with this complete tutorial.

## Prerequisites

Install Agentick (if not already done):

```bash
pip install agentick
```

## The Absolute Basics

### Step 1: Import and Create an Environment

```python
import agentick

# Create the simplest environment: GoToGoal-v0
env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="ascii", seed=42)
```

**What this does:**
- `make()` creates a new Gymnasium-compatible environment
- `"GoToGoal-v0"` is a simple navigation task (reach the goal)
- `difficulty="easy"` sets the problem difficulty (easy, medium, hard, expert)
- `render_mode="ascii"` shows the world as text (we'll cover other modes below)
- `seed=42` makes results reproducible

### Step 2: Reset the Environment

```python
obs, info = env.reset()

print("Observation:")
print(obs)
print("\nInfo:")
print(info)
```

**Output:**
```
Observation:
#####
#A.G#
#.#.#
#...#
#####

Info:
{
    'valid_actions': [0, 1, 2, 3],
    'position': (1, 1),
    'goal_position': (1, 3)
}
```

**What happens:**
- `reset()` initializes the environment and returns initial observation
- Observation shows ASCII rendering of the world (A=agent, G=goal, #=wall, .=empty)
- Info dict contains metadata like valid actions and positions

### Step 3: Take Steps in the Environment

```python
# Run a single step
action = env.action_space.sample()  # Random action
obs, reward, terminated, truncated, info = env.step(action)

print(f"Action: {action}")
print(f"Reward: {reward}")
print(f"Terminated (success): {terminated}")
print(f"Truncated (timeout): {truncated}")
print(f"Observation:\n{obs}")
```

**What happens:**
- `action_space.sample()` picks a random action from available actions (0-3 for movement)
- `step()` executes the action and returns:
  - `obs`: New observation after the action
  - `reward`: Reward signal (higher is better)
  - `terminated`: True if task succeeded
  - `truncated`: True if max steps reached
  - `info`: Additional metadata

### Step 4: Run a Complete Episode

```python
# Reset to start fresh
obs, info = env.reset(seed=42)

episode_reward = 0
episode_steps = 0

# Run until success or timeout
while True:
    # Take a random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    episode_reward += reward
    episode_steps += 1

    # Print every 5 steps
    if episode_steps % 5 == 0:
        print(f"Step {episode_steps}: Reward so far = {episode_reward:.2f}")

    # Check if episode is done
    if terminated or truncated:
        break

print(f"\nEpisode finished!")
print(f"Total reward: {episode_reward:.2f}")
print(f"Success: {info.get('success', False)}")
print(f"Steps taken: {episode_steps}")
```

**Output:**
```
Step 5: Reward so far = -0.05
Step 10: Reward so far = -0.10
Step 15: Reward so far = -0.15
Step 20: Reward so far = -0.20

Episode finished!
Total reward: -0.58
Success: False
Steps taken: 36
```

## Running Multiple Episodes

To run multiple episodes and collect statistics:

```python
import agentick

env = agentick.make("GoToGoal-v0", difficulty="easy")

n_episodes = 10
successes = 0
total_rewards = []

for episode in range(n_episodes):
    obs, info = env.reset()
    episode_reward = 0

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        if terminated or truncated:
            break

    success = info.get('success', False)
    if success:
        successes += 1
    total_rewards.append(episode_reward)

    print(f"Episode {episode+1:2d}: Return={episode_reward:7.2f}, Success={success}")

print(f"\nSummary: {successes}/{n_episodes} successes ({successes/n_episodes:.1%} success rate)")
print(f"Average return: {sum(total_rewards)/len(total_rewards):.2f}")
```

## Different Observation Modes

Agentick provides 6 observation modes for the same environment. Switch between them easily:

### 1. ASCII Text (Simple and Interpretable)

```python
env = agentick.make("GoToGoal-v0", render_mode="ascii")
obs, _ = env.reset()
print(obs)
# Output:
# #####
# #A.G#
# #.#.#
# #...#
# #####
```

### 2. Language (Natural Language Description)

```python
env = agentick.make("GoToGoal-v0", render_mode="language")
obs, _ = env.reset()
print(obs)
# Output:
# You are in a 5x5 grid room at position (1, 1). The goal is visible to your east at (1, 3).
# There are walls forming a cross pattern in the center. You can move north, south, east, or west.
```

### 3. Language Structured (Detailed Dict)

```python
env = agentick.make("GoToGoal-v0", render_mode="language_structured")
obs, _ = env.reset()
print(obs)
# Output:
# {
#     'description': '5x5 grid with agent and goal',
#     'position': (1, 1),
#     'goal_positions': [(1, 3)],
#     'obstacles': [(2, 2)],
#     'valid_actions': [0, 1, 2, 3]
# }
```

### 4. RGB Images (For Vision Models)

```python
import numpy as np

env = agentick.make("GoToGoal-v0", render_mode="rgb_array")
obs, _ = env.reset()

print(f"Image shape: {obs.shape}")  # (160, 160, 3)
print(f"Image dtype: {obs.dtype}")  # uint8
print(f"Value range: [{obs.min()}, {obs.max()}]")  # [0, 255]

# You can save the image
from PIL import Image
img = Image.fromarray(obs)
img.save("agentick_screenshot.png")
```

### 5. State Dict (For Programmatic Agents)

```python
env = agentick.make("GoToGoal-v0", render_mode="state_dict")
obs, _ = env.reset()

print(type(obs))  # <class 'dict'>
print(obs.keys())
# dict_keys(['grid', 'agent', 'entities', 'config'])

# Access structured state programmatically
print(f"Agent position: {obs['agent']['position']}")
print(f"Goal positions: {[e['position'] for e in obs['entities'] if e['type'] == 'goal']}")
```

## Different Tasks

Agentick has 40+ tasks. Try some different ones:

### Navigation Tasks

```python
# Simple navigation
env = agentick.make("GoToGoal-v0", difficulty="easy")

# Maze navigation (harder)
env = agentick.make("MazeNavigation-v0", difficulty="medium")

# Multiple goals in sequence
env = agentick.make("MultiGoalRoute-v0", difficulty="hard")

# Navigate with limited visibility
env = agentick.make("FogOfWar-v0", difficulty="expert")
```

### Memory Tasks

```python
# Remember key locations
env = agentick.make("KeyDoorPuzzle-v0", difficulty="easy")

# Recall sequences
env = agentick.make("SequenceMemory-v0", difficulty="medium")

# Follow breadcrumb trail
env = agentick.make("BreadcrumbTrail-v0", difficulty="medium")
```

### Reasoning Tasks

```python
# Push boxes to goals
env = agentick.make("SokobanPush-v0", difficulty="medium")

# Match visual symbols
env = agentick.make("SymbolMatching-v0", difficulty="easy")

# Identify causal chains
env = agentick.make("CausalChain-v0", difficulty="hard")
```

See [Task Gallery](../concepts/tasks.md) for all 40+ tasks.

## Random Agent vs Programmatic Agent

### Random Agent (Baseline)

```python
env = agentick.make("GoToGoal-v0", difficulty="easy")
obs, info = env.reset(seed=42)

for _ in range(100):
    action = env.action_space.sample()  # Pick random action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print(f"Random agent success: {info.get('success', False)}")
```

### Simple Programmatic Agent

```python
import agentick
from agentick.interfaces import BotInterface

env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="state_dict")
bot = BotInterface(env)

obs, info = env.reset(seed=42)

for step in range(100):
    # Get agent and goal positions from state
    agent_pos = tuple(obs['agent']['position'])
    goal_pos = None
    for entity in obs['entities']:
        if entity['type'] == 'goal':
            goal_pos = tuple(entity['position'])
            break

    if goal_pos is None:
        break

    # Simple heuristic: move toward goal
    dy = goal_pos[0] - agent_pos[0]
    dx = goal_pos[1] - agent_pos[1]

    if abs(dy) > abs(dx):
        action = 0 if dy < 0 else 1  # 0=north, 1=south
    else:
        action = 3 if dx < 0 else 2  # 2=east, 3=west

    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print(f"Bot success: {info.get('success', False)}")
```

## Saving and Visualizing Results

### Save Episode Trajectory

```python
import json
import agentick

env = agentick.make("GoToGoal-v0", difficulty="easy")
obs, info = env.reset(seed=42)

trajectory = {
    "episodes": [],
    "config": {
        "task": "GoToGoal-v0",
        "difficulty": "easy",
        "seed": 42
    }
}

for episode in range(3):
    obs, _ = env.reset()
    episode_data = {
        "steps": [],
        "total_reward": 0,
        "success": False
    }

    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        episode_data["steps"].append({
            "action": int(action),
            "reward": float(reward),
            "success": bool(info.get('success', False))
        })
        episode_data["total_reward"] += reward

        if terminated or truncated:
            episode_data["success"] = info.get('success', False)
            break

    trajectory["episodes"].append(episode_data)

# Save to JSON
with open("episode_trajectory.json", "w") as f:
    json.dump(trajectory, f, indent=2)

print(f"Saved trajectory with {len(trajectory['episodes'])} episodes")
```

### Render Episode to Video

```python
import agentick
from PIL import Image

env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode="rgb_array")
obs, info = env.reset(seed=42)

frames = []

for step in range(100):
    # Render current frame
    frame = env.render()
    frames.append(Image.fromarray(frame))

    # Take action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

# Save as GIF (requires pillow)
if frames:
    frames[0].save(
        "episode.gif",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0
    )
    print(f"Saved episode as GIF with {len(frames)} frames")
```

## Complete Working Example

Here's a complete script you can copy and run:

```python
#!/usr/bin/env python3
"""Complete quickstart example."""

import agentick

# Configuration
TASK = "GoToGoal-v0"
DIFFICULTY = "easy"
N_EPISODES = 5
SEED = 42

# Create environment
env = agentick.make(
    TASK,
    difficulty=DIFFICULTY,
    render_mode="ascii",
    seed=SEED
)

print(f"Running {N_EPISODES} episodes of {TASK} ({DIFFICULTY})")
print("=" * 50)

# Track statistics
all_returns = []
all_successes = []

# Run episodes
for episode in range(N_EPISODES):
    obs, info = env.reset()
    episode_return = 0
    episode_length = 0

    # Show initial state
    if episode == 0:
        print("\nInitial observation:")
        print(obs)
        print()

    # Run steps
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        episode_return += reward
        episode_length += 1

        if terminated or truncated:
            break

    success = info.get('success', False)
    all_returns.append(episode_return)
    all_successes.append(success)

    print(f"Episode {episode+1}: Return={episode_return:7.2f}, "
          f"Length={episode_length:3d}, Success={success}")

# Print summary
print("=" * 50)
print(f"\nSummary Statistics:")
print(f"  Success Rate: {sum(all_successes)}/{N_EPISODES} ({sum(all_successes)/N_EPISODES:.1%})")
print(f"  Mean Return: {sum(all_returns)/len(all_returns):.3f}")
print(f"  Max Return: {max(all_returns):.3f}")
print(f"  Min Return: {min(all_returns):.3f}")

env.close()
```

Save as `quickstart.py` and run:
```bash
python quickstart.py
```

Expected output:
```
Running 5 episodes of GoToGoal-v0 (easy)
==================================================

Initial observation:
#####
#A.G#
#.#.#
#...#
#####

Episode 1: Return= -0.58, Length= 36, Success=False
Episode 2: Return= -0.45, Length= 30, Success=True
Episode 3: Return= -0.52, Length= 34, Success=False
Episode 4: Return= -0.38, Length= 25, Success=True
Episode 5: Return= -0.49, Length= 32, Success=False
==================================================

Summary Statistics:
  Success Rate: 2/5 (40.0%)
  Mean Return: -0.48
  Max Return: -0.38
  Min Return: -0.58
```

## Next Steps

Now that you understand the basics:

1. **[First Experiment](first_experiment.md)** - Run a complete experiment with metrics and plots
2. **[Concepts - Tasks](../concepts/tasks.md)** - Learn about all 40+ available tasks
3. **[Concepts - Observations](../concepts/observations.md)** - Deep dive into observation modes
4. **[Agents - RL](../agents/rl_agents.md)** - Train agents with deep reinforcement learning
5. **[Agents - LLM](../agents/llm_agents.md)** - Use language models as agents

## Common Patterns

### Pattern: Evaluation with Multiple Seeds

```python
import agentick

env = agentick.make("GoToGoal-v0", difficulty="medium")

results = []
for seed in [42, 123, 456]:
    obs, _ = env.reset(seed=seed)
    episode_reward = 0

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        if terminated or truncated:
            break

    results.append({
        "seed": seed,
        "return": episode_reward,
        "success": info.get('success', False)
    })

for r in results:
    print(f"Seed {r['seed']}: Return={r['return']:.2f}, Success={r['success']}")
```

### Pattern: Task Sweep

```python
import agentick

tasks = ["GoToGoal-v0", "KeyDoorPuzzle-v0", "SymbolMatching-v0"]

for task_name in tasks:
    env = agentick.make(task_name, difficulty="easy")
    obs, _ = env.reset(seed=42)

    total_reward = 0
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"{task_name}: {total_reward:.2f}")
```

## Getting Help

- **Questions?** See the [FAQ](../faq.md)
- **Need API docs?** Check [API Reference](../api/index.md)
- **Examples?** Browse [examples/](https://github.com/anthropics/agentick/tree/main/examples) directory
