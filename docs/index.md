<p align="center">
  <img src="assets/agentick_banner.png" alt="Agentick" width="100%">
</p>

# Agentick

**Universal benchmark for evaluating AI agents**

Agentick provides 37 procedurally generated gridworld tasks spanning navigation, planning, reasoning, memory, generalization, and multi-agent coordination. Evaluate any agent type — RL, LLM, VLM, hybrid, or human — through a standard Gymnasium interface with multi-modal observations.

## Try It Now

The fastest way to explore Agentick is the **interactive webapp** — play tasks yourself and browse all observation modalities:

```bash
git clone https://github.com/agentick/agentick.git && cd agentick
uv sync --extra all
uv run agentick webapp          # Opens http://localhost:5000
```

## See It in Action

<div style="display: flex; gap: 10px; justify-content: center; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="https://huggingface.co/rogercc/agentick-gallery/resolve/main/iso/ProgramSynthesis-v0_easy.gif" width="256" alt="ProgramSynthesis (isometric)">
    <br><em>ProgramSynthesis</em>
  </div>
  <div style="text-align: center;">
    <img src="https://huggingface.co/rogercc/agentick-gallery/resolve/main/iso/KeyDoorPuzzle-v0_expert.gif" width="256" alt="KeyDoorPuzzle (isometric)">
    <br><em>KeyDoorPuzzle</em>
  </div>
  <div style="text-align: center;">
    <img src="https://huggingface.co/rogercc/agentick-gallery/resolve/main/iso/PackingPuzzle-v0_medium.gif" width="256" alt="PackingPuzzle (isometric)">
    <br><em>PackingPuzzle</em>
  </div>
</div>

Every task supports 6 observation modes — here's the same state seen by different agents:

**ASCII** (for LLMs):
```
#####
#@..#
#.#.#
#..G#
#####
Legend: @=agent G=goal #=wall .=empty
```

**Natural Language** (for LLMs):
```
You are at position (1,1) facing north in a 5x5 room.
A goal is visible to the southeast at distance 3.
Walls to the north and west. Path clear to the south and east.
Valid actions: move_down (1), move_right (3)
```

**State Dict** (for bots/planners):
```python
{"grid": {"height": 5, "width": 5, "terrain": [[1,1,1,1,1],[1,0,0,0,1],...]},
 "agent": {"position": [1,1], "orientation": "north", "inventory": []},
 "info": {"step_count": 0, "max_steps": 50, "valid_actions": [1, 3]}}
```

**RGB Pixels** — isometric (512x512) for VLMs and RL CNNs.

## Quick Start

```python
import agentick

env = agentick.make("GoToGoal-v0", difficulty="easy")
obs, info = env.reset(seed=42)

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

## Task Gallery

37 tasks across 6 capability categories:

| Capability | Tasks | Count |
|---|---|---|
| **Navigation** | GoToGoal, MazeNavigation, ShortestPath, DynamicObstacles, CuriosityMaze, RecursiveRooms, TimingChallenge, InstructionFollowing | 8 |
| **Planning** | SokobanPush, KeyDoorPuzzle, BacktrackPuzzle, TileSorting, PackingPuzzle, PreciseNavigation, RecipeAssembly, ToolUse, ResourceManagement | 9 |
| **Reasoning** | SwitchCircuit, RuleInduction, LightsOut, GraphColoring, SymbolMatching, ProgramSynthesis, TaskInterference, DeceptiveReward | 8 |
| **Memory** | SequenceMemory, DelayedGratification, TreasureHunt, FogOfWarExploration | 4 |
| **Generalization** | FewShotAdaptation, DistributionShift, NoisyObservation | 3 |
| **Multi-Agent** | CooperativeTransport, TagHunt, ChaseEvade, Herding, EmergentStrategy | 5 |

## Example Use Cases

### Train an RL Agent
```python
from stable_baselines3 import PPO

env = agentick.make("GoToGoal-v0", render_mode="rgb_array")  # Isometric pixels (512x512)
model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
```

### Evaluate an LLM Agent
```python
import agentick
from agentick.agents import BaseAgent, create_agent
from agentick.experiments.config import AgentConfig

env = agentick.make("GoToGoal-v0", render_mode="language")

obs, info = env.reset()
# See examples/llm/openai_text_agent.py for a complete LLM evaluation example
```

### Collect Expert Trajectories
```python
from agentick.data.collector import DataCollector
from agentick.oracles import get_oracle

env = agentick.make("GoToGoal-v0", render_mode="language")
oracle = get_oracle("GoToGoal-v0", env)
collector = DataCollector(env, oracle, record_modalities=["language"])

dataset = collector.collect(num_episodes=100, seeds=range(100))
dataset.export_to_huggingface("data/hf/", format="conversation")
```

## Learn More

- [Quickstart](getting_started/quickstart.md) — Installation and 5-minute tutorial
- [Tasks](tasks.md) — Browse all 37 tasks
- [Observations](concepts/observations.md) — All observation modes

## License

MIT License — see LICENSE file for details.
