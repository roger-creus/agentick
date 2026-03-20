---
date: 2026-03-18
authors:
  - roger
slug: introducing-agentick
---

# Introducing Agentick: A Universal Benchmark for AI Agents

**Procedurally generated tasks. Multi-modal observations. Every agent type. One benchmark.**

![Agentick Banner](../../assets/agentick_banner.png)

Agentick is an open-source benchmark for evaluating AI agents across the core challenges of sequential decision-making. It supports RL agents, LLM agents, VLM agents, hybrid systems, hand-written bots and planners and even human play - all through a standard Gymnasium interface.

<div style="display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; margin: 2em 0;">
  <img src="https://huggingface.co/rogercc/agentick-gallery/resolve/main/iso/MazeNavigation-v0_easy.gif" width="200" alt="MazeNavigation" loading="lazy" style="border-radius: 8px;">
  <img src="https://huggingface.co/rogercc/agentick-gallery/resolve/main/iso/ProgramSynthesis-v0_easy.gif" width="200" alt="ProgramSynthesis" loading="lazy" style="border-radius: 8px;">
  <img src="https://huggingface.co/rogercc/agentick-gallery/resolve/main/iso/ChaseEvade-v0_easy.gif" width="200" alt="ChaseEvade" loading="lazy" style="border-radius: 8px;">
  <img src="https://huggingface.co/rogercc/agentick-gallery/resolve/main/iso/KeyDoorPuzzle-v0_expert.gif" width="200" alt="KeyDoorPuzzle" loading="lazy" style="border-radius: 8px;">
</div>

<div style="background: linear-gradient(135deg, #161b2280, #1a2332); border: 1px solid #30363d; border-radius: 10px; padding: 24px 28px; margin: 1.5em 0; text-align: center;">
  <h3 style="margin: 0 0 8px; color: #58a6ff;">📊 The Leaderboard is Live</h3>
  <p style="margin: 0 0 12px; color: #c9d1d9;">See how current agents compare — and submit your own results.</p>
  <a href="https://roger-creus.github.io/agentick/board/" target="_blank" style="display: inline-block; padding: 10px 24px; background: #58a6ff; color: #0d1117; text-decoration: none; border-radius: 6px; font-weight: 700; margin-right: 8px;">View Leaderboard</a>
  <a href="https://roger-creus.github.io/agentick/leaderboard/" target="_blank" style="display: inline-block; padding: 10px 24px; background: transparent; color: #58a6ff; text-decoration: none; border-radius: 6px; font-weight: 600; border: 1px solid #58a6ff;">How to Submit</a>
</div>

## The Gap

The AI agents space has split into two worlds. RL researchers build agents that learn from scratch through environment interaction - PPO, DQN, SAC - but these agents are sample-inefficient, task-specific, and struggle to scale. Meanwhile, foundation model researchers prompt GPT-4, Gemini, or open-source LLMs to act as agents, leveraging internet-scale knowledge for zero-shot reasoning - but these models weren't trained for control and haven't learned from their own experience in interactive environments.

These two paradigms occupy opposite ends of a spectrum, and between them lies a rich design space of hybrid approaches: fine-tuned LLMs, RL post-training of foundation models, FM-guided reward shaping, and more.

**The problem?** There's no unified benchmark where you can meaningfully compare agents across this entire spectrum. RL benchmarks use state or pixel observations. LLM benchmarks use text. They test different capabilities with different metrics. You can't put a PPO agent and GPT-4 side by side and ask: *which one is actually better at planning?*

Agentick fills this gap.

## Capability Decomposition: What Should a General Agent Master?

Rather than producing a single aggregate score, Agentick decomposes evaluation into **six capability axes** - the core properties we believe a general autonomous agent needs to master:

| Capability | What it tests | # Tasks |
|-|-|:-:|
| **Navigation** | Spatial reasoning, pathfinding, reactive control | 8 |
| **Planning** | Multi-step lookahead, constraint satisfaction, backtracking | 9 |
| **Reasoning** | Logical inference, causal reasoning, abstraction | 8 |
| **Memory** | Information retention, temporal integration, partial observability | 4 |
| **Generalization** | Distribution shift, few-shot adaptation, noise robustness | 3 |
| **Multi-Agent** | Cooperation, competition, emergent strategy | 5 |

This decomposition lets you build **capability radar charts** showing exactly where an agent excels and where it falls short. An RL agent might dominate navigation but fail at reasoning. An LLM might ace planning but struggle with memory across long episodes. These profiles are far more informative than a single number.

Check out the <a href="https://roger-creus.github.io/agentick/board/" target="_blank">live leaderboard</a> to see how current agents compare across these axes.

## One Benchmark, Every Agent

The key design choice that makes this work: **multiple observation modes** for every task. The same underlying environment state is rendered in whatever format your agent needs:

=== "ASCII (for LLMs)"

    ```
    #####
    #@..#
    #.#.#
    #..G#
    #####
    Legend: @=agent G=goal #=wall .=empty
    ```

    Token-efficient grid representation. An LLM can parse this in a few tokens and reason about spatial relationships.

=== "Natural Language (for LLMs)"

    ```
    You are at position (1,1) facing north in a 5x5 room.
    A goal is visible to the southeast at distance 3.
    Walls to the north and west. Path clear to south and east.
    Valid actions: move_down (1), move_right (3)
    ```

    Verbose descriptions with spatial context, configurable verbosity and perspective.

=== "Isometric Pixels (for VLMs)"

    512x512 sprite-based isometric rendering. Rich visual observations for VLM agents and human evaluation.

    <img src="https://huggingface.co/rogercc/agentick-gallery/resolve/main/iso/SokobanPush-v0_easy.gif" width="300" alt="SokobanPush isometric" loading="lazy" style="border-radius: 8px;">

=== "Structured Dict (for LLM agents)"

    ```python
    {"description": "You are at (1,1) facing north...",
     "position": {"x": 1, "y": 1},
     "orientation": "north",
     "surroundings": {"north": "wall", "east": "empty", ...},
     "valid_actions": ["move_down", "move_right"],
     "inventory": [], "energy": 1.0, "step_count": 0}
    ```

    Parsed semantic fields - position, surroundings, valid actions, inventory. Useful for LLM agents that prefer structured input over free-text.

=== "State Dict (for bots/RL)"

    ```python
    {"grid": {"terrain": [[1,1,...], ...], "objects": [...]},
     "agent": {"position": [1,1], "orientation": "north"},
     "info": {"step_count": 0, "valid_actions": [1, 3]}}
    ```

    Raw numpy arrays of the full grid layers (terrain, objects, agents, metadata). For programmatic agents, planners, and MLP-based RL via the built-in `FlattenObservationWrapper`.

This means you can take the exact same task - say, SokobanPush - and evaluate a PPO agent on pixel observations, GPT-4 on ASCII text, a fine-tuned Qwen-VL on isometric renders, and a hand-coded BFS planner on the state dict. Same task, same seeds, same metrics. Fair comparison.

## Not Just Eval - Training-First Design

Most benchmarks are eval-only: you bring a pre-trained agent and measure its performance. Agentick is designed for the **full pipeline** - training, data collection, fine-tuning, and evaluation.

**Train RL agents directly:**

```python
from stable_baselines3 import PPO
import agentick

env = agentick.make("MazeNavigation-v0", render_mode="rgb_array")
model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)
```

**Collect expert trajectories from oracle policies:**

```python
from agentick.oracles import get_oracle
from agentick.data.collector import DataCollector

env = agentick.make("SokobanPush-v0", render_mode="language")
oracle = get_oracle("SokobanPush-v0", env)
collector = DataCollector(env, oracle, record_modalities=["language"])

dataset = collector.collect(num_episodes=1000, seeds=range(1000))
dataset.export_to_huggingface("data/sokoban_expert/", format="conversation")
```

**Fine-tune LLMs on expert demonstrations:**

Oracle policies are provided for all 37 tasks. Generate your own trajectories, or grab one of our pre-built datasets on HuggingFace:

| Dataset | Episodes | |
|---|---|---|
| [agentick-oracle-trajectories-120k](https://huggingface.co/datasets/rogercc/agentick-oracle-trajectories-120k) | 120K | Good starting point |
| [agentick-oracle-trajectories-250k](https://huggingface.co/datasets/rogercc/agentick-oracle-trajectories-250k) | 250K | Broader coverage |
| [agentick-oracle-trajectories-500k](https://huggingface.co/datasets/rogercc/agentick-oracle-trajectories-500k) | 500K | Full scale |

Each dataset includes per-step oracle actions, ASCII and language observations, rewards, done flags, and step info across all 37 tasks and difficulty levels. Load them directly with `datasets` and SFT your favorite open-source model.

This training-first design means Agentick isn't just measuring where agents are today - it's infrastructure for making them better.

## Coding API: Write Agents in Python

Every environment exposes a **Coding API** - a programmatic interface with spatial queries, pathfinding, entity lookups, and high-level action primitives. It's designed for hand-coded bots, code-generating LLMs, and anyone who wants to write agent logic in Python rather than training a model.

```python
from agentick.coding_api import AgentickAPI

env = agentick.make("KeyDoorPuzzle-v0", difficulty="medium")
api = AgentickAPI(env)
obs, info = env.reset(seed=42)
api.update(obs, info)

# Spatial queries
api.agent_position          # (3, 5)
api.get_nearest("key")      # EntityInfo(type="key", position=(7, 2), distance=8)
api.get_entities_of_type("door")  # [EntityInfo(...), ...]
api.is_walkable(4, 3)       # True
api.is_reachable(7, 2)      # True

# BFS pathfinding — returns action sequences
actions = api.path_to(7, 2)         # [1, 3, 3, 1, 3, ...]
actions = api.go_to_nearest("key")  # pathfind to closest key
actions = api.flee_from(5, 5)       # single action moving away

# Execute
for action in actions:
    obs, reward, done, trunc, info = env.step(action)
    api.update(obs, info)
    if done or trunc:
        break
```

The API also exposes grid inspection (`get_cell`, `get_object`, `get_walls`, `get_walkable_cells`), inventory management (`has_in_inventory`), and interaction primitives (`interact_with`).

This is how the **oracle policies** for all 37 tasks were built - coded up through this API by a frontier coding LLM with iteration and refinement. Those oracles then generate the expert trajectory datasets linked above, closing the loop from code → trajectories → SFT.

## LLM Agent Harnesses

When evaluating LLMs as agents, how you prompt matters as much as which model you use. Agentick ships with built-in **harness presets** that control the prompting strategy:

=== "Zero-Shot (Markovian)"

    Each step is independent — the model sees only the current observation with no history. Fast, token-efficient, and memoryless.

    **System prompt:**
    ```
    You are an AI agent playing grid-world tasks in the Agentick benchmark.
    Your goal is to navigate the grid and complete the task objective.

    ## Action Space
    0: NOOP  1: MOVE_UP  2: MOVE_DOWN  3: MOVE_LEFT  4: MOVE_RIGHT  5: INTERACT

    ## Task Objective
    Navigate the maze to reach the GOAL exit. Collect keys to open doors.

    Respond with ONLY the action number, nothing else.
    ```

    **Observation → Model → Response:**
    ```
    User: You are at (3,2) facing south. Key visible to the east at distance 2.
          Walls to the north and west. Valid actions: move_down, move_right

    Model: 4
    ```

=== "Chain-of-Thought (Reasoner)"

    Same single-step view, but the model reasons before acting. Trades tokens for better decisions on tasks that require planning or inference.

    **System prompt** (appended):
    ```
    IMPORTANT: Before choosing an action, reason step-by-step but be
    CONCISE (2-4 sentences max):
    1. What do you observe? What is your goal?
    2. Which action best advances you toward the goal?
    3. Output your final answer on the LAST line as: ACTION: <number>
    ```

    **Observation → Model → Response:**
    ```
    User: You are at (3,2) facing south. Key visible to the east at distance 2.
          Walls to the north and west. Valid actions: move_down, move_right

    Model: I see a key to my east at distance 2. I need to collect it to unlock
           the door blocking the maze exit. Moving right gets me closer.
           ACTION: 4
    ```

Both harnesses support any observation mode (ASCII, language, pixels for VLMs) and any backend (OpenAI, Gemini, HuggingFace, vLLM). See the <a href="https://roger-creus.github.io/agentick/agents/llm_agents/" target="_blank">LLM/VLM agents docs</a> for the full setup.

## Running Experiments

Agentick includes an <a href="https://roger-creus.github.io/agentick/experiments/" target="_blank">experiment runner</a> for reproducible evaluation. Define your setup in YAML and run:

```yaml
# eval_gpt4_navigation.yaml
name: gpt4-navigation-eval
agent:
  type: llm
  hyperparameters:
    backend: openai
    model: gpt-4o
    harness: markovian_reasoner
    observation_modes: [ascii]
tasks: navigation           # or "full", "planning", ["GoToGoal-v0", ...]
difficulties: [easy, medium, hard, expert]
n_seeds: 25
split: eval
```

```bash
uv run python -m agentick.experiments.run --config eval_gpt4_navigation.yaml
```

The runner handles seed generation, episode management, crash-safe checkpointing, metric computation, and cost tracking for API-based agents. Results include per-task success rates, normalized scores, and capability breakdowns.

For <a href="https://roger-creus.github.io/agentick/agents/rl_agents/" target="_blank">RL training</a>, use standard libraries directly — Agentick environments are Gymnasium-compatible, so SB3, CleanRL, and any Gym-compatible framework work out of the box. For <a href="https://roger-creus.github.io/agentick/agents/finetuning/" target="_blank">fine-tuning</a> LLMs on oracle trajectories, see the SFT pipeline docs.

## The Tasks

37 tasks, each procedurally generated with 4 difficulty levels (easy → expert). Every run produces a unique layout, so agents can't memorize solutions. Click a category and difficulty to explore.

<style>
.ag-tabs{display:flex;gap:6px;flex-wrap:wrap;margin:0.8em 0}
.ag-tab{padding:6px 14px;border:2px solid #e0e0e0;border-radius:20px;background:#fafafa;cursor:pointer;font-size:0.85em;font-weight:600;transition:all .15s}
.ag-tab:hover{border-color:#4051b5;color:#4051b5}
.ag-tab.active{background:#4051b5;color:#fff;border-color:#4051b5}
.ag-diff{padding:4px 12px;border:1px solid #ccc;border-radius:14px;background:#fff;cursor:pointer;font-size:0.8em;transition:all .15s}
.ag-diff:hover{border-color:#4051b5}
.ag-diff.active{background:#e8eaf6;border-color:#4051b5;color:#4051b5;font-weight:600}
.ag-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px;margin:1em 0}
.ag-card{border:1px solid #e0e0e0;border-radius:10px;overflow:hidden;background:#fff;transition:box-shadow .15s}
.ag-card:hover{box-shadow:0 2px 12px rgba(0,0,0,.1)}
.ag-card img{width:100%;aspect-ratio:1;object-fit:cover;background:#f5f5f5;display:block}
.ag-card-info{padding:8px 10px}
.ag-card-info strong{font-size:0.9em;display:block;margin-bottom:2px}
.ag-card-info span{font-size:0.78em;color:#666;line-height:1.3}
</style>

<div>
<div class="ag-tabs" id="ag-cat-tabs"></div>
<div class="ag-tabs" id="ag-diff-tabs"></div>
<div class="ag-grid" id="ag-grid"></div>
</div>

<script>
(function(){
const HF="https://huggingface.co/rogercc/agentick-gallery/resolve/main/iso/";
const DIFFS=["easy","medium","hard","expert"];
const CATS=[
{id:"navigation",label:"Navigation (8)",tasks:[
{n:"GoToGoal-v0",d:"Navigate to a visible goal, avoid guards and hazards"},
{n:"MazeNavigation-v0",d:"Solve procedural mazes with key-door gates"},
{n:"ShortestPath-v0",d:"Multi-goal traveling salesman under a step budget"},
{n:"DynamicObstacles-v0",d:"Reach goal while dodging moving NPCs"},
{n:"CuriosityMaze-v0",d:"Coverage-based exploration, no target markers"},
{n:"RecursiveRooms-v0",d:"Hierarchical nested room navigation"},
{n:"TimingChallenge-v0",d:"Cross patrol gaps with precise timing"},
{n:"InstructionFollowing-v0",d:"Find the named target among distractors"}
]},
{id:"planning",label:"Planning (9)",tasks:[
{n:"SokobanPush-v0",d:"Classic box-pushing with dead-end traps"},
{n:"KeyDoorPuzzle-v0",d:"Chained color-coded locks with backtracking"},
{n:"BacktrackPuzzle-v0",d:"Activate switches then retrace your path"},
{n:"TileSorting-v0",d:"Sliding tile puzzle (8/15-puzzle mechanics)"},
{n:"PackingPuzzle-v0",d:"Push typed pieces onto matching target slots"},
{n:"PreciseNavigation-v0",d:"Ice-sliding puzzle with interior wall segments"},
{n:"RecipeAssembly-v0",d:"Ordered ingredient collection and delivery"},
{n:"ToolUse-v0",d:"Discover scroll combinations to forge tools"},
{n:"ResourceManagement-v0",d:"Keep energy stations alive under drain pressure"}
]},
{id:"reasoning",label:"Reasoning (8)",tasks:[
{n:"SwitchCircuit-v0",d:"Non-linear switch dependencies with mutual exclusion"},
{n:"RuleInduction-v0",d:"Discover hidden rules from environmental cues"},
{n:"LightsOut-v0",d:"Classic toggle puzzle with neighbor propagation"},
{n:"GraphColoring-v0",d:"Color graph nodes with no adjacent conflicts"},
{n:"SymbolMatching-v0",d:"Match and deliver symbol pairs"},
{n:"ProgramSynthesis-v0",d:"Push gems to replicate a reference pattern"},
{n:"TaskInterference-v0",d:"Balance competing resource meters"},
{n:"DeceptiveReward-v0",d:"Resist misleading reward gradients"}
]},
{id:"memory",label:"Memory (4)",tasks:[
{n:"SequenceMemory-v0",d:"Memorize and reproduce spatial sequences"},
{n:"DelayedGratification-v0",d:"Resist nearby traps, navigate to distant goal"},
{n:"TreasureHunt-v0",d:"Triangulate hidden treasures from directional clues"},
{n:"FogOfWarExploration-v0",d:"Navigate with fog-of-war, adjacent cells only visible"}
]},
{id:"generalization",label:"Generalization (3)",tasks:[
{n:"FewShotAdaptation-v0",d:"Infer hidden rule from demo trials, then act"},
{n:"DistributionShift-v0",d:"Multi-phase maze with shifting layout and controls"},
{n:"NoisyObservation-v0",d:"Find the real goal among visual noise and ghosts"}
]},
{id:"multi_agent",label:"Multi-Agent (5)",tasks:[
{n:"CooperativeTransport-v0",d:"Push heavy boxes with NPC cooperation"},
{n:"TagHunt-v0",d:"Tag fleeing NPCs using freeze power-ups"},
{n:"ChaseEvade-v0",d:"Survive against Pacman-style ghost pursuers"},
{n:"Herding-v0",d:"Guide autonomous sheep into a pen"},
{n:"EmergentStrategy-v0",d:"Exploit NPC behaviors to unlock barriers"}
]}
];
let curCat=0,curDiff=0;
function render(){
const cat=CATS[curCat],diff=DIFFS[curDiff];
const grid=document.getElementById("ag-grid");
grid.innerHTML=cat.tasks.map(t=>`<div class="ag-card"><img src="${HF}${t.n}_${diff}.gif" alt="${t.n} ${diff}" loading="lazy"><div class="ag-card-info"><strong>${t.n.replace("-v0","")}</strong><span>${t.d}</span></div></div>`).join("");
}
function initTabs(){
const catEl=document.getElementById("ag-cat-tabs");
const diffEl=document.getElementById("ag-diff-tabs");
catEl.innerHTML=CATS.map((c,i)=>`<button class="ag-tab${i===0?" active":""}" data-i="${i}">${c.label}</button>`).join("");
diffEl.innerHTML=DIFFS.map((d,i)=>`<button class="ag-diff${i===0?" active":""}" data-i="${i}">${d.charAt(0).toUpperCase()+d.slice(1)}</button>`).join("");
catEl.addEventListener("click",e=>{const b=e.target.closest("[data-i]");if(!b)return;curCat=+b.dataset.i;catEl.querySelectorAll(".ag-tab").forEach((t,i)=>t.classList.toggle("active",i===curCat));render()});
diffEl.addEventListener("click",e=>{const b=e.target.closest("[data-i]");if(!b)return;curDiff=+b.dataset.i;diffEl.querySelectorAll(".ag-diff").forEach((t,i)=>t.classList.toggle("active",i===curDiff));render()});
}
initTabs();render();
})();
</script>

Every task scales across 4 difficulty levels. Easy tasks use 7x7 grids with simple mechanics. Expert tasks are 15-20 cell grids with multiple interacting systems, decoys, and tight step budgets. This controlled scaling lets you see exactly where an agent's capabilities break down.

## Leaderboard and Reproducibility

Agentick includes a deterministic seed system for reproducible evaluation. Seeds are derived from a SHA-256 hash of `"{task}::{difficulty}::eval"`, so every submission runs on the exact same 25 episodes per task-difficulty pair (3,800 total episodes).

Scoring is normalized against random and oracle baselines:

```
score = (agent_return - random) / (oracle - random)
```

Per-category scores break down into the six capability axes. Submit your results to appear on the <a href="https://roger-creus.github.io/agentick/board/" target="_blank">leaderboard</a> and see how your agent's capability profile compares.

## Get Started

```bash
git clone https://github.com/roger-creus/agentick.git && cd agentick
uv sync --extra all
```

```python
import agentick

env = agentick.make("GoToGoal-v0", difficulty="easy")
obs, info = env.reset(seed=42)

for _ in range(100):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        break

env.close()
```

```bash
uv run agentick webapp          # Play tasks yourself in the browser
uv run agentick list-tasks      # See all 37 tasks
```

Browse the [documentation](../../index.md), explore the [task catalog](../../tasks.md), or check out the [example configs](https://github.com/roger-creus/agentick/tree/main/examples) to get running.

## Built for the Research Community

Agentick is MIT-licensed and designed for open research. Whether you're training RL agents, evaluating foundation models, building hybrid systems, or studying what makes agents "general" - we built this to be the common ground where all of that work can be measured and compared.

We'd love to see:

- **Leaderboard submissions** from open-source and commercial models
- **New training recipes** - RL post-training, SFT, behavior cloning, curriculum learning
- **Analysis** of capability profiles across agent paradigms
- **Community contributions** - new tasks, observation modes, agent harnesses

The benchmark, documentation, examples, and leaderboard are all live. Give it a try and let us know what you find.

<div style="text-align: center; margin: 2em 0;">
  <a href="https://github.com/roger-creus/agentick" target="_blank" style="display: inline-block; padding: 12px 24px; background: #4051b5; color: white; text-decoration: none; border-radius: 6px; font-weight: bold;">View on GitHub</a>
</div>
