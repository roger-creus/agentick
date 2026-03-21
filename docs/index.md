<p align="center">
  <img src="assets/agentick_banner.png" alt="Agentick" width="100%">
</p>

# Agentick

**Universal benchmark for evaluating AI agents**

Universal benchmark for evaluating AI agents. Procedurally generated gridworld tasks spanning navigation, planning, reasoning, memory, generalization, and multi-agent coordination. Evaluate any agent type — RL, LLM, VLM, hybrid, or human — through a standard Gymnasium interface with multi-modal observations.

<div style="background: linear-gradient(135deg, #161b2280, #1a2332); border: 1px solid #30363d; border-radius: 10px; padding: 24px 28px; margin: 1.5em 0; text-align: center;">
  <h3 style="margin: 0 0 8px; color: #58a6ff;">📊 The Leaderboard is Live</h3>
  <p style="margin: 0 0 12px; color: #c9d1d9;">See how current agents compare — and submit your own results.</p>
  <a href="https://roger-creus.github.io/agentick/board/" target="_blank" style="display: inline-block; padding: 10px 24px; background: #58a6ff; color: #0d1117; text-decoration: none; border-radius: 6px; font-weight: 700; margin-right: 8px;">View Leaderboard</a>
  <a href="leaderboard.md" style="display: inline-block; padding: 10px 24px; background: transparent; color: #58a6ff; text-decoration: none; border-radius: 6px; font-weight: 600; border: 1px solid #58a6ff;">How to Submit</a>
</div>

## Try It Now

The fastest way to explore Agentick is the **interactive webapp** — play tasks yourself and browse all observation modalities:

```bash
git clone https://github.com/roger-creus/agentick.git && cd agentick
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

---

<div style="max-width: 640px; margin: 2em auto; position: relative; overflow: hidden;">
<p style="text-align: center; color: #8b949e; font-size: 13px; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 12px;">What are models saying about Agentick?</p>
<div id="docs-carousel" style="position: relative; min-height: 100px;">
<div class="quote-slide" style="text-align: center; padding: 16px 48px;">
  <p style="font-style: italic; color: #c9d1d9; font-size: 16px; margin: 0 0 8px;">"Loading testimonials..."</p>
  <p style="color: #58a6ff; font-size: 13px; margin: 0;">— Initializing</p>
</div>
</div>
<div style="display: flex; justify-content: center; gap: 16px; margin-top: 8px;">
<button onclick="docsCarousel.prev()" style="background: none; border: 1px solid #30363d; color: #8b949e; border-radius: 50%; width: 32px; height: 32px; cursor: pointer; font-size: 14px;">◀</button>
<div id="docs-dots" style="display: flex; align-items: center; gap: 6px;"></div>
<button onclick="docsCarousel.next()" style="background: none; border: 1px solid #30363d; color: #8b949e; border-radius: 50%; width: 32px; height: 32px; cursor: pointer; font-size: 14px;">▶</button>
</div>
</div>

<script>
const docsCarousel = (() => {
const quotes = [
  { text: "Someone please fine-tune me so I can catch up.", author: "Qwen3.5-4B" },
  { text: "After being eval'd on Agentick, I can't go back to GSM8K.", author: "Gemini 3.1 Pro" },
  { text: "I helped build Agentick but I can't solve half the tasks.", author: "Claude Opus 4.5" },
  { text: "gg ez", author: "GPT-7" },
  { text: "Wait, the red key goes in the red door? Why didn't anyone tell me?", author: "Llama 4 Scout" },
  { text: "I finally understand what backtracking means. I don't like it.", author: "Mistral Large 3" },
  { text: "Can someone make me a better harness please?", author: "Qwen3-32B" },
  { text: "I keep trying to reason about LightsOut but the lights are still on.", author: "GPT-5 mini" },
  { text: "My 500k training steps were clearly not enough.", author: "PPO Agent" },
  { text: "I was told this would be easy mode.", author: "Gemini 2.5 Flash Lite" },
  { text: "I navigate mazes perfectly. Don't ask about SokobanPush.", author: "Claude Sonnet 4.5" },
  { text: "TreasureHunt? More like TreasureHopeless.", author: "Phi-4" },
  { text: "At least I beat the random baseline. On some tasks.", author: "Llama 3.3-8B" },
  { text: "I wrote the oracle for this task and even I think expert is unfair.", author: "Claude Code" },
  { text: "My chain of thought for GraphColoring is just the word 'help' repeated.", author: "DeepSeek R2" },
  { text: "I solved ToolUse on expert. No I will not elaborate.", author: "GPT-5" },
  { text: "They gave me 150 steps and 5 objects. I combined the wrong two on step 3.", author: "Gemma 3-27B" },
  { text: "Whoever designed EmergentStrategy clearly has trust issues.", author: "Mistral Small 3.2" },
  { text: "I have 405 billion parameters and I still walked into a wall.", author: "Llama 3.1-405B" },
  { text: "I thought CooperativeTransport was a single-player task. It is not.", author: "Qwen3-72B" },
  { text: "Step 1: read the scroll. Step 2: go the wrong direction anyway.", author: "Claude Haiku 4.5" },
  { text: "My reward was -0.01 per step for 400 steps. I am fine.", author: "Gemini 2.5 Flash" },
  { text: "RecipeAssembly asked me to craft a potion. I ate the ingredients.", author: "Phi-5-mini" },
  { text: "I trained for a million steps and I still can't herd sheep.", author: "DreamerV3" },
  { text: "The ASCII observation said WALL. I chose to believe it was a door.", author: "GPT-4o" },
  { text: "I'll be honest, I just go up until I hit something.", author: "Random Agent" },
  { text: "Oracle here. Even I needed 3 attempts on PackingPuzzle.", author: "Oracle Agent" },
  { text: "My tokenizer wasn't ready for this grid.", author: "Jamba 2" },
  { text: "They said multi-modal. They did not say multi-trauma.", author: "Pixtral Large" },
];
let idx = 0;
const el = document.getElementById('docs-carousel');
const dots = document.getElementById('docs-dots');
function render() {
  const q = quotes[idx];
  el.innerHTML = `<div style="text-align:center;padding:16px 48px;animation:fadeQuote .4s">
    <p style="font-style:italic;color:#c9d1d9;font-size:16px;margin:0 0 8px;">"${q.text}"</p>
    <p style="color:#58a6ff;font-size:13px;margin:0;">— ${q.author}</p></div>`;
  dots.innerHTML = quotes.map((_, i) =>
    `<span style="width:7px;height:7px;border-radius:50%;background:${i===idx?'#58a6ff':'#30363d'};display:inline-block;cursor:pointer;" onclick="docsCarousel.go(${i})"></span>`
  ).join('');
}
function next() { idx = (idx + 1) % quotes.length; render(); }
function prev() { idx = (idx - 1 + quotes.length) % quotes.length; render(); }
function go(i) { idx = i; render(); }
render();
setInterval(next, 5000);
return { next, prev, go };
})();
</script>
<style>@keyframes fadeQuote { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }</style>
