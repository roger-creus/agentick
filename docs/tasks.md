# Tasks

Procedurally generated tasks across 6 categories. All tasks support 4 difficulty levels (easy/medium/hard/expert), multiple observation modes, and both sparse and dense rewards.

**Browse tasks interactively** — the best way to understand each task is to play it yourself in the webapp:

```bash
uv run python -m agentick.human.webapp   # Opens http://localhost:5000
```

Or create one programmatically:

```python
import agentick
env = agentick.make("GoToGoal-v0", difficulty="medium", render_mode="language", seed=42)
```

## Task Catalog

### Navigation (8 tasks)

| Task | Description |
|---|---|
| **GoToGoal-v0** | Navigate to a visible goal in an open grid |
| **MazeNavigation-v0** | Solve procedurally generated mazes |
| **ShortestPath-v0** | Visit multiple goals in shortest-path order |
| **DynamicObstacles-v0** | Navigate while avoiding moving obstacles |
| **CuriosityMaze-v0** | Explore a maze to reach a coverage target |
| **RecursiveRooms-v0** | Navigate hierarchical room-in-room structures |
| **TimingChallenge-v0** | Pass through moving gates at precise moments |
| **InstructionFollowing-v0** | Navigate to annotated target, avoid distractors |

### Planning (9 tasks)

| Task | Description |
|---|---|
| **SokobanPush-v0** | Push boxes onto target positions (classic Sokoban) |
| **KeyDoorPuzzle-v0** | Collect color-coded keys to unlock matching doors; hard+ requires backtracking |
| **BacktrackPuzzle-v0** | Navigate rooms, remembering visited state |
| **TileSorting-v0** | Sort numbered tiles by pushing |
| **PackingPuzzle-v0** | Push objects onto matching target slots |
| **PreciseNavigation-v0** | Navigate ice terrain where agent slides; push boxes as stops |
| **RecipeAssembly-v0** | Collect ingredients in order and deliver to station |
| **ToolUse-v0** | Discover scroll combinations to create ORB and cross river |
| **ResourceManagement-v0** | Collect resources while managing energy drain |

### Reasoning (8 tasks)

| Task | Description |
|---|---|
| **SwitchCircuit-v0** | Activate color-coded switches to open barriers; cross-zone backtracking at hard+ |
| **RuleInduction-v0** | Discover hidden rules from ICE-marked cues, then INTERACT |
| **LightsOut-v0** | Turn all lights off (classic puzzle) |
| **GraphColoring-v0** | INTERACT to cycle node colors; no adjacent nodes share colors |
| **SymbolMatching-v0** | Match symbol pairs; matched pairs disappear |
| **ProgramSynthesis-v0** | Push gems onto reference pattern shown by scrolls |
| **TaskInterference-v0** | Balance GEM/ORB meters; collecting one drains the other |
| **DeceptiveReward-v0** | Reach true goal despite misleading reward signals |

### Memory (4 tasks)

| Task | Description |
|---|---|
| **SequenceMemory-v0** | Reproduce a sequence of colors (Simon Says) |
| **DelayedGratification-v0** | Resist immediate reward for larger delayed reward |
| **TreasureHunt-v0** | Find hidden treasures using directional scroll clues |
| **FogOfWarExploration-v0** | Navigate with persistent fog; only adjacent tiles visible |

### Generalization (3 tasks)

| Task | Description |
|---|---|
| **FewShotAdaptation-v0** | Watch demos to infer hidden rule, then navigate to correct target |
| **DistributionShift-v0** | Navigate maze with toggling walls and remapped actions |
| **NoisyObservation-v0** | Navigate with corrupted observations |

### Multi-Agent (5 tasks)

| Task | Description |
|---|---|
| **CooperativeTransport-v0** | Multiple agents push boxes to goal cooperatively |
| **TagHunt-v0** | Multi-agent chase/evade competition |
| **ChaseEvade-v0** | Evade Pacman-style enemies (Chaser, Ambusher, Patrol, Erratic) |
| **Herding-v0** | Herd fleeing sheep into a pen |
| **EmergentStrategy-v0** | Lure NPC sheep onto pressure plates; different NPC behaviors |

## Difficulty Scaling

All tasks follow consistent scaling:
- **Grid size**: 5-8 (easy) → 20+ (expert)
- **Complexity**: More obstacles, constraints, and dependencies
- **Episode length**: Adjusted per difficulty