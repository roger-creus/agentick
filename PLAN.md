# PLAN.md — Task Review & Codebase Improvement Plan

## CONTEXT

The user has spent 2+ hours manually playing every task (via the human play interface) and watching PPO pixel agent benchmark videos. This document captures ALL feedback and required changes. Every item must be addressed.

**You have .env with API keys. Fix bugs as you encounter them. Run tests after every change.**

---

## GENERAL COMMENTS (APPLY ACROSS ENTIRE BENCHMARK)

### GENERAL 1: Object Diversity
Many tasks feel samey because they only use Keys (K) and Tees (T). **Add new object types** to the entity system to give tasks more visual and semantic distinction. Ideas:
- Gems/crystals (different from keys)
- Levers/buttons (different from switches)
- Potions/flasks
- Shields/armor
- Scrolls/books
- Food/apples
- Coins/tokens
- Colored orbs

Use these new objects to differentiate tasks that currently feel identical. Each task should have a visually and mechanically distinct identity.

### GENERAL 2: Action Space Audit
Review the current shared action space across all tasks. Questions to resolve:
- What are all current actions? (noop, move_up, move_down, move_left, move_right, pickup, drop, use, interact?)
- Are ALL actions fundamentally used across the benchmark?
- Are there actions that are never useful in most tasks?
- Redesign if needed: every action should be unique AND important across multiple tasks
- The action space should be shared across all tasks but every action should have meaningful use somewhere
- Remove actions that are redundant or unused; add actions if there's a clear need
- Document which tasks use which actions

---

## PER-TASK REVIEW

### ✅ PERFECT — No Changes Needed
These tasks are confirmed excellent:

| Task | Notes |
|------|-------|
| **BacktrackPuzzle** | Perfect |
| **BreadcrumbTrail** | Perfect |
| **DeceptiveReward** | Incredible design. Double-check dense reward is well-calibrated (agent not solving at 300k steps is probably fine, but verify reward signal is learnable) |
| **DynamicObstacles** | Really cool |
| **KeyDoorPuzzle** | Fantastic |
| **MazeNavigation** | Brilliant |
| **MultiRoomEscape** | Cool |

---

### 🔧 MINOR FIXES — Good Tasks Needing Tweaks

#### DelayedGratification
- **Status**: LOVE IT
- **BUG**: On harder difficulty modes, the agent spawns literally surrounded by low-reward keys. Whatever action it takes, it collides with a key at step 1 and the episode terminates immediately.
- **FIX**: During map generation, ensure there is ALWAYS a clear path of at least 2-3 empty cells around the agent's spawn position. Low-reward items must not be placed adjacent to spawn. Let the agent move before facing choices.

#### GoToGoal
- **Status**: Very cool
- **ISSUE**: Dense reward agent on medium difficulty doesn't solve it. Is the dense reward poorly designed?
- **FIX**: Review the dense reward function. Ensure it provides a clear gradient toward the goal (e.g., negative Manhattan distance potential-based shaping). If the reward landscape has local optima or plateaus, fix them.

#### Herding
- **Status**: Incredible task design, feels intuitive and smooth
- **FIX 1**: Double-check sheep-in-pen detection conditions. Something feels off when playing — verify the exact conditions for "sheep is in the pen."
- **FIX 2**: Double-check sheep movement logic (response to agent proximity, wall bouncing, etc.)
- **NOTE**: Hard and expert levels are genuinely very difficult even for humans. This is fine — it's expert difficulty.

#### CooperativeTransport
- **Status**: Pretty good
- **FIX**: Double-check NPC cooperation logic is correct (NPC moves in coordination, doesn't get stuck, responds properly to agent actions)

#### GraphColoring
- **Status**: Very cool task
- **FIX**: Thorough logic audit:
  - Are diagonals considered adjacent? (They probably shouldn't be for graph coloring — only cardinal adjacency)
  - Are keys always all available colors?
  - Are switches always all available colors?
  - Is the coloring constraint check correct?
  - Is the task always solvable with the given colors?

#### MultiGoalRoute
- **Status**: Looks sick
- **FIX 1**: Double-check reward truly emphasizes shorter paths between goals (optimal path bonus, not just visiting all goals)
- **FIX 2**: What is the role of the T tokens? They don't seem to add much. Either give them a clear mechanical role or remove them.

#### RuleInduction
- **Status**: Pretty cool
- **ISSUE**: Difficulty scaling is weak. Main scaling axis is map size, which isn't interesting. More targets don't add much challenge.
- **FIX**: Redesign difficulty scaling:
  - Easy: Simple rule (go to same color)
  - Medium: Compound rule (go to opposite color, or specific pattern)
  - Hard: Rule changes mid-episode (non-stationary)
  - Expert: Multiple interacting rules, agent must deduce from fewer examples

#### TimingChallenge
- **Status**: Pretty cool
- **FIX**: Consider making fire actually kill the agent (episode termination with negative reward) rather than just blocking. This raises stakes and makes timing more critical.

#### NoisyObservation
- **Status**: Good design foundation
- **FIX**: Make it MORE noisy. Current noise level is too mild to be meaningful. This task should be the go-to for testing robustness to observation corruption. Increase noise levels at higher difficulties. Frame it as testing whether novelty-seeking agents get distracted by aleatoric noise.

---

### 🔴 VISUAL/RENDERING BUGS — Tasks Where Observations Don't Show Key Information

#### FogOfWar
- **BUG 1**: Agent doesn't move in videos (probably just a hard task, may be fine)
- **BUG 2 (CRITICAL)**: The video renders and pixel observations do NOT show any fog. The fog is invisible. This completely defeats the purpose of the task.
- **FIX**: Fog must be visually rendered:
  - In 2D pixel mode: unexplored cells should be dark/black, partially explored cells should be dimmed/gray
  - In the observation the agent receives (rgb_array): same visual fog — the agent should NOT see the full map
  - The fog IS the observation — if the agent can see through fog in its pixels, the task is trivially easy

#### LightsOut
- **BUG (CRITICAL)**: From pixel renderings, you cannot tell which cells are "lit" vs "unlit." Everything looks the same.
- **FIX**: Lit cells and unlit cells must be visually distinct:
  - Lit cells: bright yellow/warm tone
  - Unlit cells: dark gray/blue tone
  - The difference must be obvious at a glance in the pixel rendering

#### InstructionFollowing
- **BUG (CRITICAL)**: The instruction is not visible in the agent's observations. Without seeing what the instruction is, the task is meaningless — it's just random guessing.
- **REDESIGN**: Show the instruction visually in the environment:
  - Place the "target object" locked in a cage/frame at one corner of the map
  - This visually indicates which object the agent needs to go to
  - If agent goes to any other object → negative reward + termination
  - Scale difficulty by adding more distractor objects, but always show the target one clearly
  - The agent must learn pattern recognition: "the caged object tells me where to go"

---

### 🔴 DUPLICATE/TOO-SIMILAR TASKS — Need Differentiation or Replacement

These tasks are too similar to other tasks. For each: either **redesign to be unique** or **replace entirely with a new task concept**.

#### CausalChain ↔ BreadcrumbTrail
- **PROBLEM**: CausalChain logically resembles BreadcrumbTrail too much. Basically the same thing — follow a sequence of objects.
- **FIX**: CausalChain needs to be fundamentally different. Make it truly CAUSAL:
  - Objects have cause-effect relationships (pressing switch A opens door B, which reveals key C, which opens door D)
  - The agent must figure out the causal graph, not just follow a trail
  - Difficulty scales by causal chain length and branching (parallel chains, dependencies)

#### ChaseEvade ↔ CompetitiveTag
- **PROBLEM**: Both are chase games. Very similar mechanics.
- **QUESTION**: What are the R tokens in ChaseEvade? Clarify their role.
- **FIX**: Differentiate clearly:
  - **ChaseEvade**: Pure survival — evade enemies for N steps, no objectives except staying alive. Enemies get smarter at higher difficulties.
  - **CompetitiveTag**: Offense + defense — agent must TAG NPCs while avoiding being tagged. Role switching. Strategic positioning matters.
  - OR: Replace one of them with a fundamentally different adversarial task.

#### PackingPuzzle ↔ SokobanPush
- **PROBLEM**: PackingPuzzle is basically Sokoban.
- **FIX**: Differentiate clearly:
  - **SokobanPush**: Classic Sokoban — push boxes onto targets, can't pull.
  - **PackingPuzzle**: Tetris-like — different shaped objects must fit into a constrained space. Not about pushing, about fitting/arranging.
  - OR: Replace PackingPuzzle with a genuinely different combinatorial task.
- **SOKOBAN BUG**: Never spawn boxes adjacent to walls! If a box starts next to a wall, the agent can never push it toward the center. Generation must ensure all boxes are pushable toward their targets.

#### PreciseNavigation ↔ BacktrackPuzzle
- **PROBLEM**: Looks a lot like BacktrackPuzzle.
- **FIX**: PreciseNavigation should focus on MOTOR CONTROL, not path planning:
  - Narrow corridors where you must not touch walls (like Operation board game)
  - Moving platforms / timing-based navigation
  - Precise sequences of moves where one wrong step is fatal
  - Fundamentally different from "find a path" — it's "execute a path perfectly"

#### ProgramSynthesis ↔ RecipeAssembly
- **PROBLEM**: Both feel like "do things in a specific order." Very similar.
- **FIX**: Make them genuinely distinct:
  - **ProgramSynthesis**: The agent must discover a PROGRAM (a reusable sequence) that works. E.g., the same subroutine must be applied to multiple different inputs. Think "learn a function, not a sequence."
  - **RecipeAssembly**: Collect specific ingredients in specific combinations. More about resource gathering and constraint satisfaction than sequence learning.
  - OR: Merge into one strong task and replace the other with something new.

#### SymbolMatching ↔ TaskInterference
- **PROBLEM**: Both look almost identical. Only keys and tees visible. With just 2 object types it doesn't make sense.
- **FIX**: These need fundamentally different identities:
  - **SymbolMatching**: Add MORE object types (4-6 distinct objects). Agent must match pairs. Difficulty scales by number of object types, map size, and distractor objects.
  - **TaskInterference**: Should test INTERFERENCE between competing objectives. E.g., two goals that require contradictory strategies — agent must context-switch. Not just "match symbols."

#### TileSorting
- **PROBLEM**: Too similar to previous tasks.
- **FIX**: Either redesign to be genuinely about sorting (arrange items in a specific spatial order, like sliding puzzle / 15-puzzle), or replace entirely.

#### ResourceManagement ↔ KeyDoorPuzzle
- **PROBLEM**: Very similar to KeyDoor.
- **FIX**: Resource management should be about ECONOMICS, not keys-and-doors:
  - Agent has limited resources (energy, health, inventory space)
  - Must decide what to pick up, what to skip, when to rest
  - Trade-offs: short path with hazards vs long safe path that costs more energy
  - Fundamentally about resource allocation under scarcity, not unlocking doors

#### SwitchCircuit
- **PROBLEM 1**: Looks very similar to other "touch objects in order, reach goal" tasks.
- **PROBLEM 2 (BUG)**: User went directly to the goal without touching any switches and the episode terminated. Is that supposed to give negative reward? The goal should NOT be reachable without completing the switch circuit. Switches should physically block the path to the goal (e.g., switches open doors/bridges on the path).
- **GENERAL BUG CHECK**: Verify this "go straight to goal" exploit doesn't exist in ALL tasks that have a goal. Every task with a goal should either: (a) physically block the goal until prerequisites are met, or (b) give NEGATIVE reward for reaching goal without prerequisites.
- **FIX**: Redesign so switches have PHYSICAL consequences (open bridges, deactivate hazards, power elevators). The goal is physically unreachable without the correct switch sequence.

#### ToolUse
- **BUG (CRITICAL)**: User crossed lava without any tool and reached the goal. Episode completed successfully. Completely broken — lava must kill the agent or block passage without the appropriate tool.
- **PROBLEM**: Even if the bug is fixed, this is just KeyDoor with extra steps. Tool → bypass obstacle is the same as Key → open door.
- **FIX**: Redesign ToolUse to be genuinely about TOOL USE:
  - Multiple tools with different effects (hammer breaks walls, bridge crosses water, torch reveals dark areas, shovel digs tunnels)
  - Tools have durability (N uses)
  - Agent must plan which tools to use where
  - Some tools can be combined (hammer + torch = flaming hammer that breaks ice walls)
  - This should feel like a crafting/adventure game, not a key-door puzzle

---

### 🟡 TASKS NEEDING RETHINKING

#### FewShotAdaptation
- **STATUS**: Very cool
- **QUESTION**: Isn't this a memory task? The few-shot aspect should be about rapid adaptation to NEW rules within an episode, not just remembering. Verify the task truly tests adaptation vs just memory. If it's really a memory task, rename or redesign.

#### SequenceMemory
- **STATUS**: Not bad but flawed
- **PROBLEM 1**: Is it really testing memory? One goal lights up, agent goes to it. That's just "go to the lit thing" — no memory required.
- **PROBLEM 2**: Hard maps are just bigger empty maps. No diversity or complexity.
- **FIX**: Genuine memory test:
  - Phase 1: SHOW a sequence of N locations lighting up (agent watches, cannot move or shouldn't move towards them)
  - Phase 2: Agent must REPRODUCE the sequence by visiting locations in the same order
  - Difficulty: longer sequences, more locations, maze obstacles between locations
  - THIS is a memory task. The current version is not.

#### RecursiveRooms
- **STATUS**: Very cool concept!
- **BUG**: Task generation frequently fails and produces empty maps.
- **FIX**: Fix the procedural generation to handle edge cases. Generation should never produce empty maps — add validation and retry logic.

#### DistributionShift
- **STATUS**: Cool effort, interesting direction
- **ISSUE**: Feels like it needs more core logic. What is the role of the T tokens? The shift concept isn't mechanically clear.
- **FIX**: Think about what distribution shift REALLY means for this kind of task:
  - Episode starts with rules the agent has learned (training distribution)
  - Mid-episode, rules change (test distribution): goal position swaps, reward signs flip, action mappings rotate, or terrain changes
  - The agent must detect the shift and adapt
  - T tokens should have a clear role or be removed

#### EmergentStrategy
- **STATUS**: Great design
- **BUG**: Even when the agent reaches the goal area, it doesn't terminate. It stays next to the goal but prefers not to end the episode. This suggests:
  - The goal detection condition may be broken (agent IS on goal but it doesn't register)
  - OR: Dense reward incentivizes staying alive over reaching goal (reward for surviving > reward for finishing)
- **FIX**: Debug goal detection. If the agent is on the goal cell, the episode MUST terminate with positive reward. Check if the step reward inadvertently makes non-termination preferable.

---

## EXECUTION ORDER

1. **Critical bug fixes first**: ToolUse lava bypass, SwitchCircuit goal exploit, DelayedGratification spawn flooding, EmergentStrategy goal detection, RecursiveRooms empty maps, SokobanPush wall-adjacent boxes
2. **Visual rendering fixes**: FogOfWar fog visibility, LightsOut lit/unlit rendering, InstructionFollowing instruction visibility
3. **General bug check**: Verify NO task allows reaching the goal without completing prerequisites (the SwitchCircuit exploit may exist elsewhere)
4. **General comment 2**: Action space audit across all tasks
5. **General comment 1**: Add new object types to entity system
6. **Duplicate differentiation**: CausalChain, ChaseEvade/CompetitiveTag, PackingPuzzle/SokobanPush, PreciseNavigation, ProgramSynthesis/RecipeAssembly, SymbolMatching/TaskInterference, TileSorting, ResourceManagement, SwitchCircuit, ToolUse
7. **Task rethinks**: SequenceMemory, FewShotAdaptation, DistributionShift, NoisyObservation, RuleInduction difficulty scaling
8. **Minor fixes**: GoToGoal dense reward, Herding sheep logic, CooperativeTransport NPC logic, GraphColoring logic audit, MultiGoalRoute path reward and T tokens, DeceptiveReward dense reward check
9. **Run full test suite**: Verify nothing is broken
10. **Re-run PPO benchmarks on fixed tasks**: Verify training still works, agents learn on easy/medium
