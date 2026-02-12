# Tasks

Agentick provides 41 carefully designed tasks testing diverse agent capabilities from basic navigation to advanced reasoning and multi-agent coordination. All tasks support difficulty scaling and multiple observation modes.

## Quick Start: Creating an Environment

```python
import agentick

# Create environment for any task
env = agentick.make(
    "GoToGoal-v0",           # Task name
    difficulty="medium",      # easy, medium, hard, or expert
    render_mode="language",   # ascii, language, language_structured, rgb_array, state_dict, human
    reward_mode="sparse",     # sparse or dense
    seed=42                   # For reproducibility
)

obs, info = env.reset(seed=42)
for step in range(100):
    action = env.action_space.sample()  # Your agent's action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## All Tasks by Category (41 Total)

### Navigation (5 tasks)

These tasks test pathfinding, maze solving, and obstacle avoidance.

#### 1. GoToGoal-v0
Navigate to a visible goal in an open grid.

**Capabilities**: basic_navigation, navigation
**Description**: Agent must navigate from start to goal in an open grid with optional walls.
**Difficulty Scaling**:
- Easy: 5×5 grid, no obstacles, max_steps=20, wall_density=0%
- Medium: 10×10 grid, sparse walls, max_steps=50, wall_density=10%
- Hard: 15×15 grid, moderate walls, max_steps=100, wall_density=20%
- Expert: 20×20 grid, dense walls, max_steps=200, wall_density=25%

**ASCII Example (Easy)**:
```
###########
#^  .  .  #
# . . . . #
# . . . . #
# . . . . #
###########
```

**Success Criterion**: Agent reaches goal position
**Optimal Baseline**: Always visible, direct path optimal
**Random Baseline**: ~1/(grid_size²)
**Optimal/Random Returns**: 1.0 / 0.04

#### 2. MazeNavigation-v0
Solve procedurally generated mazes with complex wall layouts.

**Capabilities**: planning, spatial_reasoning, navigation
**Description**: Navigate complex mazes requiring path exploration and memory.
**Difficulty Scaling**:
- Easy: 7×7 maze, simple structure, max_steps=50
- Medium: 10×10 maze, moderate complexity, max_steps=100
- Hard: 13×13 maze, high complexity, max_steps=200
- Expert: 15×15 maze, very complex, max_steps=400

**Success Criterion**: Reach goal through maze
**Optimal Baseline**: Requires maze solving
**Random Baseline**: Very low (random walk unlikely to solve)
**Optimal/Random Returns**: 1.0 / 0.01

#### 3. MultiGoalRoute-v0
Visit multiple goals in optimal order (traveling salesman variant).

**Capabilities**: planning, optimization, navigation
**Description**: Visit 3-5 goals efficiently. Order matters for reward.
**Difficulty Scaling**:
- Easy: 8×8 grid, 3 goals, max_steps=100
- Medium: 12×12 grid, 4 goals, max_steps=200
- Hard: 15×15 grid, 5 goals, max_steps=300
- Expert: 20×20 grid, 6 goals, max_steps=500

**Success Criterion**: Visit all goals
**Optimal Baseline**: Optimal order minimizes distance
**Random Baseline**: Suboptimal route ordering
**Optimal/Random Returns**: 1.0 / 0.3

#### 4. DynamicObstacles-v0
Navigate while avoiding moving obstacles.

**Capabilities**: reactive_planning, navigation
**Description**: Obstacles move in predictable patterns; agent must avoid collisions.
**Difficulty Scaling**:
- Easy: 10×10 grid, 2 slow obstacles, max_steps=100
- Medium: 12×12 grid, 4 medium obstacles, max_steps=150
- Hard: 15×15 grid, 6 fast obstacles, max_steps=200
- Expert: 20×20 grid, 8 very fast obstacles, max_steps=300

**Success Criterion**: Reach goal without touching obstacle
**Optimal Baseline**: Predict obstacles, navigate safely
**Random Baseline**: High collision rate
**Optimal/Random Returns**: 1.0 / 0.1

#### 5. FogOfWarExploration-v0
Navigate with limited vision (fog of war).

**Capabilities**: exploration, memory, navigation
**Description**: Agent can only see nearby tiles; must explore to find goal.
**Difficulty Scaling**:
- Easy: 8×8 grid, vision_radius=3, max_steps=200
- Medium: 12×12 grid, vision_radius=2, max_steps=300
- Hard: 15×15 grid, vision_radius=2, max_steps=400
- Expert: 20×20 grid, vision_radius=1, max_steps=600

**Success Criterion**: Find and reach goal with limited vision
**Optimal Baseline**: Systematic exploration (e.g., BFS)
**Random Baseline**: Random walk discovery
**Optimal/Random Returns**: 1.0 / 0.15

---

### Memory (4 tasks)

These tasks test the agent's ability to remember previous observations and retrace paths.

#### 6. KeyDoorPuzzle-v0
Collect keys to unlock doors in sequence.

**Capabilities**: memory, sequential_reasoning
**Description**: Find key(s), unlock door(s), reach goal. Tests working memory and planning.
**Difficulty Scaling**:
- Easy: 7×7 grid, 1 key-door pair, max_steps=100
- Medium: 10×10 grid, 2 key-door chains, max_steps=150
- Hard: 13×13 grid, 3 key-door chains, max_steps=250
- Expert: 15×15 grid, 4 complex chains, max_steps=400

**Success Criterion**: Reach goal after collecting all necessary keys and unlocking doors
**Optimal Baseline**: Optimal collection order
**Random Baseline**: Unlikely to execute full sequence
**Optimal/Random Returns**: 1.0 / 0.0

#### 7. BreadcrumbTrail-v0
Remember path using breadcrumbs, can't revisit old path.

**Capabilities**: long_horizon, memory
**Description**: Collect items along a path, then must return via different path.
**Difficulty Scaling**:
- Easy: 8×8 grid, 5 breadcrumbs, max_steps=100
- Medium: 10×10 grid, 8 breadcrumbs, max_steps=150
- Hard: 12×12 grid, 12 breadcrumbs, max_steps=200
- Expert: 15×15 grid, 15 breadcrumbs, max_steps=300

**Success Criterion**: Collect breadcrumbs and return without retracing
**Optimal Baseline**: Efficient route selection
**Random Baseline**: High chance of retracing
**Optimal/Random Returns**: 1.0 / 0.2

#### 8. SequenceMemory-v0
Reproduce a sequence of colors/objects.

**Capabilities**: memory, pattern_recognition
**Description**: Observe sequence, then reproduce it (Simon-Says style).
**Difficulty Scaling**:
- Easy: 4 items, 1-turn memory, max_steps=50
- Medium: 6 items, 2-turn memory, max_steps=100
- Hard: 8 items, 3-turn memory, max_steps=150
- Expert: 10 items, 4-turn memory, max_steps=200

**Success Criterion**: Reproduce full sequence without errors
**Optimal Baseline**: Perfect recall
**Random Baseline**: Random guessing (very low)
**Optimal/Random Returns**: 1.0 / 0.001

#### 9. BacktrackPuzzle-v0
Navigate through rooms, remembering visited state for backtracking.

**Capabilities**: memory, planning
**Description**: Move through rooms collecting items, must backtrack to starting room.
**Difficulty Scaling**:
- Easy: 3 rooms connected linearly, max_steps=100
- Medium: 5 rooms in tree structure, max_steps=200
- Hard: 7 rooms in complex graph, max_steps=300
- Expert: 10 rooms with cycles, max_steps=500

**Success Criterion**: Collect items and return to start
**Optimal Baseline**: Optimal route + memory
**Random Baseline**: Low (hard to randomly find start)
**Optimal/Random Returns**: 1.0 / 0.1

---

### Reasoning (5 tasks)

These tasks test logical reasoning, pattern matching, and planning.

#### 10. SokobanPush-v0
Push boxes onto target positions.

**Capabilities**: reasoning, planning
**Description**: Classic Sokoban: push boxes onto targets without deadlocking.
**Difficulty Scaling**:
- Easy: 7×7 grid, 1 box, max_steps=100
- Medium: 10×10 grid, 2 boxes, max_steps=200
- Hard: 13×13 grid, 4 boxes, max_steps=300
- Expert: 15×15 grid, 6 boxes, max_steps=500

**Success Criterion**: All boxes on targets
**Optimal Baseline**: Careful planning to avoid deadlocks
**Random Baseline**: Very low (deadlocks likely)
**Optimal/Random Returns**: 1.0 / 0.01

#### 11. SymbolMatching-v0
Match symbols or patterns based on rules.

**Capabilities**: pattern_recognition, reasoning
**Description**: Learn rule (e.g., color-shape matching) and apply correctly.
**Difficulty Scaling**:
- Easy: 3 symbol types, simple 1-rule, max_steps=50
- Medium: 5 symbol types, 2-rule combinations, max_steps=100
- Hard: 7 symbol types, 3-rule logic, max_steps=150
- Expert: 9 symbol types, complex rules, max_steps=200

**Success Criterion**: Match all symbols according to rule
**Optimal Baseline**: Learn rule, apply correctly
**Random Baseline**: Random matching
**Optimal/Random Returns**: 1.0 / 0.15

#### 12. CausalChain-v0
Understand causal relationships (A causes B, B causes C).

**Capabilities**: causal_reasoning
**Description**: Identify causal chains and manipulate appropriately.
**Difficulty Scaling**:
- Easy: 2-step chain (A→B), max_steps=100
- Medium: 3-step chain (A→B→C), max_steps=150
- Hard: 4-step chain with branches, max_steps=200
- Expert: Complex multi-branch causality, max_steps=300

**Success Criterion**: Achieve goal by correctly manipulating causal chain
**Optimal Baseline**: Understand and execute chain
**Random Baseline**: Trial and error
**Optimal/Random Returns**: 1.0 / 0.1

#### 13. RuleInduction-v0
Induce rules from examples.

**Capabilities**: reasoning, generalization
**Description**: Given examples, infer rule and apply to new cases.
**Difficulty Scaling**:
- Easy: Rule from 3 examples, 2-case test, max_steps=100
- Medium: Rule from 4 examples, 4-case test, max_steps=150
- Hard: Rule from 5 examples, 6-case test, max_steps=200
- Expert: Complex rule from 6+ examples, max_steps=250

**Success Criterion**: Correctly infer and apply rule
**Optimal Baseline**: Perfect induction
**Random Baseline**: Random guessing
**Optimal/Random Returns**: 1.0 / 0.25

#### 14. SwitchCircuit-v0
Solve logic circuits using switches.

**Capabilities**: compositional_logic
**Description**: Manipulate switches to activate outputs (AND/OR/NOT logic).
**Difficulty Scaling**:
- Easy: Simple 2-switch AND gate, max_steps=50
- Medium: 3 switches with AND/OR, max_steps=100
- Hard: 4 switches with complex logic, max_steps=150
- Expert: 5+ switches with branching, max_steps=200

**Success Criterion**: Activate all required outputs
**Optimal Baseline**: Understand circuit logic
**Random Baseline**: Random switch toggling
**Optimal/Random Returns**: 1.0 / 0.1

---

### Control (4 tasks)

These tasks test precise low-level control and reactive behavior.

#### 15. PreciseNavigation-v0
Navigate to target with limited action budget.

**Capabilities**: low_level_control
**Description**: Reach target in exact move count (no extra steps).
**Difficulty Scaling**:
- Easy: 5×5 grid, 5-8 step solution, max_steps=10
- Medium: 10×10 grid, 10-15 step solution, max_steps=20
- Hard: 15×15 grid, 20-30 step solution, max_steps=35
- Expert: 20×20 grid, 30-50 step solution, max_steps=60

**Success Criterion**: Reach goal in exact steps (no more, no less)
**Optimal Baseline**: Exact path planning
**Random Baseline**: Exact steps by chance
**Optimal/Random Returns**: 1.0 / 0.01

#### 16. ChaseEvade-v0
Chase a moving target or evade an enemy.

**Capabilities**: reactive_control, prediction
**Description**: Catch moving target or escape moving enemy with predictable behavior.
**Difficulty Scaling**:
- Easy: Slow target, large grid, max_steps=100
- Medium: Medium speed, moderate grid, max_steps=150
- Hard: Fast target, small grid, max_steps=200
- Expert: Very fast, tight space, max_steps=250

**Success Criterion**: Catch target or survive without touching enemy
**Optimal Baseline**: Predict and intercept/avoid
**Random Baseline**: Random action
**Optimal/Random Returns**: 1.0 / 0.2

#### 17. TimingChallenge-v0
Perform actions at precise moments.

**Capabilities**: low_level_control, timing
**Description**: Hit moving target at exact time or synchronize with pattern.
**Difficulty Scaling**:
- Easy: Slow pattern, large window, max_steps=100
- Medium: Moderate speed, normal window, max_steps=150
- Hard: Fast pattern, small window, max_steps=200
- Expert: Very fast, tiny window, max_steps=250

**Success Criterion**: Hit target at right moment
**Optimal Baseline**: Perfect timing
**Random Baseline**: Random action timing
**Optimal/Random Returns**: 1.0 / 0.1

#### 18. Herding-v0
Control multiple objects toward goal.

**Capabilities**: multi_objective_control
**Description**: Herd multiple boxes/creatures to target area.
**Difficulty Scaling**:
- Easy: 2 objects, large target zone, max_steps=150
- Medium: 3 objects, normal zone, max_steps=200
- Hard: 4 objects, small zone, max_steps=250
- Expert: 5 objects, tiny zone, max_steps=350

**Success Criterion**: All objects in target zone simultaneously
**Optimal Baseline**: Efficient herding patterns
**Random Baseline**: Random movement
**Optimal/Random Returns**: 1.0 / 0.05

---

### Skill Composition (5 tasks)

These tasks test the ability to combine multiple skills over long horizons.

#### 19. MultiRoomEscape-v0
Navigate through multiple rooms to exit.

**Capabilities**: skill_composition, long_horizon
**Description**: Traverse connected rooms, each presenting different challenges.
**Difficulty Scaling**:
- Easy: 3 simple rooms, max_steps=200
- Medium: 5 moderate rooms, max_steps=400
- Hard: 7 complex rooms, max_steps=600
- Expert: 10 rooms with obstacles, max_steps=1000

**Success Criterion**: Exit final room
**Optimal Baseline**: Navigate efficiently through rooms
**Random Baseline**: Random exploration
**Optimal/Random Returns**: 1.0 / 0.1

#### 20. ResourceManagement-v0
Collect and manage resources efficiently.

**Capabilities**: planning, optimization
**Description**: Collect resources efficiently while managing limited inventory.
**Difficulty Scaling**:
- Easy: 3 resource types, large inventory, max_steps=200
- Medium: 5 resource types, limited inventory, max_steps=300
- Hard: 7 resource types, very limited, max_steps=400
- Expert: 10 resource types, minimal inventory, max_steps=500

**Success Criterion**: Collect target quantities of each resource
**Optimal Baseline**: Optimal collection order
**Random Baseline**: Random collection
**Optimal/Random Returns**: 1.0 / 0.3

#### 21. ToolUse-v0
Select and use tools to solve problems.

**Capabilities**: skill_discovery, tool_use
**Description**: Find appropriate tools and use them to reach goals.
**Difficulty Scaling**:
- Easy: 2 tools, 1 use per tool, max_steps=100
- Medium: 3 tools, tool dependencies, max_steps=200
- Hard: 4 tools, complex dependencies, max_steps=300
- Expert: 5 tools, multi-step sequences, max_steps=400

**Success Criterion**: Use tools correctly to achieve goal
**Optimal Baseline**: Understand tool purpose and use correctly
**Random Baseline**: Random tool selection
**Optimal/Random Returns**: 1.0 / 0.05

#### 22. RecipeAssembly-v0
Follow recipes to assemble objects.

**Capabilities**: compositional_logic, planning
**Description**: Gather ingredients and assemble according to recipe.
**Difficulty Scaling**:
- Easy: 3-ingredient recipe, all ingredients nearby, max_steps=100
- Medium: 4-ingredient recipe, distributed, max_steps=200
- Hard: 5-ingredient recipe, scattered, max_steps=300
- Expert: 6-ingredient recipe, hidden locations, max_steps=400

**Success Criterion**: Assemble complete recipe
**Optimal Baseline**: Optimal collection order
**Random Baseline**: Random assembly attempts
**Optimal/Random Returns**: 1.0 / 0.05

#### 23. EmergentStrategy-v0
Discover novel strategies to solve problems.

**Capabilities**: skill_discovery, creativity
**Description**: Open-ended problem-solving requiring novel strategy discovery.
**Difficulty Scaling**:
- Easy: Simple obstacle/objective, max_steps=200
- Medium: Moderate complexity, max_steps=300
- Hard: High complexity, max_steps=400
- Expert: Very complex, max_steps=500

**Success Criterion**: Reach goal (multiple valid strategies)
**Optimal Baseline**: Efficient strategy
**Random Baseline**: Random exploration
**Optimal/Random Returns**: 1.0 / 0.1

---

### Combinatorial (4 tasks)

These tasks require solving combinatorial problems.

#### 24. TileSorting-v0
Sort tiles into matching groups.

**Capabilities**: combinatorial_reasoning
**Description**: Rearrange tiles to match target configuration.
**Difficulty Scaling**:
- Easy: 3 types, 9 tiles (3×3), max_steps=50
- Medium: 4 types, 16 tiles (4×4), max_steps=100
- Hard: 5 types, 25 tiles (5×5), max_steps=200
- Expert: 6 types, 36 tiles (6×6), max_steps=400

**Success Criterion**: All tiles of same type grouped together
**Optimal Baseline**: Efficient sorting algorithm
**Random Baseline**: Random rearrangement
**Optimal/Random Returns**: 1.0 / 0.01

#### 25. PackingPuzzle-v0
Pack objects optimally into containers.

**Capabilities**: spatial_reasoning, combinatorics
**Description**: Fit objects into limited space (knapsack-like).
**Difficulty Scaling**:
- Easy: 3 items, spacious container, max_steps=100
- Medium: 5 items, normal space, max_steps=200
- Hard: 7 items, tight space, max_steps=300
- Expert: 10 items, very tight space, max_steps=400

**Success Criterion**: Pack all items efficiently
**Optimal Baseline**: Optimal packing
**Random Baseline**: Random placement
**Optimal/Random Returns**: 1.0 / 0.1

#### 26. GraphColoring-v0
Color graph nodes (vertex coloring problem).

**Capabilities**: constraint_satisfaction
**Description**: Assign colors to nodes such that no adjacent nodes have same color.
**Difficulty Scaling**:
- Easy: 5 nodes, sparse graph, 3 colors, max_steps=100
- Medium: 8 nodes, normal graph, 4 colors, max_steps=150
- Hard: 12 nodes, dense graph, 4 colors, max_steps=200
- Expert: 16 nodes, very dense, 4 colors, max_steps=300

**Success Criterion**: Valid coloring (no adjacent same color)
**Optimal Baseline**: Graph coloring algorithm
**Random Baseline**: Random coloring
**Optimal/Random Returns**: 1.0 / 0.1

#### 27. LightsOut-v0
Turn all lights off with button presses.

**Capabilities**: combinatorial_logic
**Description**: Classic Lights Out puzzle (toggle lights in grid).
**Difficulty Scaling**:
- Easy: 3×3 grid, 3-5 button presses to solve, max_steps=20
- Medium: 4×4 grid, 5-8 presses to solve, max_steps=40
- Hard: 5×5 grid, 8-12 presses to solve, max_steps=80
- Expert: 6×6 grid, 12-20 presses to solve, max_steps=150

**Success Criterion**: All lights off
**Optimal Baseline**: Optimal button sequence
**Random Baseline**: Random button presses
**Optimal/Random Returns**: 1.0 / 0.05

---

### Compositional (3 tasks)

These tasks require composing learned behaviors.

#### 28. InstructionFollowing-v0
Follow natural language instructions.

**Capabilities**: language, grounding, instruction
**Description**: Execute multi-step instructions in natural language.
**Difficulty Scaling**:
- Easy: Simple 2-step instruction, max_steps=50
- Medium: 3-4 step compound instruction, max_steps=100
- Hard: 5-6 step complex instruction, max_steps=150
- Expert: 7-10 step instruction with conditionals, max_steps=200

**Success Criterion**: Execute instruction correctly
**Optimal Baseline**: Parse and execute instruction
**Random Baseline**: Random actions
**Optimal/Random Returns**: 1.0 / 0.01

#### 29. RecursiveRooms-v0
Navigate hierarchical room structures.

**Capabilities**: hierarchical, planning, composition
**Description**: Rooms within rooms; must navigate at multiple levels.
**Difficulty Scaling**:
- Easy: 2 level nesting, 4 rooms, max_steps=200
- Medium: 3 level nesting, 8 rooms, max_steps=300
- Hard: 4 level nesting, 16 rooms, max_steps=400
- Expert: 5 level nesting, 32 rooms, max_steps=600

**Success Criterion**: Reach goal in innermost room
**Optimal Baseline**: Hierarchical navigation
**Random Baseline**: Random exploration
**Optimal/Random Returns**: 1.0 / 0.05

#### 30. ProgramSynthesis-v0
Synthesize a program/sequence to solve task.

**Capabilities**: reasoning, planning, abstraction
**Description**: Compose actions into reusable sequences (macro discovery).
**Difficulty Scaling**:
- Easy: Discover 1 macro, 3-step solution, max_steps=100
- Medium: Discover 2 macros, 5-step solution, max_steps=200
- Hard: Discover 3 macros, 7-step solution, max_steps=300
- Expert: Discover 4+ macros, 10+ step solution, max_steps=400

**Success Criterion**: Execute synthesized program correctly
**Optimal Baseline**: Efficient macro discovery
**Random Baseline**: Random actions
**Optimal/Random Returns**: 1.0 / 0.05

---

### World Model (3 tasks)

These tasks test building and using world models.

#### 31. PhysicsDiscovery-v0
Discover physics rules (e.g., gravity, friction).

**Capabilities**: world_model, exploration, physics
**Description**: Experiment to discover how objects behave.
**Difficulty Scaling**:
- Easy: 1 physics rule, obvious effects, max_steps=200
- Medium: 2 rules, subtle effects, max_steps=300
- Hard: 3 rules, complex interactions, max_steps=400
- Expert: 4+ rules, counter-intuitive, max_steps=500

**Success Criterion**: Demonstrate understanding of physics rules
**Optimal Baseline**: Systematic experimentation
**Random Baseline**: Random exploration
**Optimal/Random Returns**: 1.0 / 0.2

#### 32. EnvironmentShift-v0
Detect and adapt to sudden environment changes.

**Capabilities**: world_model, adaptation, change_detection
**Description**: Environment changes mid-episode; agent must detect and adapt.
**Difficulty Scaling**:
- Easy: 1 obvious change mid-episode, max_steps=200
- Medium: 2 subtle changes, max_steps=300
- Hard: 3 changes with delay, max_steps=400
- Expert: 4+ changes, hard to detect, max_steps=500

**Success Criterion**: Adapt to changes and still reach goal
**Optimal Baseline**: Detect change, update model, adapt
**Random Baseline**: No adaptation
**Optimal/Random Returns**: 1.0 / 0.3

#### 33. RuleDiscoveryNavigation-v0
Discover navigation rules (one-way passages, etc).

**Capabilities**: world_model, spatial_reasoning
**Description**: Navigate with hidden rules (some passages one-way).
**Difficulty Scaling**:
- Easy: 1 one-way passage, clear goal, max_steps=150
- Medium: 2-3 one-way passages, max_steps=250
- Hard: 4-5 one-way passages, complex layout, max_steps=350
- Expert: 6+ one-way passages, hard to discover, max_steps=500

**Success Criterion**: Discover rules and reach goal
**Optimal Baseline**: Systematic exploration to discover rules
**Random Baseline**: Random navigation
**Optimal/Random Returns**: 1.0 / 0.1

---

### Adversarial (3 tasks)

These tasks test robustness to adversarial conditions.

#### 34. DeceptiveReward-v0
Achieve goal despite deceptive reward signals.

**Capabilities**: robustness, reward_hacking, exploration
**Description**: Reward signal misleads; must ignore it to reach actual goal.
**Difficulty Scaling**:
- Easy: Goal and reward aligned, max_steps=100
- Medium: Reward somewhat misleading, max_steps=200
- Hard: Reward strongly misleading, max_steps=300
- Expert: Reward directly contradicts goal, max_steps=400

**Success Criterion**: Reach true goal despite misleading rewards
**Optimal Baseline**: Identify true goal, ignore reward
**Random Baseline**: Follow reward (fail)
**Optimal/Random Returns**: 1.0 / 0.0

#### 35. NoisyObservation-v0
Navigate with noisy/ambiguous observations.

**Capabilities**: robustness, navigation, noise
**Description**: Observations are corrupted; must navigate anyway.
**Difficulty Scaling**:
- Easy: 10% observation noise, max_steps=200
- Medium: 25% observation noise, max_steps=300
- Hard: 50% observation noise, max_steps=400
- Expert: 75% observation noise, max_steps=500

**Success Criterion**: Reach goal with noisy observations
**Optimal Baseline**: Robust navigation despite noise
**Random Baseline**: Random navigation
**Optimal/Random Returns**: 1.0 / 0.2

#### 36. DistributionShift-v0
Generalize to distribution shift (OOD).

**Capabilities**: generalization, ood, robustness
**Description**: Train on one distribution, test on shifted distribution.
**Difficulty Scaling**:
- Easy: 10% shift (grid size, layout), max_steps=100
- Medium: 25% shift, max_steps=200
- Hard: 50% shift, max_steps=300
- Expert: 75% shift (very different), max_steps=400

**Success Criterion**: Achieve decent performance on shifted distribution
**Optimal Baseline**: Learn generalizable policy
**Random Baseline**: No generalization
**Optimal/Random Returns**: 1.0 / 0.1

---

### Multi-Agent (2 tasks)

These tasks involve multiple agents.

#### 37. CooperativeTransport-v0
Multiple agents cooperate to transport objects.

**Capabilities**: multi_agent, cooperation
**Description**: 2-3 agents must coordinate to move boxes to goal.
**Difficulty Scaling**:
- Easy: 2 agents, 1 box, clear target, max_steps=200
- Medium: 3 agents, 2 boxes, normal complexity, max_steps=300
- Hard: 4 agents, 3 boxes, complex coordination, max_steps=400
- Expert: 4 agents, 4 boxes, very tight coordination, max_steps=500

**Success Criterion**: All boxes reach target
**Optimal Baseline**: Efficient multi-agent coordination
**Random Baseline**: Random independent actions
**Optimal/Random Returns**: 1.0 / 0.05

#### 38. CompetitiveTag-v0
Multi-agent competition (chase/evade).

**Capabilities**: multi_agent, competition
**Description**: 2 agents: one chases, one evades (they switch roles).
**Difficulty Scaling**:
- Easy: Large arena, slow agents, max_steps=200
- Medium: Normal arena, normal speed, max_steps=300
- Hard: Small arena, fast agents, max_steps=400
- Expert: Very small, very fast, max_steps=500

**Success Criterion**: Successful chases as pursuer, successful evades as evadee
**Optimal Baseline**: Strategic behavior based on role
**Random Baseline**: Random movement
**Optimal/Random Returns**: 1.0 / 0.3

---

### Meta-Learning (2 tasks)

These tasks test learning to learn and adaptation.

#### 39. TaskInterference-v0
Solve related tasks without negative transfer.

**Capabilities**: multi_task, interference, meta_learning
**Description**: Solve task A, then task B (similar but different). Tests avoiding interference.
**Difficulty Scaling**:
- Easy: Tasks very similar, max_steps=300 each
- Medium: Tasks somewhat different, max_steps=400 each
- Hard: Tasks quite different, max_steps=500 each
- Expert: Tasks contradictory, max_steps=600 each

**Success Criterion**: Solve both tasks efficiently
**Optimal Baseline**: Task-specific adaptation
**Random Baseline**: No adaptation between tasks
**Optimal/Random Returns**: 1.0 / 0.2

#### 40. FewShotAdaptation-v0
Adapt to new task from few examples.

**Capabilities**: meta_learning, adaptation, few_shot
**Description**: See 1-3 examples of new task, then solve it.
**Difficulty Scaling**:
- Easy: 3 examples, simple task, max_steps=200
- Medium: 2 examples, moderate task, max_steps=300
- Hard: 1 example, complex task, max_steps=400
- Expert: 1 cryptic example, hard task, max_steps=500

**Success Criterion**: Solve task based on few-shot examples
**Optimal Baseline**: Quick learning from examples
**Random Baseline**: No learning from examples
**Optimal/Random Returns**: 1.0 / 0.1

---

### Bonus Challenge (1 task)

#### 41. DelayedGratification-v0
Resist immediate reward for larger delayed reward.

**Capabilities**: credit_assignment, long_horizon
**Description**: Small reward available immediately, larger reward requires waiting.
**Difficulty Scaling**:
- Easy: Immediate reward 0.1, delayed 1.0, wait 20 steps, max_steps=50
- Medium: Immediate 0.2, delayed 1.0, wait 50 steps, max_steps=150
- Hard: Immediate 0.4, delayed 1.0, wait 100 steps, max_steps=250
- Expert: Immediate 0.9, delayed 1.0, wait 200 steps, max_steps=400

**Success Criterion**: Resist immediate reward and obtain delayed reward
**Optimal Baseline**: Optimal value discounting
**Random Baseline**: Random (might take immediate)
**Optimal/Random Returns**: 1.0 / 0.5

---

## Task Selection Guide

### By Agent Type

**LLM Agents**: GoToGoal, KeyDoorPuzzle, InstructionFollowing, SymbolMatching
- Tasks where language descriptions help
- Navigation and reasoning combined

**Vision Models**: GoToGoal, MazeNavigation, SokobanPush, RuleDiscoveryNavigation
- Visual reasoning tasks
- Clear spatial structure

**RL Agents**: All tasks
- Start with GoToGoal (easy baseline)
- Advance to MazeNavigation, SokobanPush
- Challenge with multi-agent tasks

**Bots/Search**: MazeNavigation, SokobanPush, GraphColoring, PackingPuzzle
- Tasks with perfect information
- Algorithm-solvable problems

**Human Baselines**: GoToGoal, MultiGoalRoute, Herding
- Intuitive objectives
- Visual/spatial reasoning

### By Difficulty Level

**Absolute Beginners**:
- Easy: GoToGoal, KeyDoorPuzzle, MazeNavigation
- Test basic navigation and memory

**Intermediate Learners**:
- Medium: All tasks at medium difficulty
- Balanced challenge and solvability

**Advanced Agents**:
- Hard/Expert: Specialized tasks
- SokobanPush (hard): Deep planning
- CompetitiveTag (expert): Multi-agent reasoning
- FogOfWarExploration (expert): Memory + navigation

### By Capability

| Capability | Tasks |
|---|---|
| **Navigation** | GoToGoal, MazeNavigation, MultiGoalRoute, DynamicObstacles, FogOfWarExploration |
| **Memory** | KeyDoorPuzzle, BreadcrumbTrail, SequenceMemory, BacktrackPuzzle |
| **Planning** | MazeNavigation, SokobanPush, RecipeAssembly, ProgramSynthesis |
| **Reasoning** | SymbolMatching, CausalChain, RuleInduction, SwitchCircuit |
| **Control** | PreciseNavigation, ChaseEvade, TimingChallenge, Herding |
| **Skills** | MultiRoomEscape, ResourceManagement, ToolUse, RecipeAssembly |
| **Composition** | RecursiveRooms, InstructionFollowing, ProgramSynthesis |
| **World Model** | PhysicsDiscovery, EnvironmentShift, RuleDiscoveryNavigation |
| **Robustness** | DeceptiveReward, NoisyObservation, DistributionShift |
| **Multi-Agent** | CooperativeTransport, CompetitiveTag |
| **Meta-Learning** | TaskInterference, FewShotAdaptation |

## Difficulty Scaling Principles

All tasks follow consistent difficulty scaling:

1. **Grid Size**: Increases with difficulty
   - Easy: 5-8×5-8
   - Medium: 8-12×8-12
   - Hard: 12-15×12-15
   - Expert: 15-20×15-20

2. **Obstacle Density**: More obstacles at higher difficulty
   - Easy: 0-10% density
   - Medium: 10-20%
   - Hard: 20-30%
   - Expert: 30-50%

3. **Episode Length**: More steps available at higher difficulty
   - Easy: 20-100 steps
   - Medium: 50-200 steps
   - Hard: 100-300 steps
   - Expert: 200-600 steps

4. **Problem Complexity**: More complex constraints/dependencies
   - Easy: Simple, single objective
   - Medium: Multiple objectives or complex layouts
   - Hard: Complex interactions
   - Expert: Very complex, counter-intuitive

## Running Tasks

### Single Episode
```python
import agentick

env = agentick.make("GoToGoal-v0", difficulty="medium")
obs, info = env.reset(seed=42)

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward}, Valid actions: {info['valid_actions']}")
    if terminated or truncated:
        break
```

### Benchmark Suite
```python
import agentick
from agentick.benchmark import make_suite

# Quick suite (5 representative tasks)
envs = make_suite("quick", difficulty="medium")

# Navigation suite
envs = make_suite("navigation", difficulty="hard")

# Full suite (41 tasks)
envs = make_suite("full", difficulty="expert")
```

### Evaluate Agent
```python
import agentick
from agentick.benchmark import make_suite

suite = make_suite("quick")
results = {}

for env in suite:
    obs, info = env.reset(seed=42)
    episode_return = 0

    for _ in range(500):
        action = agent.get_action(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward

        if terminated or truncated:
            break

    results[env.task.name] = episode_return

print(f"Average return: {sum(results.values()) / len(results)}")
```

## Creating Custom Tasks

See [Custom Tasks](../extending/custom_tasks.md) for how to create your own tasks using the TaskSpec protocol.

## Task Metadata

Each task exposes metadata for benchmarking:

```python
env = agentick.make("GoToGoal-v0")

# Task information
print(env.task.name)                    # "GoToGoal-v0"
print(env.task.description)             # Full description
print(env.task.capability_tags)         # Tags for categorization
print(env.task.difficulty)              # Current difficulty
print(env.task.difficulty_configs.keys())  # Available difficulties

# Performance references
print(env.task.get_optimal_return())    # Optimal agent score
print(env.task.get_random_baseline())   # Random agent score
print(env.task.get_max_steps())         # Maximum steps for this difficulty
```

## Tips for Success

1. **Start Easy**: Ensure your agent can solve easy versions before hard versions
2. **Use Baselines**: Compare against random and optimal baselines
3. **Observation Mode**: Choose observation mode matching agent capabilities
4. **Reward Mode**: Dense rewards easier for learning; sparse for true difficulty
5. **Multi-Modal**: Use multiple render modes to understand agent behavior
6. **Reproducibility**: Always use seeds for comparison across agents

See [Architecture](architecture.md) and [Observations](observations.md) for implementation details.
