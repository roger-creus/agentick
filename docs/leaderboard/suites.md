# Agentick Benchmark Suites

Complete documentation of all official benchmark suites used for leaderboard evaluation.

## Quick Reference

| Suite | Tasks | Seeds | Difficulty | Time | When to Use |
|-------|-------|-------|-----------|------|------------|
| **agentick-quick-v1** | 5 | 10 | Easy | ~5 min | Quick iteration & testing |
| **agentick-core-v1** | 27 | 50 | Medium | ~2 hours | Standard baseline |
| **agentick-full-v1** | 38 | 50 | Medium | ~4-8 hours | Publication & SOTA |
| **Navigation** | 5 | 30 | All | ~1 hour | Deep-dive: navigation |
| **Memory** | 5 | 30 | All | ~1 hour | Deep-dive: memory |
| **Reasoning** | 5 | 30 | All | ~1 hour | Deep-dive: reasoning |
| **Skill** | 5 | 30 | All | ~1 hour | Deep-dive: skill discovery |
| **Control** | 4 | 30 | All | ~45 min | Deep-dive: control |
| **Combinatorial** | 4 | 30 | All | ~45 min | Deep-dive: combinatorial |
| **World Model** | 3 | 30 | All | ~30 min | World model capability |
| **Adversarial** | 3 | 30 | Medium | ~30 min | Robustness testing |
| **Meta-Learning** | 2 | 30 | Medium | ~20 min | Meta-learning |
| **Multi-Agent** | 2 | 30 | Medium | ~20 min | Coordination/competition |
| **Difficulty Scaling** | 10 | 20 | All | ~1.5 hours | Difficulty analysis |
| **Multimodal** | 10 | 20 | Medium | ~1.5 hours | Observation mode comparison |

## Suite Descriptions

### Quick Sanity Check (agentick-quick-v1)

**Purpose**: Fast validation during development iteration

- **Tasks**: 5 foundational tasks (easy difficulty)
  1. GoToGoal-v0 - Reach a target location
  2. MazeNavigation-v0 - Navigate maze to exit
  3. KeyDoorPuzzle-v0 - Find key, unlock door, reach goal
  4. SokobanPush-v0 - Push boxes to goal locations
  5. PreciseNavigation-v0 - Navigate with continuous controls

- **Configuration**:
  - Seeds: 10 deterministic
  - Episodes per seed: 1
  - Max steps: 100 (shorter episodes)
  - Difficulty: Easy

- **Scoring**: Mean of per-task normalized scores (0-1 range)

- **Time**: ~5 minutes (all 5 tasks)

- **Use When**:
  - Testing submission before full eval
  - Developing new agent ideas
  - Debugging configuration issues
  - Quick performance sanity checks

**Example**:
```bash
agentick evaluate --submission my_agent.yaml --suite agentick-quick-v1
```

---

### Core Benchmark (agentick-core-v1)

**Purpose**: Standard baseline for official comparisons

The original 27 core tasks spanning all capability dimensions.

- **Tasks** (27 total):
  - **Navigation (5)**: GoToGoal-v0, MazeNavigation-v0, FogOfWarExploration-v0, DynamicObstacles-v0, MultiGoalRoute-v0
  - **Memory (5)**: KeyDoorPuzzle-v0, SequenceMemory-v0, DelayedGratification-v0, BacktrackPuzzle-v0, BreadcrumbTrail-v0
  - **Reasoning (5)**: SokobanPush-v0, CausalChain-v0, SymbolMatching-v0, RuleInduction-v0, SwitchCircuit-v0
  - **Skill (5)**: ToolUse-v0, RecipeAssembly-v0, MultiRoomEscape-v0, EmergentStrategy-v0, ResourceManagement-v0
  - **Control (4)**: PreciseNavigation-v0, TimingChallenge-v0, ChaseEvade-v0, Herding-v0
  - **Combinatorial (2)**: LightsOut-v0, GraphColoring-v0

- **Configuration**:
  - Seeds: 50 deterministic (reproducible across runs)
  - Episodes per seed: 1
  - Max steps: Default per-task
  - Difficulty: Medium

- **Scoring**:
  1. Per-task normalized score (agent vs random/optimal baselines)
  2. Per-capability average (navigation, memory, reasoning, etc.)
  3. Overall score = equal average of 6 capability scores

- **Time**: ~2 hours

- **Use When**:
  - Establishing baseline performance
  - Comparing against historical results
  - Standard research benchmarking
  - Official leaderboard submissions

**Scoring Example**:
```
Overall Agentick Score: 0.65

Per-Capability:
  navigation:    0.70 (5 tasks)
  memory:        0.62 (5 tasks)
  reasoning:     0.68 (5 tasks)
  skill:         0.65 (5 tasks)
  control:       0.63 (4 tasks)
  combinatorial: 0.60 (2 tasks)
```

---

### Full Benchmark (agentick-full-v1)

**Purpose**: Comprehensive evaluation including advanced capabilities

The complete benchmark with all 38 official tasks.

- **Tasks** (38 total):
  - All 27 core tasks (see above)
  - **World Model (3)**: EnvironmentShift-v0, PhysicsDiscovery-v0, RuleDiscoveryNavigation-v0
  - **Adversarial (3)**: DeceptiveReward-v0, DistributionShift-v0, NoisyObservation-v0
  - **Meta-Learning (2)**: FewShotAdaptation-v0, TaskInterference-v0
  - **Multi-Agent (2)**: CooperativeTransport-v0, CompetitiveTag-v0
  - **Advanced Combinatorial (1)**: PackingPuzzle-v0

- **Configuration**:
  - Seeds: 50 deterministic
  - Episodes per seed: 1
  - Max steps: Default per-task
  - Difficulty: Medium

- **Scoring**:
  1. Per-task normalized scores
  2. Per-capability averages (10 capabilities)
  3. Overall score = equal average of 10 capability scores

- **Time**: 4-8 hours (depending on agent latency)

- **Capabilities Measured**:
  1. Navigation
  2. Memory
  3. Reasoning
  4. Skill Discovery
  5. Control
  6. Combinatorial Reasoning
  7. World Modeling
  8. Adversarial Robustness
  9. Meta-Learning
  10. Multi-Agent Coordination

- **Use When**:
  - Publishing research papers
  - Competing for SOTA
  - Comprehensive capability assessment
  - Production readiness evaluation

**Scoring Example**:
```
Overall Agentick Score: 0.62 (95% CI: 0.59-0.65)

Per-Capability:
  navigation:      0.70
  memory:          0.62
  reasoning:       0.68
  skill:           0.65
  control:         0.63
  combinatorial:   0.60
  worldmodel:      0.58
  adversarial:     0.55
  meta:            0.60
  multiagent:      0.58
```

---

## Capability-Specific Suites

These suites provide deep-dive evaluation into specific agent capabilities.

### Navigation Capability (agentick-navigation-v1)

**Focus**: Spatial reasoning, pathfinding, obstacle avoidance

- **Tasks** (5):
  - GoToGoal-v0 - Reach target location
  - MazeNavigation-v0 - Navigate through maze
  - FogOfWarExploration-v0 - Explore with limited visibility
  - DynamicObstacles-v0 - Navigate around moving obstacles
  - MultiGoalRoute-v0 - Visit multiple goals in sequence

- **Configuration**:
  - Seeds: 30
  - **Difficulty Levels**: Easy, Medium, Hard (runs all 3)
  - Episodes per seed: 1

- **Time**: ~1 hour

- **Use When**:
  - Evaluating navigation-focused agents
  - Analyzing capability across difficulty levels
  - Debugging navigation-specific issues
  - Comparing to navigation baselines

---

### Memory Capability (agentick-memory-v1)

**Focus**: State retention, recall, information integration

- **Tasks** (5):
  - KeyDoorPuzzle-v0 - Remember door key location
  - SequenceMemory-v0 - Recall item sequences
  - DelayedGratification-v0 - Wait for better rewards
  - BacktrackPuzzle-v0 - Navigate maze then backtrack
  - BreadcrumbTrail-v0 - Leave and follow trail markers

- **Configuration**:
  - Seeds: 30
  - Difficulty Levels: Easy, Medium, Hard (all 3)
  - Episodes per seed: 1

- **Time**: ~1 hour

- **Use When**:
  - Testing long-horizon memory
  - Evaluating state recall abilities
  - Comparing memory-augmented models
  - Analyzing memory capacity

---

### Reasoning Capability (agentick-reasoning-v1)

**Focus**: Logic, planning, constraint satisfaction

- **Tasks** (5):
  - SokobanPush-v0 - Block pushing puzzle
  - CausalChain-v0 - Trace cause-effect chains
  - SymbolMatching-v0 - Match symbolic patterns
  - RuleInduction-v0 - Learn and apply rules
  - SwitchCircuit-v0 - Control circuits via switches

- **Configuration**:
  - Seeds: 30
  - Difficulty Levels: Easy, Medium, Hard (all 3)
  - Episodes per seed: 1

- **Time**: ~1 hour

- **Use When**:
  - Evaluating logical reasoning
  - Testing constraint handling
  - Comparing planning capabilities
  - Analyzing rule learning

---

### Skill Discovery (agentick-skill-v1)

**Focus**: Learning and composing new behaviors

- **Tasks** (5):
  - ToolUse-v0 - Use available tools effectively
  - RecipeAssembly-v0 - Combine items into recipes
  - MultiRoomEscape-v0 - Escape multi-room environments
  - EmergentStrategy-v0 - Discover strategies emergently
  - ResourceManagement-v0 - Manage limited resources

- **Configuration**:
  - Seeds: 30
  - Difficulty Levels: Easy, Medium, Hard (all 3)
  - Episodes per seed: 1

- **Time**: ~1 hour

- **Use When**:
  - Testing skill learning and composition
  - Evaluating adaptability
  - Comparing hierarchical learning
  - Analyzing emergent behaviors

---

### Control Capability (agentick-control-v1)

**Focus**: Precise timing, continuous control, dynamics

- **Tasks** (4):
  - PreciseNavigation-v0 - Navigate with continuous controls
  - TimingChallenge-v0 - Time-sensitive actions
  - ChaseEvade-v0 - Chase or evade agents
  - Herding-v0 - Control group of entities

- **Configuration**:
  - Seeds: 30
  - Difficulty Levels: Easy, Medium, Hard (all 3)
  - Episodes per seed: 1

- **Time**: ~45 minutes

- **Use When**:
  - Evaluating fine motor control
  - Testing real-time responsiveness
  - Analyzing timing sensitivity
  - Comparing continuous control policies

---

### Combinatorial Reasoning (agentick-combinatorial-v1)

**Focus**: Combinatorial search, state spaces

- **Tasks** (4):
  - LightsOut-v0 - Lights out puzzle
  - GraphColoring-v0 - Graph coloring problem
  - TileSorting-v0 - Tile sorting puzzle
  - PackingPuzzle-v0 - 2D packing problem

- **Configuration**:
  - Seeds: 30
  - Difficulty Levels: Easy, Medium, Hard (all 3)
  - Episodes per seed: 1

- **Time**: ~45 minutes

- **Use When**:
  - Testing combinatorial search
  - Evaluating state space exploration
  - Comparing constraint solvers
  - Analyzing optimization capabilities

---

### World Model (agentick-worldmodel-v1)

**Focus**: Learning environment dynamics

- **Tasks** (3):
  - EnvironmentShift-v0 - Adapt to environment changes
  - PhysicsDiscovery-v0 - Discover physics laws
  - RuleDiscoveryNavigation-v0 - Discover navigation rules

- **Configuration**:
  - Seeds: 30
  - Difficulty: Medium
  - Episodes per seed: 1

- **Time**: ~30 minutes

- **Use When**:
  - Evaluating world modeling
  - Testing adaptation to dynamics
  - Analyzing physics understanding
  - Comparing model-based approaches

---

### Adversarial Robustness (agentick-adversarial-v1)

**Focus**: Robustness to challenging conditions

- **Tasks** (3):
  - DeceptiveReward-v0 - Misleading reward signals
  - DistributionShift-v0 - Out-of-distribution observations
  - NoisyObservation-v0 - Noisy sensor readings

- **Configuration**:
  - Seeds: 30
  - Difficulty: Medium
  - Episodes per seed: 1

- **Time**: ~30 minutes

- **Use When**:
  - Testing robustness to adversarial inputs
  - Evaluating noise tolerance
  - Comparing adversarial training
  - Production readiness assessment

---

### Meta-Learning (agentick-meta-v1)

**Focus**: Rapid adaptation to new tasks

- **Tasks** (2):
  - FewShotAdaptation-v0 - Adapt with few demonstrations
  - TaskInterference-v0 - Handle conflicting task signals

- **Configuration**:
  - Seeds: 30
  - Difficulty: Medium
  - Episodes per seed: 1

- **Time**: ~20 minutes

- **Use When**:
  - Evaluating few-shot learning
  - Testing task adaptation
  - Comparing meta-learning methods
  - Analyzing transfer learning

---

### Multi-Agent (agentick-multiagent-v1)

**Focus**: Coordination and competition

- **Tasks** (2):
  - CooperativeTransport-v0 - Coordinate to transport objects
  - CompetitiveTag-v0 - Competitive tag game

- **Configuration**:
  - Seeds: 30
  - Difficulty: Medium
  - Episodes per seed: 1

- **Time**: ~20 minutes

- **Use When**:
  - Evaluating multi-agent coordination
  - Testing competitive strategies
  - Comparing communication protocols
  - Analyzing emergent behaviors

---

## Analysis Suites

### Difficulty Scaling (agentick-difficulty-v1)

**Focus**: Performance across difficulty spectrum

Runs 10 representative tasks across all 4 difficulty levels (easy, medium, hard, extreme).

- **Tasks** (10):
  - GoToGoal-v0
  - MazeNavigation-v0
  - KeyDoorPuzzle-v0
  - SokobanPush-v0
  - ChaseEvade-v0
  - LightsOut-v0
  - SequenceMemory-v0
  - BacktrackPuzzle-v0
  - ToolUse-v0
  - RecipeAssembly-v0

- **Configuration**:
  - Seeds: 20
  - **All 4 Difficulty Levels**: Easy, Medium, Hard, Extreme
  - Episodes per seed: 1

- **Time**: ~1.5 hours

- **Generates**:
  - Difficulty scaling curves
  - Extrapolation analysis
  - Failure mode identification
  - Capability limits estimation

- **Use When**:
  - Analyzing how agent performance degrades
  - Estimating capability limits
  - Comparing to human difficulty perception
  - Publishing difficulty analysis

**Example Output**:
```
GoToGoal-v0 Performance by Difficulty:
  Easy:     0.95 ± 0.02
  Medium:   0.75 ± 0.04
  Hard:     0.45 ± 0.08
  Extreme:  0.10 ± 0.05

Degradation: Linear decline with 0.28 per difficulty step
```

---

### Multimodal Observations (agentick-multimodal-v1)

**Focus**: Performance across observation modes

Evaluates same agent with different observation formats (text, images, state dictionaries).

- **Tasks** (10):
  - GoToGoal-v0
  - MazeNavigation-v0
  - KeyDoorPuzzle-v0
  - SokobanPush-v0
  - PreciseNavigation-v0
  - ToolUse-v0
  - RecipeAssembly-v0
  - ChaseEvade-v0
  - LightsOut-v0
  - EnvironmentShift-v0

- **Configuration**:
  - Seeds: 20
  - Difficulty: Medium
  - **Multiple Observation Modes**: ascii, language, language_structured, rgb_array, state_dict
  - Episodes per seed: 1

- **Time**: ~1.5 hours per observation mode

- **Generates**:
  - Observation mode comparison
  - Modality effectiveness analysis
  - Multimodal robustness assessment

- **Use When**:
  - Testing multimodal agents
  - Comparing observation formats
  - Analyzing which modalities work best
  - Robustness across representations

**Example Output**:
```
GoToGoal-v0 Performance by Observation Mode:
  language:            0.85 ± 0.03
  language_structured: 0.83 ± 0.04
  ascii:               0.80 ± 0.04
  rgb_array:           0.75 ± 0.05
  state_dict:          0.88 ± 0.03
```

---

## Suite Specifications

### Deterministic Seeding

All official suites use deterministic, reproducible seeds:

```python
def generate_deterministic_seeds(suite_name: str, n_seeds: int) -> tuple[int, ...]:
    """
    Generate seeds deterministically from suite name.

    Anyone can regenerate the exact same seeds from the suite name.
    This prevents implicit overfitting to specific seeds.
    """
    hash_int = int(hashlib.sha256(suite_name.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(hash_int)
    seeds = tuple(int(x) for x in rng.integers(0, 2**31, size=n_seeds))
    return seeds
```

**Seed Lists**:

- **agentick-quick-v1**: `[2425698916, 1892920754, 1426916905, 3874520815, 1523149688, 583816398, 2768019595, 3259779823, 1919318886, 2282265984]`

- **agentick-core-v1**: (50 seeds) First 50 from hash of "agentick-core-v1"

- **agentick-full-v1**: (50 seeds) First 50 from hash of "agentick-full-v1"

### Immutability Guarantee

Official suites v1.0 are immutable:

- Task definitions never change
- Seeds are locked (no reseeding)
- Scoring methodology is fixed
- Once published, cannot be modified

To update: create a v2 suite with explicit version bump.

### Hash Verification

Each suite has an integrity hash:

```python
suite = get_suite("agentick-full-v1")
suite_hash = suite.compute_hash()
# "a7f2d8c1e5b9f3a2c1d4e6f8a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a"
```

This hash is reproducible and published, allowing verification that:
- You're using the exact suite definition
- Results are comparable to others
- The suite hasn't been tampered with

## Choosing the Right Suite

### Decision Tree

**Are you developing?**
→ Use **Quick** suite for fast iteration (~5 min)

**Do you need standard baseline?**
→ Use **Core** suite for official comparison (~2 hours)

**Are you publishing?**
→ Use **Full** suite for complete evaluation (~4-8 hours)

**Do you want capability analysis?**
→ Use **Capability suites** (navigation, memory, reasoning, etc.)

**Do you need difficulty analysis?**
→ Use **Difficulty Scaling** suite

**Are you testing multimodal?**
→ Use **Multimodal** suite

### Time-to-Value by Use Case

| Use Case | Suite | Time | Value |
|----------|-------|------|-------|
| **Dev iteration** | Quick | 5 min | Fast feedback |
| **Agent comparison** | Core | 2 hrs | Fair baseline |
| **SOTA submission** | Full | 4-8 hrs | Complete picture |
| **Debug navigation** | Navigation | 1 hr | Targeted analysis |
| **Production checklist** | Full + Adversarial | 5 hrs | Robustness check |
| **Difficulty research** | Difficulty Scaling | 1.5 hrs | Scaling analysis |
| **Multimodal agent** | Multimodal | 1.5 hrs | Modality comparison |

## Suite Statistics

### Aggregated across all suites

- **Total Official Tasks**: 38 unique tasks
- **Total Suite Evaluations**: 15 suites
- **Total Seeds Generated**: 1000+ deterministic seeds
- **Tasks per Suite**: 2-38
- **Difficulty Levels**: Easy, Medium, Hard, Extreme (subset per task)
- **Observation Modes**: ASCII, Language, Structured Language, RGB, State Dict

### Task Distribution

```
Navigation:    11 appearances (Core suite + Capability + Difficulty + Multimodal)
Memory:        11 appearances
Reasoning:     11 appearances
Skill:         10 appearances
Control:       9 appearances
Combinatorial: 9 appearances
WorldModel:    3 appearances
Adversarial:   3 appearances
Meta:          2 appearances
MultiAgent:    2 appearances
```

## Related Documentation

- [Scoring Methodology](scoring.md) - How tasks are scored
- [Adapters Guide](adapters.md) - How to configure agents
- [Submission Guide](submitting.md) - How to submit

---

Last updated: 2026-02-12 | Version: 1.0
