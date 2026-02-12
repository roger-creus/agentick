# Rewards

Agentick provides a flexible reward system to shape and modify agent learning signals. This guide covers reward design, shaping techniques, intrinsic motivation, and composition strategies.

## Overview

Rewards are the primary learning signal in Agentick. You can:
- Use **dense** or **sparse** rewards built into tasks
- Apply **reward shaping** to add guidance without changing optimal policy
- Compose **multiple reward signals** with configurable weights
- Wrap environments with **reward wrappers** for scaling and curriculum learning

## Dense vs Sparse Rewards

### Dense Rewards

Dense rewards provide frequent feedback, guiding the agent toward the goal.

**When to use:**
- Learning environments with sparse exploration (large state spaces)
- Complex tasks where random exploration is unlikely to succeed
- When you need rapid agent convergence
- Curriculum learning with early training stages

**Trade-offs:**
- May lead to reward hacking (optimizing learned signal instead of true goal)
- Can create suboptimal policies by encouraging local rewards
- More computation (every step incurs reward overhead)

**Example:**
```python
import agentick

# Create environment with dense rewards
env = agentick.make(
    "GoToGoal-v0",
    difficulty="medium",
    reward_mode="dense"  # Frequent feedback
)

# Dense rewards give step-wise guidance toward goal
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
# reward might be -1.0 (moved away), +0.5 (got closer), etc.
```

**Best practices:**
- Verify rewards are actually guiding learning (log and inspect)
- Compare final policy performance with sparse version
- Combine with other objectives to avoid reward hacking

### Sparse Rewards

Sparse rewards only provide feedback at task completion, reflecting true objective.

**When to use:**
- When you have efficient exploration strategies (e.g., search algorithms)
- For learning generalizable policies (less reward hacking)
- Benchmark comparisons (ground truth signal)
- When dense reward signal is hard to define

**Trade-offs:**
- Requires more episodes to learn (harder exploration problem)
- May need intrinsic motivation or reward shaping to accelerate
- Only suitable for some agent types

**Example:**
```python
# Create environment with sparse rewards
env = agentick.make(
    "GoToGoal-v0",
    difficulty="medium",
    reward_mode="sparse"  # Only at task completion
)

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
# reward is 0.0 until goal reached, then 1.0 (or episode length penalty)
```

**Best practices:**
- Use intrinsic motivation to supplement exploration
- Consider curriculum learning (start easy, increase difficulty)
- Combine with dense rewards via composite rewards
- Use optimistic exploration strategies

### Switching Between Modes

Use wrappers to switch reward modes at runtime:

```python
from agentick.wrappers import DenseRewardWrapper, SparseRewardWrapper

env = agentick.make("GoToGoal-v0")

# Switch to dense
env = DenseRewardWrapper(env)

# Or switch back to sparse
env = SparseRewardWrapper(env)
```

## Potential-Based Reward Shaping

Potential-based reward shaping (Ng et al., 1999) adds shaped rewards without changing the optimal policy.

### Theory

Given a **potential function** Φ(s) mapping states to scalars, the shaped reward is:

$$F(s, s') = \gamma \Phi(s') - \Phi(s)$$

where γ is the discount factor, s is current state, s' is next state.

**Key property**: The optimal policy under (R + F) is identical to the optimal policy under R alone, because:
- Expected value of shaped reward over trajectory telescopes
- Only initial and final potentials matter

### When to use:

- Providing dense guidance in sparse reward environments
- Incorporating domain knowledge (distance to goal, progress metrics)
- Multi-objective problems (use multiple potential functions)
- Preventing reward hacking while maintaining optimal policy

### Implementation

```python
from agentick.rewards import PotentialBasedReward

# Define a potential function (higher = better state)
def distance_to_goal(state):
    """Potential = negative distance to goal."""
    agent_pos = state["agent"].position
    goal_pos = state["goal"].position
    distance = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    return -distance  # Negative distance: closer states have higher potential

# Create shaper
shaper = PotentialBasedReward(
    potential_fn=distance_to_goal,
    gamma=0.99
)

# Use during training
env = agentick.make("GoToGoal-v0", reward_mode="sparse")
obs, info = env.reset()
shaper.reset(obs)  # Initialize for new episode

for step in range(100):
    action = agent.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    # Add shaped reward
    shaped_reward = shaper.shape_reward(reward, obs, terminated)

    # Learn with shaped reward instead of sparse reward
    agent.learn(obs, shaped_reward, terminated)

    if terminated or truncated:
        break
```

### Built-in Potential Functions

**Manhattan Distance:**
```python
from agentick.rewards import manhattan_distance_potential

goal = (5, 5)
potential = manhattan_distance_potential(goal)
# Returns Φ(s) = -|x - 5| - |y - 5|
```

**Euclidean Distance:**
```python
from agentick.rewards import euclidean_distance_potential

goal = (5.0, 5.0)
potential = euclidean_distance_potential(goal)
# Returns Φ(s) = -sqrt((x-5)² + (y-5)²)
```

**Subgoal Completion:**
```python
from agentick.rewards import subgoal_potential

subgoals = [(2, 2), (5, 5), (8, 8)]  # Must visit in order
potential = subgoal_potential(subgoals)
# Returns Φ(s) = number of completed subgoals
```

**Inventory Collection:**
```python
from agentick.rewards import inventory_potential

target_items = ["key", "door", "treasure"]
potential = inventory_potential(target_items)
# Returns Φ(s) = number of target items in inventory
```

**Composite Potential:**
```python
from agentick.rewards import composite_potential

dist_potential = euclidean_distance_potential((5, 5))
inv_potential = inventory_potential(["key"])

combined = composite_potential(
    dist_potential,
    inv_potential,
    weights=[0.7, 0.3]  # Distance guides more than inventory
)
```

### Common Pitfalls

**Problem 1: Potential function too large (diverges)**
```python
# Bad: Potential values grow unbounded
def bad_potential(state):
    steps_taken = state.get("steps", 0)
    return steps_taken ** 2  # Explosive growth!

# Good: Bound potential values
def good_potential(state):
    steps_taken = state.get("steps", 0)
    return -min(steps_taken, 100)  # Capped at -100
```

**Problem 2: Potential not differentiable (inconsistent shaping)**
```python
# Bad: Discontinuous jumps
def bad_potential(state):
    if state["agent"].position == goal:
        return 1000  # Huge jump!
    return 0

# Good: Smooth potential
def good_potential(state):
    distance = manhattan_distance(state["agent"].position, goal)
    return -distance  # Smooth gradient
```

**Problem 3: Potential doesn't correlate with success**
```python
# Bad: Potential independent of goal
def bad_potential(state):
    return random.random()  # No signal!

# Good: Potential tracks progress
def good_potential(state):
    return -manhattan_distance(state["agent"].position, goal)
```

## Intrinsic Rewards

Intrinsic motivation encourages exploration by rewarding novelty or uncertainty.

### Exploration Bonus

Rewards visiting novel states (count-based exploration).

**When to use:**
- Discrete state spaces with clear state representations
- Early exploration phase (before learning policy)
- Environments where random exploration gets stuck

**Formula:**
$$r_{intrinsic} = \frac{\beta}{N(s)^\alpha}$$

where N(s) is visit count, α ∈ {0.5, 1.0}, β is scale.

**Implementation:**
```python
from agentick.rewards import ExplorationBonus

# Count-based exploration bonus
exploration = ExplorationBonus(
    bonus_scale=1.0,
    decay_type="inverse"  # bonus ∝ 1/count
)

env = agentick.make("MazeNavigation-v0")
obs, info = env.reset()
exploration.reset()

for step in range(200):
    action = agent.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    # Add exploration bonus
    bonus = exploration.compute_bonus(obs)
    total_reward = reward + bonus

    agent.learn(obs, total_reward, terminated)

    if terminated or truncated:
        exploration.reset()
        obs, info = env.reset()
```

**Decay types:**
- `"inverse"`: bonus = 1/count (classic exploration)
- `"sqrt"`: bonus = 1/sqrt(count) (slower decay)
- `"novelty"`: bonus = 1 only if count==1 (first visit)

### Curiosity-Driven Reward

Rewards prediction errors (agent is curious about surprising states).

**When to use:**
- Continuous/high-dimensional state spaces
- When you have a world model or forward model
- Exploration in complex environments

**Formula:**
$$r_{curiosity} = \frac{|s_{true} - s_{predicted}|}{E[|s_{true} - s_{predicted}|]}$$

Normalized by recent prediction error average.

**Implementation:**
```python
from agentick.rewards import CuriosityReward

curiosity = CuriosityReward(
    reward_scale=0.1,
    prediction_window=10  # Track last 10 predictions
)

obs, info = env.reset()
curiosity.reset()

for step in range(200):
    action = agent.act(obs)
    predicted_next = agent.predict_next_state(obs, action)
    obs, reward, terminated, truncated, info = env.step(action)

    # Curiosity bonus for surprising outcomes
    curiosity_bonus = curiosity.compute_reward(predicted_next, obs)
    total_reward = reward + curiosity_bonus

    agent.learn(obs, total_reward, terminated)

    if terminated or truncated:
        curiosity.reset()
        obs, info = env.reset()
```

### Information Gain Reward

Rewards actions that maximize information gain about environment dynamics.

**Formula:**
$$r_{info} = -\log p(\text{transition | state, action})$$

Higher reward for rare, informative transitions.

**Implementation:**
```python
from agentick.rewards import InformationGainReward

info_gain = InformationGainReward(reward_scale=0.1)

obs, info = env.reset()
info_gain.reset()

for step in range(200):
    action = agent.act(obs)
    obs_next, reward, terminated, truncated, info = env.step(action)

    # Information gain bonus
    gain_bonus = info_gain.compute_reward(obs, action, obs_next)
    total_reward = reward + gain_bonus

    agent.learn(obs, total_reward, terminated)

    if terminated or truncated:
        info_gain.reset()
        obs, info = env.reset()
```

### Novelty Reward

Rewards states dissimilar to previously visited states.

**When to use:**
- Exploration of high-dimensional spaces
- Discovering diverse behaviors
- Open-ended environments

**Implementation:**
```python
from agentick.rewards import NoveltyReward

novelty = NoveltyReward(
    reward_scale=0.1,
    memory_size=100,  # Remember last 100 states
    threshold=0.1  # Only reward if dissimilarity > 0.1
)

obs, info = env.reset()
novelty.reset()

for step in range(200):
    action = agent.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    # Novelty bonus
    novelty_bonus = novelty.compute_reward(obs)
    total_reward = reward + novelty_bonus

    agent.learn(obs, total_reward, terminated)

    if terminated or truncated:
        novelty.reset()
        obs, info = env.reset()
```

## Custom Reward Functions

### Writing Your Own

Custom rewards should be functions that compute signals from states/observations:

```python
def my_reward_function(state, action, next_state, done, info):
    """Custom reward design.

    Args:
        state: Current state dict
        action: Action taken
        next_state: Resulting state
        done: Whether episode ended
        info: Info dict with metadata

    Returns:
        float: Reward value
    """
    # Example: Reward for moving toward goal
    current_dist = manhattan_distance(state["agent"].position, state["goal"].position)
    next_dist = manhattan_distance(next_state["agent"].position, state["goal"].position)

    move_reward = (current_dist - next_dist) * 0.1  # Small reward for progress

    # Example: Large reward for task completion
    task_reward = 1.0 if done and info.get("success") else 0.0

    # Example: Penalty for excessive steps
    step_penalty = -0.01 if not done else 0.0

    return move_reward + task_reward + step_penalty
```

### Best Practices

**1. Scale rewards appropriately**
```python
# Bad: Unbounded rewards cause training instability
def bad_reward(state, action, next_state, done, info):
    distance = manhattan_distance(state["agent"].pos, state["goal"].pos)
    return -distance ** 3  # Huge values!

# Good: Normalize to [-1, 1] range
def good_reward(state, action, next_state, done, info):
    max_distance = 20  # Maximum possible distance
    distance = manhattan_distance(state["agent"].pos, state["goal"].pos)
    normalized = distance / max_distance
    return -normalized  # In [-1, 0]
```

**2. Avoid reward hacking**
```python
# Bad: Can be optimized without solving task
def bad_reward(state, action, next_state, done, info):
    return -state["step_count"]  # Agent can just stand still!

# Good: Tie reward to actual progress
def good_reward(state, action, next_state, done, info):
    if info.get("success"):
        return 1.0 / (state["step_count"] + 1)  # Reward efficiency
    return 0.0
```

**3. Make reward stationary**
```python
# Bad: Reward changes based on training time
training_phase = 0
def bad_reward(state, action, next_state, done, info):
    if training_phase < 1000:
        return info.get("success", 0)
    else:
        return 1.0 if info.get("success") else -1.0

# Good: Constant reward definition
def good_reward(state, action, next_state, done, info):
    success = info.get("success", False)
    return 1.0 if success and not done else (0.0 if done else 0.0)
```

**4. Document reward structure**
```python
def documented_reward(state, action, next_state, done, info):
    """Reward for MazeNavigation task.

    Components:
    - Progress: -0.01 per step (encourages efficiency)
    - Exploration: +0.05 per new cell visited
    - Success: +1.0 upon reaching goal

    Typical range: [-1.0, 1.05] per episode
    """
    reward = 0.0

    # Progress component
    reward -= 0.01

    # Exploration component (if visited new cell)
    if info.get("visited_new_cell", False):
        reward += 0.05

    # Success component
    if done and info.get("success", False):
        reward += 1.0

    return reward
```

## Composite Rewards

Combine multiple reward signals with configurable weights.

### Why Composite Rewards?

- Mix extrinsic (task) and intrinsic (exploration) rewards
- Combine multiple shaping signals
- Balance competing objectives
- Dynamic weight adjustment for curriculum learning

### Implementation

```python
from agentick.rewards import CompositeReward

# Define individual reward functions
def task_reward(state, action, next_state, done, info):
    return 1.0 if info.get("success") else 0.0

def progress_reward(state, action, next_state, done, info):
    current_dist = manhattan_distance(state["agent"].pos, state["goal"].pos)
    next_dist = manhattan_distance(next_state["agent"].pos, state["goal"].pos)
    return (current_dist - next_dist) * 0.1

def efficiency_reward(state, action, next_state, done, info):
    return -0.01  # Small penalty for each step

# Combine them
composite = CompositeReward(
    reward_functions=[task_reward, progress_reward, efficiency_reward],
    weights=[1.0, 0.5, 1.0],  # Weight each component
    names=["task", "progress", "efficiency"]
)

# Use in training loop
obs, info = env.reset()
for step in range(200):
    action = agent.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    # Get composite reward
    shaped_reward = composite.compute_reward(obs, action, obs, terminated, info)
    agent.learn(obs, shaped_reward, terminated)

    # Inspect components
    components = composite.get_component_rewards(obs, action, obs, terminated, info)
    print(f"Task: {components['task']:.3f}, Progress: {components['progress']:.3f}")

    if terminated or truncated:
        obs, info = env.reset()

# Get statistics
stats = composite.get_statistics()
print(f"Task reward mean: {stats['task']['mean']:.3f}")
```

### Dynamic Weight Adjustment

Adjust weights during training (curriculum learning):

```python
# Start with exploration focus
composite.set_weights([0.1, 1.0, 0.5])  # Emphasize progress

# Later, shift to task focus
if episode_count > 100:
    composite.set_weights([1.0, 0.2, 0.5])  # Emphasize task reward

# Or adjust individual components
composite.set_weight("progress", 0.0)  # Disable progress reward
```

### Adding/Removing Components

```python
from agentick.rewards import ExplorationBonus

# Add exploration during training
exploration = ExplorationBonus(bonus_scale=0.1)

composite.add_reward_function(
    fn=lambda obs, action, obs_next, done, info: exploration.compute_bonus(obs_next),
    weight=0.3,
    name="exploration"
)

# Remove later when exploration is sufficient
composite.remove_reward_function("exploration")
```

## Reward Wrappers

Environment wrappers for quick reward manipulation.

### DenseRewardWrapper

Switch environment to dense reward mode:

```python
from agentick.wrappers import DenseRewardWrapper

env = agentick.make("GoToGoal-v0", reward_mode="sparse")
env = DenseRewardWrapper(env)  # Override to dense mode

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
# reward is now dense (step-wise guidance)
```

### SparseRewardWrapper

Switch environment to sparse reward mode:

```python
from agentick.wrappers import SparseRewardWrapper

env = agentick.make("GoToGoal-v0", reward_mode="dense")
env = SparseRewardWrapper(env)  # Override to sparse mode

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
# reward is now sparse (only at task completion)
```

### RewardScaleWrapper

Scale and shift rewards:

```python
from agentick.wrappers import RewardScaleWrapper

env = agentick.make("GoToGoal-v0")
env = RewardScaleWrapper(env, scale=0.1, shift=-0.01)

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
# reward = original_reward * 0.1 - 0.01
```

**Use cases:**
- Normalize reward ranges across tasks
- Prevent reward explosion
- Invert rewards (scale=-1.0)

### CurriculumWrapper

Auto-advance task difficulty on success:

```python
from agentick.wrappers import CurriculumWrapper

env = agentick.make("GoToGoal-v0", difficulty="easy")
env = CurriculumWrapper(
    env,
    success_threshold=0.8,  # Advance when 80% success rate
    window_size=10  # Over last 10 episodes
)

for episode in range(1000):
    obs, info = env.reset()
    for step in range(100):
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        agent.learn(obs, reward, terminated)
        if terminated or truncated:
            break
    # Wrapper automatically advances difficulty

print(f"Final difficulty: {env.env.difficulty}")
```

## Debugging Reward Functions

### Reward Statistics

Track reward components over time:

```python
import numpy as np

reward_history = {
    "extrinsic": [],
    "shaping": [],
    "intrinsic": []
}

for episode in range(100):
    obs, info = env.reset()
    episode_rewards = {"extrinsic": 0, "shaping": 0, "intrinsic": 0}

    for step in range(200):
        # Compute each reward component
        ext_reward = compute_extrinsic(obs, action, obs_next, done, info)
        shape_reward = compute_shaped(obs, obs_next, done)
        intr_reward = compute_intrinsic(obs)

        episode_rewards["extrinsic"] += ext_reward
        episode_rewards["shaping"] += shape_reward
        episode_rewards["intrinsic"] += intr_reward

        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    for key in episode_rewards:
        reward_history[key].append(episode_rewards[key])

# Analyze
print(f"Extrinsic mean: {np.mean(reward_history['extrinsic']):.3f}")
print(f"Shaping mean: {np.mean(reward_history['shaping']):.3f}")
print(f"Intrinsic mean: {np.mean(reward_history['intrinsic']):.3f}")
```

### Reward Correlation Analysis

Verify reward signals correlate with task success:

```python
import numpy as np
from scipy.stats import pearsonr

# Collect trajectories
episode_rewards = []
episode_successes = []

for episode in range(100):
    obs, info = env.reset()
    episode_return = 0

    for step in range(200):
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward

        if terminated or truncated:
            break

    episode_rewards.append(episode_return)
    episode_successes.append(float(info.get("success", 0)))

# Correlation
corr, p_value = pearsonr(episode_rewards, episode_successes)
print(f"Reward-Success correlation: {corr:.3f} (p={p_value:.4f})")

if corr < 0.5:
    print("WARNING: Reward signal weakly correlated with success!")
```

### Visualizing Reward Signals

Plot rewards over episode progress:

```python
import matplotlib.pyplot as plt

rewards_over_steps = []
obs, info = env.reset()

for step in range(500):
    action = agent.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    rewards_over_steps.append(reward)

    if terminated or truncated:
        break

plt.figure(figsize=(10, 4))
plt.plot(rewards_over_steps)
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Reward Signal Over Episode")
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.show()
```

## Common Pitfalls and Solutions

### Pitfall 1: Reward Scale Explosion

**Problem**: Reward values become unbounded, training becomes unstable.

**Causes:**
- Shaped rewards grow with episode length
- Composite weights sum to large values
- Missing reward caps

**Solution:**
```python
# Always clip or normalize rewards
def safe_reward(base_reward):
    # Normalize to [-1, 1]
    return np.clip(base_reward / 100.0, -1, 1)

# Or use wrapper
from agentick.wrappers import RewardScaleWrapper
env = RewardScaleWrapper(env, scale=0.01, shift=0.0)
```

### Pitfall 2: Reward Hacking

**Problem**: Agent optimizes reward instead of actual objective.

**Example**: Agent learns to collect small rewards but never reaches goal.

**Solution:**
```python
# Tie rewards directly to task success
def aligned_reward(state, action, next_state, done, info):
    # Main reward for success
    if done and info.get("success"):
        return 1.0

    # Shaping only as tie-breaker
    if info.get("success"):
        # Prefer faster solutions
        return 1.0 - 0.001 * info.get("steps", 0)

    return 0.0  # No intermediate rewards
```

### Pitfall 3: Non-Stationary Rewards

**Problem**: Reward definition changes during training.

**Causes:**
- Using training episode count in reward function
- Dynamic curriculum parameters
- Changing hyperparameters mid-run

**Solution:**
```python
# Define reward once, reuse consistently
def static_reward(state, action, next_state, done, info):
    # Independent of training progress
    return 1.0 if info.get("success") else 0.0

# Curriculum via wrapper, not reward
env = CurriculumWrapper(env, success_threshold=0.8)
```

### Pitfall 4: Overshaped Rewards

**Problem**: Too much shaping makes agent dependent on guidance, poor generalization.

**Solution:**
```python
# Use shaping sparingly
composite = CompositeReward(
    [task_reward, light_shaping],
    weights=[1.0, 0.1],  # Shaping is only 10% of signal
)

# Or disable shaping periodically
if episode % 10 == 0:
    # Use unshapped reward for evaluation
    evaluation_reward = task_reward_only()
```

See [Scoring](scoring.md) for how rewards map to benchmark scores.
