# Rewards

Agentick provides flexible reward modes and wrappers.

## Dense vs Sparse Rewards

```python
# Dense: step-wise guidance
env = agentick.make("GoToGoal-v0", reward_mode="dense")

# Sparse: reward only on success
env = agentick.make("GoToGoal-v0", reward_mode="sparse")
```

## Potential-Based Reward Shaping

```python
from agentick.rewards import PotentialBasedReward

def distance_to_goal(state):
    agent_pos = state["agent"].position
    goal_pos = state["goal"].position
    return -abs(agent_pos[0] - goal_pos[0]) - abs(agent_pos[1] - goal_pos[1])

shaper = PotentialBasedReward(potential_fn=distance_to_goal, gamma=0.99)

env = agentick.make("GoToGoal-v0", reward_mode="sparse")
obs, info = env.reset()
shaper.reset(obs)

for step in range(100):
    action = agent.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    shaped_reward = shaper.shape_reward(reward, obs, terminated)
    agent.learn(obs, shaped_reward, terminated)
```

### Built-in Potential Functions

```python
from agentick.rewards.potential import (
    manhattan_distance_potential,
    euclidean_distance_potential,
    subgoal_potential,
    inventory_potential,
    composite_potential,
)

# Manhattan distance
potential = manhattan_distance_potential(goal=(5, 5))

# Subgoal completion
potential = subgoal_potential(subgoals=[(2, 2), (5, 5)])

# Composite
dist_pot = euclidean_distance_potential((5, 5))
inv_pot = inventory_potential(["key"])
combined = composite_potential(dist_pot, inv_pot, weights=[0.7, 0.3])
```

## Intrinsic Rewards

### Exploration Bonus

```python
from agentick.rewards import ExplorationBonus

exploration = ExplorationBonus(bonus_scale=1.0, decay_type="inverse")
obs, info = env.reset()
exploration.reset()

for step in range(200):
    action = agent.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    bonus = exploration.compute_bonus(obs)
    total_reward = reward + bonus
    agent.learn(obs, total_reward, terminated)
```

**Decay types**: `"inverse"`, `"sqrt"`, `"novelty"`

### Curiosity-Driven Reward

```python
from agentick.rewards import CuriosityReward

curiosity = CuriosityReward(reward_scale=0.1, prediction_window=10)
# Rewards prediction errors
```

### Information Gain & Novelty

```python
from agentick.rewards.intrinsic import InformationGainReward, NoveltyReward

info_gain = InformationGainReward(reward_scale=0.1)
novelty = NoveltyReward(reward_scale=0.1, memory_size=100)
```

## Composite Rewards

```python
from agentick.rewards import CompositeReward

def task_reward(state, action, next_state, done, info):
    return 1.0 if info.get("success") else 0.0

def progress_reward(state, action, next_state, done, info):
    return 0.1 * distance_improvement

composite = CompositeReward(
    reward_functions=[task_reward, progress_reward],
    weights=[1.0, 0.5],
    names=["task", "progress"]
)

shaped_reward = composite.compute_reward(obs, action, obs_next, terminated, info)

# Adjust weights
composite.set_weights([1.0, 0.2])
composite.set_weight("progress", 0.0)

# Add/remove components
composite.add_reward_function(fn=exploration_fn, weight=0.3, name="exploration")
composite.remove_reward_function("exploration")
```

## Reward Wrappers

```python
from agentick.wrappers import DenseRewardWrapper, SparseRewardWrapper, RewardScaleWrapper

# Switch to dense
env = DenseRewardWrapper(env)

# Switch to sparse
env = SparseRewardWrapper(env)

# Scale rewards
env = RewardScaleWrapper(env, scale=0.1, shift=-0.01)
```

### CurriculumWrapper

```python
from agentick.wrappers import CurriculumWrapper

env = agentick.make("GoToGoal-v0", difficulty="easy")
env = CurriculumWrapper(
    env,
    success_threshold=0.8,  # Advance when 80% success
    window_size=10
)
# Auto-advances difficulty during training
```

See [Scoring](scoring.md) for how rewards map to scores.
