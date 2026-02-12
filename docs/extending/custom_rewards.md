# Custom Reward Functions

Learn how to design and implement effective reward functions for your tasks.

## Reward Function Basics

### Sparse vs Dense Rewards

```python
from agentick.tasks.base import TaskSpec

class MyTask(TaskSpec):
    # Sparse reward: only give reward on success
    def compute_sparse_reward(self, old_state, action, new_state, info) -> float:
        """Sparse reward: 1.0 on success, 0.0 otherwise."""
        if self.check_success(new_state):
            return 1.0
        return 0.0

    # Dense reward: provide continuous feedback
    def compute_dense_reward(self, old_state, action, new_state, info) -> float:
        """Dense reward: encourage progress toward goal."""
        reward = 0.0

        # Step penalty (encourages efficiency)
        reward -= 0.01

        # Progress reward (guides learning)
        reward += self._compute_progress_reward(old_state, new_state)

        # Success bonus
        if self.check_success(new_state):
            reward += 1.0

        return reward
```

## Reward Shaping Patterns

### 1. Distance-Based Rewards

Reward agent for getting closer to the goal:

```python
import numpy as np

def distance_reward(self, old_state, new_state, goal_pos, scale=0.1):
    """Reward for moving closer to goal."""
    old_pos = old_state["agent"].position
    new_pos = new_state["agent"].position

    old_dist = np.linalg.norm(np.array(old_pos) - np.array(goal_pos))
    new_dist = np.linalg.norm(np.array(new_pos) - np.array(goal_pos))

    return (old_dist - new_dist) * scale
```

**Used for**: Navigation, reaching goals

### 2. Step Efficiency Penalty

Encourage agents to solve tasks quickly:

```python
def efficiency_penalty(self, scale=-0.01):
    """Penalty for each step taken."""
    return scale  # Negative reward per step

def compute_dense_reward(self, old_state, action, new_state, info):
    reward = 0.0
    reward += self.efficiency_penalty(scale=-0.01)

    if self.check_success(new_state):
        reward += 1.0

    return reward
```

**Used for**: Encouraging efficient solutions

### 3. Exploration Bonus

```python
from agentick.rewards.intrinsic import ExplorationBonus

class ExplorationTask(TaskSpec):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.exploration = ExplorationBonus(bonus_scale=0.1)

    def compute_dense_reward(self, old_state, action, new_state, info) -> float:
        reward = 0.0

        # Extrinsic reward
        reward -= 0.01  # Step cost

        # Intrinsic reward for novelty
        bonus = self.exploration.compute_bonus(new_state)
        reward += bonus

        return reward
```

**Used for**: Exploration-heavy tasks

### 4. Potential-Based Reward Shaping

Provably optimal policy-preserving shaping:

```python
from agentick.rewards.potential import PotentialBasedReward

class GoalTask(TaskSpec):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Define potential function (distance to goal)
        def potential(state):
            if "agent" in state and "goal" in state:
                agent_pos = np.array(state["agent"].position)
                goal_pos = np.array(state["goal"].position)
                return -np.linalg.norm(agent_pos - goal_pos)
            return 0.0

        self.reward_shaper = PotentialBasedReward(
            potential_fn=potential,
            gamma=0.99,
        )

    def generate(self, seed):
        grid, config = self._create_grid(seed)
        self.goal_position = config["goal_position"]
        return grid, config

    def compute_dense_reward(self, old_state, action, new_state, info) -> float:
        # Base sparse reward
        base_reward = 1.0 if self.check_success(new_state) else 0.0

        # Apply potential-based shaping
        shaped_reward = self.reward_shaper.shape_reward(
            base_reward,
            new_state,
            info.get("terminated", False),
        )

        return shaped_reward
```

**Used for**: Goal-reaching with guaranteed policy preservation

### 5. Composing Multiple Rewards

```python
from agentick.rewards.composite import CompositeReward

class MultiObjectiveTask(TaskSpec):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Combine multiple reward signals
        self.reward_components = {
            "distance": 0.5,      # Weight: 50%
            "efficiency": 0.3,    # Weight: 30%
            "exploration": 0.2,   # Weight: 20%
        }

    def compute_dense_reward(self, old_state, action, new_state, info) -> float:
        total_reward = 0.0

        # Distance reward
        distance_reward = self._compute_distance_reward(old_state, new_state)
        total_reward += distance_reward * self.reward_components["distance"]

        # Efficiency reward
        efficiency_reward = -0.01  # Step penalty
        total_reward += efficiency_reward * self.reward_components["efficiency"]

        # Exploration reward
        exploration_reward = self._compute_exploration_reward(new_state)
        total_reward += exploration_reward * self.reward_components["exploration"]

        # Success bonus
        if self.check_success(new_state):
            total_reward += 1.0

        return total_reward
```

**Used for**: Multi-objective optimization

## Intrinsic Rewards

### Curiosity-Driven Learning

```python
from agentick.rewards.intrinsic import CuriosityReward

class CuriousTask(TaskSpec):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.curiosity = CuriosityReward(reward_scale=0.1)

    def compute_dense_reward(self, old_state, action, new_state, info) -> float:
        # Base reward
        reward = -0.01

        # Curiosity bonus (reward for surprising states)
        predicted_next = self._predict_next_state(new_state)
        curiosity = self.curiosity.compute_reward(predicted_next, new_state)
        reward += curiosity

        return reward

    def _predict_next_state(self, state):
        """Simple prediction model."""
        # In practice, this would be a learned model
        return state
```

**Used for**: Exploration without explicit task reward

### Information Gain Reward

```python
from agentick.rewards.intrinsic import InformationGainReward

class DiscoveryTask(TaskSpec):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.info_gain = InformationGainReward(reward_scale=0.1)

    def compute_dense_reward(self, old_state, action, new_state, info) -> float:
        # Reward actions that maximize information gain
        reward = self.info_gain.compute_reward(old_state, action, new_state)
        return reward
```

**Used for**: Active learning, information seeking

### Novelty Reward

```python
from agentick.rewards.intrinsic import NoveltyReward

class ExplorationTask(TaskSpec):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.novelty = NoveltyReward(
            reward_scale=0.1,
            memory_size=100,
            threshold=0.1,
        )

    def compute_dense_reward(self, old_state, action, new_state, info) -> float:
        reward = -0.01  # Step cost

        # Bonus for visiting novel states
        novelty_bonus = self.novelty.compute_reward(new_state)
        reward += novelty_bonus

        return reward
```

**Used for**: Exploration tasks where novelty matters

## Practical Examples

### Example 1: Simple Navigation

```python
class NavigationTask(TaskSpec):
    def compute_dense_reward(self, old_state, action, new_state, info) -> float:
        """Reward: progress toward goal minus step cost."""
        reward = 0.0

        # Step cost (efficiency)
        reward -= 0.01

        # Distance reward
        if "agent" in old_state and "agent" in new_state:
            old_dist = self._manhattan_distance(
                old_state["agent"].position,
                self.goal_pos,
            )
            new_dist = self._manhattan_distance(
                new_state["agent"].position,
                self.goal_pos,
            )
            reward += (old_dist - new_dist) * 0.1

        # Success bonus
        if self.check_success(new_state):
            reward += 1.0

        return reward

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
```

### Example 2: Puzzle Solving

```python
class PuzzleTask(TaskSpec):
    def compute_dense_reward(self, old_state, action, new_state, info) -> float:
        """Reward: progress in puzzle minus step cost."""
        reward = 0.0

        # Step penalty
        reward -= 0.01

        # Progress reward: number of correctly placed pieces
        old_progress = self._count_correct_pieces(old_state)
        new_progress = self._count_correct_pieces(new_state)

        if new_progress > old_progress:
            reward += 0.5  # Reward for progress

        # Success bonus
        if self.check_success(new_state):
            reward += 1.0

        return reward

    def _count_correct_pieces(self, state):
        """Count how many puzzle pieces are correctly placed."""
        count = 0
        for pos, piece in state["pieces"].items():
            if piece.position == piece.target_position:
                count += 1
        return count
```

### Example 3: Multi-Goal Collection

```python
class CollectionTask(TaskSpec):
    def compute_dense_reward(self, old_state, action, new_state, info) -> float:
        """Reward: collection progress with efficiency penalty."""
        reward = 0.0

        # Step cost
        reward -= 0.01

        # Collection rewards
        old_collected = info.get("items_collected_before", 0)
        new_collected = info.get("items_collected_after", 0)

        if new_collected > old_collected:
            # Bonus for each item collected
            reward += (new_collected - old_collected) * 0.5

        # Distance to nearest uncollected item
        if "agent" in new_state:
            nearest_dist = self._distance_to_nearest_goal(new_state)
            if nearest_dist < 5:
                reward += 0.1  # Bonus when close to items

        # Success bonus (all items collected)
        if self.check_success(new_state):
            reward += 1.0

        return reward
```

## Testing Reward Functions

### Validate Reward Distribution

```python
def test_reward_distribution():
    """Verify rewards are in reasonable range."""
    task = MyTask(difficulty="easy")

    rewards = []
    for seed in range(100):
        obs, info = task.generate(seed)

        for _ in range(100):
            action = np.random.randint(0, 4)

            # Create dummy states
            old_state = {"agent": AgentState(position=(1, 1))}
            new_state = {"agent": AgentState(position=(2, 1))}

            reward = task.compute_dense_reward(old_state, action, new_state, info)
            rewards.append(reward)

    rewards = np.array(rewards)

    # Check statistics
    print(f"Mean reward: {rewards.mean():.4f}")
    print(f"Std reward: {rewards.std():.4f}")
    print(f"Min reward: {rewards.min():.4f}")
    print(f"Max reward: {rewards.max():.4f}")

    # Rewards should be reasonable
    assert rewards.mean() > -1.0, "Mean reward too negative"
    assert rewards.std() > 0.01, "Reward signal too weak"
```

### Test Alignment with Success

```python
def test_reward_alignment():
    """Verify successful episodes have higher cumulative reward."""
    task = MyTask(difficulty="easy")

    successful_returns = []
    failed_returns = []

    for seed in range(50):
        obs, info = task.generate(seed)

        cumulative_reward = 0.0
        success = False

        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            cumulative_reward += reward

            if terminated or truncated:
                success = info.get("success", False)
                break

        if success:
            successful_returns.append(cumulative_reward)
        else:
            failed_returns.append(cumulative_reward)

    # Successful episodes should have higher returns
    assert np.mean(successful_returns) > np.mean(failed_returns), \
        "Reward function not aligned with success!"

    print(f"Successful return: {np.mean(successful_returns):.3f}")
    print(f"Failed return: {np.mean(failed_returns):.3f}")
```

## Common Pitfalls

### Pitfall 1: Overly Sparse Rewards
```python
# BAD: No learning signal
def bad_reward(self, old_state, action, new_state, info):
    if self.check_success(new_state):
        return 1.0
    return 0.0

# GOOD: Provide guidance
def good_reward(self, old_state, action, new_state, info):
    reward = 0.0
    reward -= 0.01  # Step cost
    reward += self._progress_reward(old_state, new_state)
    if self.check_success(new_state):
        reward += 1.0
    return reward
```

### Pitfall 2: Reward Hacking
```python
# BAD: Agent games the reward
def bad_reward(self, old_state, action, new_state, info):
    return len(new_state["agent"].inventory)  # Collect anything!

# GOOD: Specific target rewards
def good_reward(self, old_state, action, new_state, info):
    # Only reward collecting target items
    target_items = [item for item in new_state["agent"].inventory
                    if item.type in self.target_types]
    return len(target_items) * 0.1
```

### Pitfall 3: Conflicting Objectives
```python
# BAD: Contradictory goals
def bad_reward(self, old_state, action, new_state, info):
    reward = 0.0
    reward += self._distance_to_goal(new_state)  # Get closer
    reward -= self._distance_to_goal(new_state)  # Stay away (contradiction!)
    return reward

# GOOD: Aligned objectives
def good_reward(self, old_state, action, new_state, info):
    reward = 0.0
    reward -= 0.01  # Efficient
    reward += self._distance_to_goal(new_state) * 0.1  # Guided
    if self.check_success(new_state):
        reward += 1.0
    return reward
```

## Best Practices

1. **Scale rewards appropriately**: Typically [-1, 1] range
2. **Use step penalty**: -0.01 per step is common
3. **Provide progress signal**: Help agent learn with dense rewards
4. **Test with random agent**: Random agent should get near 0 reward
5. **Balance difficulty levels**: Adjust reward scales for different difficulties
6. **Document reward design**: Explain why each component exists
7. **Verify learning**: Check that agents improve with your reward

## Reward Function Checklist

```python
class RewardChecklist:
    """Verify reward function quality."""

    @staticmethod
    def verify_task_rewards(task, n_episodes=100):
        """Run verification checks."""
        task_env = agentick.make(task)

        results = {
            "mean_return": [],
            "success_rate": 0.0,
            "reward_range": {"min": float("inf"), "max": float("-inf")},
        }

        successes = 0
        for seed in range(n_episodes):
            obs, info = task_env.reset(seed=seed)
            episode_return = 0.0

            for _ in range(task_env.spec.max_episode_steps):
                action = task_env.action_space.sample()
                obs, reward, terminated, truncated, info = task_env.step(action)

                episode_return += reward
                results["reward_range"]["min"] = min(results["reward_range"]["min"], reward)
                results["reward_range"]["max"] = max(results["reward_range"]["max"], reward)

                if terminated or truncated:
                    if info.get("success"):
                        successes += 1
                    break

            results["mean_return"].append(episode_return)

        results["success_rate"] = successes / n_episodes
        results["mean_return"] = np.mean(results["mean_return"])

        # Verification
        print(f"✓ Mean return: {results['mean_return']:.3f}")
        print(f"✓ Success rate: {results['success_rate']:.2%}")
        print(f"✓ Reward range: [{results['reward_range']['min']:.3f}, {results['reward_range']['max']:.3f}]")

        return results
```
