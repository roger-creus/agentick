# Custom Agents

Build custom agents for Agentick tasks using search, planning, heuristics, and hybrid approaches. This guide covers the agent protocol, complete working examples, and best practices.

## Agent Protocol

All custom agents must implement this interface:

```python
from typing import Any, Dict, Optional
import numpy as np


class Agent:
    """Standard agent interface for Agentick."""

    def reset(self) -> None:
        """Reset agent state for new episode.

        Called at the start of each episode to reset internal state,
        clear memory, reset timers, etc.
        """
        pass

    def act(self, observation: Any, info: Dict) -> int:
        """Select action based on observation.

        Args:
            observation: Current environment observation
            info: Info dict from environment (valid_actions, grid, positions, etc.)

        Returns:
            Action index (0-based)
        """
        raise NotImplementedError

    def update(
        self,
        observation: Any,
        action: int,
        reward: float,
        next_observation: Any,
        done: bool,
        info: Dict,
    ) -> None:
        """Optional: Update agent after step.

        Called after each environment step. Use for learning,
        planning, state updates, etc.
        """
        pass
```

## Example 1: Greedy/Reactive Agent

Simplest agent: move toward goal greedily.

```python
"""Greedy agent that moves toward goal."""

import numpy as np


class GreedyAgent:
    """Always move toward the goal."""

    def __init__(self):
        self.goal_pos = None
        self.agent_pos = None

    def reset(self):
        """Reset for new episode."""
        self.goal_pos = None
        self.agent_pos = None

    def act(self, observation, info):
        """Select action moving toward goal."""
        # Extract positions from info
        if self.agent_pos is None:
            self.agent_pos = np.array(info.get("agent_pos", [0, 0]))
        if self.goal_pos is None:
            self.goal_pos = np.array(info.get("goal_pos", [0, 0]))

        # Update agent position
        self.agent_pos = np.array(info.get("agent_pos", self.agent_pos))

        # Compute direction
        diff = self.goal_pos - self.agent_pos

        # Choose action: prioritize vertical movement
        if abs(diff[0]) > abs(diff[1]):
            # Move horizontally
            return 3 if diff[1] > 0 else 2  # RIGHT or LEFT
        else:
            # Move vertically
            return 1 if diff[0] > 0 else 0  # DOWN or UP

    def update(self, *args):
        """Greedy agent doesn't learn."""
        pass


# Test
def test_greedy_agent():
    import agentick

    env = agentick.make("GoToGoal-v0", difficulty="easy")
    agent = GreedyAgent()

    obs, info = env.reset()
    agent.reset()

    total_reward = 0
    for step in range(100):
        action = agent.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    print(f"Greedy agent reward: {total_reward}, Success: {info['success']}")
    env.close()
```

## Example 2: Search-Based Agent (BFS)

Find optimal path using breadth-first search.

```python
"""BFS-based agent for optimal pathfinding."""

from collections import deque
import numpy as np


class BFSAgent:
    """Use BFS to find optimal path to goal."""

    def __init__(self):
        self.plan = []
        self.grid = None
        self.agent_pos = None
        self.goal_pos = None

    def reset(self):
        """Reset for new episode."""
        self.plan = []
        self.grid = None
        self.agent_pos = None
        self.goal_pos = None

    def act(self, observation, info):
        """Execute plan, compute new plan if needed."""
        # Extract state
        self.agent_pos = tuple(info.get("agent_pos", [0, 0]))
        self.goal_pos = tuple(info.get("goal_pos", [0, 0]))
        self.grid = info.get("grid", None)

        # Compute new plan if empty
        if not self.plan:
            self.plan = self._compute_path()

        # Execute plan
        if self.plan:
            action = self.plan.pop(0)
            return action
        else:
            return 0  # NOOP if no plan

    def _compute_path(self):
        """BFS to find path to goal."""
        if self.grid is None or self.agent_pos is None or self.goal_pos is None:
            return []

        queue = deque([(self.agent_pos, [])])
        visited = {self.agent_pos}

        while queue:
            pos, path = queue.popleft()

            # Check if reached goal
            if pos == self.goal_pos:
                return path

            # Try all 4 directions: UP, DOWN, LEFT, RIGHT
            actions = [
                (0, -1, 0),  # UP
                (0, 1, 1),   # DOWN
                (-1, 0, 2),  # LEFT
                (1, 0, 3),   # RIGHT
            ]

            for dx, dy, action in actions:
                next_pos = (pos[0] + dx, pos[1] + dy)

                if self._is_walkable(next_pos) and next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [action]))

        return []  # No path found

    def _is_walkable(self, pos):
        """Check if position is walkable."""
        if self.grid is None:
            return True

        h, w = self.grid.shape
        if not (0 <= pos[0] < w and 0 <= pos[1] < h):
            return False

        # 0=empty, 1=wall
        return self.grid[pos[1], pos[0]] == 0

    def update(self, *args):
        """BFS agent doesn't learn."""
        pass


# Test
def test_bfs_agent():
    import agentick

    env = agentick.make("GoToGoal-v0", difficulty="easy")
    agent = BFSAgent()

    obs, info = env.reset()
    agent.reset()

    total_reward = 0
    steps = 0

    for step in range(100):
        action = agent.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    print(f"BFS agent: reward={total_reward}, steps={steps}, "
          f"success={info['success']}")
    env.close()
```

## Example 3: MCTS Agent

Monte Carlo Tree Search for complex planning.

```python
"""MCTS-based agent for planning."""

import numpy as np
from collections import defaultdict


class MCTSAgent:
    """Monte Carlo Tree Search agent."""

    def __init__(self, n_simulations=100, exploration_constant=1.41):
        self.n_simulations = n_simulations
        self.exploration_constant = exploration_constant

        # MCTS tree
        self.tree = {}
        self.visit_counts = defaultdict(int)
        self.value_sums = defaultdict(float)

    def reset(self):
        """Reset for new episode."""
        self.tree = {}
        self.visit_counts.clear()
        self.value_sums.clear()

    def act(self, observation, info):
        """Select action using MCTS."""
        state = self._state_key(info)

        # Run simulations
        for _ in range(self.n_simulations):
            self._simulate(state, info)

        # Select best action
        best_action = None
        best_value = -np.inf

        for action in range(6):  # Assuming 6 actions
            child_state = self._get_child_state(state, action)
            value = self._get_value(child_state)

            if value > best_value:
                best_value = value
                best_action = action

        return best_action or 0

    def _simulate(self, state, info, depth=0, max_depth=20):
        """Run single simulation."""
        if depth >= max_depth:
            return 0.0

        # Check if terminal
        if info.get("done", False):
            return 1.0 if info.get("success", False) else 0.0

        # Check if state in tree
        if state not in self.tree:
            # Leaf node - expand
            self.tree[state] = list(range(6))  # All actions
            return self._rollout(info, depth)

        # Select action using UCB
        action = self._select_action_ucb(state)

        # Simulate action (dummy transition)
        reward = self._estimate_reward(action, info)
        value = reward + 0.99 * self._simulate(state, info, depth + 1)

        # Backup
        child_state = self._get_child_state(state, action)
        self.visit_counts[child_state] += 1
        self.value_sums[child_state] += value

        return value

    def _select_action_ucb(self, state):
        """Select action using UCB1."""
        best_action = None
        best_ucb = -np.inf

        for action in self.tree[state]:
            child_state = self._get_child_state(state, action)
            visits = self.visit_counts[child_state]
            value = self.value_sums[child_state] / max(visits, 1)

            # UCB formula
            ucb = value + self.exploration_constant * np.sqrt(
                np.log(sum(self.visit_counts[self._get_child_state(state, a)]
                          for a in self.tree[state]))
            ) / max(visits, 1)

            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action

        return best_action or 0

    def _rollout(self, info, depth):
        """Random rollout from current state."""
        reward = 0.0
        for _ in range(10):
            action = np.random.randint(0, 6)
            reward += self._estimate_reward(action, info)
        return min(reward / 10, 1.0)

    def _estimate_reward(self, action, info):
        """Estimate immediate reward (heuristic)."""
        # Reward for moving toward goal
        agent_pos = np.array(info.get("agent_pos", [0, 0]))
        goal_pos = np.array(info.get("goal_pos", [0, 0]))
        current_dist = np.linalg.norm(goal_pos - agent_pos)

        # Estimate next position
        next_pos = agent_pos.copy()
        if action == 0:  # UP
            next_pos[1] -= 1
        elif action == 1:  # DOWN
            next_pos[1] += 1
        elif action == 2:  # LEFT
            next_pos[0] -= 1
        elif action == 3:  # RIGHT
            next_pos[0] += 1

        next_dist = np.linalg.norm(goal_pos - next_pos)

        # Reward if moving closer
        return 0.1 if next_dist < current_dist else -0.05

    def _state_key(self, info):
        """Create hashable state key."""
        agent_pos = tuple(info.get("agent_pos", [0, 0]))
        goal_pos = tuple(info.get("goal_pos", [0, 0]))
        return (agent_pos, goal_pos)

    def _get_child_state(self, state, action):
        """Get child state (dummy)."""
        return (state, action)

    def _get_value(self, state):
        """Get average value of state."""
        if self.visit_counts[state] == 0:
            return 0.0
        return self.value_sums[state] / self.visit_counts[state]
```

## Example 4: Learning Agent (Q-Learning)

Agent that learns from experience.

```python
"""Q-Learning agent."""

import numpy as np
from collections import defaultdict


class QLearningAgent:
    """Tabular Q-Learning agent."""

    def __init__(self, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        # Q-table: (state, action) -> value
        self.q_table = defaultdict(lambda: np.zeros(6))  # 6 actions

    def reset(self):
        """Reset for new episode."""
        pass

    def act(self, observation, info):
        """Select action (epsilon-greedy)."""
        state = self._state_key(info)

        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 6)

        # Greedy
        return np.argmax(self.q_table[state])

    def update(self, observation, action, reward, next_observation, done, info):
        """Update Q-table."""
        state = self._state_key(info)

        # Get next Q-value
        next_state_key = tuple(next_observation) if isinstance(next_observation, (list, tuple)) else state
        next_q_max = np.max(self.q_table[next_state_key]) if not done else 0.0

        # Q-learning update
        current_q = self.q_table[state][action]
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_q_max - current_q
        )
        self.q_table[state][action] = new_q

    def _state_key(self, info):
        """Create state representation."""
        agent_pos = tuple(info.get("agent_pos", [0, 0]))
        goal_pos = tuple(info.get("goal_pos", [0, 0]))
        return (agent_pos, goal_pos)

    def train_episode(self, env):
        """Train for one episode."""
        obs, info = env.reset()
        self.reset()

        done = False
        episode_reward = 0

        while not done:
            action = self.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            self.update(obs, action, reward, obs, done, info)

        return episode_reward


# Training
def train_qlearning_agent():
    import agentick

    env = agentick.make("GoToGoal-v0", difficulty="easy")
    agent = QLearningAgent()

    for episode in range(100):
        reward = agent.train_episode(env)
        if episode % 20 == 0:
            print(f"Episode {episode}: reward={reward}")

    # Test trained agent
    print("\nTesting trained agent:")
    for _ in range(5):
        obs, info = env.reset()
        agent.reset()
        agent.epsilon = 0  # Greedy only

        total_reward = 0
        for step in range(100):
            action = agent.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        print(f"  Test: reward={total_reward}, success={info['success']}")

    env.close()
```

## Example 5: Hybrid Agent

Combines multiple strategies.

```python
"""Hybrid agent using multiple strategies."""

import numpy as np


class HybridAgent:
    """Combine search, learning, and heuristics."""

    def __init__(self):
        self.bfs_agent = BFSAgent()
        self.qlearning_agent = QLearningAgent()
        self.greedy_agent = GreedyAgent()

        self.episode_step = 0
        self.strategy = "bfs"  # Current strategy

    def reset(self):
        """Reset for new episode."""
        self.bfs_agent.reset()
        self.qlearning_agent.reset()
        self.greedy_agent.reset()
        self.episode_step = 0

    def act(self, observation, info):
        """Select action using best strategy."""
        self.episode_step += 1

        # Strategy selection based on state and step count
        if self.episode_step < 10 and info.get("grid") is not None:
            # Use BFS early (guaranteed optimal)
            self.strategy = "bfs"
            return self.bfs_agent.act(observation, info)
        elif self.episode_step % 3 == 0:
            # Occasionally use learned policy
            self.strategy = "qlearning"
            self.qlearning_agent.epsilon = 0
            return self.qlearning_agent.act(observation, info)
        else:
            # Use greedy as fallback
            self.strategy = "greedy"
            return self.greedy_agent.act(observation, info)

    def update(self, observation, action, reward, next_obs, done, info):
        """Update learning components."""
        self.qlearning_agent.update(observation, action, reward, next_obs, done, info)


# Test
def test_hybrid_agent():
    import agentick

    env = agentick.make("GoToGoal-v0", difficulty="easy")
    agent = HybridAgent()

    obs, info = env.reset()
    agent.reset()

    total_reward = 0
    strategies_used = {"bfs": 0, "qlearning": 0, "greedy": 0}

    for step in range(100):
        action = agent.act(obs, info)
        strategies_used[agent.strategy] += 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        agent.update(obs, action, reward, obs, terminated or truncated, info)

        if terminated or truncated:
            break

    print(f"Hybrid agent: reward={total_reward}, success={info['success']}")
    print(f"Strategies used: {strategies_used}")
    env.close()
```

## Agent Interface Implementation

Complete integration with Agentick benchmark:

```python
"""Full agent implementation with Agentick integration."""

import agentick
from typing import Any, Dict


class MyCustomAgent:
    """Full-featured custom agent."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize agent with config."""
        self.config = config or {}
        self.episode_data = []
        self.reset()

    def reset(self) -> None:
        """Reset agent state."""
        self.episode_data = []
        self.step_count = 0
        self.total_reward = 0

    def act(self, observation: Any, info: Dict) -> int:
        """Select action."""
        # Implement your policy here
        valid_actions = info.get("valid_actions", [])
        return valid_actions[0] if valid_actions else 0

    def update(
        self,
        observation: Any,
        action: int,
        reward: float,
        next_observation: Any,
        done: bool,
        info: Dict,
    ) -> None:
        """Update after each step."""
        self.step_count += 1
        self.total_reward += reward

        self.episode_data.append({
            "step": self.step_count,
            "action": action,
            "reward": reward,
            "done": done,
        })


# Evaluation
def evaluate_custom_agent():
    """Evaluate custom agent on benchmark."""
    env = agentick.make("GoToGoal-v0", difficulty="easy")
    agent = MyCustomAgent()

    results = []

    for episode in range(10):
        obs, info = env.reset()
        agent.reset()

        done = False
        episode_reward = 0

        while not done:
            action = agent.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.update(obs, action, reward, obs, done, info)
            episode_reward += reward

        results.append({
            "episode": episode,
            "reward": episode_reward,
            "success": info.get("success", False),
            "steps": agent.step_count,
        })

    # Print results
    success_rate = sum(r["success"] for r in results) / len(results)
    mean_reward = sum(r["reward"] for r in results) / len(results)

    print(f"\nResults (10 episodes):")
    print(f"  Success rate: {success_rate:.1%}")
    print(f"  Mean reward: {mean_reward:.2f}")

    env.close()

    return results
```

## Testing Your Agent

```python
"""Comprehensive agent testing."""

import agentick
import numpy as np


def test_agent(agent_class, agent_config=None):
    """Test agent on standard evaluation."""
    print(f"\nTesting {agent_class.__name__}")
    print("=" * 50)

    # Test on different tasks
    tasks = ["GoToGoal-v0"]
    difficulties = ["easy", "medium", "hard"]

    for task in tasks:
        for difficulty in difficulties:
            env = agentick.make(task, difficulty=difficulty)
            agent = agent_class(**(agent_config or {}))

            rewards = []
            successes = []
            steps_list = []

            for episode in range(5):
                obs, info = env.reset()
                agent.reset()

                episode_reward = 0
                steps = 0
                done = False

                while not done and steps < 200:
                    action = agent.act(obs, info)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    steps += 1

                    agent.update(obs, action, reward, obs, done, info)

                rewards.append(episode_reward)
                successes.append(info.get("success", False))
                steps_list.append(steps)

            env.close()

            success_rate = np.mean(successes)
            mean_reward = np.mean(rewards)
            mean_steps = np.mean(steps_list)

            print(f"\n{task} ({difficulty}):")
            print(f"  Success: {success_rate:.1%}")
            print(f"  Reward: {mean_reward:.2f} ± {np.std(rewards):.2f}")
            print(f"  Steps: {mean_steps:.1f} ± {np.std(steps_list):.1f}")


# Run tests
if __name__ == "__main__":
    test_agent(GreedyAgent)
    test_agent(BFSAgent)
    test_agent(QLearningAgent, {"learning_rate": 0.1})
    test_agent(HybridAgent)
```

## Best Practices

1. **Implement reset()** - Clear all state between episodes
2. **Handle edge cases** - Missing info keys, invalid observations
3. **Make deterministic** - Set seeds for reproducibility
4. **Document assumptions** - About observation format, action space
5. **Test thoroughly** - Multiple tasks, seeds, difficulties
6. **Optimize performance** - Consider computational cost
7. **Add logging** - Track decisions for debugging
8. **Parameterize** - Use config for hyperparameters
9. **Error handling** - Gracefully handle failures
10. **Validate output** - Check action validity

## Common Pitfalls

```python
# Bad: Assumes specific observation format
def bad_act(self, obs, info):
    return obs[0, 0]  # Crashes if obs is dict

# Good: Handles multiple formats
def good_act(self, obs, info):
    if isinstance(obs, dict):
        return obs.get("position", [0, 0])
    else:
        return 0

# Bad: No reset
class BadAgent:
    def act(self, obs, info):
        self.state += 1  # NameError: state not defined
        return 0

# Good: Proper reset
class GoodAgent:
    def reset(self):
        self.state = 0

    def act(self, obs, info):
        self.state += 1
        return 0
```

## Advanced Topics

### State Abstraction

```python
def create_state_representation(info, abstract_level="medium"):
    """Create abstract state for generalization."""
    agent_pos = tuple(info.get("agent_pos", [0, 0]))
    goal_pos = tuple(info.get("goal_pos", [0, 0]))

    if abstract_level == "low":
        # Full resolution
        return (agent_pos, goal_pos)
    elif abstract_level == "medium":
        # Coarse grid
        agent_abstract = (agent_pos[0] // 2, agent_pos[1] // 2)
        goal_abstract = (goal_pos[0] // 2, goal_pos[1] // 2)
        return (agent_abstract, goal_abstract)
    else:  # high
        # Direction only
        direction = "up" if agent_pos[1] < goal_pos[1] else "down"
        return (direction,)
```

### Memory and Planning

```python
class MemoryAgent:
    """Agent with memory of visited states."""

    def __init__(self):
        self.visited_states = set()
        self.longest_path = []

    def act(self, observation, info):
        """Use memory to improve decisions."""
        state = tuple(info.get("agent_pos", [0, 0]))
        self.visited_states.add(state)

        # Avoid revisiting if possible
        if len(self.visited_states) > 10:
            # Switch strategy
            return self._explore()

        return self._greedy_toward_goal(info)
```
