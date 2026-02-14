"""
Complete example: Creating a custom task as a Gymnasium environment.

Demonstrates:
- Building a custom gridworld environment
- Custom entities and mechanics
- Reward shaping
- Registration and testing

Usage:
    uv run python examples/advanced/custom_task.py
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CollectGemsEnv(gym.Env):
    """
    Custom task: Collect all gems in the shortest path.

    Goal: Agent must collect all gems before reaching the exit.

    Entities:
    - Agent (A) - player character
    - Gems (G) - must collect all
    - Exit (E) - reach after collecting gems
    - Walls (#) - obstacles
    - Lava (X) - instant failure if stepped on

    Reward structure:
    - +0.5 per gem collected
    - +1.0 for reaching exit with all gems
    - -1.0 for stepping on lava
    - -0.01 per step (encourages efficiency)
    - -0.5 for reaching exit without all gems

    Success condition:
    - Collect all gems AND reach exit
    """

    metadata = {"render_modes": ["ascii", "human"]}

    EMPTY = 0
    WALL = 1
    GEM = 2
    LAVA = 3
    EXIT = 4
    AGENT = 5

    def __init__(
        self,
        grid_size: int = 8,
        num_gems: int = 3,
        num_walls: int = 5,
        num_lava: int = 2,
        difficulty: str = "easy",
        render_mode: str | None = "ascii",
        max_steps: int = 100,
        **kwargs,
    ):
        """
        Initialize CollectGems environment.

        Args:
            grid_size: Size of the grid
            num_gems: Number of gems to place
            num_walls: Number of wall obstacles
            num_lava: Number of lava traps
            difficulty: Difficulty level
            render_mode: How to render the environment
            max_steps: Maximum steps per episode
        """
        super().__init__()

        # Adjust parameters based on difficulty
        if difficulty == "easy":
            grid_size, num_gems, num_walls, num_lava = 6, 2, 3, 1
        elif difficulty == "medium":
            grid_size, num_gems, num_walls, num_lava = 8, 3, 5, 2
        elif difficulty == "hard":
            grid_size, num_gems, num_walls, num_lava = 10, 4, 8, 3
        elif difficulty == "expert":
            grid_size, num_gems, num_walls, num_lava = 12, 5, 12, 4

        self.grid_size = grid_size
        self.num_gems = num_gems
        self.num_walls = num_walls
        self.num_lava = num_lava
        self.max_steps = max_steps
        self.render_mode = render_mode

        # 4 movement directions: up, right, down, left
        self.action_space = spaces.Discrete(4)
        # Observation is the rendered text
        self.observation_space = spaces.Text(min_length=1, max_length=100000)

        # State
        self.grid = None
        self.agent_pos = (1, 1)
        self.exit_pos = (0, 0)
        self.gems_collected = 0
        self.total_gems = 0
        self.step_count = 0

    def reset(self, seed=None, options=None):
        """Reset the environment and generate new level."""
        super().reset(seed=seed)

        self.step_count = 0
        self.gems_collected = 0
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # Walls around border
        self.grid[0, :] = self.WALL
        self.grid[-1, :] = self.WALL
        self.grid[:, 0] = self.WALL
        self.grid[:, -1] = self.WALL

        # Place agent in bottom-left interior
        self.agent_pos = (1, self.grid_size - 2)
        self.grid[self.agent_pos[1], self.agent_pos[0]] = self.AGENT

        # Place exit in top-right interior
        self.exit_pos = (self.grid_size - 2, 1)
        self.grid[self.exit_pos[1], self.exit_pos[0]] = self.EXIT

        occupied = {self.agent_pos, self.exit_pos}

        # Place gems
        self.total_gems = self.num_gems
        for _ in range(self.num_gems):
            pos = self._place_random(occupied)
            if pos:
                self.grid[pos[1], pos[0]] = self.GEM
                occupied.add(pos)

        # Place walls
        for _ in range(self.num_walls):
            pos = self._place_random(occupied)
            if pos:
                self.grid[pos[1], pos[0]] = self.WALL
                occupied.add(pos)

        # Place lava
        for _ in range(self.num_lava):
            pos = self._place_random(occupied)
            if pos:
                self.grid[pos[1], pos[0]] = self.LAVA
                occupied.add(pos)

        return self._get_obs(), self._get_info()

    def _place_random(self, occupied, max_attempts=100):
        """Place an entity at a random unoccupied interior position."""
        for _ in range(max_attempts):
            x = self.np_random.integers(1, self.grid_size - 1)
            y = self.np_random.integers(1, self.grid_size - 1)
            if (x, y) not in occupied:
                return (x, y)
        return None

    def step(self, action):
        """Take a step in the environment."""
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        dx, dy = directions[action]
        new_x = self.agent_pos[0] + dx
        new_y = self.agent_pos[1] + dy

        reward = -0.01  # Step penalty
        terminated = False

        if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
            cell = self.grid[new_y, new_x]

            if cell == self.WALL:
                pass  # Can't move into wall
            elif cell == self.GEM:
                self.gems_collected += 1
                self._move_agent(new_x, new_y)
                reward = 0.5
            elif cell == self.LAVA:
                self._move_agent(new_x, new_y)
                reward = -1.0
                terminated = True
            elif cell == self.EXIT:
                self._move_agent(new_x, new_y)
                if self.gems_collected >= self.total_gems:
                    reward = 1.0
                    terminated = True
                else:
                    reward = -0.5
                    terminated = True
            else:
                self._move_agent(new_x, new_y)

        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        info = self._get_info()
        info["success"] = terminated and reward > 0

        return self._get_obs(), reward, terminated, truncated, info

    def _move_agent(self, new_x, new_y):
        """Move agent to new position."""
        self.grid[self.agent_pos[1], self.agent_pos[0]] = self.EMPTY
        self.agent_pos = (new_x, new_y)
        self.grid[new_y, new_x] = self.AGENT

    def _get_obs(self):
        """Render grid as text."""
        symbols = {
            self.EMPTY: ". ",
            self.WALL: "# ",
            self.GEM: "G ",
            self.LAVA: "X ",
            self.EXIT: "E ",
            self.AGENT: "A ",
        }
        lines = []
        for y in range(self.grid_size):
            line = ""
            for x in range(self.grid_size):
                line += symbols.get(self.grid[y, x], "? ")
            lines.append(line)
        return "\n".join(lines)

    def _get_info(self):
        """Get info dictionary."""
        return {
            "step_count": self.step_count,
            "agent_pos": self.agent_pos,
            "gems_collected": self.gems_collected,
            "total_gems": self.total_gems,
            "exit_pos": self.exit_pos,
        }

    def render(self):
        """Render the environment."""
        obs = self._get_obs()
        if self.render_mode == "human":
            print(obs)
        return obs


def register_collect_gems_task():
    """Register the CollectGems task with Gymnasium."""
    gym.register(
        id="CollectGems-v0",
        entry_point="examples.advanced.custom_task:CollectGemsEnv",
    )
    print("CollectGems-v0 task registered!")


def test_custom_task():
    """Test the custom task."""
    print("\n" + "=" * 60)
    print("Testing CollectGems-v0 Custom Task")
    print("=" * 60)

    # Create directly (no registration needed for direct use)
    env = CollectGemsEnv(difficulty="easy", render_mode="ascii")

    # Run a test episode
    obs, info = env.reset(seed=42)

    print("\nInitial state:")
    print(obs)
    print(f"Gems to collect: {info['total_gems']}")
    print(f"Exit position: {info['exit_pos']}")

    # Run random agent
    print("\nRunning random agent...")
    total_reward = 0.0
    max_gems = 0

    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        max_gems = max(max_gems, info["gems_collected"])

        if terminated or truncated:
            break

    print("\nEpisode finished!")
    print(f"Steps: {step + 1}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Gems collected: {max_gems}/{info['total_gems']}")
    print(f"Success: {info.get('success', False)}")

    print("\nFinal state:")
    print(obs)

    env.close()

    print("\n" + "=" * 60)
    print("Custom task test complete!")
    print("=" * 60)


def main():
    """Run custom task demonstration."""
    test_custom_task()

    print("\nYou can also register and use via gym.make():")
    print("  register_collect_gems_task()")
    print("  env = gym.make('CollectGems-v0', difficulty='medium')")


if __name__ == "__main__":
    main()
