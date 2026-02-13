"""
Complete example: Creating a custom Agentick task.

Demonstrates:
- Extending GridWorldEnv
- Custom entities and mechanics
- Reward shaping
- Registration and testing
"""

import gymnasium as gym

from agentick.core.entity import Entity
from agentick.core.env import AgentickEnv
from agentick.tasks.registry import register_task


class CollectGemsEnv(AgentickEnv):
    """
    Custom task: Collect all gems in the shortest path.

    Goal: Agent must collect all gems before reaching the exit.

    Entities:
    - Agent (blue square)
    - Gems (yellow diamonds) - must collect all
    - Exit (green circle) - reach after collecting gems
    - Walls (gray blocks) - obstacles
    - Lava (red) - instant failure if stepped on

    Reward structure:
    - +0.5 per gem collected
    - +1.0 for reaching exit with all gems
    - -1.0 for stepping on lava
    - -0.01 per step (encourages efficiency)
    - -0.5 for reaching exit without all gems

    Success condition:
    - Collect all gems AND reach exit
    """

    def __init__(
        self,
        grid_size: int = 8,
        num_gems: int = 3,
        num_walls: int = 5,
        num_lava: int = 2,
        difficulty: str = "easy",
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
        """
        # Adjust parameters based on difficulty
        if difficulty == "easy":
            grid_size = 6
            num_gems = 2
            num_walls = 3
            num_lava = 1
        elif difficulty == "medium":
            grid_size = 8
            num_gems = 3
            num_walls = 5
            num_lava = 2
        elif difficulty == "hard":
            grid_size = 10
            num_gems = 4
            num_walls = 8
            num_lava = 3
        elif difficulty == "expert":
            grid_size = 12
            num_gems = 5
            num_walls = 12
            num_lava = 4

        super().__init__(grid_size=grid_size, max_steps=100, **kwargs)

        self.num_gems = num_gems
        self.num_walls = num_walls
        self.num_lava = num_lava
        self.difficulty = difficulty

        # Track game state
        self.gems_collected = 0
        self.total_gems = 0
        self.exit_pos = None

    def reset(self, seed=None, options=None):
        """Reset the environment and generate new level."""
        super().reset(seed=seed)

        # Reset state
        self.gems_collected = 0
        self.grid.clear()

        # Place agent in bottom-left corner
        self.agent_pos = (1, self.grid.height - 2)
        self.grid.set(
            self.agent_pos[0],
            self.agent_pos[1],
            Entity(id="agent", entity_type="agent", position=self.agent_pos, properties={"color": "blue"}),
        )

        # Place exit in top-right corner
        self.exit_pos = (self.grid.width - 2, 1)
        self.grid.set(
            self.exit_pos[0],
            self.exit_pos[1],
            Entity(id="goal", entity_type="goal", position=self.exit_pos, properties={"color": "green"}),
        )

        # Place gems randomly
        self.total_gems = self.num_gems
        gems_placed = 0
        attempts = 0
        max_attempts = 100

        while gems_placed < self.num_gems and attempts < max_attempts:
            x = self.np_random.integers(1, self.grid.width - 1)
            y = self.np_random.integers(1, self.grid.height - 1)

            # Don't place on agent or exit
            if (x, y) not in [self.agent_pos, self.exit_pos]:
                if self.grid.get(x, y) is None:
                    self.grid.set(x, y, Entity(id=f"gem_{gems_placed}", entity_type="key", position=(x, y), properties={"color": "yellow"}))
                    gems_placed += 1

            attempts += 1

        # Place walls
        walls_placed = 0
        attempts = 0

        while walls_placed < self.num_walls and attempts < max_attempts:
            x = self.np_random.integers(1, self.grid.width - 1)
            y = self.np_random.integers(1, self.grid.height - 1)

            # Don't block important positions
            if (x, y) not in [self.agent_pos, self.exit_pos]:
                if self.grid.get(x, y) is None:
                    self.grid.set(x, y, Entity(id=f"wall_{walls_placed}", entity_type="wall", position=(x, y), properties={"color": "gray"}))
                    walls_placed += 1

            attempts += 1

        # Place lava traps
        lava_placed = 0
        attempts = 0

        while lava_placed < self.num_lava and attempts < max_attempts:
            x = self.np_random.integers(1, self.grid.width - 1)
            y = self.np_random.integers(1, self.grid.height - 1)

            # Don't place near agent or exit
            if (x, y) not in [self.agent_pos, self.exit_pos]:
                if abs(x - self.agent_pos[0]) > 1 or abs(y - self.agent_pos[1]) > 1:
                    if self.grid.get(x, y) is None:
                        self.grid.set(x, y, Entity(id=f"lava_{lava_placed}", entity_type="lava", position=(x, y), properties={"color": "red"}))
                        lava_placed += 1

            attempts += 1

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):
        """Take a step in the environment."""
        # Get new position based on action
        dx, dy = self._action_to_direction(action)
        new_x = self.agent_pos[0] + dx
        new_y = self.agent_pos[1] + dy

        reward = 0
        terminated = False

        # Check if move is valid
        if not self.grid.in_bounds(new_x, new_y):
            # Hit boundary - no move, small penalty
            reward = -0.01
        else:
            entity = self.grid.get(new_x, new_y)

            if entity is None:
                # Empty space - move and small step penalty
                self._move_agent(new_x, new_y)
                reward = -0.01

            elif entity.entity_type == "wall":
                # Hit wall - no move, small penalty
                reward = -0.01

            elif entity.entity_type == "key":
                # Collect gem!
                self.gems_collected += 1
                self._move_agent(new_x, new_y)
                reward = 0.5

            elif entity.entity_type == "lava":
                # Stepped on lava - instant failure
                self._move_agent(new_x, new_y)
                reward = -1.0
                terminated = True

            elif entity.entity_type == "goal":
                # Reached exit
                self._move_agent(new_x, new_y)

                if self.gems_collected >= self.total_gems:
                    # Success! All gems collected
                    reward = 1.0
                    terminated = True
                else:
                    # Reached exit but missing gems
                    reward = -0.5
                    terminated = True

        # Increment step counter
        self.step_count += 1

        # Check if max steps reached
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()
        info = self._get_info()
        info["gems_collected"] = self.gems_collected
        info["total_gems"] = self.total_gems
        info["success"] = terminated and reward > 0

        return obs, reward, terminated, truncated, info

    def _move_agent(self, new_x: int, new_y: int):
        """Move agent to new position."""
        # Clear old position
        self.grid.set(self.agent_pos[0], self.agent_pos[1], None)

        # Update position
        self.agent_pos = (new_x, new_y)

        # Set new position (might overwrite entity like gem)
        self.grid.set(new_x, new_y, Entity(id="agent", entity_type="agent", position=(new_x, new_y), properties={"color": "blue"}))

    def _action_to_direction(self, action: int) -> tuple[int, int]:
        """Convert action to direction vector."""
        # 0: up, 1: right, 2: down, 3: left
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        return directions[action]

    def _get_obs(self):
        """Get current observation."""
        if self.render_mode == "ascii":
            return self.grid.render_ascii()
        elif self.render_mode == "rgb_array":
            return self.grid.render_rgb()
        else:
            return self.grid.to_numpy()

    def _get_info(self) -> dict:
        """Get info dictionary."""
        return {
            "step_count": self.step_count,
            "agent_pos": self.agent_pos,
            "gems_collected": self.gems_collected,
            "total_gems": self.total_gems,
            "exit_pos": self.exit_pos,
        }


def register_collect_gems_task():
    """Register the CollectGems task."""
    # Register with gymnasium
    gym.register(
        id="CollectGems-v0",
        entry_point=lambda **kwargs: CollectGemsEnv(**kwargs),
    )

    # Register with agentick
    register_task(
        task_id="CollectGems-v0",
        task_class=CollectGemsEnv,
        description="Collect all gems before reaching the exit",
        capabilities=["navigation", "planning", "collection"],
    )

    print("✓ CollectGems-v0 task registered!")


def test_custom_task():
    """Test the custom task."""
    print("\n" + "=" * 60)
    print("Testing CollectGems-v0 Custom Task")
    print("=" * 60)

    # Register the task
    register_collect_gems_task()

    # Create environment
    env = gym.make("CollectGems-v0", difficulty="easy", render_mode="ascii")

    # Run a test episode
    obs, info = env.reset(seed=42)

    print("\nInitial state:")
    print(obs)
    print(f"Gems to collect: {info['total_gems']}")
    print(f"Exit position: {info['exit_pos']}")

    # Run random agent
    print("\nRunning random agent...")
    total_reward = 0
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

    print("\nNow you can use this task in experiments:")
    print("  env = agentick.make('CollectGems-v0', difficulty='medium')")
    print("  # or in ExperimentConfig: tasks=['CollectGems-v0']")


if __name__ == "__main__":
    main()
