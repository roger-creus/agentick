"""Rule Discovery Navigation - Discover hidden grid rules.

Agent must discover hidden rules like "red tiles teleport" or
"stepping on blue after green opens doors".
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task(
    "RuleDiscoveryNavigation-v0",
    tags=["world_model", "navigation", "exploration", "rule_discovery"],
)
class RuleDiscoveryNavigationTask(TaskSpec):
    """Test world-model reasoning by discovering hidden grid rules through interaction.

    The agent must navigate a grid governed by hidden rules that are not
    communicated in advance. Rules may include teleportation tiles,
    multi-step tile sequences that unlock doors, penalty tiles, or
    prerequisite object collection. The agent must discover these rules
    through experimentation, build a causal model of how the grid
    responds to different actions, and then leverage that understanding
    to reach the goal. The number of special tiles and rule complexity
    scale with difficulty.

    Difficulty Levels:
        - easy: 7x7 grid with 2 special tiles and simple rules
          (complexity 1), 150 max steps.
        - medium: 10x10 grid with 3 special tiles and moderate rules
          (complexity 2), 250 max steps.
        - hard: 13x13 grid with 4 special tiles and complex rules
          (complexity 3), 350 max steps.
        - expert: 15x15 grid with 5 special tiles and highly complex
          rules (complexity 4) requiring deep causal reasoning,
          500 max steps.

    Capabilities Tested:
        - world_model: The agent must build and update an internal model
          of hidden rules governing tile interactions and effects.
        - exploration: The agent must experiment with special tiles to
          discover their hidden effects and trigger conditions.
        - navigation: The agent must navigate through the grid while
          accounting for discovered rule effects.
        - reasoning: The agent must reason causally about how tile
          interactions produce observed effects to infer the rules.

    Example:
        >>> env = agentick.make("RuleDiscoveryNavigation-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Discover hidden tile rules and use them to reach the goal
    """

    name = "RuleDiscoveryNavigation-v0"
    description = "Discover hidden rules and navigate to goal"
    capability_tags = ["world_model", "exploration", "navigation", "reasoning"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=150,
            params={"n_special_tiles": 2, "rule_complexity": 1},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=10,
            max_steps=250,
            params={"n_special_tiles": 3, "rule_complexity": 2},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=350,
            params={"n_special_tiles": 4, "rule_complexity": 3},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=15,
            max_steps=500,
            params={"n_special_tiles": 5, "rule_complexity": 4},
        ),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_rule = None
        self.rule_state = {}  # Track state for multi-step rules

    def generate(self, seed):
        """Generate a rule discovery navigation task instance.

        Creates a walled grid with randomly placed interior walls, a
        goal, and special tiles governed by a hidden rule. The rule is
        selected based on the configured complexity level and applied to
        the grid via special tile placement. The generator retries up to
        10 times to ensure solvability via flood fill, falling back to
        a simple open layout with no hidden rule if all attempts fail.

        Args:
            seed: Random seed for reproducible procedural generation.

        Returns:
            tuple: (grid, metadata) where grid is the initial Grid state
                with walls, goal, and special tiles, and metadata
                contains agent_start, goal_positions, max_steps, and
                the hidden_rule specification.
        """
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        params = self.difficulty_config.params

        max_attempts = 10
        for attempt in range(max_attempts):
            grid = Grid(size, size)
            # Walls
            grid.terrain[0, :] = CellType.WALL
            grid.terrain[-1, :] = CellType.WALL
            grid.terrain[:, 0] = CellType.WALL
            grid.terrain[:, -1] = CellType.WALL

            # Add internal walls
            for _ in range(size // 2):
                x, y = rng.integers(1, size - 1, 2)
                grid.terrain[y, x] = CellType.WALL

            agent_pos = (1, 1)
            goal_pos = (size - 2, size - 2)

            # Verify solvable
            reachable = grid.flood_fill(agent_pos)
            if goal_pos not in reachable:
                continue

            # Place goal
            grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

            # Choose and apply hidden rule
            self.hidden_rule = self._choose_rule(rng, params["rule_complexity"])
            self._apply_rule_to_grid(grid, rng, params["n_special_tiles"])

            return grid, {
                "agent_start": agent_pos,
                "goal_positions": [goal_pos],
                "max_steps": self.get_max_steps(),
                "hidden_rule": self.hidden_rule,
            }

        # Fallback
        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL
        agent_pos = (1, 1)
        goal_pos = (size - 2, size - 2)
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL
        self.hidden_rule = {"type": "none"}

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "max_steps": self.get_max_steps(),
            "hidden_rule": self.hidden_rule,
        }

    def _choose_rule(self, rng, complexity):
        """Choose hidden rule based on complexity."""
        rules = [
            {"type": "teleport", "trigger_tile": "ice"},  # Ice tiles teleport
            {"type": "sequence", "sequence": ["ice", "water"]},  # Ice then water unlocks doors
            {"type": "avoid", "penalty_tile": "hazard"},  # Hazard tiles give penalty
            {
                "type": "collect_first",
                "required_object": "key",
            },  # Must collect key before goal counts
        ]

        if complexity >= len(rules):
            return rules[-1]
        return rules[complexity - 1]

    def _apply_rule_to_grid(self, grid, rng, n_special):
        """Apply special tiles based on rule."""
        size = grid.width

        if self.hidden_rule["type"] == "teleport":
            # Place ice tiles
            for _ in range(n_special):
                x, y = rng.integers(1, size - 1, 2)
                if grid.terrain[y, x] == CellType.EMPTY:
                    grid.terrain[y, x] = CellType.ICE

        elif self.hidden_rule["type"] == "sequence":
            # Place ice and water tiles
            for _ in range(n_special // 2):
                x, y = rng.integers(1, size - 1, 2)
                if grid.terrain[y, x] == CellType.EMPTY:
                    grid.terrain[y, x] = CellType.ICE
            for _ in range(n_special // 2):
                x, y = rng.integers(1, size - 1, 2)
                if grid.terrain[y, x] == CellType.EMPTY:
                    grid.terrain[y, x] = CellType.WATER

        elif self.hidden_rule["type"] == "collect_first":
            # Place key
            x, y = rng.integers(1, size - 1, 2)
            grid.objects[y, x] = ObjectType.KEY

    def compute_dense_reward(self, old_state, action, new_state, info):
        """Potential-based shaping toward goal."""
        if "agent" not in new_state or "goal_positions" not in info:
            return -0.01

        agent_pos = new_state["agent"].position
        goals = info["goal_positions"]
        if not goals:
            return -0.01

        goal_pos = goals[0]
        distance = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])

        # Bonus for discovering/using rule
        rule_bonus = 0.0
        if self.hidden_rule["type"] == "teleport":
            # Bonus for stepping on ice
            x, y = agent_pos
            if "grid" in new_state and new_state["grid"].terrain[y, x] == CellType.ICE:
                rule_bonus = 0.1

        return -0.01 - distance * 0.001 + rule_bonus

    def check_success(self, state):
        """Check if the task objective is complete.

        The task succeeds when the agent reaches the goal cell. For
        rules of type 'collect_first', the agent must also have the
        required key in its inventory. Other rule types only require
        reaching the goal position.

        Args:
            state: Current state dict containing 'grid' and 'agent' keys.

        Returns:
            True if the agent is on the goal cell and any rule-specific
            prerequisites are satisfied, False otherwise.
        """
        if "grid" not in state or "agent" not in state:
            return False

        x, y = state["agent"].position
        on_goal = state["grid"].objects[y, x] == ObjectType.GOAL

        # Check rule requirements
        if self.hidden_rule["type"] == "collect_first":
            # Must have collected key
            has_key = any(item.entity_type == "key" for item in state["agent"].inventory)
            return on_goal and has_key

        return on_goal

    def get_optimal_return(self, difficulty=None):
        """Get the optimal (maximum possible) return for this task.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Optimal return of 1.0 (sparse success reward).
        """
        return 1.0

    def get_random_baseline(self, difficulty=None):
        """Get expected return for a random agent baseline.

        A random agent cannot discover hidden rules or satisfy rule
        prerequisites, yielding near-zero expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
