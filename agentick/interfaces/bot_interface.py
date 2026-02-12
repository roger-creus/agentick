"""API for programmatic bots."""


class BotInterface:
    """API for programmatic bots: exposes get_state(), get_valid_actions(), etc."""

    def __init__(self, env):
        self.env = env

    def get_grid(self):
        """Get the raw grid."""
        return self.env.grid

    def get_agent_position(self):
        """Get agent position."""
        return self.env.agent.position

    def get_goal_positions(self):
        """Get goal positions."""
        return self.env.task_config.get("goal_positions", [])

    def get_valid_actions(self):
        """Get list of valid action indices."""
        mask = self.env.get_valid_actions()
        return [i for i, valid in enumerate(mask) if valid]

    def get_shortest_path(self, target):
        """Get shortest path to target position."""
        agent_pos = self.get_agent_position()
        return self.env.grid.bfs(agent_pos, target)

    def step(self, action):
        """Take a step in the environment."""
        return self.env.step(action)
