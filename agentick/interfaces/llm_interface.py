"""Helper interface for LLM agents."""


class LLMAgentInterface:
    """Helper for LLM agents: format observations as prompts, parse text actions."""

    def __init__(self, env):
        self.env = env

    def format_prompt(self, obs, task_description=None):
        """Format observation into an LLM prompt."""
        prompt = "# Gridworld Task\n\n"

        if task_description:
            prompt += f"**Task:** {task_description}\n\n"

        prompt += "## Current State\n"
        prompt += f"{obs}\n\n"

        # Add valid actions
        info = self.env._get_info()
        valid_actions = info.get("valid_actions", [])
        prompt += "## Available Actions\n"
        for action in valid_actions:
            prompt += f"- {action}\n"

        prompt += "\nWhat action should the agent take? Respond with just the action name."

        return prompt

    def parse_action(self, llm_response):
        """Parse LLM text response into a discrete action integer."""
        action_name = llm_response.strip().lower().replace(" ", "_")
        try:
            return self.env.action_space_obj.parse_action_name(action_name)
        except ValueError:
            # Default to NOOP if can't parse
            return 0
