"""Helper interface for VLM agents."""

from PIL import Image


class VLMAgentInterface:
    """Helper for VLM agents: render pixel obs + text prompt, parse actions."""

    def __init__(self, env):
        self.env = env

    def format_observation(self, task_description=None):
        """Return dict with 'image' (PIL.Image) and 'prompt' (str)."""
        pixel_obs = self.env.get_pixel_observation()
        image = Image.fromarray(pixel_obs)

        prompt = "You are controlling an agent in a gridworld environment.\n"
        if task_description:
            prompt += f"Task: {task_description}\n"
        prompt += "\nWhat action should the agent take?"

        return {"image": image, "prompt": prompt}

    def parse_action(self, vlm_response):
        """Parse VLM text response into action."""
        action_name = vlm_response.strip().lower().replace(" ", "_")
        try:
            return self.env.action_space_obj.parse_action_name(action_name)
        except ValueError:
            return 0
