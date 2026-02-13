"""
HuggingFace local model agent for Agentick tasks.

This example demonstrates:
- Using a local HuggingFace model for text-based tasks
- Running inference without API calls
- Processing text observations

Requirements:
    - uv sync --extra llm
    - GPU recommended for faster inference

Usage:
    uv run python examples/llm/huggingface_local_agent.py
"""

from dotenv import load_dotenv

import agentick

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not available. Install with: uv sync --extra llm")


class HuggingFaceAgent:
    """Local HuggingFace model agent."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507-FP8",
        device: str | None = None,
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers package not installed. Run: uv sync --extra llm")

        print(f"Loading model: {model_name}")
        print("This may take a few minutes on first run...")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        print(f"Using device: {device}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )

        if device == "cpu":
            self.model = self.model.to(device)

        self.model.eval()
        print("Model loaded!")

    def get_action(
        self,
        observation: str,
        task_description: str,
        action_space_description: str,
    ) -> str:
        """Get action from local model."""
        # Create prompt with few-shot examples for better small model performance
        prompt = f"""You are an AI agent. Choose the best action.

Task: {task_description}
Actions: {action_space_description}

Example 1:
Observation: You are at (1,1). Goal is at (3,3).
Action: right

Example 2:
Observation: You are at (2,2). Goal is at (2,1).
Action: left

Example 3:
Observation: You are at (3,3). This is the goal!
Action: done

Now your turn:
Observation: {observation}
Action:"""

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        action_text = response.strip().split("\n")[0].strip()

        return action_text


def parse_action(action_text: str, action_list: list[str]) -> int:
    """Parse action text to action index."""
    action_text = action_text.lower().strip()

    # Try exact match
    for i, action in enumerate(action_list):
        if action.lower() == action_text:
            return i

    # Try partial match
    for i, action in enumerate(action_list):
        if action_text in action.lower() or action.lower() in action_text:
            return i

    # Default to first action (usually noop)
    print(f"Could not parse action '{action_text}', using default")
    return 0


def main():
    """Run HuggingFace local model agent."""
    # Load environment variables (for potential HF_TOKEN usage)
    load_dotenv()

    print("HuggingFace Local Model Agent")
    print("=" * 80)

    if not TRANSFORMERS_AVAILABLE:
        return

    # Create environment (use language for text observations)
    env_id = "GoToGoal-v0"
    print(f"\nEnvironment: {env_id}")

    env = agentick.make(env_id, difficulty="easy", render_mode="language")

    # Get action descriptions
    action_list = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]

    # Create agent
    agent = HuggingFaceAgent(model_name="Qwen/Qwen2.5-0.5B-Instruct")

    # Run episodes
    num_episodes = 3
    print(f"\nRunning {num_episodes} episodes...")
    print()

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        max_steps = 20

        print(f"Episode {episode + 1}:")

        while not done and steps < max_steps:
            # Get action from model
            action_text = agent.get_action(
                observation=obs,
                task_description="Navigate to the goal position in a grid world",
                action_space_description=", ".join(action_list),
            )

            # Parse action
            action = parse_action(action_text, action_list)

            print(f"  Step {steps + 1}: Action = {action_text} ({action_list[action]})")

            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        success = info.get("success", False)
        print(f"  Result: reward={total_reward:.2f}, steps={steps}, success={success}")
        print()

    env.close()

    print("=" * 80)
    print("Done!")
    print("\n💡 Note: This runs a local model, no API calls required.")
    print("Models are cached in ~/.cache/huggingface/hub/")


if __name__ == "__main__":
    main()
