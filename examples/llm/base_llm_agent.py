"""Base class for LLM agents.

Provides common functionality for prompt construction, action parsing, and retry logic.
"""

from abc import ABC, abstractmethod


class BaseLLMAgent(ABC):
    """Abstract base class for LLM-based agents."""

    def __init__(self, temperature: float = 0.0, max_retries: int = 3):
        """
        Initialize base LLM agent.

        Args:
            temperature: Sampling temperature (0 = deterministic)
            max_retries: Maximum retries for failed API calls or invalid actions
        """
        self.temperature = temperature
        self.max_retries = max_retries
        self.action_history: list[tuple[str, int, float]] = []

    @abstractmethod
    def _call_llm(self, prompt: str) -> str:
        """Call LLM API (must be implemented by subclass)."""
        pass

    def act(self, observation: str, valid_actions: list[str]) -> int:
        """
        Choose action based on observation.

        Args:
            observation: Text observation from environment
            valid_actions: List of valid action names

        Returns:
            Action index
        """
        prompt = self._construct_prompt(observation, valid_actions)

        for attempt in range(self.max_retries):
            try:
                response = self._call_llm(prompt)
                action_idx = self._parse_action(response, valid_actions)

                if action_idx is not None:
                    return action_idx

                # Invalid action, add feedback and retry
                prompt += f"\n\nYour response '{response}' was invalid. Valid actions are: {valid_actions}. Please respond with exactly one of these action names."

            except Exception as e:
                if attempt == self.max_retries - 1:
                    # Final attempt failed, return random action
                    print(f"LLM agent failed after {self.max_retries} attempts: {e}")
                    return 0  # Default to first action
                continue

        return 0  # Fallback

    def _construct_prompt(self, observation: str, valid_actions: list[str]) -> str:
        """Construct prompt for LLM."""
        base_prompt = f"""You are an AI agent playing a gridworld game. Your goal is to navigate and complete tasks.

Observation:
{observation}

Valid actions: {", ".join(valid_actions)}

Instructions:
- Analyze the observation carefully
- Consider your goal and current situation
- Choose the best action from the valid actions
- Respond with ONLY the action name, nothing else

Your action:"""

        # Add action history context if available
        if self.action_history:
            recent_history = self.action_history[-5:]  # Last 5 actions
            history_str = "\n".join(
                [
                    f"Step {i}: {action} (reward: {reward:.2f})"
                    for i, (obs, action, reward) in enumerate(recent_history)
                ]
            )
            base_prompt = base_prompt.replace(
                "Observation:", f"Recent history:\n{history_str}\n\nObservation:"
            )

        return base_prompt

    def _parse_action(self, response: str, valid_actions: list[str]) -> int | None:
        """
        Parse LLM response to extract action.

        Args:
            response: LLM response text
            valid_actions: List of valid action names

        Returns:
            Action index or None if invalid
        """
        # Clean response
        response = response.strip().lower()

        # Try exact match
        for i, action in enumerate(valid_actions):
            if response == action.lower():
                return i

        # Try partial match
        for i, action in enumerate(valid_actions):
            if action.lower() in response:
                return i

        # Try to extract from common formats
        if "move" in response:
            if "up" in response and "move_up" in [a.lower() for a in valid_actions]:
                return [a.lower() for a in valid_actions].index("move_up")
            elif "down" in response and "move_down" in [a.lower() for a in valid_actions]:
                return [a.lower() for a in valid_actions].index("move_down")
            elif "left" in response and "move_left" in [a.lower() for a in valid_actions]:
                return [a.lower() for a in valid_actions].index("move_left")
            elif "right" in response and "move_right" in [a.lower() for a in valid_actions]:
                return [a.lower() for a in valid_actions].index("move_right")

        return None

    def record_step(self, observation: str, action: int, reward: float) -> None:
        """Record step for history."""
        self.action_history.append((observation, action, reward))

    def reset(self) -> None:
        """Reset agent for new episode."""
        self.action_history.clear()
