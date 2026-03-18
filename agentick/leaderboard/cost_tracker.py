"""API cost tracking and estimation."""

from __future__ import annotations

from typing import Any

# Pricing per 1M tokens (as of 2025 - update as needed)
MODEL_PRICING = {
    # OpenAI
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # Google
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.0-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.0-pro": {"input": 2.50, "output": 10.00},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    # OpenAI
    "gpt-5-mini": {"input": 0.125, "output": 1.00},
}


class CostTracker:
    """Track API call costs."""

    def __init__(self, model_name: str):
        """
        Initialize cost tracker.

        Args:
            model_name: Model name to track costs for
        """
        self.model_name = model_name
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0

        # Get pricing
        self.pricing = self._get_pricing(model_name)

    def _get_pricing(self, model_name: str) -> dict[str, float]:
        """Get pricing for model."""
        # Try exact match
        if model_name in MODEL_PRICING:
            return MODEL_PRICING[model_name]

        # Try fuzzy match (e.g., "gpt-4o-2024-08-06" -> "gpt-4o")
        for key in MODEL_PRICING:
            if model_name.startswith(key):
                return MODEL_PRICING[key]

        # Default: assume same as GPT-4o
        return MODEL_PRICING["gpt-4o"]

    def add_call(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int | None = None,
    ):
        """
        Add an API call to the tracker.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            total_tokens: Total tokens (if input/output not available)
        """
        if total_tokens is not None and input_tokens == 0 and output_tokens == 0:
            # Estimate split (roughly 2:1 input:output)
            input_tokens = int(total_tokens * 0.67)
            output_tokens = int(total_tokens * 0.33)

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_calls += 1

    def get_total_cost(self) -> float:
        """
        Get total cost in USD.

        Returns:
            Total cost in dollars
        """
        input_cost = (self.total_input_tokens / 1_000_000) * self.pricing["input"]
        output_cost = (self.total_output_tokens / 1_000_000) * self.pricing["output"]

        return input_cost + output_cost

    def get_report(self) -> dict[str, Any]:
        """
        Get detailed cost report.

        Returns:
            Dictionary with cost breakdown
        """
        total_cost = self.get_total_cost()

        return {
            "model": self.model_name,
            "total_calls": self.total_calls,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_cost_usd": total_cost,
            "cost_per_call": total_cost / max(1, self.total_calls),
        }

    def print_report(self):
        """Print cost report to console."""
        report = self.get_report()

        print("\n=== API Cost Report ===")
        print(f"Model: {report['model']}")
        print(f"Total Calls: {report['total_calls']}")
        print(f"Total Tokens: {report['total_tokens']:,}")
        print(f"  Input: {report['input_tokens']:,}")
        print(f"  Output: {report['output_tokens']:,}")
        print(f"Total Cost: ${report['total_cost_usd']:.2f}")
        print(f"Cost per Call: ${report['cost_per_call']:.4f}")
