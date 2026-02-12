"""LLM-specific logging with cost tracking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LLMLogger:
    """Comprehensive LLM logging."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.calls: list[dict[str, Any]] = []
        self.total_tokens = {"prompt": 0, "completion": 0}
        self.total_cost = 0.0

    def log_call(
        self,
        prompt: str,
        system_prompt: str,
        response: str,
        parsed_action: Any,
        parse_success: bool,
        token_count: dict[str, int],
        latency_ms: float,
        cost: float,
    ) -> None:
        """Log LLM API call."""
        call_data = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "response": response,
            "parsed_action": str(parsed_action),
            "parse_success": parse_success,
            "token_count": token_count,
            "latency_ms": latency_ms,
            "cost": cost,
        }

        self.calls.append(call_data)
        self.total_tokens["prompt"] += token_count.get("prompt", 0)
        self.total_tokens["completion"] += token_count.get("completion", 0)
        self.total_cost += cost

    def save(self) -> None:
        """Save log."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save full log
        log_path = self.output_dir / "llm_calls.json"
        with open(log_path, "w") as f:
            json.dump(
                {
                    "calls": self.calls,
                    "total_tokens": self.total_tokens,
                    "total_cost": self.total_cost,
                },
                f,
                indent=2,
            )

        # Save readable transcript
        self.export_transcript(self.output_dir / "llm_transcript.txt")

    def export_transcript(self, output_path: str | Path) -> None:
        """Export readable conversation transcript."""
        with open(output_path, "w") as f:
            for i, call in enumerate(self.calls):
                f.write(f"=== Call {i + 1} ===\n")
                f.write(f"[SYSTEM] {call['system_prompt']}\n\n")
                f.write(f"[USER] {call['prompt']}\n\n")
                f.write(f"[ASSISTANT] {call['response']}\n\n")
                f.write(f"[PARSED] {call['parsed_action']} ")
                f.write("✓\n" if call["parse_success"] else "✗\n")
                f.write(f"[TOKENS] {call['token_count']}\n")
                f.write(f"[COST] ${call['cost']:.4f}\n")
                f.write("---\n\n")

            f.write("\n=== Summary ===\n")
            f.write(f"Total calls: {len(self.calls)}\n")
            f.write(f"Total tokens: {self.total_tokens}\n")
            f.write(f"Total cost: ${self.total_cost:.4f}\n")
