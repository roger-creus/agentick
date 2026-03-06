"""Batched multi-episode runner for vLLM and other batching backends."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from agentick.agents.backends.base import ModelBackend
from agentick.agents.harness import HarnessPreset
from agentick.core.types import ActionType
from agentick.leaderboard.adapters.prompt_templates import parse_action_from_text


@dataclass
class EpisodeResult:
    """Result from a single episode in a batch."""

    seed: int
    seed_idx: int
    episode_idx: int
    total_reward: float = 0.0
    episode_length: int = 0
    success: bool = False
    steps: list[dict[str, Any]] = field(default_factory=list)
    call_log: list[dict[str, Any]] = field(default_factory=list)
    agent_stats: dict[str, Any] = field(default_factory=dict)


class BatchedEpisodeRunner:
    """Run N environments in lockstep, batching LLM calls.

    Uses ``backend.generate_batch()`` for a single batched vLLM call per step
    across all active environments.
    """

    def __init__(
        self,
        backend: ModelBackend,
        harness_cls: type[HarnessPreset],
        harness_kwargs: dict[str, Any],
        obs_modes: list[str],
    ):
        self.backend = backend
        self.harness_cls = harness_cls
        self.harness_kwargs = harness_kwargs
        self.obs_modes = obs_modes

    def run_batch(
        self,
        envs: list[Any],
        seeds: list[int],
        seed_indices: list[int],
        episode_indices: list[int],
    ) -> list[EpisodeResult]:
        """Run N episodes in parallel with batched inference.

        Args:
            envs: List of Gymnasium environments (already created).
            seeds: Per-env seed for reset.
            seed_indices: Per-env seed index (for logging).
            episode_indices: Per-env episode index (for logging).

        Returns:
            List of EpisodeResult, one per environment.
        """
        n = len(envs)
        assert len(seeds) == n

        # Create per-env harness instances and results
        harnesses: list[HarnessPreset] = [
            self.harness_cls(**self.harness_kwargs) for _ in range(n)
        ]
        results: list[EpisodeResult] = [
            EpisodeResult(
                seed=seeds[i],
                seed_idx=seed_indices[i],
                episode_idx=episode_indices[i],
            )
            for i in range(n)
        ]

        # Reset all envs
        observations: list[Any] = []
        infos: list[dict[str, Any]] = []
        for i in range(n):
            obs, info = envs[i].reset(seed=seeds[i])
            observations.append(obs)
            infos.append(info)
            harnesses[i].reset()

        # Track which envs are still active
        active = [True] * n
        terminated = [False] * n
        truncated = [False] * n

        # Tracking for per-env token stats
        total_input_tokens = [0] * n
        total_output_tokens = [0] * n
        total_latency = [0.0] * n
        total_calls = [0] * n

        while any(active):
            # Collect indices and messages for active envs
            active_indices = [i for i in range(n) if active[i]]

            # Build messages for each active env
            all_messages = []
            for i in active_indices:
                info_with_modes = {**infos[i], "_obs_modes": self.obs_modes}
                messages = harnesses[i].build_messages(
                    observations[i], info_with_modes, self.obs_modes
                )
                all_messages.append(messages)

            # Batched inference
            start = time.time()
            responses = self.backend.generate_batch(all_messages)
            batch_latency = time.time() - start

            # Process responses and step active envs
            for idx_in_batch, env_idx in enumerate(active_indices):
                response = responses[idx_in_batch]
                messages = all_messages[idx_in_batch]

                # Parse action
                action = parse_action_from_text(response.text, list(range(6)))

                # Extract reasoning
                reasoning = None
                if "ACTION:" in response.text:
                    parts = response.text.rsplit("ACTION:", 1)
                    if parts[0].strip():
                        reasoning = parts[0].strip()

                try:
                    action_name = ActionType(action).name
                except ValueError:
                    action_name = f"ACTION_{action}"

                # Extract observation text from last user message
                user_content = messages[-1]["content"]
                if isinstance(user_content, list):
                    obs_text = "\n".join(
                        b["text"] for b in user_content if b.get("type") == "text"
                    )
                    has_image = any(b.get("type") == "image" for b in user_content)
                else:
                    obs_text = user_content
                    has_image = False

                # Update stats
                total_calls[env_idx] += 1
                total_input_tokens[env_idx] += response.input_tokens
                total_output_tokens[env_idx] += response.output_tokens
                total_latency[env_idx] += batch_latency / len(active_indices)

                results[env_idx].call_log.append(
                    {
                        "step": infos[env_idx].get("step", 0),
                        "observation": obs_text,
                        "has_image": has_image,
                        "response": response.text,
                        "reasoning": reasoning,
                        "parsed_action": action,
                        "action_name": action_name,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "latency": batch_latency / len(active_indices),
                    }
                )

                # Step env
                obs, reward, term, trunc, info = envs[env_idx].step(action)
                observations[env_idx] = obs
                infos[env_idx] = info
                terminated[env_idx] = term
                truncated[env_idx] = trunc

                results[env_idx].total_reward += reward
                results[env_idx].episode_length += 1
                results[env_idx].steps.append(
                    {
                        "step": results[env_idx].episode_length,
                        "action": int(action),
                        "reward": float(reward),
                        "terminated": bool(term),
                        "truncated": bool(trunc),
                    }
                )

                # Record in harness
                info_with_modes = {**infos[env_idx], "_obs_modes": self.obs_modes}
                harnesses[env_idx].record_step(
                    observations[env_idx],
                    info_with_modes,
                    action,
                    response.text,
                    reward,
                )

                # Check if done
                if term or trunc:
                    active[env_idx] = False
                    results[env_idx].success = bool(info.get("success", False))

        # Compile agent stats
        for i in range(n):
            results[i].agent_stats = {
                "total_calls": total_calls[i],
                "total_tokens": total_input_tokens[i] + total_output_tokens[i],
                "total_input_tokens": total_input_tokens[i],
                "total_output_tokens": total_output_tokens[i],
                "total_latency": total_latency[i],
                "mean_latency": (
                    total_latency[i] / total_calls[i] if total_calls[i] > 0 else 0.0
                ),
            }

        return results
