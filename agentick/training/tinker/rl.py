"""Tinker RL — Reinforcement learning for LLMs via the Tinker API.

Wraps agentick environments in Tinker's RL environment interface and runs
PPO or REINFORCE training using environment rewards as the learning signal.

Supports the SFT warmstart -> RL fine-tune pipeline: first train with
:mod:`agentick.training.tinker.sft`, then continue with RL on live
environment interactions.

.. note::
    Tinker must be installed separately (``pip install tinker``) and a
    Tinker API key must be set (``TINKER_API_KEY``).

Example::

    from agentick.training.tinker.rl import TinkerRLTrainer

    trainer = TinkerRLTrainer(
        base_model="Qwen/Qwen2.5-7B-Instruct",
        task_id="GoToGoal-v0",
        difficulty="medium",
        rank=32,
    )
    trainer.train(num_episodes=100, learning_rate=1e-5)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_TINKER_AVAILABLE = False
try:
    import tinker
    import tinker.types as tinker_types

    _TINKER_AVAILABLE = True
except ImportError:
    tinker = None  # type: ignore[assignment]
    tinker_types = None  # type: ignore[assignment]


def _require_tinker() -> None:
    if not _TINKER_AVAILABLE:
        raise ImportError(
            "Tinker is not installed. Install with: pip install tinker\n"
            "See https://tinker-docs.thinkingmachines.ai/install"
        )


class AgentickTinkerEnv:
    """Wraps an agentick env in Tinker's RL environment interface.

    Tinker RL environments are token-based: the agent receives
    tokenized observations and produces tokenized actions.
    This wrapper converts between text observations and token sequences.

    Corresponds to Tinker's ``Env`` interface::

        class Env:
            async def initial_observation(self) -> tuple[Observation, StopCondition]:
                ...
            async def step(self, action: Action) -> StepResult:
                ...
    """

    def __init__(
        self,
        env: Any,
        tokenizer: Any,
        max_response_tokens: int = 32,
    ) -> None:
        self.env = env
        self.tokenizer = tokenizer
        self.max_response_tokens = max_response_tokens
        self._obs = None
        self._info: dict[str, Any] = {}
        self._done = False
        self._total_reward = 0.0

    async def initial_observation(self) -> tuple[Any, Any]:
        """Reset env and return tokenized initial observation."""
        obs, info = self.env.reset()
        self._obs = obs
        self._info = info
        self._done = False
        self._total_reward = 0.0

        obs_text = obs if isinstance(obs, str) else json.dumps(obs)
        prompt = f"Observation: {obs_text}\nAction: "
        tokens = self.tokenizer.encode(prompt)

        observation = tinker_types.ModelInput.from_ints(tokens=tokens)
        stop = tinker_types.StopCondition(
            max_tokens=self.max_response_tokens,
            stop_strings=["\n"],
        )
        return observation, stop

    async def step(self, action: Any) -> Any:
        """Process agent's token action, step env, return next observation.

        Args:
            action: ``TokensWithLogprobs`` from the model.

        Returns:
            ``StepResult`` with next observation, reward, done, and logprobs.
        """
        # Decode action tokens to text
        action_text = self.tokenizer.decode(action.tokens).strip()

        # Parse action integer
        match = re.search(r"\b(\d+)\b", action_text)
        action_int = int(match.group(1)) if match else 0

        # Step the environment
        obs, reward, done, truncated, info = self.env.step(action_int)
        self._obs = obs
        self._info = info
        self._done = done or truncated
        self._total_reward += reward

        # Build next observation
        obs_text = obs if isinstance(obs, str) else json.dumps(obs)
        prompt = f"Observation: {obs_text}\nAction: "
        tokens = self.tokenizer.encode(prompt)

        next_obs = tinker_types.ModelInput.from_ints(tokens=tokens)
        stop = tinker_types.StopCondition(
            max_tokens=self.max_response_tokens,
            stop_strings=["\n"],
        )

        return {
            "observation": next_obs,
            "stop": stop,
            "reward": float(reward),
            "done": self._done,
            "info": {
                "success": info.get("success", False),
                "total_reward": self._total_reward,
            },
        }


class TinkerRLTrainer:
    """RL trainer using Tinker's API with agentick environments.

    Supports both PPO and REINFORCE (importance_sampling) loss functions.
    Can warm-start from an SFT checkpoint.

    Args:
        base_model: HuggingFace model name.
        task_id: Agentick task identifier.
        difficulty: Task difficulty level.
        rank: LoRA rank.
        loss_fn: ``"ppo"`` or ``"importance_sampling"`` (REINFORCE).
        output_dir: Directory for logs and checkpoints.
    """

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        task_id: str = "GoToGoal-v0",
        difficulty: str = "easy",
        rank: int = 32,
        loss_fn: str = "ppo",
        output_dir: str | Path = "models/tinker_rl",
        render_mode: str = "language",
    ) -> None:
        _require_tinker()

        self.base_model = base_model
        self.task_id = task_id
        self.difficulty = difficulty
        self.rank = rank
        self.loss_fn = loss_fn
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.render_mode = render_mode

        self._training_client = None
        self._sampling_client = None
        self._tokenizer = None

    def train(
        self,
        num_episodes: int = 100,
        learning_rate: float = 1e-5,
        gamma: float = 1.0,
        clip_low: float = 0.8,
        clip_high: float = 1.2,
    ) -> dict[str, Any]:
        """Run RL training loop.

        For each episode:
        1. Create agentick env, wrap in AgentickTinkerEnv
        2. Sample actions from current policy
        3. Compute advantages from environment rewards
        4. Update with PPO or REINFORCE

        Args:
            num_episodes: Total training episodes.
            learning_rate: Adam learning rate.
            gamma: Discount factor for reward computation.
            clip_low: PPO clip lower bound.
            clip_high: PPO clip upper bound.

        Returns:
            Training metrics.
        """
        _require_tinker()
        import agentick

        # Initialize Tinker client
        service_client = tinker.ServiceClient()
        self._training_client = service_client.create_lora_training_client(
            base_model=self.base_model,
            rank=self.rank,
        )
        self._tokenizer = self._training_client.get_tokenizer()

        # Save initial weights for sampling
        self._sampling_client = self._training_client.save_weights_and_get_sampling_client(
            name="rl_policy",
        )

        metrics = {
            "episode_rewards": [],
            "episode_successes": [],
            "losses": [],
        }

        for ep in range(num_episodes):
            # Create environment
            env = agentick.make(
                self.task_id,
                difficulty=self.difficulty,
                render_mode=self.render_mode,
            )

            # Run episode and collect rollout data
            rollout = self._collect_rollout(env)
            env.close()

            metrics["episode_rewards"].append(rollout["total_reward"])
            metrics["episode_successes"].append(rollout["success"])

            # Compute advantages (simple: reward - baseline)
            baseline = np.mean(metrics["episode_rewards"]) if metrics["episode_rewards"] else 0.0
            advantage = rollout["total_reward"] - baseline

            # Update policy
            if rollout["data"]:
                loss = self._update_policy(
                    rollout["data"],
                    advantage,
                    learning_rate,
                    clip_low,
                    clip_high,
                )
                metrics["losses"].append(loss)

            # Periodically refresh sampling client
            if (ep + 1) % 10 == 0:
                self._sampling_client = self._training_client.save_weights_and_get_sampling_client(
                    name=f"rl_policy_ep{ep + 1}",
                )
                success_rate = np.mean(metrics["episode_successes"][-10:])
                avg_reward = np.mean(metrics["episode_rewards"][-10:])
                print(
                    f"Episode {ep + 1}/{num_episodes}: "
                    f"reward={avg_reward:.3f} success={success_rate:.0%}"
                )

        # Save final
        self._sampling_client = self._training_client.save_weights_and_get_sampling_client(
            name="rl_final",
        )

        # Save metadata
        meta = {
            "base_model": self.base_model,
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "rank": self.rank,
            "loss_fn": self.loss_fn,
            "num_episodes": num_episodes,
            "final_success_rate": float(np.mean(metrics["episode_successes"][-10:])),
            "final_avg_reward": float(np.mean(metrics["episode_rewards"][-10:])),
        }
        with open(self.output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Save reward curve
        with open(self.output_dir / "metrics.jsonl", "w") as f:
            for ep_idx in range(len(metrics["episode_rewards"])):
                entry = {
                    "episode": ep_idx,
                    "reward/total": metrics["episode_rewards"][ep_idx],
                    "success": metrics["episode_successes"][ep_idx],
                }
                if ep_idx < len(metrics["losses"]):
                    entry["loss"] = metrics["losses"][ep_idx]
                f.write(json.dumps(entry) + "\n")

        return metrics

    def _collect_rollout(self, env: Any) -> dict[str, Any]:
        """Run one episode, collecting actions and rewards."""
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        data = []  # (model_input_tokens, action_tokens, logprobs, reward)

        while not done:
            # Build prompt
            obs_text = obs if isinstance(obs, str) else json.dumps(obs)
            prompt = f"Observation: {obs_text}\nAction: "
            tokens = self.tokenizer.encode(prompt)
            model_input = tinker_types.ModelInput.from_ints(tokens=tokens)

            # Sample from policy
            result = self._sampling_client.sample(
                prompt=model_input,
                num_samples=1,
                sampling_params=tinker_types.SamplingParams(max_tokens=32),
            )

            seq = result.sequences[0]
            action_text = self.tokenizer.decode(seq.tokens).strip()

            # Parse action
            match = re.search(r"\b(\d+)\b", action_text)
            action_int = int(match.group(1)) if match else 0

            # Step env
            obs, reward, done_flag, truncated, info = env.step(action_int)
            done = done_flag or truncated
            total_reward += reward

            # Store rollout data
            data.append(
                {
                    "input_tokens": tokens,
                    "action_tokens": seq.tokens,
                    "logprobs": seq.logprobs if hasattr(seq, "logprobs") else None,
                    "reward": reward,
                }
            )

        return {
            "total_reward": total_reward,
            "success": info.get("success", False),
            "data": data,
        }

    def _update_policy(
        self,
        rollout_data: list[dict],
        advantage: float,
        learning_rate: float,
        clip_low: float,
        clip_high: float,
    ) -> float:
        """Perform a policy gradient update using rollout data."""
        datums = []
        for step_data in rollout_data:
            input_tokens = step_data["input_tokens"]
            action_tokens = step_data["action_tokens"]

            # Full sequence: input + action
            full_tokens = list(input_tokens) + list(action_tokens)
            target_tokens = [-100] * len(input_tokens) + list(action_tokens)
            target_tokens = target_tokens[1:]

            advantages = [0.0] * len(input_tokens) + [advantage] * len(action_tokens)
            advantages = advantages[1:]

            loss_fn_inputs: dict[str, Any] = {
                "target_tokens": tinker_types.TensorData.from_list(
                    target_tokens,
                    dtype="int64",
                ),
                "advantages": tinker_types.TensorData.from_list(
                    advantages,
                    dtype="float32",
                ),
            }

            if step_data.get("logprobs") is not None:
                loss_fn_inputs["logprobs"] = tinker_types.TensorData.from_list(
                    list(step_data["logprobs"]),
                    dtype="float32",
                )

            datums.append(
                tinker_types.Datum(
                    model_input=tinker_types.ModelInput.from_ints(tokens=full_tokens),
                    loss_fn_inputs=loss_fn_inputs,
                )
            )

        # Forward-backward with PPO or importance_sampling
        loss_fn_config = {}
        if self.loss_fn == "ppo":
            loss_fn_config = {
                "clip_low_threshold": clip_low,
                "clip_high_threshold": clip_high,
            }

        fwdbwd_kwargs: dict[str, Any] = {}
        if loss_fn_config:
            fwdbwd_kwargs["loss_fn_config"] = loss_fn_config
        fwdbwd_future = self._training_client.forward_backward(
            datums,
            loss_fn=self.loss_fn,
            **fwdbwd_kwargs,
        )
        optim_future = self._training_client.optim_step(
            tinker_types.AdamParams(learning_rate=learning_rate),
        )

        fwdbwd_result = fwdbwd_future.result()
        optim_future.result()

        # Compute average loss
        try:
            logprobs = np.concatenate(
                [out["logprobs"].tolist() for out in fwdbwd_result.loss_fn_outputs]
            )
            loss = float(-np.mean(logprobs))
        except Exception:
            loss = 0.0

        return loss

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    def as_agent(self) -> Any:
        """Wrap the RL-trained model as an agentick agent."""
        if self._sampling_client is None:
            raise RuntimeError("Must call train() first")
        from agentick.training.tinker.sft import TinkerSFTAgent

        return TinkerSFTAgent(
            sampling_client=self._sampling_client,
            tokenizer=self._tokenizer,
        )
