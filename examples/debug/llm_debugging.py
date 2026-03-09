#!/usr/bin/env python3
"""LLM Debugging Tool — Record episodes and generate an interactive HTML viewer.

Captures multi-modal observations (ASCII, language, pixel renders) alongside the
full LLM prompt/response exchange for each step.

Usage:
    # Basic (ascii + isometric):
    uv run python examples/debug/llm_debugging.py \
        --task GoToGoal-v0 \
        --config examples/experiments/configs/qwen3_4b_ascii_nonmarkov_reasoner.yaml \
        --difficulty easy --n-episodes 5

    # All modalities:
    uv run python examples/debug/llm_debugging.py \
        --task GoToGoal-v0 \
        --config examples/experiments/configs/qwen3_4b_ascii_nonmarkov_reasoner.yaml \
        --difficulty easy --n-episodes 1 \
        --modalities ascii,language,rgb_array
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from _shared import (
    capture_renders,
    generate_html,
    renders_to_html_data,
    serialise_messages,
)

# ---------------------------------------------------------------------------
# Multimodal observation injection (mirrors ExperimentRunner._inject_secondary_obs)
# ---------------------------------------------------------------------------
_TEXT_MODES = ("language", "ascii", "language_structured")


def _inject_secondary_obs(
    env: Any, info: dict[str, Any], obs_modes: list[str],
) -> None:
    """Inject secondary text renderings into info for multimodal agents."""
    render_mode = getattr(env, "render_mode", None)
    for mode in obs_modes:
        if mode != render_mode and mode in _TEXT_MODES:
            info[f"obs_{mode}"] = env.render_in_mode(mode)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
def collect_episodes(
    task_name: str,
    config_path: str,
    difficulty: str,
    n_episodes: int,
    start_seed: int,
    modalities: list[str],
    harness_override: str | None = None,
) -> list[dict[str, Any]]:
    """Run episodes, capturing everything for debugging."""
    import agentick
    from agentick.agents.factory import create_agent
    from agentick.experiments.config import load_config

    config = load_config(config_path)

    # Allow CLI harness override (reuse same model with different harness)
    if harness_override:
        config.agent.hyperparameters["harness"] = harness_override

    harness_name = config.agent.hyperparameters.get("harness", "unknown")
    model_name = config.agent.hyperparameters.get("model", "unknown")

    print(f"Creating agent: {model_name} / {harness_name}")
    agent = create_agent(config.agent)
    if agent is None:
        print("ERROR: create_agent returned None (non-LLM agent type?)")
        sys.exit(1)

    # Determine primary render mode (same logic as ExperimentRunner):
    # prefer rgb_array if the agent uses it, so text modes come via render_in_mode
    obs_modes = agent.observation_modes
    if "rgb_array" in obs_modes:
        primary_render = "rgb_array"
    else:
        primary_render = obs_modes[0]
    print(f"Primary render mode: {primary_render}  (agent obs_modes: {obs_modes})")

    episodes_data: list[dict[str, Any]] = []

    for ep_idx in range(n_episodes):
        seed = start_seed + ep_idx
        print(f"\n{'='*60}")
        print(f"Episode {ep_idx + 1}/{n_episodes}  seed={seed}")
        print(f"{'='*60}")

        env = agentick.make(
            task_name, difficulty=difficulty, render_mode=primary_render, seed=seed,
        )
        obs, info = env.reset(seed=seed)
        _inject_secondary_obs(env, info, obs_modes)
        agent.reset()

        steps_data: list[dict[str, Any]] = []
        total_reward = 0.0
        done = False
        step_idx = 0

        while not done:
            # Capture all requested modalities for the HTML viewer
            raw_renders = capture_renders(env, obs, modalities)
            render_data = renders_to_html_data(raw_renders)

            # Capture the full messages sent to the LLM
            info_with_modes = {**info, "_obs_modes": obs_modes}
            messages = agent.harness.build_messages(obs, info_with_modes, obs_modes)
            messages_display = serialise_messages(messages)

            # Agent acts (info already has secondary obs injected)
            action = agent.act(obs, info)
            call = agent.call_log[-1]

            step_data: dict[str, Any] = {
                "step": step_idx,
                "messages": messages_display,
                "renders": render_data,
                "response": call["response"],
                "reasoning": call.get("reasoning"),
                "parsed_action": call["parsed_action"],
                "action_name": call["action_name"],
                "input_tokens": call["input_tokens"],
                "output_tokens": call["output_tokens"],
                "latency": round(call["latency"], 3),
            }

            # Step env and inject secondary obs for next iteration
            obs, reward, terminated, truncated, info = env.step(action)
            _inject_secondary_obs(env, info, obs_modes)

            step_data["reward"] = float(reward)
            total_reward += float(reward)
            step_data["cumulative_reward"] = round(total_reward, 4)
            step_data["done"] = terminated or truncated
            step_data["terminated"] = terminated
            step_data["truncated"] = truncated

            steps_data.append(step_data)
            step_idx += 1
            done = terminated or truncated

            action_str = call["action_name"]
            print(
                f"  step {step_idx:3d}  action={action_str:<12s}"
                f"  r={reward:.2f}  cum={total_reward:.2f}",
                end="",
            )
            if done:
                print(f"  {'SUCCESS' if info.get('success') else 'FAIL'}")
            else:
                print()

        success = bool(info.get("success", False))
        episodes_data.append({
            "episode": ep_idx,
            "seed": seed,
            "task": task_name,
            "difficulty": difficulty,
            "harness": harness_name,
            "model": model_name,
            "total_reward": round(total_reward, 4),
            "n_steps": len(steps_data),
            "success": success,
            "steps": steps_data,
        })

        env.close()

    return episodes_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="LLM Debugging — record episodes and generate an interactive HTML viewer"
    )
    parser.add_argument("--task", required=True, help="Task name (e.g. GoToGoal-v0)")
    parser.add_argument(
        "--config", required=True,
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--difficulty", default="easy",
        choices=["easy", "medium", "hard", "expert"],
    )
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument(
        "--start-seed", type=int, default=0,
        help="Starting seed (increments per episode)",
    )
    parser.add_argument(
        "--modalities", default="ascii,language,rgb_array",
        help="Comma-separated render modes to capture (ascii,language,rgb_array,rgb_array_flat)",
    )
    parser.add_argument(
        "--harness", default=None,
        help=(
            "Override the harness from the YAML config "
            "(markovian_zero_shot, non_markovian_zero_shot, "
            "markovian_reasoner, non_markovian_reasoner)"
        ),
    )
    parser.add_argument(
        "--output", default=None,
        help="Output HTML filename (auto-generated if omitted)",
    )
    args = parser.parse_args()

    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    output = args.output or f"debug_{args.task}_{args.difficulty}.html"

    print(f"Task:       {args.task}")
    print(f"Config:     {args.config}")
    print(f"Difficulty: {args.difficulty}")
    if args.harness:
        print(f"Harness:    {args.harness} (override)")
    print(f"Episodes:   {args.n_episodes}")
    print(f"Modalities: {modalities}")
    print(f"Seeds:      {args.start_seed}..{args.start_seed + args.n_episodes - 1}")
    print(f"Output:     {output}")

    episodes = collect_episodes(
        task_name=args.task,
        config_path=args.config,
        difficulty=args.difficulty,
        n_episodes=args.n_episodes,
        start_seed=args.start_seed,
        modalities=modalities,
        harness_override=args.harness,
    )

    generate_html(episodes, output, title="Agentick LLM Debugger")

    # Summary
    successes = sum(1 for e in episodes if e["success"])
    print(f"\nSummary: {successes}/{len(episodes)} episodes succeeded")
    for ep in episodes:
        status = "SUCCESS" if ep["success"] else "FAIL"
        print(
            f"  Ep {ep['episode']+1} (seed {ep['seed']}): {status}"
            f"  steps={ep['n_steps']}  reward={ep['total_reward']}"
        )


if __name__ == "__main__":
    main()
