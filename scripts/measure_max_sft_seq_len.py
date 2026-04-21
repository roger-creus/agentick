"""Measure the maximum tokenized SFT sequence length across all tasks/difficulties.

Runs the oracle for ~5 steps per (task, difficulty), builds the full
SFT chat row, applies the Qwen3.5 tokenizer, and reports the max length.

Usage:
    uv run python scripts/measure_max_sft_seq_len.py
"""

from __future__ import annotations


def main():
    from transformers import AutoTokenizer

    import agentick
    from agentick.agents.prompt_templates import (
        SYSTEM_PROMPT,
        format_observation_to_text,
    )
    from agentick.oracles import get_oracle, list_oracles
    from agentick.tasks.descriptions import get_task_description

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)

    max_len = 0
    max_row = None
    for task in list_oracles():
        for diff in ["easy", "medium", "hard", "expert"]:
            try:
                env = agentick.make(task, difficulty=diff, render_mode="ascii")
            except Exception:
                continue
            try:
                oracle = get_oracle(task, env)
            except Exception:
                env.close()
                continue

            obs, info = env.reset(seed=0)
            oracle.reset(obs, info)

            for _ in range(5):
                ascii_render = env.unwrapped.render_in_mode("ascii")
                sys_msg = SYSTEM_PROMPT.format(task_description=get_task_description(task))
                user_msg = format_observation_to_text(
                    str(ascii_render),
                    {"task_name": task, "step": 0},
                    "ascii",
                )
                messages = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": "0"},
                ]
                rendered = tok.apply_chat_template(messages, tokenize=False)
                n = len(tok.encode(rendered))
                if n > max_len:
                    max_len = n
                    max_row = (task, diff)

                action = oracle.act(obs, info)
                obs, _, done, trunc, info = env.step(action)
                oracle.update(obs, info)
                if done or trunc:
                    break
            env.close()

    print(f"Max tokenized SFT sequence length: {max_len} tokens")
    print(f"Worst-case task/difficulty: {max_row}")


if __name__ == "__main__":
    main()
