"""Measure tokenized SFT sequence lengths on the actual HF datasets.

The SFT/eval formatter strips ANSI color codes before tokenization, so raw
`ascii_render` byte length substantially overstates model context length. This
script streams each split, keeps the longest raw ASCII rows, then tokenizes
those candidates with the same chat template used by training.

Usage:
    uv run python scripts/measure_max_sft_seq_len.py
"""

from __future__ import annotations

import argparse

DEFAULT_DATASETS = [
    "rogercc/agentick-oracle-trajectories-120k",
    "rogercc/agentick-oracle-trajectories-250k",
    "rogercc/agentick-oracle-trajectories-500k",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    from agentick.agents.prompt_templates import (
        SYSTEM_PROMPT,
        format_observation_to_text,
    )
    from agentick.tasks.descriptions import get_task_description

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    global_max: tuple[int, tuple] | None = None
    for repo in args.datasets:
        longest = []
        ds = load_dataset(repo, split=args.split, streaming=True)
        for i, example in enumerate(ds):
            raw_len = len(str(example["ascii_render"]))
            if len(longest) < args.top_k or raw_len > longest[0][0]:
                longest.append((raw_len, i, example))
                longest = sorted(longest, key=lambda x: x[0])[-args.top_k:]
            if (i + 1) % 100_000 == 0:
                print(f"{repo}: scanned {i + 1}, raw max={longest[-1][0]}", flush=True)

        repo_max: tuple[int, tuple] | None = None
        for raw_len, idx, example in longest:
            sys_msg = SYSTEM_PROMPT.format(
                task_description=get_task_description(example["task"])
            )
            user_msg = format_observation_to_text(
                str(example["ascii_render"]),
                {"task_name": example["task"], "step": example["step"]},
                "ascii",
            )
            rendered = tok.apply_chat_template(
                [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": str(example["action_int"])},
                ],
                tokenize=False,
            )
            n_tokens = len(tok.encode(rendered))
            item = (
                repo,
                idx,
                example["task"],
                example["difficulty"],
                example["step"],
                raw_len,
                len(user_msg),
            )
            if repo_max is None or n_tokens > repo_max[0]:
                repo_max = (n_tokens, item)
            if global_max is None or n_tokens > global_max[0]:
                global_max = (n_tokens, item)

        print(f"{repo}: max tokenized length = {repo_max[0]} | {repo_max[1]}")

    print(f"global max tokenized length = {global_max[0]} | {global_max[1]}")


if __name__ == "__main__":
    main()
