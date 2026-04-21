"""Audit tests for examples/data_and_finetuning/sft_with_trl.py.

These tests verify the SFT pipeline produces training data that exactly
matches the eval-time prompt format and trains with assistant-only loss.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the example script importable as a module
SFT_DIR = Path(__file__).resolve().parents[2] / "examples" / "data_and_finetuning"
sys.path.insert(0, str(SFT_DIR))


def test_sft_module_imports():
    """Sanity: sft_with_trl.py imports without errors."""
    import sft_with_trl  # noqa: F401


def test_sft_build_chat_dataset_matches_eval_prompt():
    """End-to-end: SFT-built user content must match eval-time prompt byte-for-byte,
    including ANSI-color stripping for ASCII observations.
    """
    import sft_with_trl

    from agentick.agents.prompt_templates import format_observation_to_text

    # ASCII observation with ANSI color codes — the realistic dataset content
    raw_ascii = "\x1b[31m#\x1b[0m\x1b[32m.\x1b[0m.A\n....\n..G."
    task_name = "GoToGoal-v0"
    step = 7

    # Build a one-row fake HF dataset
    from datasets import Dataset
    fake_ds = Dataset.from_list([{
        "task": task_name,
        "step": step,
        "ascii_render": raw_ascii,
        "language_render": "ignored",
        "action_int": 4,
    }])

    chat_ds = sft_with_trl.build_chat_dataset(fake_ds, modality="ascii", max_steps_per_episode=None)
    assert len(chat_ds) == 1
    messages = chat_ds[0]["messages"]
    assert messages[1]["role"] == "user"
    sft_user = messages[1]["content"]

    # Eval would call format_observation_to_text on the same raw ASCII with ANSI codes
    info = {"task_name": task_name, "step": step}
    eval_user = format_observation_to_text(raw_ascii, info, "ascii")

    assert sft_user == eval_user, (
        f"SFT-built user content diverges from eval prompt:\n"
        f"SFT:\n{sft_user!r}\n\nEVAL:\n{eval_user!r}"
    )


def test_sft_system_prompt_matches_eval_system_prompt():
    """SFT SYSTEM_PROMPT must match the eval harness SYSTEM_PROMPT."""
    import sft_with_trl

    from agentick.agents.prompt_templates import SYSTEM_PROMPT as EVAL_SYSTEM

    assert sft_with_trl.SYSTEM_PROMPT == EVAL_SYSTEM, (
        "SFT SYSTEM_PROMPT drifted from agentick.agents.prompt_templates.SYSTEM_PROMPT — "
        "training/eval will see different system messages"
    )


def test_sft_config_uses_assistant_only_loss():
    """SFTConfig must be configured with assistant_only_loss (or equivalent).

    Either TRL's newer `assistant_only_loss=True` flag or explicit
    completion-only data collator must be in the training_args.
    """
    import inspect

    import sft_with_trl

    src = inspect.getsource(sft_with_trl.main)

    # Accept either the modern TRL kwarg or an explicit data collator.
    has_assistant_only = "assistant_only_loss=True" in src
    has_completion_collator = "DataCollatorForCompletionOnlyLM" in src

    assert has_assistant_only or has_completion_collator, (
        "SFT training must mask non-assistant tokens from the loss. "
        "Add `assistant_only_loss=True` to SFTConfig or pass a "
        "DataCollatorForCompletionOnlyLM to SFTTrainer."
    )
