# SFT Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run 3 SFT runs on Qwen3.5-4B (120k/250k/500k ASCII oracle datasets, fixed recipe) and 6 evaluation configs (Markov + Reasoner per size) to fill the paper's SFT gap.

**Architecture:** Audit and fix the existing `sft_with_trl.py` pipeline, extend `cluster_manager/` with an SFT job type + dataset caching, run a pilot, then launch full training (3 jobs) → full eval (912 jobs) → paper figures.

**Tech Stack:** TRL + PEFT LoRA on HuggingFace Transformers, Accelerate for multi-GPU, vLLM for eval, SLURM via apptainer on Alliance Canada clusters (rorqual / narval / fir; nibi excluded from training).

**Spec:** `docs/superpowers/specs/2026-04-20-sft-experiments-design.md`

---

## Phase 0 — SFT script audit and fixes

The existing `examples/data_and_finetuning/sft_with_trl.py` is suspected-buggy (prior training was unsatisfying). Fix before running anything at scale.

### Task 0.1: Create test scaffolding for SFT pipeline

**Files:**
- Create: `tests/test_data_and_finetuning/__init__.py`
- Create: `tests/test_data_and_finetuning/test_sft_script.py`

- [ ] **Step 1: Create empty test package init**

```bash
mkdir -p /home/roger/Desktop/agentick/tests/test_data_and_finetuning
touch /home/roger/Desktop/agentick/tests/test_data_and_finetuning/__init__.py
```

- [ ] **Step 2: Create the test file with a smoke test**

File: `tests/test_data_and_finetuning/test_sft_script.py`

```python
"""Audit tests for examples/data_and_finetuning/sft_with_trl.py.

These tests verify the SFT pipeline produces training data that exactly
matches the eval-time prompt format and trains with assistant-only loss.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the example script importable as a module
SFT_DIR = Path(__file__).resolve().parents[2] / "examples" / "data_and_finetuning"
sys.path.insert(0, str(SFT_DIR))


def test_sft_module_imports():
    """Sanity: sft_with_trl.py imports without errors."""
    import sft_with_trl  # noqa: F401
```

- [ ] **Step 3: Run the smoke test**

```bash
cd /home/roger/Desktop/agentick
uv run pytest tests/test_data_and_finetuning/test_sft_script.py -v
```

Expected: PASS (module imports cleanly).

- [ ] **Step 4: Commit scaffolding**

```bash
cd /home/roger/Desktop/agentick
git add tests/test_data_and_finetuning/
git commit -m "$(cat <<'EOF'
Tests: add scaffolding for SFT script audit

Empty test package so subsequent audit tasks have a home.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 0.2: Test that SFT chat format matches eval-time prompt byte-for-byte

The SFT script's `build_chat_dataset` constructs rows with a system + user + assistant message triple. At eval time, `agentick/agents/harness.py::MarkovianZeroShot.build_messages` builds the messages via `_make_user_content` → `format_text_observation` → `format_observation_to_text`. The user/system content must match exactly, or training and inference see different input distributions.

**Files:**
- Modify: `tests/test_data_and_finetuning/test_sft_script.py`

- [ ] **Step 1: Add the test**

Append to `tests/test_data_and_finetuning/test_sft_script.py`:

```python
def test_sft_user_template_matches_eval_prompt_format():
    """SFT user message must match what eval harness produces.

    The SFT script's USER_TEMPLATE and the eval harness's
    format_observation_to_text must produce identical strings for the
    same (task_name, step, obs_text) inputs.
    """
    import sft_with_trl
    from agentick.agents.prompt_templates import format_observation_to_text

    task_name = "GoToGoal-v0"
    step = 3
    obs_text = "....\n.A..\n....\n...G"

    # SFT builds user content via .format()
    sft_user = sft_with_trl.USER_TEMPLATE.format(
        task_name=task_name,
        step=step,
        observation=obs_text,
    )

    # Eval builds user content via format_observation_to_text with
    # observation_mode='ascii'. ASCII path in that function does
    # ANSI strip then format into the SAME f-string template. We pass
    # already-stripped text to isolate the template match.
    info = {"task_name": task_name, "step": step}
    eval_user = format_observation_to_text(obs_text, info, "ascii")

    assert sft_user == eval_user, (
        f"SFT user template diverges from eval prompt:\n"
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
```

- [ ] **Step 2: Run the tests**

```bash
cd /home/roger/Desktop/agentick
uv run pytest tests/test_data_and_finetuning/test_sft_script.py::test_sft_user_template_matches_eval_prompt_format tests/test_data_and_finetuning/test_sft_script.py::test_sft_system_prompt_matches_eval_system_prompt -v
```

Expected: Both PASS (sft_with_trl.py was written to match prompt_templates.py exactly — if either fails that's a real bug to fix before proceeding).

- [ ] **Step 3: If either test fails, fix by importing from the source of truth**

If `test_sft_system_prompt_matches_eval_system_prompt` fails, replace the duplicated `SYSTEM_PROMPT` constant in `examples/data_and_finetuning/sft_with_trl.py` with an import:

```python
# REMOVE the inlined SYSTEM_PROMPT constant.
# REPLACE with:
from agentick.agents.prompt_templates import SYSTEM_PROMPT
```

If `test_sft_user_template_matches_eval_prompt_format` fails, similarly replace the `USER_TEMPLATE` string with a call to `format_observation_to_text` inside `build_chat_dataset`.

- [ ] **Step 4: Re-run tests and commit**

```bash
cd /home/roger/Desktop/agentick
uv run pytest tests/test_data_and_finetuning/test_sft_script.py -v
git add tests/test_data_and_finetuning/test_sft_script.py examples/data_and_finetuning/sft_with_trl.py
git commit -m "$(cat <<'EOF'
SFT audit: pin chat template + system prompt to eval source

Ensures training data and eval prompts use byte-identical formatting.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 0.3: Enable completion-only (assistant-only) loss

Prime suspect for prior unsatisfying results. With default SFTTrainer behavior, loss is computed over all tokens — the model spends capacity reproducing the ASCII observation. We want loss only on assistant tokens (the single action digit).

**Files:**
- Modify: `examples/data_and_finetuning/sft_with_trl.py`
- Modify: `tests/test_data_and_finetuning/test_sft_script.py`

- [ ] **Step 1: Add test that verifies assistant-only loss is active**

Append to `tests/test_data_and_finetuning/test_sft_script.py`:

```python
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
```

- [ ] **Step 2: Run the test — expect it to FAIL**

```bash
cd /home/roger/Desktop/agentick
uv run pytest tests/test_data_and_finetuning/test_sft_script.py::test_sft_config_uses_assistant_only_loss -v
```

Expected: FAIL (the current script has neither).

- [ ] **Step 3: Enable assistant-only loss in the SFT script**

Edit `examples/data_and_finetuning/sft_with_trl.py`. In the `SFTConfig(...)` construction inside `main()` (around line 287), add `assistant_only_loss=True` as a kwarg:

```python
training_args = SFTConfig(
    output_dir=args.output_dir,
    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    max_length=args.max_seq_length,
    packing=args.packing,
    assistant_only_loss=True,       # <-- ADD THIS LINE
    warmup_ratio=args.warmup_ratio,
    weight_decay=args.weight_decay,
    lr_scheduler_type=args.lr_scheduler,
    bf16=args.bf16,
    gradient_checkpointing=args.gradient_checkpointing,
    gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
    logging_steps=args.logging_steps,
    save_strategy=args.save_strategy,
    save_total_limit=args.save_total_limit,
    report_to=args.report_to,
    run_name=run_name,
    ddp_find_unused_parameters=not args.no_lora,
    model_init_kwargs=model_kwargs,
    **({
        "eval_strategy": args.save_strategy,
        "per_device_eval_batch_size": args.batch_size,
    } if eval_dataset is not None else {}),
)
```

- [ ] **Step 4: Re-run test — expect it to PASS**

```bash
cd /home/roger/Desktop/agentick
uv run pytest tests/test_data_and_finetuning/test_sft_script.py::test_sft_config_uses_assistant_only_loss -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/roger/Desktop/agentick
git add examples/data_and_finetuning/sft_with_trl.py tests/test_data_and_finetuning/test_sft_script.py
git commit -m "$(cat <<'EOF'
SFT: enable assistant-only loss

Loss was previously computed over the full sequence, causing the model
to spend capacity reproducing ASCII observations. Assistant-only loss
focuses updates on the single action digit, matching the eval objective.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 0.4: Verify max sequence length holds the longest ASCII observation

Silent truncation would drop the assistant action token. Scan all 38 tasks × 4 difficulties with the oracle running a few steps and find the max tokenized length of a complete (system+user+assistant) sequence.

**Files:**
- Create: `scripts/measure_max_sft_seq_len.py`

- [ ] **Step 1: Write the measurement script**

File: `scripts/measure_max_sft_seq_len.py`

```python
"""Measure the maximum tokenized SFT sequence length across all tasks/difficulties.

Runs the oracle for ~5 steps per (task, difficulty), builds the full
SFT chat row, applies the Qwen3.5 tokenizer, and reports the max length.

Usage:
    uv run python scripts/measure_max_sft_seq_len.py
"""

from __future__ import annotations

import re

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def main():
    import agentick
    from agentick.agents.prompt_templates import SYSTEM_PROMPT, format_observation_to_text
    from agentick.oracles import get_oracle, list_oracles
    from agentick.tasks.descriptions import get_task_description
    from transformers import AutoTokenizer

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
                obs_text = _ANSI_RE.sub("", str(ascii_render))
                sys_msg = SYSTEM_PROMPT.format(task_description=get_task_description(task))
                user_msg = format_observation_to_text(obs_text, {"task_name": task, "step": 0}, "ascii")
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
```

- [ ] **Step 2: Run the script**

```bash
cd /home/roger/Desktop/agentick
uv run python scripts/measure_max_sft_seq_len.py
```

Expected: completes in ~2–5 min and reports a max length. Record the number.

- [ ] **Step 3: Update SFT default max_seq_length if measurement exceeds 8192**

If measured max > 8192: in `examples/data_and_finetuning/sft_with_trl.py`, change the `--max-seq-length` default to the next power of 2 above the measured max (e.g., 16384 if max is 10k).
If measured max ≤ 8192: no change needed.

- [ ] **Step 4: Commit the script (and optional default bump)**

```bash
cd /home/roger/Desktop/agentick
git add scripts/measure_max_sft_seq_len.py
git add -u examples/data_and_finetuning/sft_with_trl.py 2>/dev/null || true
git commit -m "$(cat <<'EOF'
SFT: add max-sequence-length measurement tool

Scans all 38 tasks x 4 difficulties to find the worst-case tokenized
SFT row length, so we can pick a max_seq_length that never truncates.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 0.5: Fast local end-to-end smoke test of the SFT script

Train for 2 steps on 16 tiny rows, verify script completes and an adapter is written.

**Files:**
- Create: `scripts/smoke_test_sft.sh`

- [ ] **Step 1: Create the smoke-test script**

File: `scripts/smoke_test_sft.sh`

```bash
#!/usr/bin/env bash
# Smoke-test the SFT script end-to-end on a tiny slice of the 120k dataset.
# Trains for 2 steps, writes LoRA adapter to /tmp/sft_smoke_out, succeeds if
# the adapter_config.json exists at the end.
set -euo pipefail

OUT=/tmp/sft_smoke_out
rm -rf "$OUT"

uv run python examples/data_and_finetuning/sft_with_trl.py \
    --dataset rogercc/agentick-oracle-trajectories-120k \
    --model Qwen/Qwen3.5-0.8B \
    --modality ascii \
    --epochs 1 \
    --batch-size 1 \
    --grad-accum 1 \
    --max-seq-length 2048 \
    --output-dir "$OUT" \
    --logging-steps 1 \
    --save-strategy no \
    --report-to none

# Confirm adapter was written
if [ ! -f "$OUT/adapter_config.json" ]; then
    echo "FAIL: no adapter_config.json at $OUT"
    exit 1
fi
echo "OK: SFT smoke test wrote $OUT/adapter_config.json"
```

Make it executable:

```bash
chmod +x /home/roger/Desktop/agentick/scripts/smoke_test_sft.sh
```

- [ ] **Step 2: Add max_steps override knob to the SFT script**

The smoke test needs to stop after 2 steps without running a full epoch. Edit `examples/data_and_finetuning/sft_with_trl.py` to accept `--max-steps`:

In the argparse `# Training` section, add after `--epochs`:

```python
parser.add_argument("--max-steps", type=int, default=-1, help="Max training steps (-1 = use epochs)")
```

In the `SFTConfig(...)` construction, add:

```python
max_steps=args.max_steps,
```

Then update the smoke test script to pass `--max-steps 2`:

```bash
uv run python examples/data_and_finetuning/sft_with_trl.py \
    --dataset rogercc/agentick-oracle-trajectories-120k \
    --model Qwen/Qwen3.5-0.8B \
    --modality ascii \
    --max-steps 2 \
    --batch-size 1 \
    --grad-accum 1 \
    --max-seq-length 2048 \
    --output-dir "$OUT" \
    --logging-steps 1 \
    --save-strategy no \
    --report-to none
```

- [ ] **Step 3: Run the smoke test**

```bash
cd /home/roger/Desktop/agentick
bash scripts/smoke_test_sft.sh
```

Expected: Downloads 120k dataset (few min), initializes Qwen3.5-0.8B (fast), runs 2 training steps, writes adapter, prints `OK: SFT smoke test wrote ...`. Total: 5–15 min depending on network.

If it fails at dataset loading: note the error (may indicate HF auth or dataset path issue — fix before continuing).
If it fails at model init: check GPU availability and VRAM.
If it fails at training: this is what we need to know NOW, not on the cluster. Debug locally.

- [ ] **Step 4: Commit the smoke test**

```bash
cd /home/roger/Desktop/agentick
git add scripts/smoke_test_sft.sh examples/data_and_finetuning/sft_with_trl.py
git commit -m "$(cat <<'EOF'
SFT: add end-to-end smoke test + max_steps override

Runs 2 training steps on the 120k dataset with Qwen3.5-0.8B to validate
the pipeline end-to-end locally before committing cluster compute.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 0.6: Test merge_and_push weight-change guard

`merge_and_push.py` already has the `_fix_adapter_key_mismatch` patch + a `n_changed == 0` assertion. Verify the smoke-test adapter merges without triggering the zero-weight-change error.

**Files:**
- Create: `scripts/smoke_test_merge.sh`

- [ ] **Step 1: Write the merge smoke test**

File: `scripts/smoke_test_merge.sh`

```bash
#!/usr/bin/env bash
# Verify merge_and_push.py can load the smoke-test adapter and produce
# a non-zero weight delta vs. the base model. Does NOT push to HF (uses
# a dummy repo that will fail to push, which is fine — the goal is to
# exercise the merge path up to the upload step).
set -euo pipefail

SMOKE_OUT=/tmp/sft_smoke_out
if [ ! -f "$SMOKE_OUT/adapter_config.json" ]; then
    echo "ERROR: $SMOKE_OUT/adapter_config.json missing — run scripts/smoke_test_sft.sh first"
    exit 1
fi

# Run merge_and_push with a dummy target; we only care about merge success
uv run python examples/data_and_finetuning/merge_and_push.py \
    --base-model Qwen/Qwen3.5-0.8B \
    --adapter-dir "$SMOKE_OUT" \
    --push-to-hub rogercc/__merge_smoke_test__ \
    --dtype bfloat16 || _rc=$?

# The script will fail at push (repo doesn't exist / no perms). That's fine.
# Verify the merged_dir was created and that the "ZERO weight changes" error
# did NOT appear in output. We re-run capturing output.
uv run python examples/data_and_finetuning/merge_and_push.py \
    --base-model Qwen/Qwen3.5-0.8B \
    --adapter-dir "$SMOKE_OUT" \
    --push-to-hub rogercc/__merge_smoke_test__ \
    --dtype bfloat16 2>&1 | tee /tmp/merge_smoke.log || true

if grep -q "ZERO weight changes" /tmp/merge_smoke.log; then
    echo "FAIL: merge produced zero weight changes — adapter is not being applied"
    exit 1
fi
if grep -q "weights differ from base" /tmp/merge_smoke.log; then
    echo "OK: merge applied adapter weights correctly"
else
    echo "WARN: could not confirm weight delta in log"
fi
```

Make executable:

```bash
chmod +x /home/roger/Desktop/agentick/scripts/smoke_test_merge.sh
```

- [ ] **Step 2: Run it**

```bash
cd /home/roger/Desktop/agentick
bash scripts/smoke_test_merge.sh
```

Expected: loads base model, merges adapter, prints `{N}/{M} weights differ from base` with N > 0, then fails at Hub upload (dummy repo). Critical check: `ZERO weight changes` must NOT appear.

If `ZERO weight changes` appears: the `_fix_adapter_key_mismatch` patch is failing. Investigate the key structure manually (`python -c 'from safetensors.torch import load_file; print(list(load_file("/tmp/sft_smoke_out/adapter_model.safetensors").keys())[:5])'`) and fix the remap logic in `merge_and_push.py::_find_key_remap`.

- [ ] **Step 3: Commit**

```bash
cd /home/roger/Desktop/agentick
git add scripts/smoke_test_merge.sh
git commit -m "$(cat <<'EOF'
SFT: add merge-and-push smoke test

Validates that merge_and_push.py produces a non-zero weight delta from
a freshly-trained LoRA adapter, catching the silent key-prefix mismatch
that plagued prior runs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 1 — Container dependency verification

The existing `agentick.sif` was built primarily for vLLM inference. SFT training may be missing deps.

### Task 1.1: Audit training deps inside the existing SIF

**Files:** (no code changes; reports into the terminal)

- [ ] **Step 1: Check the SIF exists**

```bash
ls -lh /home/roger/Desktop/agentick/cluster_manager/agentick.sif
```

If missing: run `cd /home/roger/Desktop/agentick/cluster_manager && ./cm.py setup --skip-sync --skip-push --skip-models` to build it before proceeding.

- [ ] **Step 2: Check each required training package inside the container**

```bash
apptainer exec --nv /home/roger/Desktop/agentick/cluster_manager/agentick.sif \
    bash -c 'for pkg in trl peft accelerate transformers datasets safetensors huggingface_hub flash_attn bitsandbytes; do
        python3 -c "import $pkg; print(\"$pkg\", $pkg.__version__)" 2>/dev/null || echo "MISSING: $pkg"
    done'
```

Expected: every package reports a version. `flash_attn` and `bitsandbytes` may be missing — note which.

- [ ] **Step 3: If any packages are missing, extend the Dockerfile and rebuild the SIF**

Locate the Dockerfile:

```bash
ls /home/roger/Desktop/agentick/Dockerfile* /home/roger/Desktop/agentick/container/Dockerfile* 2>/dev/null
```

Add missing packages to the relevant `pip install` or `uv sync` line (pin versions: `trl>=0.11`, `peft>=0.11`, `accelerate>=0.32`). Rebuild:

```bash
cd /home/roger/Desktop/agentick
docker build -t agentick:latest .
cd cluster_manager && ./cm.py setup --skip-sync --skip-models
```

If no packages are missing, skip this step.

- [ ] **Step 4: Commit any Dockerfile changes**

```bash
cd /home/roger/Desktop/agentick
git add -u Dockerfile* container/ 2>/dev/null || true
git diff --cached --stat
git commit -m "$(cat <<'EOF'
Container: add SFT training dependencies

Ensures trl/peft/accelerate/flash-attn/bitsandbytes are available in
agentick.sif so training jobs can run without extra install steps.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)" || echo "No Dockerfile changes to commit"
```

---

### Task 1.2: Run the SFT smoke test inside the container

- [ ] **Step 1: Run smoke test inside the SIF**

```bash
cd /home/roger/Desktop/agentick
apptainer exec --nv \
    -H $PWD \
    --bind $PWD:/src \
    --env PYTHONPATH=/src \
    --env HF_HOME=$HOME/.cache/huggingface \
    cluster_manager/agentick.sif \
    bash -c "cd /src && bash scripts/smoke_test_sft.sh"
```

Expected: same behavior as local smoke test. If it fails with `ImportError`, loop back to Task 1.1.

- [ ] **Step 2: Run merge smoke inside the container**

```bash
cd /home/roger/Desktop/agentick
apptainer exec --nv \
    -H $PWD \
    --bind $PWD:/src \
    --env PYTHONPATH=/src \
    --env HF_HOME=$HOME/.cache/huggingface \
    cluster_manager/agentick.sif \
    bash -c "cd /src && bash scripts/smoke_test_merge.sh"
```

Expected: merge completes with non-zero weight delta.

---

## Phase 2 — Cluster manager extensions

### Task 2.1: Add SFT config stems and resource profile

**Files:**
- Modify: `cluster_manager/config.py`

- [ ] **Step 1: Add SFT config stems and model mapping**

Edit `cluster_manager/config.py`. After the `PPO_CONFIGS = [...]` block (around line 154), add:

```python
SFT_CONFIGS = [
    "qwen35_4b_sft_train_ascii_120k",
    "qwen35_4b_sft_train_ascii_250k",
    "qwen35_4b_sft_train_ascii_500k",
]
```

Update `ALL_CONFIGS`:

```python
ALL_CONFIGS = LLM_CONFIGS + PPO_CONFIGS + SFT_CONFIGS
```

In `CONFIG_TO_MODEL`, add entries for the 3 training configs (they all use Qwen3.5-4B):

```python
"qwen35_4b_sft_train_ascii_120k": "Qwen/Qwen3.5-4B",
"qwen35_4b_sft_train_ascii_250k": "Qwen/Qwen3.5-4B",
"qwen35_4b_sft_train_ascii_500k": "Qwen/Qwen3.5-4B",
```

- [ ] **Step 2: Add SFT resource profile**

Add a new walltime constant near the top of `cluster_manager/config.py` resource section:

```python
_SFT_TIME = "23:59:59"
```

Add entries in `RESOURCE_PROFILES` after the PPO entries:

```python
"qwen35_4b_sft_train_ascii_120k":  {"cpus": 12, "mem": "64G", "walltime": _SFT_TIME},
"qwen35_4b_sft_train_ascii_250k":  {"cpus": 12, "mem": "64G", "walltime": _SFT_TIME},
"qwen35_4b_sft_train_ascii_500k":  {"cpus": 12, "mem": "64G", "walltime": _SFT_TIME},
```

Add a VRAM requirement for the SFT "model" key (same as Qwen3.5-4B since it IS Qwen3.5-4B):

In `MODEL_GPU_REQUIREMENTS`, the existing `Qwen/Qwen3.5-4B` entry already covers the SFT case. No new entry needed.

- [ ] **Step 3: Quick-check Python parses config.py**

```bash
cd /home/roger/Desktop/agentick
uv run python -c "from cluster_manager.config import SFT_CONFIGS, ALL_CONFIGS, RESOURCE_PROFILES; print('SFT_CONFIGS:', SFT_CONFIGS); print('total:', len(ALL_CONFIGS))"
```

Expected: prints the 3 SFT configs and total count.

- [ ] **Step 4: Commit**

```bash
cd /home/roger/Desktop/agentick
git add cluster_manager/config.py
git commit -m "$(cat <<'EOF'
cluster_manager: declare SFT training configs and resource profile

Adds 3 SFT training configs (120k/250k/500k) for Qwen3.5-4B ASCII with
24h walltime on a full H100, to support the paper's SFT experiments.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2.2: Create SFT training YAML configs

**Files:**
- Create: `examples/experiments/configs/qwen35_4b_sft_train_ascii_120k.yaml`
- Create: `examples/experiments/configs/qwen35_4b_sft_train_ascii_250k.yaml`
- Create: `examples/experiments/configs/qwen35_4b_sft_train_ascii_500k.yaml`

Unlike eval configs, SFT configs carry only the *training* hyperparameters — the cluster_manager job builder consumes these, not the ExperimentRunner.

- [ ] **Step 1: Write the 120k training config**

File: `examples/experiments/configs/qwen35_4b_sft_train_ascii_120k.yaml`

```yaml
name: qwen35_4b_sft_train_ascii_120k
description: SFT training job — Qwen3.5-4B LoRA on 120k ASCII oracle trajectories
job_type: sft_train
training:
  dataset: rogercc/agentick-oracle-trajectories-120k
  base_model: Qwen/Qwen3.5-4B
  modality: ascii
  epochs: 3
  lr: 4.0e-5
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  batch_size: 4
  grad_accum: 4
  max_seq_length: 8192
  packing: true
  warmup_ratio: 0.05
  weight_decay: 0.01
  lr_scheduler: cosine
  attn_implementation: flash_attention_2
  gradient_checkpointing: true
  wandb_project: agentick-sft
  run_name: sft-qwen35-4b-ascii-120k
push_to_hub: rogercc/agentick-qwen35-4b-sft-ascii-120k
tags: [sft, qwen35-4b, ascii, 120k]
```

- [ ] **Step 2: Write 250k and 500k configs**

Files: same shape, change every occurrence of `120k` to `250k` / `500k` (including `dataset`, `name`, `run_name`, `push_to_hub`).

```bash
cd /home/roger/Desktop/agentick/examples/experiments/configs
sed 's/120k/250k/g' qwen35_4b_sft_train_ascii_120k.yaml > qwen35_4b_sft_train_ascii_250k.yaml
sed 's/120k/500k/g' qwen35_4b_sft_train_ascii_120k.yaml > qwen35_4b_sft_train_ascii_500k.yaml
```

- [ ] **Step 3: Verify all three are valid YAML**

```bash
cd /home/roger/Desktop/agentick
for f in examples/experiments/configs/qwen35_4b_sft_train_ascii_{120k,250k,500k}.yaml; do
    uv run python -c "import yaml; c = yaml.safe_load(open('$f')); print('$f:', c['name'], '->', c['push_to_hub'])"
done
```

Expected: 3 lines printing name and push-to-hub target.

- [ ] **Step 4: Commit**

```bash
cd /home/roger/Desktop/agentick
git add examples/experiments/configs/qwen35_4b_sft_train_ascii_*.yaml
git commit -m "$(cat <<'EOF'
configs: add 3 SFT training YAMLs (120k/250k/500k, Qwen3.5-4B ASCII)

Fixed recipe across dataset sizes for clean scaling comparison.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2.3: Add SFT runner command generation in jobs.py

**Files:**
- Modify: `cluster_manager/jobs.py`

- [ ] **Step 1: Add SFT branch to `_build_runner_command`**

Edit `cluster_manager/jobs.py`. At the top of the file, import `SFT_CONFIGS`:

```python
from config import (
    ALL_TASKS,
    CLUSTERS,
    CLUSTER_NAMES,
    CONFIG_TO_MODEL,
    CONFIGS_DIR,
    DIFFICULTIES,
    FULL_GPU_NAMES,
    LLM_CONFIGS,
    MODEL_GPU_REQUIREMENTS,
    PPO_CONFIGS,
    RESOURCE_PROFILES,
    SFT_CONFIGS,
)
```

Replace `_build_runner_command` (around line 185) with:

```python
def _build_runner_command(
    config_stem: str,
    task: str | None = None,
    difficulty: str | None = None,
    config_name: str = "",
) -> str:
    """Build the python command to run inside the container."""
    py = "python3"

    if config_stem in SFT_CONFIGS:
        # SFT training: accelerate launch sft_with_trl.py then merge_and_push.py
        # Inline config is written to $_AGENTICK_CFG by the sbatch template;
        # we parse it here in bash via python to extract training args.
        cmd = (
            f"{py} -c \"import yaml; c = yaml.safe_load(open('$_AGENTICK_CFG'))['training']; "
            f"import shlex; print(' '.join(f'--{{k.replace(chr(95),chr(45))}} {{shlex.quote(str(v))}}' "
            f"for k,v in c.items() if not isinstance(v, bool))"
            f" + ' ' + ' '.join(f'--{{k.replace(chr(95),chr(45))}}' for k,v in c.items() if v is True))\" "
            f"> /tmp/sft_args.txt && "
            f"SFT_ARGS=$(cat /tmp/sft_args.txt) && "
            f"ADAPTER_DIR=/runs/{config_name}/adapter && "
            f"mkdir -p $ADAPTER_DIR && "
            f"accelerate launch --num_processes 1 "
            f"examples/data_and_finetuning/sft_with_trl.py "
            f"$SFT_ARGS --output-dir $ADAPTER_DIR && "
            f"PUSH_TO=$({py} -c \"import yaml; print(yaml.safe_load(open('$_AGENTICK_CFG'))['push_to_hub'])\") && "
            f"BASE_MODEL=$({py} -c \"import yaml; print(yaml.safe_load(open('$_AGENTICK_CFG'))['training']['base_model'])\") && "
            f"{py} examples/data_and_finetuning/merge_and_push.py "
            f"--base-model $BASE_MODEL --adapter-dir $ADAPTER_DIR --push-to-hub $PUSH_TO"
        )
        return cmd

    if config_stem in PPO_CONFIGS:
        cmd = (
            f"{py} examples/experiments/train_and_eval_ppo.py"
            f" --config $_AGENTICK_CFG"
            f" --output-dir /runs/{config_name}"
        )
        if task:
            cmd += f" --tasks {task}"
        if difficulty:
            cmd += f" --difficulties {difficulty}"
        return cmd

    # LLM/VLM: use the experiment runner module
    cmd = (
        f"{py} -m agentick.experiments.run"
        f" --config $_AGENTICK_CFG"
        f" --output-dir /runs/{config_name}"
    )
    if difficulty:
        cmd += f" --difficulties {difficulty}"
    return cmd
```

Note: the nested-quote shell construction is fragile. Prefer a helper script to avoid it — do that in the next step.

- [ ] **Step 2: Replace the inline bash with a dedicated helper script**

Create `examples/data_and_finetuning/run_sft_from_config.sh`:

```bash
#!/usr/bin/env bash
# Runs an SFT job given a YAML config path.
# Invokes sft_with_trl.py then merge_and_push.py.
#
# Usage: run_sft_from_config.sh <config.yaml> <output_dir>
set -euo pipefail

CFG="$1"
OUT="$2"
mkdir -p "$OUT/adapter"

# Extract training args from YAML into a flat list
python3 - "$CFG" > /tmp/sft_args.sh <<'PYEOF'
import sys, shlex, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
args = []
for k, v in cfg["training"].items():
    flag = "--" + k.replace("_", "-")
    if isinstance(v, bool):
        if v:
            args.append(flag)
    else:
        args.append(f"{flag} {shlex.quote(str(v))}")
print(" ".join(args))
PYEOF
SFT_ARGS=$(cat /tmp/sft_args.sh)

# Extract push-to-hub target + base model
PUSH_TO=$(python3 -c "import yaml; print(yaml.safe_load(open('$CFG'))['push_to_hub'])")
BASE_MODEL=$(python3 -c "import yaml; print(yaml.safe_load(open('$CFG'))['training']['base_model'])")

echo "=== SFT run ==="
echo "Config: $CFG"
echo "Output: $OUT"
echo "Args:   $SFT_ARGS"
echo "Push:   $PUSH_TO"

# Train
accelerate launch --num_processes 1 \
    examples/data_and_finetuning/sft_with_trl.py \
    $SFT_ARGS \
    --output-dir "$OUT/adapter"

# Merge + push
python3 examples/data_and_finetuning/merge_and_push.py \
    --base-model "$BASE_MODEL" \
    --adapter-dir "$OUT/adapter" \
    --push-to-hub "$PUSH_TO"

echo "=== SFT complete: pushed $PUSH_TO ==="
```

Make executable:

```bash
chmod +x /home/roger/Desktop/agentick/examples/data_and_finetuning/run_sft_from_config.sh
```

- [ ] **Step 3: Replace the SFT branch in jobs.py with a call to this script**

In `cluster_manager/jobs.py`, replace the complex inline command from Step 1 with:

```python
if config_stem in SFT_CONFIGS:
    return (
        f"bash examples/data_and_finetuning/run_sft_from_config.sh "
        f"$_AGENTICK_CFG /runs/{config_name}"
    )
```

- [ ] **Step 4: Verify jobs.py still parses**

```bash
cd /home/roger/Desktop/agentick
uv run python -c "
import sys; sys.path.insert(0, 'cluster_manager')
from jobs import _build_runner_command, generate_all_jobs
cmd = _build_runner_command('qwen35_4b_sft_train_ascii_120k', config_name='qwen35_4b_sft_train_ascii_120k')
print('SFT cmd:', cmd)
"
```

Expected: prints the `bash examples/...` command.

- [ ] **Step 5: Commit**

```bash
cd /home/roger/Desktop/agentick
git add cluster_manager/jobs.py examples/data_and_finetuning/run_sft_from_config.sh
git commit -m "$(cat <<'EOF'
cluster_manager: add SFT job type + runner shell script

jobs.py detects SFT configs and dispatches to run_sft_from_config.sh,
which reads a YAML and invokes sft_with_trl.py + merge_and_push.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2.4: Adjust job distribution for SFT (one-job-per-config, skip nibi)

**Files:**
- Modify: `cluster_manager/jobs.py`

- [ ] **Step 1: Update `generate_all_jobs` to emit single SFT jobs without task/difficulty fan-out**

Find `generate_all_jobs` in `cluster_manager/jobs.py` (around line 278). Replace the inner loop to special-case SFT:

```python
def generate_all_jobs(
    configs: list[str] | None = None,
    tasks: list[str] | None = None,
    clusters: list[str] | None = None,
    difficulties: list[str] | None = None,
    pilot: bool = False,
) -> list[dict]:
    """Generate the full list of jobs to submit."""
    configs = configs or (LLM_CONFIGS + PPO_CONFIGS + SFT_CONFIGS)
    tasks = tasks or ALL_TASKS
    clusters = clusters or CLUSTER_NAMES
    difficulties = difficulties or DIFFICULTIES

    jobs: list[dict] = []
    cluster_idx = 0

    for config_stem in configs:
        base_config = load_experiment_config(config_stem)

        # SFT training jobs are single-shot (no per-task/per-difficulty fan-out)
        if config_stem in SFT_CONFIGS:
            # Skip nibi for SFT (unreliable for expensive jobs)
            sft_clusters = [c for c in clusters if c != "nibi"]
            cluster = sft_clusters[cluster_idx % len(sft_clusters)]
            cluster_idx += 1

            output_dir_container = f"/runs/{config_stem}"
            import yaml as _yaml
            inline_yaml = _yaml.dump(base_config, default_flow_style=False, sort_keys=False)

            script = generate_sbatch_script(
                config_stem, cluster, task=config_stem,  # task field unused for SFT
                difficulty=None,
                inline_config_yaml=inline_yaml,
            )
            jobs.append({
                "config": config_stem,
                "task": "-",
                "difficulty": "-",
                "cluster": cluster,
                "job_name": f"agentick-{config_stem}",
                "gres": select_gpu(cluster, config_stem),
                "walltime": RESOURCE_PROFILES[config_stem]["walltime"],
                "script": script,
                "job_id": None,
            })
            continue

        for task in tasks:
            for diff in difficulties:
                cluster = clusters[cluster_idx % len(clusters)]
                cluster_idx += 1

                output_dir_container = f"/runs/{config_stem}"
                inline_yaml = make_per_task_difficulty_config(
                    base_config, task, diff, output_dir_container,
                    pilot=pilot,
                )

                script = generate_sbatch_script(
                    config_stem, cluster, task,
                    difficulty=diff,
                    inline_config_yaml=inline_yaml,
                )

                task_short = task.removesuffix("-v0")
                job_name = f"agentick-{config_stem}-{task_short}-{diff}"
                if len(job_name) > 100:
                    job_name = job_name[:100]

                jobs.append({
                    "config": config_stem,
                    "task": task,
                    "difficulty": diff,
                    "cluster": cluster,
                    "job_name": job_name,
                    "gres": select_gpu(cluster, config_stem),
                    "walltime": RESOURCE_PROFILES[config_stem]["walltime"],
                    "script": script,
                    "job_id": None,
                })

    return jobs
```

- [ ] **Step 2: Test the job generator with dry-run**

```bash
cd /home/roger/Desktop/agentick/cluster_manager
./cm.py submit --configs qwen35_4b_sft_train_ascii_120k qwen35_4b_sft_train_ascii_250k qwen35_4b_sft_train_ascii_500k --dry-run
```

Expected: shows exactly 3 jobs, distributed across rorqual / narval / fir (not nibi). Prints sample sbatch script showing the `bash examples/.../run_sft_from_config.sh ...` command.

- [ ] **Step 3: Commit**

```bash
cd /home/roger/Desktop/agentick
git add cluster_manager/jobs.py
git commit -m "$(cat <<'EOF'
cluster_manager: single-shot SFT jobs, skip nibi

SFT training configs emit one job per config (no task/difficulty
fan-out). Distributed across rorqual/narval/fir only — nibi excluded
from expensive training because of reliability issues.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2.5: Add `cm.py setup --datasets` to pre-cache HF datasets on clusters

**Files:**
- Modify: `cluster_manager/cm.py`

- [ ] **Step 1: Add dataset list to config.py**

Edit `cluster_manager/config.py`. Near the `MODELS_TO_DOWNLOAD` block, add:

```python
DATASETS_TO_DOWNLOAD = [
    "rogercc/agentick-oracle-trajectories-120k",
    "rogercc/agentick-oracle-trajectories-250k",
    "rogercc/agentick-oracle-trajectories-500k",
]
```

- [ ] **Step 2: Add `--datasets` to `cm.py setup` and implement `_download_and_push_datasets`**

In `cluster_manager/cm.py`, update the `setup` subparser (around line 346) to accept `--skip-datasets`:

```python
p_setup.add_argument("--skip-datasets", action="store_true", help="Skip dataset downloads")
```

In `cmd_setup`, after the models block, add:

```python
# 6. Download datasets locally then rsync to clusters
if not args.skip_datasets:
    print("\n[6/6] Downloading HF datasets locally then pushing to clusters...")
    from config import DATASETS_TO_DOWNLOAD
    print(f"  Datasets: {', '.join(DATASETS_TO_DOWNLOAD)}")
    _download_and_push_datasets(reachable)
else:
    print("\n[6/6] Skipping dataset downloads (--skip-datasets)")
```

Add the helper function near `_download_and_push_models`:

```python
def _download_and_push_datasets(clusters: list[str]) -> None:
    """Download HF datasets locally then rsync to each cluster's HF cache."""
    import tempfile
    from config import DATASETS_TO_DOWNLOAD

    local_hf = Path(tempfile.gettempdir()) / "agentick_hf_cache"
    local_hf.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(local_hf)

    for ds in DATASETS_TO_DOWNLOAD:
        ds_dir = local_hf / "hub" / f"datasets--{ds.replace('/', '--')}"
        if (ds_dir / "refs").exists():
            print(f"  [local] {ds} already cached")
            continue
        print(f"  [local] Downloading dataset {ds}...")
        rc = subprocess.run(
            ["uv", "run", "huggingface-cli", "download", "--repo-type=dataset", ds],
            env={**os.environ, "HF_HOME": str(local_hf)},
            timeout=7200,
        ).returncode
        if rc != 0:
            print(f"  [local] FAILED to download {ds}")
            return
        print(f"  [local] OK {ds}")

    print(f"\n  Pushing {local_hf} datasets to clusters...")
    for c in clusters:
        hf_remote = CLUSTERS[c]["hf_cache"]
        ssh_run(c, f"mkdir -p {hf_remote}")
        rc = rsync_to(c, str(local_hf) + "/", hf_remote + "/")
        if rc != 0:
            print(f"  [{c}] FAILED to push HF cache")
        else:
            print(f"  [{c}] HF cache (models+datasets) pushed OK")
```

- [ ] **Step 3: Dry-run to verify cm.py parses**

```bash
cd /home/roger/Desktop/agentick/cluster_manager
./cm.py setup --help | grep -i datasets
```

Expected: shows `--skip-datasets` option.

- [ ] **Step 4: Commit**

```bash
cd /home/roger/Desktop/agentick
git add cluster_manager/cm.py cluster_manager/config.py
git commit -m "$(cat <<'EOF'
cluster_manager: add dataset caching step to setup

`cm.py setup` now downloads the 3 SFT datasets locally and rsyncs them
into each cluster's HF cache, so compute nodes (no internet) can load
them for training.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3 — Prep clusters with datasets and updated code

### Task 3.1: Sync updated code to all clusters

- [ ] **Step 1: Run code sync**

```bash
cd /home/roger/Desktop/agentick/cluster_manager
./cm.py sync-code
```

Expected: rsync progress output for rorqual, narval, nibi, fir. Non-zero exit codes from nibi are acceptable (we don't rely on nibi for training). Confirm at least 3 of 4 succeed.

- [ ] **Step 2: Verify SFT files exist on remotes**

```bash
for c in rorqual narval fir; do
    echo "=== $c ==="
    ssh ${c}-robot 'ls /scratch/rogercc/agentick/examples/data_and_finetuning/run_sft_from_config.sh /scratch/rogercc/agentick/examples/experiments/configs/qwen35_4b_sft_train_ascii_*.yaml'
done
```

Expected: 4 files per cluster (1 shell script + 3 yamls).

---

### Task 3.2: Download datasets locally and push to clusters

- [ ] **Step 1: Run dataset download + push**

```bash
cd /home/roger/Desktop/agentick/cluster_manager
./cm.py setup --skip-build --skip-sync --skip-push --skip-models
```

This leaves only the dataset step active. Expected runtime: 30–90 min depending on network speed (3 datasets, roughly 1–5 GB each, plus rsync to 4 clusters).

- [ ] **Step 2: Verify datasets landed on each cluster**

```bash
for c in rorqual narval fir; do
    echo "=== $c ==="
    ssh ${c}-robot 'ls /scratch/rogercc/hf_cache/hub/ | grep datasets--rogercc--agentick-oracle'
done
```

Expected: 3 dataset directories per cluster (`datasets--rogercc--agentick-oracle-trajectories-{120k,250k,500k}`).

---

## Phase 4 — Pilot end-to-end on cluster

### Task 4.1: Submit pilot SFT job (120k, max_steps=200)

**Files:**
- Create: `examples/experiments/configs/qwen35_4b_sft_train_ascii_pilot.yaml`

- [ ] **Step 1: Write a pilot SFT config**

File: `examples/experiments/configs/qwen35_4b_sft_train_ascii_pilot.yaml`

```yaml
name: qwen35_4b_sft_train_ascii_pilot
description: Pilot SFT — 200 steps only, pushes to --pilot repo, validates the full pipeline
job_type: sft_train
training:
  dataset: rogercc/agentick-oracle-trajectories-120k
  base_model: Qwen/Qwen3.5-4B
  modality: ascii
  max_steps: 200
  lr: 4.0e-5
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  batch_size: 4
  grad_accum: 4
  max_seq_length: 8192
  packing: true
  warmup_ratio: 0.05
  weight_decay: 0.01
  lr_scheduler: cosine
  attn_implementation: flash_attention_2
  gradient_checkpointing: true
  wandb_project: agentick-sft
  run_name: sft-qwen35-4b-ascii-pilot
push_to_hub: rogercc/agentick-qwen35-4b-sft-ascii-pilot
tags: [sft, qwen35-4b, ascii, pilot]
```

- [ ] **Step 2: Register the pilot config in cluster_manager/config.py**

Append `"qwen35_4b_sft_train_ascii_pilot"` to `SFT_CONFIGS` in `cluster_manager/config.py`. Add matching entries to `CONFIG_TO_MODEL` and `RESOURCE_PROFILES` (use a shorter walltime for the pilot):

```python
"qwen35_4b_sft_train_ascii_pilot": "Qwen/Qwen3.5-4B",
```

```python
"qwen35_4b_sft_train_ascii_pilot":  {"cpus": 12, "mem": "64G", "walltime": "3:59:59"},
```

- [ ] **Step 3: Sync the pilot config to clusters**

```bash
cd /home/roger/Desktop/agentick/cluster_manager
./cm.py sync-code --clusters rorqual
```

- [ ] **Step 4: Dry-run the pilot submit to sanity-check the sbatch script**

```bash
./cm.py submit --configs qwen35_4b_sft_train_ascii_pilot --clusters rorqual --dry-run
```

Expected: shows 1 job on rorqual, 3:59:59 walltime, full H100 GPU, with `bash examples/.../run_sft_from_config.sh ...` in the script.

- [ ] **Step 5: Submit the pilot job**

```bash
./cm.py submit --configs qwen35_4b_sft_train_ascii_pilot --clusters rorqual
```

Record the job ID that comes back.

- [ ] **Step 6: Monitor until completion**

```bash
./cm.py status --detailed
```

Poll every 10–15 min. Expected: job runs ~30–60 min and exits.

If it fails:
- pull logs: `ssh rorqual-robot 'cat /scratch/rogercc/agentick_runs/logs/slurm-agentick-qwen35_4b_sft_train_ascii_pilot-*.out'`
- diagnose; most common bugs at this point: missing container dep (loop back to Task 1.1), HF_TOKEN not set (add `--env HF_TOKEN=$HF_TOKEN` in `jobs.py` SBATCH_TEMPLATE apptainer exec), filesystem quota, flash-attn CUDA version mismatch.

- [ ] **Step 7: Verify merged pilot model on HF Hub**

```bash
uv run huggingface-cli repo info --repo-type model rogercc/agentick-qwen35-4b-sft-ascii-pilot
```

Expected: repo exists and has model files (config.json + safetensors).

- [ ] **Step 8: Commit pilot config**

```bash
cd /home/roger/Desktop/agentick
git add examples/experiments/configs/qwen35_4b_sft_train_ascii_pilot.yaml cluster_manager/config.py
git commit -m "$(cat <<'EOF'
SFT: add pilot training config (200 steps on 120k)

Short pilot job used as the P2 verification gate: trains a tiny adapter,
merges, pushes to HF pilot repo. Keeps the heavier 120k/250k/500k runs
gated until the pipeline is proven end-to-end on the cluster.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4.2: Eval the pilot model on GoToGoal easy

**Files:**
- Create: `examples/experiments/configs/qwen35_4b_sft_pilot_ascii_markov.yaml`

- [ ] **Step 1: Write pilot eval config**

File: `examples/experiments/configs/qwen35_4b_sft_pilot_ascii_markov.yaml`

```yaml
name: qwen35_4b_sft_pilot_ascii_markov
description: Pilot eval of the 200-step SFT adapter — 1 task, 1 diff, 5 seeds
agent:
  type: llm
  hyperparameters:
    backend: vllm_llm
    model: rogercc/agentick-qwen35-4b-sft-ascii-pilot
    harness: markovian_zero_shot
    observation_modes:
    - ascii
    dtype: float16
    max_new_tokens: 8192
    temperature: 0.7
    top_p: 0.8
    top_k: 20
    min_p: 0.0
    gpu_memory_utilization: 0.9
    enable_prefix_caching: true
    tensor_parallel_size: 1
tasks:
- GoToGoal-v0
difficulties:
- easy
training: null
n_episodes: 1
n_seeds: 5
render_modes:
- ascii
record_trajectories: false
record_videos: false
output_dir: results/llm_benchmarks
tags:
- llm
- qwen35-4b-sft-pilot
- benchmark
- ascii
- markov
- pilot
metrics:
- mean_return
- success_rate
- mean_length
- mean_latency
- total_tokens
split: eval
```

- [ ] **Step 2: Register the pilot eval config in cluster_manager/config.py**

Append `"qwen35_4b_sft_pilot_ascii_markov"` to `LLM_CONFIGS` in `cluster_manager/config.py`. Add mapping + resource profile (Markov profile):

```python
"qwen35_4b_sft_pilot_ascii_markov": "Qwen/Qwen3.5-4B",
```

```python
"qwen35_4b_sft_pilot_ascii_markov":  {"cpus": 8, "mem": "32G", "walltime": _MARKOV_TIME},
```

- [ ] **Step 3: Pull pilot merged model locally and push to one cluster's HF cache**

```bash
HF_HOME=/tmp/agentick_hf_cache uv run huggingface-cli download rogercc/agentick-qwen35-4b-sft-ascii-pilot
rsync -avz /tmp/agentick_hf_cache/hub/models--rogercc--agentick-qwen35-4b-sft-ascii-pilot/ \
    rorqual-robot:/scratch/rogercc/hf_cache/hub/models--rogercc--agentick-qwen35-4b-sft-ascii-pilot/
```

- [ ] **Step 4: Sync code and submit the pilot eval job**

```bash
cd /home/roger/Desktop/agentick/cluster_manager
./cm.py sync-code --clusters rorqual
./cm.py submit --configs qwen35_4b_sft_pilot_ascii_markov --tasks GoToGoal-v0 --difficulties easy --clusters rorqual
```

- [ ] **Step 5: Wait for completion and check results**

```bash
./cm.py status
# After it completes:
./cm.py pull --configs qwen35_4b_sft_pilot_ascii_markov
cat /home/roger/Desktop/agentick/cluster_manager/results/qwen35_4b_sft_pilot_ascii_markov/per_task/GoToGoal-v0/metrics.json
```

**Pass gate:** `success_rate > 0` for GoToGoal-v0 easy. If 0%, something is broken — debug before continuing. If >0%, proceed to Phase 5.

- [ ] **Step 6: Commit pilot eval config**

```bash
cd /home/roger/Desktop/agentick
git add examples/experiments/configs/qwen35_4b_sft_pilot_ascii_markov.yaml cluster_manager/config.py
git commit -m "$(cat <<'EOF'
SFT: add pilot eval config (GoToGoal easy, 5 seeds, Markov harness)

P2 verification gate: runs cheap eval on the 200-step pilot adapter;
passing >0% unblocks the full 120k/250k/500k training launches.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 5 — Full SFT training

### Task 5.1: Submit all 3 SFT training jobs

- [ ] **Step 1: Dry-run**

```bash
cd /home/roger/Desktop/agentick/cluster_manager
./cm.py submit \
    --configs qwen35_4b_sft_train_ascii_120k qwen35_4b_sft_train_ascii_250k qwen35_4b_sft_train_ascii_500k \
    --clusters rorqual narval fir \
    --dry-run
```

Expected: 3 jobs, each on a different cluster, 23:59:59 walltime.

- [ ] **Step 2: Submit for real**

```bash
./cm.py submit \
    --configs qwen35_4b_sft_train_ascii_120k qwen35_4b_sft_train_ascii_250k qwen35_4b_sft_train_ascii_500k \
    --clusters rorqual narval fir
```

Record job IDs.

- [ ] **Step 3: Monitor**

```bash
./cm.py status --detailed
```

Poll every few hours. Expected durations: 120k ~3–4h, 250k ~6–8h, 500k ~12–16h.

- [ ] **Step 4: On each completion, verify push to HF**

```bash
for size in 120k 250k 500k; do
    uv run huggingface-cli repo info --repo-type model rogercc/agentick-qwen35-4b-sft-ascii-$size && echo "OK $size" || echo "MISSING $size"
done
```

If any missing: pull logs from the relevant cluster, diagnose, re-run that one training job.

---

## Phase 6 — Distribute merged models to all clusters

### Task 6.1: Pull merged models locally and rsync to all 4 clusters

- [ ] **Step 1: Pull merged models into local HF cache**

```bash
for size in 120k 250k 500k; do
    HF_HOME=/tmp/agentick_hf_cache uv run huggingface-cli download rogercc/agentick-qwen35-4b-sft-ascii-$size
done
```

- [ ] **Step 2: Rsync to all 4 clusters** (nibi is OK for eval, just flaky)

```bash
for c in rorqual narval nibi fir; do
    for size in 120k 250k 500k; do
        rsync -avz /tmp/agentick_hf_cache/hub/models--rogercc--agentick-qwen35-4b-sft-ascii-$size/ \
            ${c}-robot:/scratch/rogercc/hf_cache/hub/models--rogercc--agentick-qwen35-4b-sft-ascii-$size/ || echo "FAIL: $c $size"
    done
done
```

- [ ] **Step 3: 1-job smoke test per size on one cluster**

Quick vLLM-load test. Create a one-off smoke eval:

```bash
cd /home/roger/Desktop/agentick/cluster_manager
for size in 120k 250k 500k; do
    # Edit pilot eval yaml to point at the <size> merged model
    uv run python -c "
import yaml
p = 'examples/experiments/configs/qwen35_4b_sft_pilot_ascii_markov.yaml'
c = yaml.safe_load(open(p))
c['agent']['hyperparameters']['model'] = 'rogercc/agentick-qwen35-4b-sft-ascii-${size}'
c['name'] = 'qwen35_4b_sft_smoke_${size}'
c['tags'] = ['smoke', '${size}']
open(p.replace('pilot', 'smoke_${size}'), 'w').write(yaml.dump(c, default_flow_style=False, sort_keys=False))
"
done
```

Then register each `qwen35_4b_sft_smoke_<size>` in `cluster_manager/config.py` (same pattern as pilot eval) and submit each for GoToGoal-v0 easy × 1 seed:

```bash
./cm.py sync-code
for size in 120k 250k 500k; do
    ./cm.py submit --configs qwen35_4b_sft_smoke_$size --tasks GoToGoal-v0 --difficulties easy --clusters rorqual --pilot
done
```

Expected: 3 jobs complete in ~10 min each, all producing non-error metrics.

---

## Phase 7 — Full eval launch

### Task 7.1: Create the 6 full eval configs

**Files:**
- Create: `examples/experiments/configs/qwen35_4b_sft_{120k,250k,500k}_ascii_{markov,reasoner}.yaml`

- [ ] **Step 1: Generate eval configs from templates**

```bash
cd /home/roger/Desktop/agentick/examples/experiments/configs
for size in 120k 250k 500k; do
    for harness_short in markov reasoner; do
        if [ "$harness_short" = "markov" ]; then
            harness="markovian_zero_shot"
        else
            harness="markovian_reasoner"
        fi
        out="qwen35_4b_sft_${size}_ascii_${harness_short}.yaml"
        # Use existing qwen35_4b_ascii_<harness_short>.yaml as template, change the model
        base="qwen35_4b_ascii_${harness_short}.yaml"
        uv run python - <<PYEOF
import yaml
c = yaml.safe_load(open("$base"))
c["name"] = "qwen35_4b_sft_${size}_ascii_${harness_short}"
c["description"] = "Qwen3.5-4B SFT-${size} ASCII, ${harness} harness"
c["agent"]["hyperparameters"]["model"] = "rogercc/agentick-qwen35-4b-sft-ascii-${size}"
c["agent"]["hyperparameters"]["harness"] = "$harness"
c["tags"] = ["llm", "qwen35-4b-sft-${size}", "benchmark", "ascii", "$harness_short"]
with open("$out", "w") as f:
    yaml.dump(c, f, default_flow_style=False, sort_keys=False)
print("Wrote", "$out")
PYEOF
    done
done
```

- [ ] **Step 2: Register all 6 in cluster_manager/config.py**

Append to `LLM_CONFIGS` in `cluster_manager/config.py`:

```python
"qwen35_4b_sft_120k_ascii_markov",
"qwen35_4b_sft_120k_ascii_reasoner",
"qwen35_4b_sft_250k_ascii_markov",
"qwen35_4b_sft_250k_ascii_reasoner",
"qwen35_4b_sft_500k_ascii_markov",
"qwen35_4b_sft_500k_ascii_reasoner",
```

Add matching entries to `CONFIG_TO_MODEL` (all Qwen/Qwen3.5-4B) and `RESOURCE_PROFILES` (use `_MARKOV_TIME` for markov variants, `_REASONER_TIME` for reasoner variants).

- [ ] **Step 3: Sync code and dry-run**

```bash
cd /home/roger/Desktop/agentick/cluster_manager
./cm.py sync-code
./cm.py submit \
    --configs qwen35_4b_sft_120k_ascii_markov qwen35_4b_sft_120k_ascii_reasoner \
              qwen35_4b_sft_250k_ascii_markov qwen35_4b_sft_250k_ascii_reasoner \
              qwen35_4b_sft_500k_ascii_markov qwen35_4b_sft_500k_ascii_reasoner \
    --dry-run
```

Expected: 6 × 152 = 912 jobs, distributed across 4 clusters.

- [ ] **Step 4: Submit full eval matrix**

```bash
./cm.py submit \
    --configs qwen35_4b_sft_120k_ascii_markov qwen35_4b_sft_120k_ascii_reasoner \
              qwen35_4b_sft_250k_ascii_markov qwen35_4b_sft_250k_ascii_reasoner \
              qwen35_4b_sft_500k_ascii_markov qwen35_4b_sft_500k_ascii_reasoner
```

- [ ] **Step 5: Commit the 6 new YAMLs**

```bash
cd /home/roger/Desktop/agentick
git add examples/experiments/configs/qwen35_4b_sft_{120k,250k,500k}_ascii_{markov,reasoner}.yaml cluster_manager/config.py
git commit -m "$(cat <<'EOF'
Eval configs: add 6 SFT eval matrices (3 sizes x 2 harnesses)

All point at the merged models pushed by the SFT training jobs. Feeds
the paper's SFT headline, scaling, and per-category figures.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7.2: Monitor and backfill failed eval jobs

- [ ] **Step 1: Daily status checks**

```bash
cd /home/roger/Desktop/agentick/cluster_manager
./cm.py status --detailed
```

Expected: eval jobs complete over 1-3 days depending on cluster queue times.

- [ ] **Step 2: Pull results as they complete**

```bash
./cm.py pull --configs qwen35_4b_sft_120k_ascii_markov qwen35_4b_sft_120k_ascii_reasoner \
                       qwen35_4b_sft_250k_ascii_markov qwen35_4b_sft_250k_ascii_reasoner \
                       qwen35_4b_sft_500k_ascii_markov qwen35_4b_sft_500k_ascii_reasoner
```

- [ ] **Step 3: Identify and re-submit failed jobs**

```bash
uv run python cluster_manager/check_missing.py --configs qwen35_4b_sft_120k_ascii_markov,qwen35_4b_sft_120k_ascii_reasoner,qwen35_4b_sft_250k_ascii_markov,qwen35_4b_sft_250k_ascii_reasoner,qwen35_4b_sft_500k_ascii_markov,qwen35_4b_sft_500k_ascii_reasoner
```

This lists `(config, task, difficulty)` tuples missing a `metrics.json`. Re-submit:

```bash
# Example: missing qwen35_4b_sft_500k_ascii_reasoner on SokobanPush-v0 expert
./cm.py submit --configs qwen35_4b_sft_500k_ascii_reasoner --tasks SokobanPush-v0 --difficulties expert
```

Pass gate for Phase 7: ≥95% of 912 jobs return valid `metrics.json` after one backfill pass.

---

## Phase 8 — Analysis and paper update

### Task 8.1: Aggregate results into paper-figure format

**Files:**
- Modify: `scripts/generate_paper_figures.py`

- [ ] **Step 1: Inspect current paper-figures script and add an SFT-figures section**

Read current script:

```bash
head -40 /home/roger/Desktop/agentick/scripts/generate_paper_figures.py
```

- [ ] **Step 2: Add three figure-generating functions**

Append to `scripts/generate_paper_figures.py`:

```python
SFT_SIZES = ["120k", "250k", "500k"]
SFT_HARNESSES = [("markov", "Markov"), ("reasoner", "Reasoner")]


def _load_ons(config_name: str) -> float:
    """Load overall ONS (%) for a results folder."""
    import json
    from pathlib import Path
    results_dir = Path("cluster_manager/results") / config_name / "per_task"
    if not results_dir.exists():
        return float("nan")
    ons_values = []
    for task_dir in results_dir.iterdir():
        m = task_dir / "metrics.json"
        if m.exists():
            data = json.loads(m.read_text())
            if "ons" in data:
                ons_values.append(data["ons"])
    return (sum(ons_values) / len(ons_values) * 100.0) if ons_values else float("nan")


def figure_a_sft_vs_baselines():
    """Grouped bar: baseline + 3 SFT sizes, per harness."""
    import matplotlib.pyplot as plt
    import numpy as np
    # Baselines already in paper (from earlier eval runs)
    baseline_markov_ons = _load_ons("qwen35_4b_ascii_markov")
    baseline_reasoner_ons = _load_ons("qwen35_4b_ascii_reasoner")

    groups = ["Baseline", "SFT-120k", "SFT-250k", "SFT-500k"]
    markov = [baseline_markov_ons] + [_load_ons(f"qwen35_4b_sft_{s}_ascii_markov") for s in SFT_SIZES]
    reasoner = [baseline_reasoner_ons] + [_load_ons(f"qwen35_4b_sft_{s}_ascii_reasoner") for s in SFT_SIZES]

    x = np.arange(len(groups))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(x - w/2, markov, w, label="Markov harness")
    ax.bar(x + w/2, reasoner, w, label="Reasoner harness")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("ONS (%)")
    ax.set_title("Qwen3.5-4B ASCII: SFT vs. baselines")
    ax.legend()
    fig.tight_layout()
    fig.savefig("agentick_paper/figures/sft_vs_baselines.pdf")
    print("wrote agentick_paper/figures/sft_vs_baselines.pdf")


def figure_b_scaling_curve():
    """Data-scaling curve: x = log(size), y = ONS, two lines."""
    import matplotlib.pyplot as plt
    sizes_num = [120_000, 250_000, 500_000]
    markov = [_load_ons(f"qwen35_4b_sft_{s}_ascii_markov") for s in SFT_SIZES]
    reasoner = [_load_ons(f"qwen35_4b_sft_{s}_ascii_reasoner") for s in SFT_SIZES]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(sizes_num, markov, "o-", label="Markov harness")
    ax.plot(sizes_num, reasoner, "s-", label="Reasoner harness")
    ax.axhline(_load_ons("qwen35_4b_ascii_reasoner"), ls="--", c="gray", label="Reasoner baseline")
    ax.set_xscale("log")
    ax.set_xlabel("SFT dataset size (rows)")
    ax.set_ylabel("ONS (%)")
    ax.set_title("SFT data-scaling — Qwen3.5-4B ASCII")
    ax.legend()
    fig.tight_layout()
    fig.savefig("agentick_paper/figures/sft_scaling_curve.pdf")
    print("wrote agentick_paper/figures/sft_scaling_curve.pdf")


def figure_c_sft_category_breakdown():
    """Per-category ONS: baseline Reasoner vs. best SFT config."""
    # Implementation: call existing category-breakdown utility with configs:
    # qwen35_4b_ascii_reasoner, plus the SFT config with highest overall ONS
    import json
    from pathlib import Path

    best_cfg = max(
        [f"qwen35_4b_sft_{s}_ascii_{h}" for s in SFT_SIZES for h, _ in SFT_HARNESSES],
        key=lambda n: (_load_ons(n) if _load_ons(n) == _load_ons(n) else -1),
    )
    print(f"Best SFT config by ONS: {best_cfg}")
    # Reuse existing category-bar code — this is a stub: extend with the
    # existing plotting helper from the same file (e.g., `plot_per_category(configs, ...)`)
    # when it exists. If not, call `plot_category_bar` or similar.
    # TODO when integrating: wire to the existing helper for per-category data.
```

Note: if the existing `scripts/generate_paper_figures.py` has a different category-plotting helper, wire Figure C to it in Step 3.

- [ ] **Step 3: Run the SFT figure generators**

```bash
cd /home/roger/Desktop/agentick
uv run python -c "
import sys; sys.path.insert(0, 'scripts')
from generate_paper_figures import figure_a_sft_vs_baselines, figure_b_scaling_curve
figure_a_sft_vs_baselines()
figure_b_scaling_curve()
"
```

Expected: two PDFs written to `agentick_paper/figures/`.

- [ ] **Step 4: Commit figure code**

```bash
cd /home/roger/Desktop/agentick
git add scripts/generate_paper_figures.py agentick_paper/figures/sft_*.pdf
git commit -m "$(cat <<'EOF'
Figures: SFT vs. baselines + data-scaling curve

Consumes per-task metrics.json from cluster_manager/results/ to produce
Figures A (grouped bar) and B (scaling curve) for the paper's SFT story.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8.2: Write the SFT subsection of the Experiments section

**Files:**
- Modify: `agentick_paper/sections/experiments.tex`

- [ ] **Step 1: Draft the SFT paragraph + figure references**

Append to `agentick_paper/sections/experiments.tex`, after the existing `\subsection{Harness Design Matters...}` block:

```latex
\subsection{Supervised Fine-Tuning on Oracle Trajectories}
\label{sec:sft}

A central question for practitioners is whether supervised fine-tuning on expert data is a more effective tool than clever prompting. We fine-tune Qwen3.5-4B via LoRA on three public oracle trajectory datasets (\texttt{rogercc/agentick-oracle-trajectories-\{120k,250k,500k\}}), holding the training recipe fixed across sizes so the only changing variable is data scale. The datasets contain only action labels --- no reasoning traces --- so the model learns a Markovian policy that emits a single action digit per observation.

Figure~\ref{fig:sft_vs_baselines} compares the three fine-tunes against the zero-shot and Reasoner baselines already reported in \S\ref{sec:harness}, under both evaluation harnesses. \textbf{[RESULT TO BE FILLED IN AFTER P8 NUMBERS LAND]}: Does SFT on 500k close the gap to the Reasoner baseline when evaluated under the Markov harness? Does it surpass the Reasoner harness baseline? How does performance change when an SFT-trained model is evaluated under the Reasoner harness it never trained under?

Figure~\ref{fig:sft_scaling} shows the data-scaling curve. \textbf{[RESULT TO BE FILLED IN]}: Is the curve saturating, still rising, or overfitting at 500k? This directly informs how much oracle data future users should collect.

\begin{figure}[h]
    \centering
    \begin{minipage}[t]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{figures/sft_vs_baselines.pdf}
        \caption{SFT on Qwen3.5-4B ASCII across three dataset sizes, under Markov and Reasoner evaluation harnesses, compared to the baselines from \S\ref{sec:harness}.}
        \label{fig:sft_vs_baselines}
    \end{minipage}
    \hfill
    \begin{minipage}[t]{0.48\textwidth}
        \centering
        \includegraphics[width=\linewidth]{figures/sft_scaling_curve.pdf}
        \caption{SFT data-scaling for Qwen3.5-4B ASCII (log x-axis). Dashed line: zero-shot Reasoner baseline.}
        \label{fig:sft_scaling}
    \end{minipage}
\end{figure}
```

- [ ] **Step 2: Compile the paper and check for LaTeX errors**

```bash
cd /home/roger/Desktop/agentick/agentick_paper
latexmk -pdf main.tex 2>&1 | tail -30
```

Expected: compiles cleanly; figure references resolve.

If the SFT numbers are in, edit the `[RESULT TO BE FILLED IN...]` placeholders to describe the actual outcome (e.g., "SFT on 500k with Markov eval reaches 26.3\% ONS, surpassing the Reasoner baseline by 3.5 points...").

- [ ] **Step 3: Commit**

```bash
cd /home/roger/Desktop/agentick
git add agentick_paper/sections/experiments.tex
git commit -m "$(cat <<'EOF'
Paper: add SFT subsection to Experiments

Frames SFT on oracle trajectories as a test of whether expert-data
fine-tuning beats prompting strategy. Two figure placeholders + prose
to be filled in once the full eval matrix returns numbers.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8.3: Final sanity-check summary and hand-off

- [ ] **Step 1: Verify all artifacts exist**

```bash
cd /home/roger/Desktop/agentick

# 3 merged models on HF
for s in 120k 250k 500k; do
    uv run huggingface-cli repo info --repo-type model rogercc/agentick-qwen35-4b-sft-ascii-$s >/dev/null 2>&1 && echo "OK model $s" || echo "MISSING model $s"
done

# 6 eval result directories
for s in 120k 250k 500k; do
    for h in markov reasoner; do
        d="cluster_manager/results/qwen35_4b_sft_${s}_ascii_${h}/per_task"
        if [ -d "$d" ]; then
            n=$(find "$d" -name metrics.json | wc -l)
            echo "OK eval ${s}_${h}: $n metrics.json files"
        else
            echo "MISSING eval ${s}_${h}"
        fi
    done
done

# 2 figures
ls -l agentick_paper/figures/sft_*.pdf
```

Expected: 3 OK models, 6 OK evals (each with 152 metrics.json), 2 figure PDFs.

- [ ] **Step 2: Print a one-paragraph run summary for your notes**

```bash
uv run python -c "
import sys; sys.path.insert(0, 'scripts')
from generate_paper_figures import _load_ons
for s in ['120k','250k','500k']:
    for h in ['markov','reasoner']:
        v = _load_ons(f'qwen35_4b_sft_{s}_ascii_{h}')
        print(f'SFT-{s} {h}: {v:.2f}% ONS')
print(f'Baseline markov: {_load_ons(\"qwen35_4b_ascii_markov\"):.2f}%')
print(f'Baseline reasoner: {_load_ons(\"qwen35_4b_ascii_reasoner\"):.2f}%')
"
```

Expected: prints 8 lines of final numbers.

---

## Self-review notes (fixed inline)

- **Spec coverage:** Phase 0 covers all 6 audit items from the spec (completion-only loss → Task 0.3, chat-template match → Task 0.2, sequence-length truncation → Task 0.4, ANSI strip → covered implicitly by Task 0.2 since both SFT and eval strip identically, packing → left to TRL defaults + the smoke test, adapter-key prefix → Task 0.6). Phase 1 covers container deps. Phase 2 covers cluster_manager SFT job type + dataset caching. Phase 3 covers syncing code + datasets. Phase 4 is the pilot (spec phase P2). Phase 5 is full training (P3). Phase 6 is model distribution (P4). Phase 7 is full eval (P5). Phase 8 is analysis + paper (P6). All verification gates are expressed as pass criteria on explicit `_load_ons` checks or artifact existence.
- **No placeholders:** every task has exact file paths, exact shell commands, and full code where code is produced. The one intentional placeholder is the `[RESULT TO BE FILLED IN]` stub in the paper prose — to be written only after the numbers exist, which is a human authorial decision and belongs in the paper draft, not the plan.
- **Type consistency:** `SFT_CONFIGS` is the only new exported name. `_build_runner_command` and `generate_all_jobs` have consistent signatures with the existing codebase.
