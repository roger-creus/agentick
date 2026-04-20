# SFT Experiments for the Agentick Paper — Design

**Date:** 2026-04-20
**Owner:** roger (led by Claude)
**Status:** Approved — implementation plan next

## Motivation

The paper's Experiments section covers frontier LLMs, PPO from scratch, and open-weight Qwen under zero-shot and Reasoner harnesses, but contains **zero SFT results**. A prior SFT attempt on `rogercc/agentick-qwen35-4b-sft-ascii` was unsatisfying; root cause unknown. This project fills the gap by running a clean, reproducible SFT data-scaling study on Qwen3.5-4B with ASCII observations, evaluated under both inference harnesses.

## Research questions

**A — Can oracle-action SFT on a small open-weight model beat its own Reasoner baseline?**
The paper currently shows Qwen3.5-4B + Reasoner at 22.8% ONS via clever prompting alone. If SFT on oracle trajectories (action labels only, no CoT) can match or exceed that with a *Markov* harness at inference, that is a strong paper-worthy result: the scaffolding-vs-learning tradeoff.

**B — How does SFT performance scale with dataset size?**
Three public datasets exist: `rogercc/agentick-oracle-trajectories-{120k,250k,500k}` (row counts, not episodes). Running all three with an identical recipe produces a data-scaling curve at fixed compute-per-step, which directly informs how much oracle data future users should collect.

Out-of-scope (to keep the study clean): model-size scaling, observation-modality scaling, reasoning-label SFT, harness optimization, train/eval-on-disjoint-difficulties.

## Experiment matrix

| # | Training | Eval harness | Notes |
|---|---|---|---|
| 1 | — (baseline) | Markov | Already in paper: 2.3% ONS |
| 2 | — (baseline) | Reasoner | Already in paper: 22.8% ONS |
| 3 | SFT on 120k | Markov | New |
| 4 | SFT on 120k | Reasoner | New — tests harness transfer |
| 5 | SFT on 250k | Markov | New |
| 6 | SFT on 250k | Reasoner | New |
| 7 | SFT on 500k | Markov | New |
| 8 | SFT on 500k | Reasoner | New |

**3 SFT training runs + 6 new eval configs × 152 jobs = 912 new eval jobs.**

## Training recipe (fixed across all three sizes)

| Knob | Value |
|---|---|
| Base model | `Qwen/Qwen3.5-4B` |
| Modality | ASCII only |
| LoRA | r=16, α=32, dropout=0.05, target_modules=all-linear |
| Epochs | 3 |
| LR | 4e-5, cosine schedule, 5% warmup |
| Per-device batch × grad-accum | 4 × 4 = 16 effective on 1 GPU |
| Max sequence length | 8192 |
| Precision | bf16 |
| Gradient checkpointing | on |
| Packing | **on** (must be verified leak-free) |
| Completion-only loss | **on** (loss only on assistant tokens) |
| Eval during training | every epoch, on the dataset's `test` split |

**Per-run compute (1× H100 80GB):** ~3–4 h for 120k, ~6–8 h for 250k, ~12–16 h for 500k. All fit under a 23:59:59 walltime.

**Why fixed recipe, not tuned per size:** a single-variable (dataset size) comparison is the cleanest story; tuning per point would make scaling claims suspect. Accepted tradeoff: each individual run may not be at its absolute ceiling.

## Components to build / change

### (a) `examples/data_and_finetuning/sft_with_trl.py` — audit + fixes

Expected bug list (fixed before any training job runs):

1. **Completion-only loss is not set.** The current script applies standard causal LM loss over the full sequence, so the model is learning to reproduce the ASCII observation too, not just emit the action. Fix: enable TRL's completion-only loss (via `SFTConfig(completion_only_loss=True)` if supported in the pinned version, or a `DataCollatorForCompletionOnlyLM` with a response template that matches Qwen3.5's chat template). This is the prime suspect for last time's poor results.
2. **Chat-template match.** Byte-compare the built row after `apply_chat_template` against the exact eval-time prompt produced by `agentick/leaderboard/adapters/prompt_templates.py`. They must match (modulo trailing whitespace).
3. **Sequence-length truncation.** Find the longest ASCII observation across 38 × 4 task/difficulty combinations and confirm it fits in `max_seq_length=8192` after tokenization. Bump if needed.
4. **ANSI escape stripping.** Verified by hash of stripped render.
5. **Packing correctness.** TRL's packing in older versions could leak attention across packed examples. Pin a known-good version; add a synthetic assertion test.
6. **Adapter-key prefix mismatch.** `merge_and_push.py` already contains a workaround (`_fix_adapter_key_mismatch`) for VL-first Qwen models. Verify it fires for Qwen3.5-4B and that the merged weights yield identical logits to `base + adapter` on a held-out batch.

### (b) `cluster_manager/` — add SFT job type

- `SFT_CONFIGS = ["qwen35_4b_sft_ascii_120k", "..._250k", "..._500k"]` in `config.py`
- `SFT_TIME = "23:59:59"` resource profile, 1 CPU-heavy node, full H100
- `_build_runner_command` branch in `jobs.py` emitting `accelerate launch sft_with_trl.py ... && python merge_and_push.py ...` inside the container
- Extend `cm.py setup` with a `--datasets` step: download the three HF datasets locally once, then rsync to `/scratch/rogercc/hf_cache/` on each reachable cluster (same pattern as `_download_and_push_models`)
- SFT jobs distributed across **rorqual / narval / fir only** (nibi excluded due to reliability concerns; evals may still use nibi)

### (c) `examples/experiments/configs/` — 6 new eval YAMLs

`qwen35_4b_sft_{120k,250k,500k}_ascii_{markov,reasoner}.yaml`, each pointing at `rogercc/agentick-qwen35-4b-sft-ascii-<size>` — based on existing `qwen35_4b_sft_ascii_markov.yaml` template, only the `model:` field changes.

## Data flow

1. **Local → clusters (one-shot, training prep).** `cm.py setup --datasets` downloads the three HF datasets and rsyncs to each cluster's HF cache. Compute nodes have no internet.
2. **Training (per dataset).** SLURM job launches `accelerate launch sft_with_trl.py` against the cached dataset, trains LoRA, runs `merge_and_push.py` in the same job which merges and pushes the merged model to `rogercc/agentick-qwen35-4b-sft-ascii-<size>` (public repo).
3. **HF hub → clusters (one-shot, eval prep).** After each training job finishes, pull the merged model locally and rsync to all 4 clusters' HF caches.
4. **Eval.** `cm.py submit --configs qwen35_4b_sft_<size>_ascii_<harness>` schedules 152 jobs per eval config across all 4 clusters. vLLM reads the merged model from the local HF cache with `HF_HUB_OFFLINE=1`.
5. **Results back.** `cm.py pull` rsyncs `metrics.json` files. `merge_results.py` aggregates. Paper-figure script produces Figures A/B/C.

## Analysis — three figures for the paper

- **Figure A (headline):** grouped bar chart. For each harness {Markov, Reasoner}, four bars: baseline + SFT-120k + SFT-250k + SFT-500k. Shows whether SFT beats the existing Reasoner baseline; shows whether Markov-harness SFT can close the gap to Reasoner-harness baseline.
- **Figure B (scaling curve):** x-axis log(dataset size); two lines, one per eval harness; y-axis ONS; baseline as x=0 reference. Directly answers research question B.
- **Figure C (per-category):** radar or grouped bar, baseline Reasoner vs. best SFT config (eval harness chosen by overall ONS), per capability category. Tells the reader *what* SFT transfers.

## Execution phases with verification gates

| Phase | What | Pass gate to move on |
|---|---|---|
| P0 — Audit | Code-review + fix `sft_with_trl.py` per the 6-point list | All 6 audit checks pass locally on a 10-step synthetic run |
| P1 — Containers | Verify `trl`, `peft`, `accelerate`, `flash-attn`, `datasets` work inside `agentick.sif` for training (not just eval) | 10-step training inside the container locally runs without crash |
| P2 — Pilot | SFT on 120k, 200 steps, merge, push `rogercc/agentick-qwen35-4b-sft-ascii-pilot`, eval GoToGoal-v0 easy × 5 seeds under Markov | Eval returns success rate > 0% without errors |
| P3 — Full training | 3 SFT jobs in parallel (rorqual / narval / fir) | All 3 exit 0; train loss drops ≥2× from start; wandb curves monotone |
| P4 — Model distribution | Pull each merged model locally, rsync to all 4 cluster HF caches | vLLM 1-job smoke test per size returns valid response |
| P5 — Full eval | 6 configs × 152 jobs = 912 jobs | ≥95% valid `metrics.json` after one re-run pass on stragglers |
| P6 — Paper | Generate Figures A/B/C, update `agentick_paper/sections/experiments.tex` | Figures committed; paper diff reviewed |

## Explicit non-goals

- Tuning hyperparameters per dataset size.
- Comparing to Qwen3.5 at 0.8B / 2B or to Qwen3-4B.
- Language or pixel observation modalities.
- Training with reasoning-label data (none available in the datasets).
- Cross-difficulty or cross-task generalization studies.
- Retraining PPO or re-running any baseline already in the paper.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| SFT helps on Markov but not Reasoner (since Reasoner expects multi-token reasoning and we trained single-digit outputs) | Report honestly; this is a genuine finding, not a bug |
| SFT regresses vs. baseline on some categories (overfitting to oracle heuristics) | Figure C will surface this; report per-category |
| HF push fails inside SLURM job (HF token missing) | Bake HF_TOKEN into container env via apptainer `--env`; verify in P1 |
| Container missing training deps | Detected in P1; either extend Dockerfile and rebuild SIF, or install at container runtime via a requirements file |
| 500k training run exceeds 23:59:59 | Fallback: 2 epochs instead of 3 for all sizes (still fixed recipe — dataset size is the only variable) |
| Merged-model load fails in vLLM after push | Detected in P4 smoke test; worst case, re-merge with explicit dtype handling |

## References

- SFT script: `examples/data_and_finetuning/sft_with_trl.py`
- Merge script: `examples/data_and_finetuning/merge_and_push.py`
- Cluster manager: `cluster_manager/cm.py`, `cluster_manager/config.py`, `cluster_manager/jobs.py`
- Existing SFT eval configs (template): `examples/experiments/configs/qwen35_4b_sft_ascii_{markov,reasoner}.yaml`
- Paper experiments section: `agentick_paper/sections/experiments.tex`
- HF datasets: `rogercc/agentick-oracle-trajectories-{120k,250k,500k}`
