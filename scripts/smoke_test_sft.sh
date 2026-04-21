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
    --max-steps 2 \
    --batch-size 1 \
    --grad-accum 1 \
    --max-seq-length 2048 \
    --output-dir "$OUT" \
    --logging-steps 1 \
    --save-strategy no \
    --report-to none

if [ ! -f "$OUT/adapter_config.json" ]; then
    echo "FAIL: no adapter_config.json at $OUT"
    exit 1
fi
echo "OK: SFT smoke test wrote $OUT/adapter_config.json"
