#!/usr/bin/env bash
# Verify merge_and_push.py can load the smoke-test adapter and produce
# a non-zero weight delta vs. the base model.
#
# Runs merge_and_push.py with a dummy push-to-hub target. The push will
# fail (no such repo / no write perms) but that's fine — we only care
# about exercising the merge path and confirming:
#   (a) no "ZERO weight changes" error from merge_and_push's own guard
#   (b) it logs "{N}/{M} weights differ from base" with N > 0
#
# Must be run AFTER scripts/smoke_test_sft.sh has produced the adapter.
set -euo pipefail

SMOKE_OUT=/tmp/sft_smoke_out
LOG=/tmp/merge_smoke.log

if [ ! -f "$SMOKE_OUT/adapter_config.json" ]; then
    echo "ERROR: $SMOKE_OUT/adapter_config.json missing — run scripts/smoke_test_sft.sh first"
    exit 1
fi

# Run merge_and_push.py — capture output. The push step at the end will
# fail (we expect that), but the merge must succeed first.
uv run python examples/data_and_finetuning/merge_and_push.py \
    --base-model Qwen/Qwen3.5-0.8B \
    --adapter-dir "$SMOKE_OUT" \
    --push-to-hub rogercc/__merge_smoke_test__ \
    --dtype bfloat16 2>&1 | tee "$LOG" || true

if grep -q "ZERO weight changes" "$LOG"; then
    echo "FAIL: merge produced zero weight changes — adapter is not being applied"
    exit 1
fi

if ! grep -q "weights differ from base" "$LOG"; then
    echo "FAIL: could not confirm weight delta in log — expected 'N/M weights differ from base'"
    exit 1
fi

echo "OK: merge applied adapter weights correctly"
