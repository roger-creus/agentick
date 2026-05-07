#!/usr/bin/env bash
# Run a full SFT job given a YAML config path.
# Invokes sft_with_trl.py for training, then merge_and_push.py.
#
# Usage:
#   run_sft_from_config.sh <config.yaml> <output_dir>
#
# The YAML is expected to have shape:
#   training:
#     dataset: ...
#     model: ...
#     ... other sft_with_trl.py flags ...
#   push_to_hub: <repo_id>
set -euo pipefail

CFG="$1"
OUT="$2"
PYTHON="${PYTHON:-python}"
TMP_ARGS="$(mktemp)"
trap 'rm -f "$TMP_ARGS"' EXIT

if compgen -G "$OUT/adapter/checkpoint-*" >/dev/null; then
    echo "Found existing checkpoints under $OUT/adapter; preserving adapter dir for resume."
else
    rm -rf "$OUT/adapter"
fi
mkdir -p "$OUT/adapter"

# Extract the `training:` dict as a flat argv list for sft_with_trl.py
"$PYTHON" - "$CFG" > "$TMP_ARGS" <<'PYEOF'
import sys
import shlex

import yaml
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
SFT_ARGS=$(cat "$TMP_ARGS")

PUSH_TO=$("$PYTHON" -c "import yaml; print(yaml.safe_load(open('$CFG'))['push_to_hub'])")
BASE_MODEL=$("$PYTHON" -c "import yaml; print(yaml.safe_load(open('$CFG'))['training']['model'])")
DATASET_ID=$("$PYTHON" -c "import yaml; print(yaml.safe_load(open('$CFG'))['training']['dataset'])")

# Resolve the HF dataset ID AND model ID to local snapshot paths so
# load_dataset / AutoTokenizer / AutoModel don't try to hit the Hub.
# Requires the dataset and model to already be in HF_HOME (via ./cm.py setup).
RESOLVED_DATASET=$("$PYTHON" - "$DATASET_ID" "dataset" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
try:
    path = snapshot_download(
        repo_id=sys.argv[1],
        repo_type=sys.argv[2],
        local_files_only=True,
    )
    print(path)
except Exception as e:
    print(f"# failed to resolve locally: {e}", file=sys.stderr)
    print(sys.argv[1])
PYEOF
)
echo "Resolved dataset: $DATASET_ID -> $RESOLVED_DATASET"

RESOLVED_MODEL=$("$PYTHON" - "$BASE_MODEL" "model" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
try:
    path = snapshot_download(
        repo_id=sys.argv[1],
        repo_type=sys.argv[2],
        local_files_only=True,
    )
    print(path)
except Exception as e:
    print(f"# failed to resolve locally: {e}", file=sys.stderr)
    print(sys.argv[1])
PYEOF
)
echo "Resolved model:   $BASE_MODEL -> $RESOLVED_MODEL"

# Rewrite SFT_ARGS: replace HF IDs with local snapshot paths.
SFT_ARGS=$(echo "$SFT_ARGS" | sed "s|--dataset $DATASET_ID|--dataset $RESOLVED_DATASET|")
SFT_ARGS=$(echo "$SFT_ARGS" | sed "s|--model $BASE_MODEL|--model $RESOLVED_MODEL|")

echo "=== SFT run ==="
echo "Config: $CFG"
echo "Output: $OUT"
echo "Args:   $SFT_ARGS"
echo "Push:   $PUSH_TO"

# HF_HUB_OFFLINE=1 (set by the eval sbatch template) makes load_dataset
# raise OfflineModeIsEnabled even when the dataset is cached. Swap to
# the library-specific offline flags instead:
# - HF_DATASETS_OFFLINE=1 → datasets uses cache, no Hub check
# - TRANSFORMERS_OFFLINE=1 → transformers uses cache, no Hub check
# Explicit HF_HUB_OFFLINE=0 overrides any inherited value.
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== env (post-override) ==="
echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-unset}"
echo "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-unset}"
echo "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-unset}"
echo "HF_HOME=${HF_HOME:-unset}"
echo "==="

if [ "${SFT_NUM_PROCESSES:-1}" = "1" ]; then
    "$PYTHON" examples/data_and_finetuning/sft_with_trl.py \
        $SFT_ARGS \
        --output-dir "$OUT/adapter"
else
    torchrun --standalone --nnodes=1 --nproc_per_node "${SFT_NUM_PROCESSES}" \
        examples/data_and_finetuning/sft_with_trl.py \
        $SFT_ARGS \
        --output-dir "$OUT/adapter"
fi

echo "=== SFT training complete. Adapter at: $OUT/adapter ==="
ls -lh "$OUT/adapter"

# Merge + push to HF Hub. Some managed compute nodes are offline, so push is
# best-effort: if it fails, the adapter is still available in the output dir.
# Set HF_TOKEN in the environment when pushing to a private or owned repo.

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true

echo "=== Attempting merge + push to $PUSH_TO (best-effort) ==="
if "$PYTHON" examples/data_and_finetuning/merge_and_push.py \
        --base-model "$RESOLVED_MODEL" \
        --base-model-id "$BASE_MODEL" \
        --adapter-dir "$OUT/adapter" \
        --push-to-hub "$PUSH_TO" 2>&1; then
    echo "=== SFT complete: pushed $PUSH_TO ==="
else
    echo "=== WARN: merge_and_push failed (likely offline compute node)."
    echo "=== Adapter saved at $OUT/adapter — pull + push manually:"
    echo "===   rsync -avz <remote>:$OUT/adapter/ /tmp/adapter/"
    echo "===   python examples/data_and_finetuning/merge_and_push.py \\"
    echo "===       --base-model $BASE_MODEL --adapter-dir /tmp/adapter --push-to-hub $PUSH_TO"
    echo "=== Exiting 0 so cluster sees the training portion as successful."
    exit 0
fi
