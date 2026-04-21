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
mkdir -p "$OUT/adapter"

# Extract the `training:` dict as a flat argv list for sft_with_trl.py
python3 - "$CFG" > /tmp/sft_args.sh <<'PYEOF'
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
SFT_ARGS=$(cat /tmp/sft_args.sh)

PUSH_TO=$(python3 -c "import yaml; print(yaml.safe_load(open('$CFG'))['push_to_hub'])")
BASE_MODEL=$(python3 -c "import yaml; print(yaml.safe_load(open('$CFG'))['training']['model'])")

echo "=== SFT run ==="
echo "Config: $CFG"
echo "Output: $OUT"
echo "Args:   $SFT_ARGS"
echo "Push:   $PUSH_TO"

accelerate launch --num_processes 1 \
    examples/data_and_finetuning/sft_with_trl.py \
    $SFT_ARGS \
    --output-dir "$OUT/adapter"

python3 examples/data_and_finetuning/merge_and_push.py \
    --base-model "$BASE_MODEL" \
    --adapter-dir "$OUT/adapter" \
    --push-to-hub "$PUSH_TO"

echo "=== SFT complete: pushed $PUSH_TO ==="
