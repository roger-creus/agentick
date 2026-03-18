#!/usr/bin/env bash
# Generate oracle trajectory datasets at different scales and push to HuggingFace.
# Each dataset has train + test splits using different deterministic seeds.
#
# Usage:
#   bash scripts/generate_datasets.sh          # all 4 sizes
#   bash scripts/generate_datasets.sh 100k     # just the 100k

set -euo pipefail

SCRIPT="examples/data_and_finetuning/collect_oracle_trajectories.py"
HF_PREFIX="rogercc/agentick-oracle-trajectories"

generate() {
    local size="$1" train_eps="$2" test_eps="$3"
    echo ""
    echo "============================================================"
    echo "  Generating ${size} dataset (${train_eps} train + ${test_eps} test episodes)"
    echo "============================================================"
    uv run python "$SCRIPT" \
        --n-episodes "$train_eps" \
        --n-test-episodes "$test_eps" \
        --push-to-hub "${HF_PREFIX}-${size}"
}

TARGET="${1:-all}"

case "$TARGET" in
    50k)   generate 50k  12 12 ;;
    100k)  generate 100k 25 25 ;;
    200k)  generate 200k 50 25 ;;
    400k)  generate 400k 100 25 ;;
    all)
        generate 50k  12 12
        generate 100k 25 25
        generate 200k 50 25
        generate 400k 100 25
        ;;
    *)
        echo "Usage: $0 [50k|100k|200k|400k|all]"
        exit 1
        ;;
esac

echo ""
echo "Done! Datasets pushed to https://huggingface.co/datasets/${HF_PREFIX}-*"
