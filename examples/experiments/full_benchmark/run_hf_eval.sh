#!/usr/bin/env bash
# Convenience wrapper for running HuggingFace LLM/VLM evaluation configs.
# Supports filtering by model, harness, and observation mode.
#
# Usage:
#   bash run_hf_eval.sh                                   # run all 24 configs
#   bash run_hf_eval.sh --model qwen3_4b                  # only Qwen3-4B configs (6)
#   bash run_hf_eval.sh --harness markov                  # only markovian zero-shot (8)
#   bash run_hf_eval.sh --obs lang                        # only language obs (12)
#   bash run_hf_eval.sh --model qwen3_vl4b --harness reasoner  # combined filters

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/configs"

# Parse arguments
MODEL_FILTER=""
HARNESS_FILTER=""
OBS_FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL_FILTER="$2"; shift 2 ;;
        --harness)
            HARNESS_FILTER="$2"; shift 2 ;;
        --obs)
            OBS_FILTER="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--model PATTERN] [--harness PATTERN] [--obs PATTERN]"
            echo ""
            echo "Filters (substring match on config filename):"
            echo "  --model    Model filter: qwen3_4b, qwen3_30b_a3b, qwen3_vl4b, qwen3_vl8b"
            echo "  --harness  Harness filter: markov, nonmarkov, reasoner"
            echo "  --obs      Observation filter: lang, ascii"
            echo ""
            echo "Examples:"
            echo "  $0                                        # all 24 configs"
            echo "  $0 --model qwen3_4b                       # 6 configs for Qwen3-4B"
            echo "  $0 --model qwen3_vl4b --harness markov    # 2 configs"
            exit 0 ;;
        *)
            echo "Unknown option: $1 (use --help for usage)"; exit 1 ;;
    esac
done

# Collect matching configs
CONFIGS=()
for config in "${CONFIG_DIR}"/qwen3_*.yaml; do
    name=$(basename "${config}" .yaml)

    # Apply filters
    if [[ -n "${MODEL_FILTER}" ]]; then
        # Match model prefix (before obs mode): e.g. qwen3_4b, qwen3_vl4b
        if [[ "${name}" != *"${MODEL_FILTER}"* ]]; then
            continue
        fi
    fi
    if [[ -n "${HARNESS_FILTER}" ]]; then
        if [[ "${name}" != *"${HARNESS_FILTER}"* ]]; then
            continue
        fi
    fi
    if [[ -n "${OBS_FILTER}" ]]; then
        if [[ "${name}" != *"_${OBS_FILTER}_"* ]]; then
            continue
        fi
    fi

    CONFIGS+=("${config}")
done

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "No configs matched the given filters."
    echo "  --model=${MODEL_FILTER:-<any>} --harness=${HARNESS_FILTER:-<any>} --obs=${OBS_FILTER:-<any>}"
    exit 1
fi

TOTAL=${#CONFIGS[@]}

echo "=========================================="
echo "  Agentick HuggingFace Evaluation"
echo "  Matched ${TOTAL} config(s)"
echo "=========================================="
echo ""

CURRENT=0
FAILED=0

for config in "${CONFIGS[@]}"; do
    CURRENT=$((CURRENT + 1))
    name=$(basename "${config}" .yaml)
    echo ">>> [${CURRENT}/${TOTAL}] Running: ${name}"
    uv run python -m agentick.experiments.run --config "${config}" || {
        echo "    FAILED: ${name} (continuing...)"
        FAILED=$((FAILED + 1))
    }
    echo ""
done

echo "=========================================="
echo "  Done: ${TOTAL} configs, $((TOTAL - FAILED)) succeeded, ${FAILED} failed"
echo "=========================================="
