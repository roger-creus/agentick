#!/usr/bin/env bash
# Run all local HuggingFace LLM/VLM benchmark configs sequentially.
# These run models locally -- no API keys required, but need GPU.
#
# Usage:
#   bash examples/experiments/full_benchmark/run_llm_benchmark.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/configs"

echo "=========================================="
echo "  Agentick Local LLM/VLM Benchmark"
echo "  24 configs: 4 models x 2 obs x 3 harnesses"
echo "=========================================="
echo ""

CONFIGS=(
    # --- Qwen3-4B-Instruct (LLM) ---
    "${CONFIG_DIR}/qwen3_4b_lang_markov.yaml"
    "${CONFIG_DIR}/qwen3_4b_lang_nonmarkov.yaml"
    "${CONFIG_DIR}/qwen3_4b_lang_reasoner.yaml"
    "${CONFIG_DIR}/qwen3_4b_ascii_markov.yaml"
    "${CONFIG_DIR}/qwen3_4b_ascii_nonmarkov.yaml"
    "${CONFIG_DIR}/qwen3_4b_ascii_reasoner.yaml"

    # --- Qwen3-30B-A3B-Instruct-FP8 (LLM, MoE) ---
    "${CONFIG_DIR}/qwen3_30b_a3b_lang_markov.yaml"
    "${CONFIG_DIR}/qwen3_30b_a3b_lang_nonmarkov.yaml"
    "${CONFIG_DIR}/qwen3_30b_a3b_lang_reasoner.yaml"
    "${CONFIG_DIR}/qwen3_30b_a3b_ascii_markov.yaml"
    "${CONFIG_DIR}/qwen3_30b_a3b_ascii_nonmarkov.yaml"
    "${CONFIG_DIR}/qwen3_30b_a3b_ascii_reasoner.yaml"

    # --- Qwen3-VL-4B-Instruct (VLM) ---
    "${CONFIG_DIR}/qwen3_vl4b_lang_markov.yaml"
    "${CONFIG_DIR}/qwen3_vl4b_lang_nonmarkov.yaml"
    "${CONFIG_DIR}/qwen3_vl4b_lang_reasoner.yaml"
    "${CONFIG_DIR}/qwen3_vl4b_ascii_markov.yaml"
    "${CONFIG_DIR}/qwen3_vl4b_ascii_nonmarkov.yaml"
    "${CONFIG_DIR}/qwen3_vl4b_ascii_reasoner.yaml"

    # --- Qwen3-VL-8B-Instruct (VLM) ---
    "${CONFIG_DIR}/qwen3_vl8b_lang_markov.yaml"
    "${CONFIG_DIR}/qwen3_vl8b_lang_nonmarkov.yaml"
    "${CONFIG_DIR}/qwen3_vl8b_lang_reasoner.yaml"
    "${CONFIG_DIR}/qwen3_vl8b_ascii_markov.yaml"
    "${CONFIG_DIR}/qwen3_vl8b_ascii_nonmarkov.yaml"
    "${CONFIG_DIR}/qwen3_vl8b_ascii_reasoner.yaml"
)

TOTAL=${#CONFIGS[@]}
CURRENT=0

for config in "${CONFIGS[@]}"; do
    CURRENT=$((CURRENT + 1))
    name=$(basename "${config}" .yaml)
    echo ">>> [${CURRENT}/${TOTAL}] Running: ${name}"
    echo "    Config: ${config}"
    uv run python -m agentick.experiments.run --config "${config}" || {
        echo "    FAILED: ${name} (continuing...)"
    }
    echo ""
done

echo "=========================================="
echo "  All ${TOTAL} local benchmarks complete."
echo "=========================================="
