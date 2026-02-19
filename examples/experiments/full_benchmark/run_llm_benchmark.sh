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
echo "=========================================="
echo ""

CONFIGS=(
    "${CONFIG_DIR}/qwen3_4b_language.yaml"
    "${CONFIG_DIR}/qwen3_4b_ascii.yaml"
    "${CONFIG_DIR}/qwen3_8b_language.yaml"
    "${CONFIG_DIR}/qwen3_vl_4b.yaml"
)

for config in "${CONFIGS[@]}"; do
    name=$(basename "${config}" .yaml)
    echo ">>> Running: ${name}"
    echo "    Config: ${config}"
    uv run python -m agentick.experiments.run --config "${config}" || {
        echo "    FAILED: ${name} (continuing...)"
    }
    echo ""
done

echo "=========================================="
echo "  All local benchmarks complete."
echo "=========================================="
