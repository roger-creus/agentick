#!/usr/bin/env bash
# Run all API-based (OpenAI, Anthropic) benchmark configs.
# Requires API keys to be set in the environment.
#
# Usage:
#   export OPENAI_API_KEY=sk-...
#   export ANTHROPIC_API_KEY=sk-ant-...
#   bash examples/experiments/full_benchmark/run_api_benchmark.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/configs"

echo "=========================================="
echo "  Agentick API Agent Benchmark"
echo "=========================================="
echo ""

# Check API keys
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "WARNING: OPENAI_API_KEY not set -- OpenAI configs will be skipped."
fi
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "WARNING: ANTHROPIC_API_KEY not set -- Anthropic configs will be skipped."
fi
echo ""

# Claude configs
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    for config in "${CONFIG_DIR}"/claude_sonnet_*.yaml; do
        name=$(basename "${config}" .yaml)
        echo ">>> Running: ${name}"
        uv run python -m agentick.experiments.run --config "${config}" || {
            echo "    FAILED: ${name} (continuing...)"
        }
        echo ""
    done
fi

# GPT-4o configs
if [ -n "${OPENAI_API_KEY:-}" ]; then
    for config in "${CONFIG_DIR}"/gpt4o_*.yaml; do
        name=$(basename "${config}" .yaml)
        echo ">>> Running: ${name}"
        uv run python -m agentick.experiments.run --config "${config}" || {
            echo "    FAILED: ${name} (continuing...)"
        }
        echo ""
    done
fi

echo "=========================================="
echo "  All API benchmarks complete."
echo "=========================================="
