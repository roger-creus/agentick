#!/bin/bash
# Run ALL benchmarks in sequence
# Estimated times:
#   Random:    ~2 min  (no deps)
#   Greedy:    ~5 min  (no deps)
#   PPO:       ~2-4 hours (GPU, no API keys)
#   OpenAI:    ~1 hour ($10-20 per agent)
#   Anthropic: ~1 hour ($10-20 per agent)
#
# Total: 4-6 hours, ~$50-80 in API costs

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"

echo "========================================"
echo "FULL BENCHMARK PIPELINE"
echo "========================================"
echo "This will run ALL agent benchmarks."
echo "Estimated time: 4-6 hours"
echo "Estimated cost: \$50-80 (for LLM agents)"
echo ""
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

echo ""
echo "[1/7] Running random agent baseline..."
uv run python examples/experiments/full_benchmark/run_single_benchmark.py \
    examples/experiments/full_benchmark/configs/random_agent.yaml

echo ""
echo "[2/7] Running greedy agent baseline..."
uv run python examples/experiments/full_benchmark/run_single_benchmark.py \
    examples/experiments/full_benchmark/configs/oracle_agent.yaml

echo ""
echo "[3/7] Training and evaluating PPO (dense rewards)..."
uv run python examples/experiments/full_benchmark/train_and_eval_ppo.py \
    --config examples/experiments/full_benchmark/configs/ppo_pixels_dense.yaml

echo ""
echo "[3b/7] Training and evaluating PPO (sparse rewards)..."
uv run python examples/experiments/full_benchmark/train_and_eval_ppo.py \
    --config examples/experiments/full_benchmark/configs/ppo_pixels_sparse.yaml

echo ""
echo "[4/7] Running OpenAI text agent..."
echo "(Skipping - requires OPENAI_API_KEY in .env)"
echo "To run: uv run python examples/experiments/full_benchmark/run_single_benchmark.py examples/experiments/full_benchmark/configs/openai_text.yaml"

echo ""
echo "[5/7] Running OpenAI vision agent..."
echo "(Skipping - requires OPENAI_API_KEY in .env)"
echo "To run: uv run python examples/experiments/full_benchmark/run_single_benchmark.py examples/experiments/full_benchmark/configs/openai_vision.yaml"

echo ""
echo "[6/7] Running Anthropic text agent..."
echo "(Skipping - requires ANTHROPIC_API_KEY in .env)"
echo "To run: uv run python examples/experiments/full_benchmark/run_single_benchmark.py examples/experiments/full_benchmark/configs/anthropic_text.yaml"

echo ""
echo "[7/7] Running Anthropic vision agent..."
echo "(Skipping - requires ANTHROPIC_API_KEY in .env)"
echo "To run: uv run python examples/experiments/full_benchmark/run_single_benchmark.py examples/experiments/full_benchmark/configs/anthropic_vision.yaml"

echo ""
echo "========================================"
echo "BASELINE BENCHMARKS COMPLETE"
echo "========================================"
echo ""
echo "Results saved to: results/full_benchmark/"
echo ""
echo "Next steps:"
echo "  1. Run LLM agents (if you have API keys)"
echo "  2. Generate plots: uv run python examples/experiments/full_benchmark/plot_all_results.py"
echo "  3. Generate report: uv run python examples/experiments/full_benchmark/generate_report.py"
echo ""
