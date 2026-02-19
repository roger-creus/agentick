#!/bin/bash
# Run full PPO training benchmark (dense + sparse) and generate plots.
#
# Usage:
#   bash examples/experiments/full_benchmark/run_ppo_benchmark.sh
#
# Estimated time: 4-8 hours depending on GPU

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"

echo "========================================"
echo "PPO PIXEL TRAINING BENCHMARK"
echo "========================================"
echo "This will train PPO on all 38 tasks x 4 difficulties."
echo "  - Dense rewards: 152 runs x 300k steps"
echo "  - Sparse rewards: 152 runs x 300k steps"
echo ""
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

echo ""
echo "[1/3] Training PPO with dense rewards..."
uv run python examples/experiments/full_benchmark/train_and_eval_ppo.py \
    --config examples/experiments/full_benchmark/configs/ppo_pixels_dense.yaml

echo ""
echo "[2/3] Training PPO with sparse rewards..."
uv run python examples/experiments/full_benchmark/train_and_eval_ppo.py \
    --config examples/experiments/full_benchmark/configs/ppo_pixels_sparse.yaml

echo ""
echo "[3/3] Generating comparison plots..."
DENSE_DIR=$(ls -td results/ppo_benchmarks/ppo-pixels-dense-300k_* 2>/dev/null | head -1)
SPARSE_DIR=$(ls -td results/ppo_benchmarks/ppo-pixels-sparse-300k_* 2>/dev/null | head -1)

if [ -n "$DENSE_DIR" ] && [ -n "$SPARSE_DIR" ]; then
    uv run python examples/experiments/full_benchmark/plot_ppo_results.py \
        --results-dir "$DENSE_DIR" \
        --compare "$SPARSE_DIR"
elif [ -n "$DENSE_DIR" ]; then
    uv run python examples/experiments/full_benchmark/plot_ppo_results.py \
        --results-dir "$DENSE_DIR"
fi

echo ""
echo "========================================"
echo "PPO BENCHMARK COMPLETE"
echo "========================================"
echo ""
echo "Results:"
[ -n "$DENSE_DIR" ] && echo "  Dense:  $DENSE_DIR"
[ -n "$SPARSE_DIR" ] && echo "  Sparse: $SPARSE_DIR"
echo ""
echo "TensorBoard:"
[ -n "$DENSE_DIR" ] && echo "  tensorboard --logdir $DENSE_DIR/tensorboard"
echo ""
