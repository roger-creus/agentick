#!/bin/bash
# 06_run_rl_eval.sh - Evaluate trained RL agents
#
# Prerequisites: Run 05_train_rl.sh first
# Estimated time: ~30 minutes
# Estimated cost: $0
# Output: leaderboard_data/entries/ppo_v1.json, dqn_v1.json

set -e  # Exit on error

echo "========================================"
echo "Evaluating Trained RL Agents"
echo "========================================"
echo ""
echo "This script evaluates:"
echo "  - Trained PPO agent"
echo "  - Trained DQN agent"
echo ""
echo "Time: ~30 minutes"
echo "Cost: $0"
echo ""

# Check for trained models
if [ ! -d "checkpoints/ppo_leaderboard" ] && [ ! -d "checkpoints/dqn_leaderboard" ]; then
    echo "❌ Error: No trained models found"
    echo "Run 05_train_rl.sh first to train agents"
    exit 1
fi

# Create output directories
mkdir -p leaderboard_data/entries
mkdir -p logs

# Timestamp for logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/06_rl_eval_${TIMESTAMP}.log"

echo "Logs will be saved to: $LOG_FILE"
echo ""

# Evaluate PPO
if [ -d "checkpoints/ppo_leaderboard" ]; then
    echo "1/2: Evaluating PPO agent..."
    uv run agentick evaluate \
        --submission leaderboard_data/submissions/ppo_trained.yaml \
        --suite agentick-core-v1 \
        --output leaderboard_data/entries/ \
        2>&1 | tee -a "$LOG_FILE"
    echo ""
else
    echo "⚠️  Skipping PPO: model not found"
fi

# Evaluate DQN
if [ -d "checkpoints/dqn_leaderboard" ]; then
    echo "2/2: Evaluating DQN agent..."
    uv run agentick evaluate \
        --submission leaderboard_data/submissions/dqn_trained.yaml \
        --suite agentick-core-v1 \
        --output leaderboard_data/entries/ \
        2>&1 | tee -a "$LOG_FILE"
    echo ""
else
    echo "⚠️  Skipping DQN: model not found"
fi

echo "========================================"
echo "✓ RL evaluations complete!"
echo "========================================"
echo ""
echo "Results saved to: leaderboard_data/entries/"
ls -lh leaderboard_data/entries/ppo*.json leaderboard_data/entries/dqn*.json 2>/dev/null || echo "No results found"
echo ""
echo "Next step: Run 07_generate_leaderboard.py to compile results:"
echo "  python scripts/populate_leaderboard/07_generate_leaderboard.py"
