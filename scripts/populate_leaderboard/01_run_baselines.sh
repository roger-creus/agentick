#!/bin/bash
# 01_run_baselines.sh - Run baseline agents (Random, Greedy, Oracle)
#
# Prerequisites: None (no API keys, no GPU required)
# Estimated time: ~30 minutes
# Estimated cost: $0
# Output: leaderboard_data/entries/random_agent_v1.json, greedy_agent_v1.json, oracle_agent_v1.json

set -e  # Exit on error

echo "========================================"
echo "Running Baseline Agents"
echo "========================================"
echo ""
echo "This script evaluates:"
echo "  - Random agent (action space sampling)"
echo "  - Greedy agent (heuristic-based)"
echo "  - Oracle agent (optimal solution, upper bound)"
echo ""
echo "Time: ~30 minutes"
echo "Cost: $0"
echo ""

# Create output directories
mkdir -p leaderboard_data/entries
mkdir -p logs

# Timestamp for logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/01_baselines_${TIMESTAMP}.log"

echo "Logs will be saved to: $LOG_FILE"
echo ""

# Run evaluations
echo "1/3: Evaluating Random agent..."
uv run agentick evaluate \
    --submission leaderboard_data/submissions/random_agent.yaml \
    --suite agentick-core-v1 \
    --output leaderboard_data/entries/ \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "2/3: Evaluating Greedy agent..."
uv run agentick evaluate \
    --submission leaderboard_data/submissions/greedy_agent.yaml \
    --suite agentick-core-v1 \
    --output leaderboard_data/entries/ \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "3/3: Evaluating Oracle agent..."
uv run agentick evaluate \
    --submission leaderboard_data/submissions/oracle_agent.yaml \
    --suite agentick-core-v1 \
    --output leaderboard_data/entries/ \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "========================================"
echo "✓ Baseline evaluations complete!"
echo "========================================"
echo ""
echo "Results saved to: leaderboard_data/entries/"
ls -lh leaderboard_data/entries/*.json
echo ""
echo "Next step: Run 02_run_openai.sh (requires OPENAI_API_KEY, ~$20)"
echo "  or skip to step 7 for leaderboard generation:"
echo "  python scripts/populate_leaderboard/07_generate_leaderboard.py"
