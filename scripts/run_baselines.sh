#!/bin/bash
# Run baseline agents on all suites

set -e

echo "=== Running Baselines on Agentick Suites ==="

# Create output directory
mkdir -p leaderboard_data/baselines

# Run random and greedy baselines on quick suite (for testing)
python3 -c "
from agentick.leaderboard.baselines import compute_baselines_for_suite

# Quick suite for testing
print('Computing baselines for quick suite...')
compute_baselines_for_suite(
    'agentick-quick-v1',
    output_dir='leaderboard_data/baselines',
    run_random=True,
    run_greedy=True,
    run_oracle=True,
    n_episodes_random=10,
    n_episodes_oracle=5,
)

print('✓ Baselines complete!')
"

echo "✓ Baseline computation complete"
echo "Results saved to leaderboard_data/baselines/"
