#!/bin/bash
# 08_generate_site.sh - Build static leaderboard website
#
# Prerequisites: 07_generate_leaderboard.py completed
# Estimated time: < 1 minute
# Output: leaderboard_site/index.html

set -e

echo "========================================"
echo "Building Leaderboard Site"
echo "========================================"
echo ""

# Check if leaderboard data exists
if [ ! -f "leaderboard_data/leaderboard.json" ]; then
    echo "Error: leaderboard.json not found"
    echo "Run: python scripts/populate_leaderboard/07_generate_leaderboard.py"
    exit 1
fi

# Generate site
echo "Generating static HTML..."
uv run python -m agentick.leaderboard.site.generate \
    --data-file leaderboard_data/leaderboard.json \
    --output-dir leaderboard_site/

echo ""
echo "✓ Leaderboard site generated!"
echo ""
echo "Preview locally:"
echo "  cd leaderboard_site && python -m http.server 8080"
echo "  Visit: http://localhost:8080"
echo ""
echo "Deploy:"
echo "  See HOSTING.md for deployment instructions"
echo "  - GitHub Pages (free)"
echo "  - Vercel/Netlify (free)"
echo "  - Custom domain (~$12/year)"
