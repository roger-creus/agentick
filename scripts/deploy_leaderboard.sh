#!/bin/bash
# Deploy leaderboard site

set -e

echo "=== Deploying Leaderboard Site ==="

# Generate site
echo "Generating site..."
python3 -m agentick.leaderboard.site.generate

# Copy to deployment directory (could be GitHub Pages, Vercel, etc.)
echo "Site generated in leaderboard_site/"
echo "Deploy manually to your hosting service"

# Example GitHub Pages deployment:
# git add leaderboard_site/
# git commit -m "Update leaderboard"
# git subtree push --prefix leaderboard_site origin gh-pages

echo "✓ Site ready for deployment"
