#!/bin/bash
set -euo pipefail

# Paper Intelligence Cloud — Migration Script
# Uploads locally processed papers to the cloud backend.
#
# Usage: ./migrate.sh [papers_dir]
#   papers_dir: path to papers directory (default: ~/Documents/papers)

PAPERS_DIR="${1:-$HOME/Documents/papers}"
API_URL="${PAPER_INTELLIGENCE_URL:-}"
API_TOKEN="${PAPER_INTELLIGENCE_TOKEN:-}"

# Try to get API token from 1Password if not set
if [ -z "$API_TOKEN" ]; then
  API_TOKEN=$(op read "op://CLI Secrets/paper-intelligence-api-token/token" 2>/dev/null || true)
fi

if [ -z "$API_URL" ]; then
  echo "Error: Set PAPER_INTELLIGENCE_URL to your Worker URL"
  echo "  export PAPER_INTELLIGENCE_URL=https://paper-intelligence.<subdomain>.workers.dev"
  exit 1
fi

if [ -z "$API_TOKEN" ]; then
  echo "Error: Set PAPER_INTELLIGENCE_TOKEN or configure 1Password CLI"
  exit 1
fi

echo "Migrating papers from: $PAPERS_DIR"
echo "Target: $API_URL"
echo ""

SUCCESS=0
SKIP=0
FAIL=0

for dir in "$PAPERS_DIR"/*/; do
  name=$(basename "$dir")
  md="$dir/paper.md"

  if [ ! -f "$md" ]; then
    echo "SKIP  $name (no paper.md)"
    SKIP=$((SKIP + 1))
    continue
  fi

  echo -n "UPLOAD $name... "

  RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/papers" \
    -H "Authorization: Bearer $API_TOKEN" \
    -H "Content-Type: application/json" \
    -d "$(jq -n --arg name "$name" --rawfile markdown "$md" \
      '{name: $name, markdown: $markdown}')")

  HTTP_CODE=$(echo "$RESPONSE" | tail -1)
  BODY=$(echo "$RESPONSE" | sed '$d')

  if [ "$HTTP_CODE" = "202" ]; then
    PAPER_ID=$(echo "$BODY" | jq -r '.id')
    echo "OK (id: $PAPER_ID)"
    SUCCESS=$((SUCCESS + 1))
  else
    echo "FAIL (HTTP $HTTP_CODE)"
    echo "  $BODY"
    FAIL=$((FAIL + 1))
  fi
done

echo ""
echo "Done: $SUCCESS uploaded, $SKIP skipped, $FAIL failed"
