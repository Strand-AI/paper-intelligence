#!/bin/bash
set -euo pipefail

# Paper Intelligence Cloud — Setup Script
# Creates Cloudflare resources and deploys the Worker.
#
# Prerequisites:
#   - npm install (in this directory)
#   - wrangler login (authenticated to Cloudflare)
#   - op CLI configured (for API token)

echo "=== Paper Intelligence Cloud Setup ==="
echo ""

# 1. Create D1 database
echo "Creating D1 database..."
D1_OUTPUT=$(npx wrangler d1 create paper-intelligence 2>&1) || true
echo "$D1_OUTPUT"

# Extract database ID
D1_ID=$(echo "$D1_OUTPUT" | grep -o '"database_id": "[^"]*"' | head -1 | cut -d'"' -f4)
if [ -z "$D1_ID" ]; then
  echo "Could not extract D1 database ID. If it already exists, find it with:"
  echo "  npx wrangler d1 list"
  echo "Then update wrangler.jsonc manually."
else
  echo "D1 database ID: $D1_ID"
  # Update wrangler.jsonc
  sed -i '' "s/\"database_id\": \"TODO\"/\"database_id\": \"$D1_ID\"/" wrangler.jsonc
  echo "Updated wrangler.jsonc with D1 database ID."
fi
echo ""

# 2. Create R2 bucket
echo "Creating R2 bucket..."
npx wrangler r2 bucket create paper-intelligence 2>&1 || echo "(may already exist)"
echo ""

# 3. Create Vectorize index
echo "Creating Vectorize index..."
npx wrangler vectorize create paper-intelligence \
  --dimensions=384 \
  --metric=cosine 2>&1 || echo "(may already exist)"
echo ""

# 4. Create metadata index for paper_id filtering
echo "Creating Vectorize metadata index..."
npx wrangler vectorize create-metadata-index paper-intelligence \
  --property-name=paper_id \
  --type=string 2>&1 || echo "(may already exist)"
echo ""

# 5. Apply D1 schema
echo "Applying D1 schema..."
npx wrangler d1 execute paper-intelligence --file=schema.sql
echo ""

# 6. Set API token secret
echo "Setting API_TOKEN secret..."
API_TOKEN=$(op read "op://CLI Secrets/paper-intelligence-api-token/token" 2>/dev/null || true)
if [ -n "$API_TOKEN" ]; then
  echo "$API_TOKEN" | npx wrangler secret put API_TOKEN
  echo "API_TOKEN set from 1Password."
else
  echo "Could not read from 1Password. Set manually:"
  echo "  npx wrangler secret put API_TOKEN"
fi
echo ""

# 7. Deploy
echo "Deploying Worker..."
npx wrangler deploy
echo ""

echo "=== Setup complete ==="
echo ""
echo "MCP endpoint: https://paper-intelligence.<your-subdomain>.workers.dev/mcp"
echo ""
echo "Add to Claude Code:"
echo "  claude mcp add paper-intelligence-cloud \\"
echo "    --transport http \\"
echo "    --header 'Authorization: Bearer <your-token>' \\"
echo "    https://paper-intelligence.<your-subdomain>.workers.dev/mcp"
