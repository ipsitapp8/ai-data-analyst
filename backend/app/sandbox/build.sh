#!/usr/bin/env bash
# Builds the sandbox Docker image used to run all AI-generated analysis code.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker build -t ai-data-analyst-sandbox:latest -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR"
echo "Built image: ai-data-analyst-sandbox:latest"
