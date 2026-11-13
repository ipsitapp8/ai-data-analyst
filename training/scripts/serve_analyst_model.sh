#!/usr/bin/env bash
# Serve the analyst model locally (OpenAI-compatible API on 127.0.0.1:8080), which the
# AI Analyst page and /api/analyst use when it is running.
# Requests must name the model "default_model" (the planner does) so the adapter is applied.
#   ANALYST_ADAPTER=none   serve the base model without the adapter (for comparisons)
set -euo pipefail
cd "$(dirname "$0")/../.."
ADAPTER="${ANALYST_ADAPTER:-training/adapters/analyst_qwen3_4b_lora}"
ARGS=(--model mlx-community/Qwen3-4B-Instruct-2507-4bit --host 127.0.0.1 --port "${ANALYST_PORT:-8080}")
if [ "$ADAPTER" != "none" ]; then
  ARGS+=(--adapter-path "$ADAPTER")
fi
exec .venv313/bin/python -m mlx_lm server "${ARGS[@]}"
