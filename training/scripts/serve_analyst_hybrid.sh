#!/usr/bin/env bash
# Serve the Silt HYBRID analyst models (OpenAI-compatible, loopback only):
#   planner + narration : v4 adapter (exp_v4_fresh_4b_r16 it4000)      -> 127.0.0.1:8080  (ANALYST_MODEL_URL)
#   compound decomposer : P1 adapter (exp_p1_decomp_4b_r16 it3400)      -> 127.0.0.1:8081  (ANALYST_DECOMPOSER_URL)
# The adapters are used in place, read-only, and their weights are hash-checked before serving.
# Rollback to v1: training/scripts/rollback_to_v1.sh  (v1's own launcher, serve_analyst_model.sh, is unchanged).
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
A=data/training_expansion/adapters
PLANNER="$A/exp_v4_fresh_4b_r16_eval_it4000"
DECOMPOSER="$A/exp_p1_decomp_4b_r16_eval_it3400"
PLANNER_SHA=1f62f22e5e3bcd32       # first 16 hex of adapters.safetensors SHA-256 (verified 2026-11-26)
DECOMPOSER_SHA=fc6824efc4268391
LOGS=data/training_expansion/promotion/logs
mkdir -p "$LOGS"
for pair in "$PLANNER:$PLANNER_SHA" "$DECOMPOSER:$DECOMPOSER_SHA"; do
  dir=${pair%%:*}; want=${pair##*:}
  got=$(shasum -a 256 "$dir/adapters.safetensors" | cut -c1-16)
  if [ "$got" != "$want" ]; then echo "adapter hash mismatch for $dir ($got, expected $want); refusing to serve" >&2; exit 1; fi
done
BASE=mlx-community/Qwen3-4B-Instruct-2507-4bit
.venv313/bin/python -m mlx_lm server --model $BASE --adapter-path "$PLANNER" --host 127.0.0.1 --port 8080 > "$LOGS/hybrid_planner_8080.log" 2>&1 &
P1=$!
.venv313/bin/python -m mlx_lm server --model $BASE --adapter-path "$DECOMPOSER" --host 127.0.0.1 --port 8081 > "$LOGS/hybrid_decomposer_8081.log" 2>&1 &
P2=$!
echo "$P1" > "$LOGS/hybrid_planner.pid"; echo "$P2" > "$LOGS/hybrid_decomposer.pid"
trap 'kill $P1 $P2 2>/dev/null' INT TERM
for port in 8080 8081; do
  for _ in $(seq 1 90); do curl -s "http://127.0.0.1:$port/v1/models" >/dev/null && break; sleep 2; done
  curl -s "http://127.0.0.1:$port/v1/models" >/dev/null || { echo "model server on $port did not start" >&2; kill $P1 $P2; exit 1; }
done
# warm both models (the first request loads the weights)
for port in 8080 8081; do
  curl -s "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"default_model","messages":[{"role":"user","content":"ok"}],"max_tokens":1}' >/dev/null
done
echo "hybrid model servers ready: planner v4 on :8080 (pid $P1), decomposer P1 it3400 on :8081 (pid $P2)"
wait
