#!/usr/bin/env bash
# Start the whole Silt app with the HYBRID models: both model servers, the API (:8000) and the Streamlit UI (:8501).
# Fails loudly -- never silently model-off -- if the backend cannot reach both model servers.
#   bash training/scripts/start_demo.sh          (stop: bash training/scripts/stop_demo.sh)
set -euo pipefail
cd "$(dirname "$0")/../.."
LOGS=data/training_expansion/promotion/logs; mkdir -p "$LOGS"
if ! curl -s http://127.0.0.1:8080/v1/models >/dev/null || ! curl -s http://127.0.0.1:8081/v1/models >/dev/null; then
  nohup bash training/scripts/serve_analyst_hybrid.sh > "$LOGS/hybrid_servers.log" 2>&1 < /dev/null &
  for _ in $(seq 1 150); do grep -q "hybrid model servers ready" "$LOGS/hybrid_servers.log" 2>/dev/null && break
    grep -q "refusing\|did not start" "$LOGS/hybrid_servers.log" 2>/dev/null && { cat "$LOGS/hybrid_servers.log"; exit 1; }; sleep 2; done
fi
grep -q "hybrid model servers ready" "$LOGS/hybrid_servers.log" 2>/dev/null || curl -s http://127.0.0.1:8081/v1/models >/dev/null \
  || { echo "model servers not ready; see $LOGS" >&2; exit 1; }
nohup .venv313/bin/python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 > "$LOGS/app_backend_8000.log" 2>&1 < /dev/null &
echo $! > "$LOGS/app_backend.pid"
for _ in $(seq 1 60); do curl -s http://127.0.0.1:8000/api/health >/dev/null && break; sleep 1; done
nohup .venv313/bin/python -m streamlit run frontend/Home.py --server.port 8501 --server.headless true \
  --browser.gatherUsageStats false > "$LOGS/app_frontend_8501.log" 2>&1 < /dev/null &
echo $! > "$LOGS/app_frontend.pid"
for _ in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:8501 && break; sleep 1; done
.venv313/bin/python -I - <<'PY'
import os, sys, urllib.request
sys.path.insert(0, "backend")
from app import config  # noqa: F401  (loads .env exactly as the backend does)
for name, url in (("planner", os.getenv("ANALYST_MODEL_URL")), ("decomposer", os.getenv("ANALYST_DECOMPOSER_URL"))):
    if not url:
        sys.exit(f"{name} URL is not configured in .env")
    try:
        urllib.request.urlopen(url.rstrip("/") + "/models", timeout=3)
    except Exception as e:
        sys.exit(f"{name} at {url} is NOT reachable ({e}); the Copilot would run without it")
    print(f"{name}: {url} reachable")
PY
echo "Silt is up: UI http://localhost:8501  API http://127.0.0.1:8000  (hybrid: v4 planner :8080, P1 it3400 decomposer :8081)"
