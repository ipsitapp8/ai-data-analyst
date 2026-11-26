#!/usr/bin/env bash
# Roll the Copilot back to the v1 adapter: stop everything, remove the hybrid lines from .env (a copy of the current
# .env is kept), then serve v1 with its unchanged launcher on :8080 and restart the API and UI.
set -euo pipefail
cd "$(dirname "$0")/../.."
bash training/scripts/stop_demo.sh
cp -p .env ".env.bak-hybrid-$(date +%Y%m%d-%H%M)"
python3 - <<'PY'
import re
s = open(".env").read()
s = re.sub(r"\n# --- Silt hybrid model configuration.*?\n(?:ANALYST_(?:MODEL_URL|DECOMPOSER_URL|MODEL_LABEL)=.*\n?)+", "", s, flags=re.S)
open(".env", "w").write(s)
PY
grep -q "^ANALYST_DECOMPOSER_URL" .env && { echo "could not remove the hybrid lines from .env" >&2; exit 1; }
LOGS=data/training_expansion/promotion/logs
nohup bash training/scripts/serve_analyst_model.sh > "$LOGS/v1_server_8080.log" 2>&1 &
for _ in $(seq 1 90); do curl -s http://127.0.0.1:8080/v1/models >/dev/null && break; sleep 2; done
nohup .venv313/bin/python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 > "$LOGS/app_backend_8000.log" 2>&1 &
nohup .venv313/bin/python -m streamlit run frontend/Home.py --server.port 8501 --server.headless true > "$LOGS/app_frontend_8501.log" 2>&1 < /dev/null &
echo "rolled back: v1 on :8080, API :8000, UI :8501"
