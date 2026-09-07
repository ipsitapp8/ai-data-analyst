#!/usr/bin/env bash
# Boots both processes in one container and exits if either dies, so the
# platform restarts us instead of serving a half-dead app.
set -uo pipefail

echo "[start] launching backend on 127.0.0.1:8000"
uvicorn app.main:app --app-dir /app/backend --host 127.0.0.1 --port 8000 &
backend_pid=$!

# Don't start the UI until the API actually answers -- otherwise the first page
# load races the backend and greets the user with connection errors.
for _ in $(seq 1 90); do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2)" 2>/dev/null; then
    echo "[start] backend healthy"
    break
  fi
  if ! kill -0 "$backend_pid" 2>/dev/null; then
    echo "[start] backend exited during startup" >&2
    exit 1
  fi
  sleep 1
done

# Render (and most other PaaS Docker platforms) injects PORT and expects the
# service to bind it; Hugging Face Spaces doesn't set PORT and expects 7860.
# Preferring PORT when present lets the same image serve either without a
# platform-specific build.
listen_port="${PORT:-${STREAMLIT_SERVER_PORT:-7860}}"
echo "[start] launching streamlit on 0.0.0.0:${listen_port}"
streamlit run /app/frontend/Home.py \
  --server.port "${listen_port}" \
  --server.address 0.0.0.0 &
frontend_pid=$!

# First one to exit takes the container with it.
wait -n "$backend_pid" "$frontend_pid"
echo "[start] a process exited; shutting down" >&2
kill -TERM "$backend_pid" "$frontend_pid" 2>/dev/null
exit 1
