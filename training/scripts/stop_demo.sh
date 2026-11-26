#!/usr/bin/env bash
# Stop the UI, the API and the hybrid model servers started by start_demo.sh.
cd "$(dirname "$0")/../.."
for p in 8501 8000 8080 8081; do pid=$(lsof -nP -iTCP:$p -sTCP:LISTEN -t 2>/dev/null); [ -n "$pid" ] && kill $pid && echo "stopped :$p (pid $pid)"; done; exit 0
