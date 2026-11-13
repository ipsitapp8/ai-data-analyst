"""Which model is actually being served, read from the model server itself.

The label in planner.MODEL_LABEL is a claim; this checks it. The process that
listens on the model port is found with lsof, and its command line gives the
base model and the --adapter-path; the adapter weights are hashed and compared
with the frozen v1 adapter. Cached for a minute (hashing ~70 MB is not free).
"""
from __future__ import annotations

import hashlib
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

V1_ADAPTER = "training/adapters/analyst_qwen3_4b_lora"
V1_SHA256 = "38b91af3327099b10d6f08f7a23fe3a9d9e9facdbea047e2e1c12f5e02a65b17"
REPO = Path(__file__).resolve().parents[3]
_CACHE: dict = {}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def served(model_url: str) -> dict:
    now = time.monotonic()
    if _CACHE.get("at") and now - _CACHE["at"] < 60:
        return _CACHE["info"]
    info = {"verified": False}
    try:
        port = urlparse(model_url).port or 8080
        pids = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], capture_output=True, text=True,
                              timeout=5).stdout.split()
        if not pids:
            info["reason"] = "no process is listening on the model port"
        else:
            cmd = subprocess.run(["ps", "-o", "command=", "-p", pids[0]], capture_output=True, text=True, timeout=5).stdout
            args = shlex.split(cmd)
            def opt(name):
                return args[args.index(name) + 1] if name in args and args.index(name) + 1 < len(args) else None
            base, adapter = opt("--model"), opt("--adapter-path")
            info.update(pid=int(pids[0]), base_model=base, adapter=adapter, server="mlx_lm" if "mlx_lm" in cmd else cmd[:60])
            if adapter:
                path = Path(adapter) if os.path.isabs(adapter) else REPO / adapter
                weights = path / "adapters.safetensors"
                digest = _sha(weights) if weights.exists() else None
                info.update(adapter_sha256=digest, adapter_is_v1=(digest == V1_SHA256 and
                                                                    re.sub(r"/+$", "", adapter).endswith(V1_ADAPTER)))
            else:
                info["adapter_is_v1"] = False
            info["verified"] = True
    except Exception as e:  # noqa: BLE001 - identity is informational; never break the page for it
        info["reason"] = f"{type(e).__name__}"
    _CACHE.update(at=now, info=info)
    return info
