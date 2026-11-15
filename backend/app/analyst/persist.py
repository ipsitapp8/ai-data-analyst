"""Persistence of a conversation's analytical memory across restarts: the ordered result register and the open
target/scenario objective (app/analyst/register.py, objectives.py). Nothing else -- trained models, scopes and
per-turn state stay in process.

Isolation: one file per full memory key (team : user : dataset : document : conversation), stored under the team's
and user's own directory as the SHA-256 of the key, and loaded only when the key stored inside the file equals the
key asked for. Files are written atomically, readable by the server's user only (0600, directories 0700), and expire
after RETENTION_DAYS. The database is not touched.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

RETENTION_DAYS = 30
VERSION = 1
KEYS = ("register", "objective", "objectives")


def root() -> Path:
    from app.config import DATA_DIR  # noqa: PLC0415
    return Path(os.getenv("ANALYST_MEMORY_DIR") or DATA_DIR / "analyst_memory")


def _safe(part: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(part))[:64] or "_"


def path_for(key: str) -> Path:
    parts = key.split(":")
    team, user = (parts + ["_", "_"])[:2]
    return root() / _safe(team) / _safe(user) / f"{hashlib.sha256(key.encode()).hexdigest()}.json"


def _default(x):
    if hasattr(x, "item"):
        return x.item()                 # numpy scalars
    return str(x)


def save(key: str, memory: dict) -> None:
    payload = {k: memory.get(k) for k in KEYS if memory.get(k) is not None}
    if not payload:
        return
    p = path_for(key)
    try:
        p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        text = json.dumps({"version": VERSION, "key": key, "saved_at": time.time(), **payload}, default=_default)
        # a unique temporary file in the same directory, flushed to disk, then renamed over the old file: a crash or
        # a failed write leaves the previous memory intact, never a half-written file
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, p)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except (OSError, TypeError, ValueError):
        logger.exception("could not persist analyst memory")       # no content is logged


def load(key: str) -> dict:
    p = path_for(key)
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError, UnicodeDecodeError):
        return {}                       # missing, empty, truncated or malformed: nothing is restored
    if not isinstance(data, dict) or data.get("key") != key or data.get("version") != VERSION:
        return {}                       # another conversation's file (or an old format): never loaded
    try:
        saved = float(data.get("saved_at"))
    except (TypeError, ValueError):
        return {}                       # no readable save time: its age cannot be checked, so it is not trusted
    if time.time() - saved > RETENTION_DAYS * 86400:
        try:
            p.unlink()
        except OSError:
            pass
        return {}
    return _valid(data)


def _valid(data: dict) -> dict:
    """Only well-formed memory is restored: a register of entries with their number and question, objectives as
    objects. Anything else is dropped, never half-used."""
    out = {}
    reg = data.get("register")
    if isinstance(reg, list) and all(isinstance(e, dict) and isinstance(e.get("n"), int) and "question" in e
                                     and "kind" in e for e in reg) and \
            all(a["n"] < b["n"] for a, b in zip(reg, reg[1:])):
        out["register"] = reg
    obj = data.get("objective")
    if isinstance(obj, dict) and isinstance(obj.get("params"), dict) and "type" in obj:
        out["objective"] = obj
    objs = data.get("objectives")
    if isinstance(objs, list) and all(isinstance(o, dict) and isinstance(o.get("params"), dict) and "type" in o
                                      for o in objs):
        out["objectives"] = objs
    return out


def forget(key: str) -> None:
    try:
        path_for(key).unlink()
    except OSError:
        pass
