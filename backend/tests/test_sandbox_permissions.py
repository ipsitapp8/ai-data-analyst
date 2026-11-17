"""The unprivileged sandbox user must be able to reach its run directory whatever umask the app runs under."""
import os
import stat

import pytest

from app import config
from app.sandbox import runner

posix_only = pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")


@posix_only
def test_run_directories_stay_traversable_under_a_strict_umask(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORKSPACES_DIR", tmp_path / "workspaces")
    old = os.umask(0o077)               # what a hardened container may use: new directories are root-only
    try:
        r = runner.SubprocessSandboxRunner()
        ws = r.workspace_for(2)
        exec_dir = ws / "runs" / "s0_a1_deadbeef"
        (exec_dir / "out").mkdir(parents=True)
        assert stat.S_IMODE(ws.stat().st_mode) == 0o700, "the strict umask really does make the directory root-only"
        runner._make_traversable(exec_dir, ws)
    finally:
        os.umask(old)
    for path in (config.WORKSPACES_DIR, ws, ws / "runs", exec_dir, ws / "data"):
        assert stat.S_IMODE(path.stat().st_mode) & 0o111 == 0o111, f"{path} must be searchable by the sandbox user"
    for path in (ws, ws / "runs", ws / "data"):
        assert not stat.S_IMODE(path.stat().st_mode) & 0o022, "nothing is made writable"
        assert not stat.S_IMODE(path.stat().st_mode) & 0o004, "and nothing is made listable"
