"""Sandboxed execution of agent-generated Python code.

Contract the generated code must follow (see agents/prompts.py):
  - Read the dataset from /workspace/data/input.csv
  - Print exactly one line "RESULT_JSON:<json>" with any structured result
  - Save any Plotly figures via fig.write_json("/workspace/output/<name>.json")

DockerSandboxRunner is the real, isolated execution path: a `docker run`
with no network access, capped memory/CPU, and a non-root user, invoked
fresh per attempt. SubprocessSandboxRunner is a dev-only fallback with no
real isolation guarantees, used only if Docker is unavailable.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from app import config

RESULT_MARKER = "RESULT_JSON:"


@dataclass
class SandboxResult:
    success: bool
    stdout: str
    stderr: str
    exit_code: int | None
    result: dict | None
    chart_paths: list[str] = field(default_factory=list)
    timed_out: bool = False


class SandboxRunner:
    """Common workspace management shared by both backends."""

    def workspace_for(self, question_id: int) -> Path:
        ws = config.WORKSPACES_DIR / f"q{question_id}"
        (ws / "data").mkdir(parents=True, exist_ok=True)
        (ws / "output").mkdir(parents=True, exist_ok=True)
        return ws

    def seed_input_data(self, question_id: int, source_csv_path: str) -> Path:
        ws = self.workspace_for(question_id)
        dest = ws / "data" / "input.csv"
        if not dest.exists():
            shutil.copyfile(source_csv_path, dest)
        return dest

    def _parse_stdout(self, stdout: str) -> dict | None:
        for line in stdout.splitlines():
            if line.startswith(RESULT_MARKER):
                raw = line[len(RESULT_MARKER):].strip()
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return {"_unparseable_result": raw}
        return None

    def run(self, question_id: int, code: str, step_index: int, attempt: int) -> SandboxResult:
        raise NotImplementedError


class DockerSandboxRunner(SandboxRunner):
    def run(self, question_id: int, code: str, step_index: int, attempt: int) -> SandboxResult:
        ws = self.workspace_for(question_id)
        output_dir = ws / "output"
        before = {p.name for p in output_dir.glob("*")}

        script_path = ws / f"step{step_index}_attempt{attempt}.py"
        script_path.write_text(code, encoding="utf-8")

        docker_bin = shutil.which("docker") or "docker"
        cmd = [
            docker_bin, "run", "--rm",
            "--network", "none",
            "--memory", config.SANDBOX_MEMORY_LIMIT,
            "--cpus", config.SANDBOX_CPU_LIMIT,
            "-v", f"{ws}:/workspace",
            "-w", "/workspace",
            config.SANDBOX_IMAGE,
            f"/workspace/{script_path.name}",
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.SANDBOX_TIMEOUT_SECONDS,
            )
            stdout, stderr, exit_code, timed_out = proc.stdout, proc.stderr, proc.returncode, False
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout or ""
            stderr = (e.stderr or "") + "\n[sandbox] execution timed out"
            exit_code = None
            timed_out = True
        except FileNotFoundError:
            return SandboxResult(
                success=False, stdout="", stderr=(
                    "[sandbox] Docker executable not found. Install/start Docker Desktop, "
                    "or set SANDBOX_BACKEND=subprocess for an (unsandboxed) dev fallback."
                ), exit_code=None, result=None,
            )

        after = {p.name for p in output_dir.glob("*")}
        new_files = sorted(after - before)
        chart_paths = [str((output_dir / f).resolve()) for f in new_files if f.endswith(".json")]

        success = (exit_code == 0) and not timed_out
        return SandboxResult(
            success=success,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            result=self._parse_stdout(stdout) if success else None,
            chart_paths=chart_paths,
            timed_out=timed_out,
        )


def _sandbox_env(workspace: Path) -> dict[str, str]:
    """Minimal environment handed to agent-generated code.

    Built from scratch rather than inherited from os.environ on purpose: this
    process holds GEMINI_API_KEY / LLAMA_API_KEY, and generated code has no
    business seeing them. Everything a legitimate analysis script needs is
    listed here explicitly; anything not listed simply isn't visible.
    """
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(workspace),
        "TMPDIR": str(workspace / "tmp"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        # matplotlib: headless backend, and a config dir it can actually write to
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(workspace / "tmp"),
        # keep BLAS from spawning a thread per core on a shared box
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    if os.name == "nt":
        # CPython won't start on Windows without these.
        for key in ("SYSTEMROOT", "COMSPEC", "PATHEXT", "TEMP", "TMP", "USERPROFILE"):
            if key in os.environ:
                env[key] = os.environ[key]
    return env


def _parse_size(text: str, default: int) -> int:
    """'512m' / '2g' / '1048576' -> bytes. Falls back to default if unparseable."""
    text = (text or "").strip().lower()
    if not text:
        return default
    units = {"k": 1024, "m": 1024 ** 2, "g": 1024 ** 3}
    mult = units.get(text[-1], 1)
    digits = text[:-1] if mult > 1 else text
    try:
        return int(float(digits) * mult)
    except ValueError:
        return default


def _preexec_limits():
    """POSIX-only resource caps applied in the child between fork and exec.

    Address space is deliberately opt-in (SANDBOX_ADDRESS_SPACE_LIMIT): numpy and
    OpenBLAS reserve a large virtual arena at import time, so an RLIMIT_AS sized
    like a container memory limit makes `import pandas` itself fail. CPU, file
    size and process count are safe to always apply.
    """
    if os.name == "nt":
        return None

    import resource

    cpu_seconds = config.SANDBOX_TIMEOUT_SECONDS + 5
    fsize = _parse_size(config.SANDBOX_MAX_OUTPUT_BYTES, 256 * 1024 ** 2)
    address_space = _parse_size(config.SANDBOX_ADDRESS_SPACE_LIMIT, 0)

    def _apply() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
        resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if address_space:
            resource.setrlimit(resource.RLIMIT_AS, (address_space, address_space))

    return _apply


def _drop_privileges_kwargs() -> dict:
    """Run the child as an unprivileged user, when we're root and one is configured.

    On a deployed box this is what keeps generated code away from app.db and the
    application source: that user owns nothing outside its own workspace.
    """
    user = config.SANDBOX_RUN_AS_USER
    if not user or os.name == "nt":
        return {}
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        # Not root -- we can't switch user, and asking would just raise.
        return {}

    import pwd

    try:
        entry = pwd.getpwnam(user)
    except KeyError:
        return {}
    # Set the group and clear supplementary groups too: `user=` alone only calls
    # setuid, which would leave the child sitting in root's groups.
    return {"user": entry.pw_uid, "group": entry.pw_gid, "extra_groups": []}


def _chown_workspace(workspace: Path) -> None:
    """Hand the workspace to the sandbox user so the dropped-privilege child can write.

    No-op unless we're actually going to switch user. Only the workspace moves --
    the database, uploads and application source stay owned by us.
    """
    kwargs = _drop_privileges_kwargs()
    if "user" not in kwargs:
        return
    uid, gid = kwargs["user"], kwargs["group"]
    for path in [workspace, *workspace.rglob("*")]:
        try:
            os.chown(path, uid, gid)
        except OSError:
            pass


class SubprocessSandboxRunner(SandboxRunner):
    """Fallback for hosts that can't give us Docker (most PaaS containers).

    Weaker than the Docker path and not a substitute for it -- there is no
    filesystem or network namespace here. What it does guarantee: generated code
    gets a scrubbed environment (no API keys), runs in its own session under
    CPU/file-size/process caps, is killed as a process group on timeout, and --
    where the host allows it -- runs as a separate unprivileged user.
    """

    def run(self, question_id: int, code: str, step_index: int, attempt: int) -> SandboxResult:
        ws = self.workspace_for(question_id)
        output_dir = ws / "output"
        (ws / "tmp").mkdir(exist_ok=True)
        before = {p.name for p in output_dir.glob("*")}

        script_path = ws / f"step{step_index}_attempt{attempt}.py"
        script_path.write_text(code, encoding="utf-8")
        _chown_workspace(ws)

        popen_kwargs = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "cwd": str(ws),
            "env": _sandbox_env(ws),
            # own process group, so a timeout kills grandchildren too
            "start_new_session": True,
            **_drop_privileges_kwargs(),
        }
        preexec = _preexec_limits()
        if preexec is not None:
            popen_kwargs["preexec_fn"] = preexec

        # Not -I/-E: those ignore inherited PYTHON* vars, which is moot now that we
        # pass an explicit env dict (strictly stronger), and -E breaks interpreters
        # whose stdlib lookup leans on the environment -- notably a venv with a
        # stale pyvenv.cfg, where -E fails to import `encodings` and nothing runs.
        flags = ["-s", "-B"]
        if sys.version_info >= (3, 11):
            flags.append("-P")  # keep the workspace dir off sys.path
        proc = subprocess.Popen([sys.executable, *flags, str(script_path)], **popen_kwargs)
        try:
            stdout, stderr = proc.communicate(timeout=config.SANDBOX_TIMEOUT_SECONDS)
            exit_code, timed_out = proc.returncode, False
        except subprocess.TimeoutExpired:
            _kill_process_group(proc)
            stdout, stderr = proc.communicate()
            stderr = (stderr or "") + "\n[sandbox] execution timed out"
            exit_code, timed_out = None, True

        after = {p.name for p in output_dir.glob("*")}
        new_files = sorted(after - before)
        chart_paths = [str((output_dir / f).resolve()) for f in new_files if f.endswith(".json")]

        success = (exit_code == 0) and not timed_out
        return SandboxResult(
            success=success,
            stdout=stdout or "",
            stderr=stderr or "",
            exit_code=exit_code,
            result=self._parse_stdout(stdout or "") if success else None,
            chart_paths=chart_paths,
            timed_out=timed_out,
        )


def _kill_process_group(proc: subprocess.Popen) -> None:
    """SIGKILL the child's whole session; fall back to the bare child."""
    try:
        if os.name == "nt":
            proc.kill()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def get_runner() -> SandboxRunner:
    if config.SANDBOX_BACKEND == "subprocess":
        return SubprocessSandboxRunner()
    return DockerSandboxRunner()


def docker_image_available() -> bool:
    docker_bin = shutil.which("docker")
    if not docker_bin:
        return False
    try:
        proc = subprocess.run(
            [docker_bin, "image", "inspect", config.SANDBOX_IMAGE],
            capture_output=True, text=True, timeout=10,
        )
        return proc.returncode == 0
    except Exception:
        return False
