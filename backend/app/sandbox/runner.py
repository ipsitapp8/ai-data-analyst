"""Sandboxed execution of agent-generated Python code.

Contract the generated code must follow (see agents/prompts.py):
  - Read the dataset from /workspace/data/input.csv
  - Print exactly one line "RESULT_JSON:<json>" with any structured result
  - Print exactly one line "DATA_SLICE_JSON:<json>" with the {columns, rows}
    subset of the dataframe the result was computed from
  - Save any Plotly figures via fig.write_json("/workspace/output/<name>.json")

Generated code is hostile input: it is written by a model that read a user's
question and a user's CSV. Two runners exist and only one is a boundary.

DockerSandboxRunner -- the security boundary. One fresh container per
execution with: no network, read-only root filesystem, all capabilities
dropped, no-new-privileges, pid / memory / CPU / file-size limits, a non-root
user, the dataset mounted read-only and a private output directory that
exists only for that execution. Set SANDBOX_DOCKER_RUNTIME=runsc to run the
container under gVisor.

SubprocessSandboxRunner -- NOT a boundary. A child process on the host shares
its kernel, filesystem and network. It exists for local development and for
tests of the pipeline's own logic, and is refused unless
SANDBOX_ALLOW_UNSAFE_SUBPROCESS=true.

If the configured boundary is not available the run fails (SandboxUnavailable);
nothing ever falls back to a weaker runner on its own.

Every execution, on either runner:
  - gets its own directory, deleted afterwards, so one run cannot read what
    another wrote;
  - is killed on timeout, on cancellation, and when it prints too much;
  - has its artifacts validated before anything leaves that directory.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app import config, runtime

logger = logging.getLogger(__name__)

RESULT_MARKER = "RESULT_JSON:"
DATA_SLICE_MARKER = "DATA_SLICE_JSON:"
_ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,79}\.json$")
_POLL_SECONDS = 0.2


class SandboxUnavailable(runtime.RunAborted):
    """The configured isolation boundary cannot be used. Fail closed."""

    failure_class = "sandbox_unavailable"
    user_message = ("The secure code sandbox is not available, so this analysis was not run. "
                    "Ask an administrator to check the sandbox configuration.")


@dataclass
class SandboxResult:
    success: bool
    stdout: str
    stderr: str
    exit_code: int | None
    result: dict | None
    chart_paths: list[str] = field(default_factory=list)
    timed_out: bool = False
    data_slice: dict | None = None
    cancelled: bool = False
    output_truncated: bool = False
    rejected_artifacts: list[str] = field(default_factory=list)
    isolation: str = ""


@dataclass
class _RawRun:
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool = False
    cancelled: bool = False
    overflow: bool = False


class _Pump(threading.Thread):
    """Drain one pipe, keeping at most `limit` bytes and counting the rest."""

    def __init__(self, stream, limit: int):
        super().__init__(daemon=True)
        self.stream, self.limit = stream, limit
        self.chunks: list[bytes] = []
        self.kept = 0
        self.total = 0

    def run(self) -> None:
        try:
            while True:
                chunk = self.stream.read(65536)
                if not chunk:
                    break
                self.total += len(chunk)
                room = self.limit - self.kept
                if room > 0:
                    self.chunks.append(chunk[:room])
                    self.kept += min(len(chunk), room)
        except (OSError, ValueError):
            pass

    def text(self) -> str:
        return b"".join(self.chunks).decode("utf-8", errors="replace")


def _supervise(proc: subprocess.Popen, kill, timeout: float, output_limit: int) -> _RawRun:
    """Wait for the child while enforcing the wall clock, cancellation and the
    output cap. `kill` must terminate the whole execution, not just `proc`."""
    out, err = _Pump(proc.stdout, output_limit), _Pump(proc.stderr, output_limit)
    out.start()
    err.start()
    deadline = time.monotonic() + timeout
    timed_out = cancelled = overflow = False
    while proc.poll() is None:
        if time.monotonic() >= deadline:
            timed_out = True
        elif runtime.is_cancel_requested():
            cancelled = True
        elif out.total > output_limit * 4 or err.total > output_limit * 4:
            overflow = True
        if timed_out or cancelled or overflow:
            kill()
            break
        time.sleep(_POLL_SECONDS)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    out.join(timeout=5)
    err.join(timeout=5)
    stderr = err.text()
    if timed_out:
        stderr += "\n[sandbox] execution timed out"
    if cancelled:
        stderr += "\n[sandbox] execution cancelled"
    if overflow:
        stderr += "\n[sandbox] execution stopped: too much output"
    exit_code = None if (timed_out or cancelled or overflow) else proc.returncode
    return _RawRun(out.text(), stderr, exit_code, timed_out, cancelled,
                   overflow or out.total > out.kept or err.total > err.kept)


class SandboxRunner:
    """Workspace layout, artifact handling and bookkeeping shared by both runners.

        <WORKSPACES_DIR>/q<id>/data/input.csv        dataset copy, read-only
        <WORKSPACES_DIR>/q<id>/runs/<run id>/         one execution; deleted after
        <WORKSPACES_DIR>/q<id>/artifacts/<run id>/    validated charts
    """

    name = "base"
    isolation = "none"

    # ------------------------------------------------------------ workspace --
    def workspace_for(self, question_id: int) -> Path:
        ws = config.WORKSPACES_DIR / f"q{int(question_id)}"
        for sub in ("data", "runs", "artifacts"):
            (ws / sub).mkdir(parents=True, exist_ok=True)
        return ws

    def seed_input_data(self, question_id: int, source_csv_path: str) -> Path:
        ws = self.workspace_for(question_id)
        dest = ws / "data" / "input.csv"
        if not dest.exists():
            shutil.copyfile(source_csv_path, dest)
            try:
                os.chmod(dest, 0o444)
            except OSError:
                pass
        return dest

    def cleanup_workspace(self, question_id: int) -> None:
        """Remove everything a question's executions left on disk."""
        ws = config.WORKSPACES_DIR / f"q{int(question_id)}"
        shutil.rmtree(ws, ignore_errors=True)

    # --------------------------------------------------------------- output --
    def _parse_stdout(self, stdout: str, marker: str = RESULT_MARKER) -> dict | None:
        for line in stdout.splitlines():
            if line.startswith(marker):
                raw = line[len(marker):].strip()
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return {"_unparseable_result": raw[:2000]}
        return None

    def _collect_artifacts(self, out_dir: Path, dest_dir: Path) -> tuple[list[str], list[str]]:
        """Validate what the execution wrote and copy the good files out.

        Nothing is trusted: symlinks and non-regular files are skipped (never
        followed), names must be plain, sizes are capped, and a file is only
        kept if it parses as a Plotly figure. Returns (kept paths, rejections)."""
        kept: list[str] = []
        rejected: list[str] = []
        try:
            entries = sorted(os.scandir(out_dir), key=lambda e: e.name)
        except OSError:
            return kept, rejected
        for entry in entries:
            name = entry.name
            if len(kept) >= config.SANDBOX_MAX_ARTIFACTS:
                rejected.append(f"{name[:80]}: too many artifacts")
                continue
            try:
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    rejected.append(f"{name[:80]}: not a regular file")
                    continue
                if not _ARTIFACT_NAME.match(name):
                    rejected.append(f"{name[:80]}: name or type not allowed")
                    continue
                size = entry.stat(follow_symlinks=False).st_size
                if size > config.SANDBOX_MAX_ARTIFACT_BYTES:
                    rejected.append(f"{name}: larger than {config.SANDBOX_MAX_ARTIFACT_BYTES} bytes")
                    continue
                with open(entry.path, "rb") as f:
                    payload = json.loads(f.read().decode("utf-8"))
                if not (isinstance(payload, dict) and isinstance(payload.get("data"), list)):
                    rejected.append(f"{name}: not a Plotly figure")
                    continue
            except (OSError, ValueError, UnicodeDecodeError):
                rejected.append(f"{name[:80]}: unreadable or not valid JSON")
                continue
            dest_dir.mkdir(parents=True, exist_ok=True)
            target = dest_dir / name
            # Re-serialise rather than copy bytes: what is stored is exactly
            # what was validated.
            target.write_text(json.dumps(payload), encoding="utf-8")
            kept.append(str(target.resolve()))
        return kept, rejected

    # ------------------------------------------------------------------ run --
    def _execute(self, ws: Path, exec_dir: Path, code: str, timeout: float) -> _RawRun:
        raise NotImplementedError

    def ensure_available(self) -> None:
        """Raise SandboxUnavailable if this runner must not be used."""

    def run(self, question_id: int, code: str, step_index: int, attempt: int) -> SandboxResult:
        runtime.checkpoint(about_to="sandbox")
        self.ensure_available()
        started = time.monotonic()
        if not isinstance(code, str) or "\x00" in code or len(code.encode("utf-8", "replace")) > config.SANDBOX_MAX_CODE_BYTES:
            return SandboxResult(False, "", "[sandbox] script rejected: empty, too large or not text",
                                 None, None, isolation=self.isolation)

        ws = self.workspace_for(question_id)
        run_id = f"s{step_index}_a{attempt}_{uuid.uuid4().hex[:8]}"
        exec_dir = ws / "runs" / run_id
        out_dir = exec_dir / "out"
        out_dir.mkdir(parents=True)
        timeout = runtime.remaining_seconds(float(config.SANDBOX_TIMEOUT_SECONDS))
        try:
            raw = self._execute(ws, exec_dir, code, timeout)
            charts, rejected = self._collect_artifacts(out_dir, ws / "artifacts" / run_id)
        finally:
            shutil.rmtree(exec_dir, ignore_errors=True)

        success = raw.exit_code == 0 and not (raw.timed_out or raw.cancelled)
        stderr = raw.stderr
        if raw.exit_code in (137, -9) and not raw.timed_out:
            stderr += "\n[sandbox] process was killed (memory or another resource limit)"
        result = SandboxResult(
            success=success,
            stdout=raw.stdout,
            stderr=stderr,
            exit_code=raw.exit_code,
            result=self._parse_stdout(raw.stdout) if success else None,
            chart_paths=charts if success else [],
            timed_out=raw.timed_out,
            data_slice=self._parse_stdout(raw.stdout, DATA_SLICE_MARKER) if success else None,
            cancelled=raw.cancelled,
            output_truncated=raw.overflow,
            rejected_artifacts=rejected,
            isolation=self.isolation,
        )
        runtime.note_sandbox_run(self.name, step_index, attempt, success, raw.timed_out,
                                 int((time.monotonic() - started) * 1000),
                                 {"rejected_artifacts": len(rejected), "output_truncated": raw.overflow})
        if raw.cancelled:
            runtime.checkpoint()  # raises RunCancelled / DeadlineExceeded
        return result


# ---------------------------------------------------------------- docker --

_docker_ok_until = 0.0


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
    except Exception:  # noqa: BLE001
        return False


class DockerSandboxRunner(SandboxRunner):
    name = "docker"

    @property
    def isolation(self) -> str:  # type: ignore[override]
        runtime_name = config.SANDBOX_DOCKER_RUNTIME
        return f"docker container ({runtime_name})" if runtime_name else "docker container"

    def ensure_available(self) -> None:
        global _docker_ok_until
        if time.monotonic() < _docker_ok_until:
            return
        if not docker_image_available():
            raise SandboxUnavailable(
                f"Docker or the sandbox image '{config.SANDBOX_IMAGE}' is not available")
        _docker_ok_until = time.monotonic() + 60

    def _command(self, ws: Path, exec_dir: Path, container: str) -> list[str]:
        docker_bin = shutil.which("docker") or "docker"
        fsize = _parse_size(config.SANDBOX_MAX_OUTPUT_BYTES, 256 * 1024 ** 2)
        cmd = [
            docker_bin, "run", "--rm", "--name", container,
            "--network", "none",              # no egress at all: no internet, no host, no metadata endpoint
            "--read-only",                    # root filesystem is immutable
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", str(config.SANDBOX_PIDS_LIMIT),
            "--memory", config.SANDBOX_MEMORY_LIMIT,
            "--memory-swap", config.SANDBOX_MEMORY_LIMIT,  # equal to --memory: no swap
            "--cpus", config.SANDBOX_CPU_LIMIT,
            "--ulimit", f"fsize={fsize}:{fsize}",
            "--ulimit", "nofile=256:256",
            "--ulimit", "core=0:0",
            "--ipc", "none",
            "--tmpfs", f"/tmp:rw,noexec,nosuid,nodev,size={config.SANDBOX_TMPFS_SIZE}",
            "-e", "HOME=/tmp", "-e", "MPLCONFIGDIR=/tmp", "-e", "MPLBACKEND=Agg",
            "-e", "OPENBLAS_NUM_THREADS=1", "-e", "OMP_NUM_THREADS=1",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-v", f"{ws / 'data'}:/workspace/data:ro",
            "-v", f"{exec_dir / 'out'}:/workspace/output:rw",
            "-v", f"{exec_dir / 'main.py'}:/workspace/main.py:ro",
            "-w", "/tmp",
        ]
        if config.SANDBOX_DOCKER_RUNTIME:
            cmd += ["--runtime", config.SANDBOX_DOCKER_RUNTIME]
        # -I: isolated mode (no user site, no PYTHON* variables, cwd not on sys.path)
        cmd += [config.SANDBOX_IMAGE, "-I", "-B", "/workspace/main.py"]
        return cmd

    def _execute(self, ws: Path, exec_dir: Path, code: str, timeout: float) -> _RawRun:
        (exec_dir / "main.py").write_text(code, encoding="utf-8")
        try:
            # The container user is not the host user; it needs to write here.
            os.chmod(exec_dir / "out", 0o777)
        except OSError:
            pass
        container = f"aida-sbx-{uuid.uuid4().hex[:16]}"
        docker_bin = shutil.which("docker") or "docker"
        try:
            proc = subprocess.Popen(self._command(ws, exec_dir, container),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError as e:
            raise SandboxUnavailable("Docker executable not found") from e

        def kill() -> None:
            # Killing the `docker run` client does not stop the container.
            subprocess.run([docker_bin, "kill", container], capture_output=True, timeout=20)
            try:
                proc.kill()
            except OSError:
                pass

        try:
            return _supervise(proc, kill, timeout, config.SANDBOX_MAX_STDOUT_BYTES)
        finally:
            # --rm normally removes it; this covers a client that died first.
            subprocess.run([docker_bin, "rm", "-f", container], capture_output=True, timeout=20)


# ------------------------------------------------------------ subprocess --

def _sandbox_env(workspace: Path) -> dict[str, str]:
    """Minimal environment handed to agent-generated code.

    Built from scratch rather than inherited from os.environ on purpose: this
    process holds GEMINI_API_KEY / LLAMA_API_KEY, and generated code has no
    business seeing them."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(workspace),
        "TMPDIR": str(workspace / "tmp"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(workspace / "tmp"),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    if os.name == "nt":
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
    """POSIX-only resource caps applied in the child between fork and exec."""
    if os.name == "nt":
        return None
    import resource

    cpu_seconds = config.SANDBOX_TIMEOUT_SECONDS + 5
    fsize = _parse_size(config.SANDBOX_MAX_OUTPUT_BYTES, 256 * 1024 ** 2)
    address_space = _parse_size(config.SANDBOX_ADDRESS_SPACE_LIMIT, 0)

    def _apply() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))
        except (ValueError, OSError):
            pass  # some platforms refuse to lower it for the current user
        if address_space:
            resource.setrlimit(resource.RLIMIT_AS, (address_space, address_space))

    return _apply


def _drop_privileges_kwargs() -> dict:
    """Run the child as an unprivileged user, when we're root and one is configured."""
    user = config.SANDBOX_RUN_AS_USER
    if not user or os.name == "nt":
        return {}
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        return {}
    import pwd

    try:
        entry = pwd.getpwnam(user)
    except KeyError:
        return {}
    return {"user": entry.pw_uid, "group": entry.pw_gid, "extra_groups": []}


def _chown_tree(root: Path) -> None:
    kwargs = _drop_privileges_kwargs()
    if "user" not in kwargs:
        return
    uid, gid = kwargs["user"], kwargs["group"]
    for path in [root, *root.rglob("*")]:
        try:
            os.chown(path, uid, gid)
        except OSError:
            pass


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


class SubprocessSandboxRunner(SandboxRunner):
    """Development-only runner. NOT a security boundary: the child shares the
    host kernel, filesystem and network. What it does do: scrubbed environment
    (no API keys), own process group killed on timeout, CPU / file-size /
    process caps, optional drop to an unprivileged user, per-run directory."""

    name = "subprocess"
    isolation = "none (unsafe development runner)"

    def ensure_available(self) -> None:
        if not config.SANDBOX_ALLOW_UNSAFE_SUBPROCESS:
            raise SandboxUnavailable(
                "SANDBOX_BACKEND=subprocess is not an isolation boundary and is disabled. "
                "Use SANDBOX_BACKEND=docker, or set SANDBOX_ALLOW_UNSAFE_SUBPROCESS=true "
                "for local development only.")

    def _execute(self, ws: Path, exec_dir: Path, code: str, timeout: float) -> _RawRun:
        (exec_dir / "tmp").mkdir(exist_ok=True)
        # Generated scripts use the container paths; map them onto this run's
        # directory. One pass over the original text: chained str.replace would
        # rewrite a host path that itself happens to contain "/workspace".
        targets = {"/workspace/data": (ws / "data").as_posix(),
                   "/workspace/output": (exec_dir / "out").as_posix(),
                   "/workspace": exec_dir.as_posix()}
        host_code = re.sub(r"/workspace(?:/data|/output)?(?![A-Za-z0-9_])", lambda m: targets[m.group(0)], code)
        script_path = exec_dir / "main.py"
        script_path.write_text(host_code, encoding="utf-8")
        _chown_tree(exec_dir)

        popen_kwargs = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "cwd": str(exec_dir),
            "env": _sandbox_env(exec_dir),
            "start_new_session": True,  # own process group, so a timeout kills grandchildren too
            **_drop_privileges_kwargs(),
        }
        preexec = _preexec_limits()
        if preexec is not None:
            popen_kwargs["preexec_fn"] = preexec
        flags = ["-s", "-B"]
        if sys.version_info >= (3, 11):
            flags.append("-P")  # keep the run directory off sys.path
        proc = subprocess.Popen([sys.executable, *flags, str(script_path)], **popen_kwargs)
        return _supervise(proc, lambda: _kill_process_group(proc), timeout, config.SANDBOX_MAX_STDOUT_BYTES)


def get_runner() -> SandboxRunner:
    """The configured runner. Raises SandboxUnavailable instead of ever
    returning a weaker runner than the one asked for."""
    backend = (config.SANDBOX_BACKEND or "").strip().lower()
    if backend == "docker":
        return DockerSandboxRunner()
    if backend == "subprocess":
        runner = SubprocessSandboxRunner()
        runner.ensure_available()
        return runner
    raise SandboxUnavailable(f"Unknown SANDBOX_BACKEND '{backend}'")


def sandbox_status() -> dict:
    """What the health endpoint and the Settings page report."""
    backend = (config.SANDBOX_BACKEND or "").strip().lower()
    try:
        runner = get_runner()
        runner.ensure_available()
        return {"backend": backend, "ready": True, "isolation": runner.isolation,
                "is_security_boundary": backend == "docker", "reason": ""}
    except SandboxUnavailable as e:
        return {"backend": backend, "ready": False, "isolation": "unavailable",
                "is_security_boundary": False, "reason": str(e)}


def sweep_workspaces(max_age_days: int, keep_question_ids: set[int]) -> int:
    """Delete workspaces older than `max_age_days` that no live job needs."""
    removed = 0
    cutoff = time.time() - max_age_days * 86400
    try:
        entries = list(os.scandir(config.WORKSPACES_DIR))
    except OSError:
        return 0
    for entry in entries:
        m = re.fullmatch(r"q(\d+)", entry.name)
        if not m or not entry.is_dir(follow_symlinks=False):
            continue
        if int(m.group(1)) in keep_question_ids:
            continue
        try:
            if entry.stat(follow_symlinks=False).st_mtime < cutoff:
                shutil.rmtree(entry.path, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed
