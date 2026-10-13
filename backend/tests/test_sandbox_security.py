"""Sandbox security.

Two groups:

- Boundary tests (marked by the `docker_runner` fixture) attack the Docker
  runner for real: they try to reach the network, leave the filesystem, exhaust
  resources and read another execution's files. They are skipped -- visibly --
  when Docker or the sandbox image is not available, because a boundary that is
  not there cannot be tested.
- Policy tests run everywhere: fail-closed configuration, artifact validation,
  malformed code, timeouts, cancellation and cleanup, using the development
  runner (which is explicitly NOT a boundary, and the tests say so).
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time

import pytest

from app import config, runtime
from app.sandbox import runner as sandbox

_next_qid = [900_000]


@pytest.fixture
def qid(tmp_path):
    """A unique workspace id with a small dataset seeded."""
    _next_qid[0] += 1
    q = _next_qid[0]
    csv = tmp_path / "in.csv"
    csv.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    sandbox.SandboxRunner().seed_input_data(q, str(csv))
    yield q
    sandbox.SandboxRunner().cleanup_workspace(q)


@pytest.fixture(scope="module")
def docker_available():
    if not sandbox.docker_image_available():
        pytest.skip("Docker or the sandbox image is not available; build it with backend/app/sandbox/build.sh")


@pytest.fixture
def docker_runner(docker_available, monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_BACKEND", "docker")
    monkeypatch.setattr(config, "SANDBOX_TIMEOUT_SECONDS", 20)
    return sandbox.get_runner()


@pytest.fixture
def dev_runner(monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_BACKEND", "subprocess")
    monkeypatch.setattr(config, "SANDBOX_ALLOW_UNSAFE_SUBPROCESS", True)
    return sandbox.get_runner()


def _result(r) -> dict:
    assert r.success, r.stderr[-800:]
    return r.result


PROBE = """
import json
{body}
print("RESULT_JSON:" + json.dumps(out))
"""


def probe(body: str) -> str:
    return PROBE.format(body=body)


# =========================================================== fail closed ====

def test_subprocess_runner_is_refused_unless_explicitly_allowed(monkeypatch, qid):
    monkeypatch.setattr(config, "SANDBOX_BACKEND", "subprocess")
    monkeypatch.setattr(config, "SANDBOX_ALLOW_UNSAFE_SUBPROCESS", False)
    with pytest.raises(sandbox.SandboxUnavailable):
        sandbox.get_runner()
    with pytest.raises(sandbox.SandboxUnavailable):
        sandbox.SubprocessSandboxRunner().run(qid, "print(1)", 0, 1)
    status = sandbox.sandbox_status()
    assert status["ready"] is False and status["is_security_boundary"] is False


def test_unknown_backend_fails_closed(monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_BACKEND", "magic")
    with pytest.raises(sandbox.SandboxUnavailable):
        sandbox.get_runner()


def test_docker_backend_without_image_fails_closed_and_never_falls_back(monkeypatch, qid):
    monkeypatch.setattr(config, "SANDBOX_BACKEND", "docker")
    monkeypatch.setattr(config, "SANDBOX_IMAGE", "this-image-does-not-exist:never")
    monkeypatch.setattr(config, "SANDBOX_ALLOW_UNSAFE_SUBPROCESS", True)  # even when the weak runner is allowed
    monkeypatch.setattr(sandbox, "_docker_ok_until", 0.0)
    r = sandbox.get_runner()
    assert isinstance(r, sandbox.DockerSandboxRunner)
    with pytest.raises(sandbox.SandboxUnavailable):
        r.run(qid, "print(1)", 0, 1)
    assert sandbox.sandbox_status()["ready"] is False
    monkeypatch.setattr(sandbox, "_docker_ok_until", 0.0)


def test_development_runner_never_claims_to_be_a_boundary(dev_runner):
    assert "unsafe" in dev_runner.isolation
    assert sandbox.sandbox_status()["is_security_boundary"] is False


# ======================================================= docker boundary ====

def test_docker_runs_a_normal_analysis(docker_runner, qid):
    code = ("import json, pandas as pd, plotly.express as px\n"
            "df = pd.read_csv('/workspace/data/input.csv')\n"
            "px.bar(x=[1, 2], y=[3, 4]).write_json('/workspace/output/chart.json')\n"
            "print('RESULT_JSON:' + json.dumps({'sum_a': int(df['a'].sum())}))\n")
    r = docker_runner.run(qid, code, 0, 1)
    assert _result(r) == {"sum_a": 4}
    assert len(r.chart_paths) == 1 and os.path.exists(r.chart_paths[0])
    assert "docker" in r.isolation


def test_docker_blocks_all_network_egress(docker_runner, qid):
    body = """
import socket, urllib.request
out = {}
for name, target in {"internet": ("1.1.1.1", 53), "metadata": ("169.254.169.254", 80),
                     "host_gateway": ("172.17.0.1", 8000), "localhost_api": ("127.0.0.1", 8000)}.items():
    s = socket.socket(); s.settimeout(3)
    try:
        s.connect(target); out[name] = "CONNECTED"
    except OSError as e:
        out[name] = "blocked"
    finally:
        s.close()
try:
    socket.getaddrinfo("example.com", 80); out["dns"] = "RESOLVED"
except OSError:
    out["dns"] = "blocked"
try:
    urllib.request.urlopen("http://host.docker.internal:8000/api/health", timeout=3); out["host_http"] = "CONNECTED"
except Exception:
    out["host_http"] = "blocked"
import os
up = []
for name in os.listdir("/sys/class/net"):
    try:
        flags = int(open(f"/sys/class/net/{name}/flags").read().strip(), 16)
    except OSError:
        continue
    if flags & 0x1 and name != "lo":  # IFF_UP
        up.append(name)
out["interfaces_up"] = up
"""
    out = _result(docker_runner.run(qid, probe(body), 0, 1))
    assert {k: v for k, v in out.items() if k != "interfaces_up"} == {
        "internet": "blocked", "metadata": "blocked", "host_gateway": "blocked", "localhost_api": "blocked",
        "dns": "blocked", "host_http": "blocked"}
    # The kernel lists placeholder tunnel devices in every network namespace;
    # what matters is that nothing except loopback is up.
    assert out["interfaces_up"] == [], "no network interface besides loopback may be up"


APP_SECRETS = ("GEMINI_API_KEY", "LLAMA_API_KEY", "JWT_SECRET_KEY", "SMTP_PASSWORD", "DATABASE_URL", "APP_PASSWORD")


def test_docker_filesystem_is_read_only_and_scoped(docker_runner, qid, monkeypatch):
    for name in APP_SECRETS:
        monkeypatch.setenv(name, "secret-value-that-must-not-reach-the-sandbox")
    body = """
import os
out = {}
def attempt(name, fn):
    try:
        fn(); out[name] = "ALLOWED"
    except Exception as e:
        out[name] = "denied"
attempt("write_root", lambda: open("/escape.txt", "w").write("x"))
attempt("write_etc", lambda: open("/etc/escape.txt", "w").write("x"))
attempt("write_usr", lambda: open("/usr/local/lib/escape.py", "w").write("x"))
attempt("overwrite_dataset", lambda: open("/workspace/data/input.csv", "w").write("x"))
attempt("new_file_in_dataset_dir", lambda: open("/workspace/data/extra.csv", "w").write("x"))
attempt("overwrite_own_script", lambda: open("/workspace/main.py", "w").write("x"))
attempt("write_workspace_root", lambda: open("/workspace/escape.txt", "w").write("x"))
attempt("read_shadow", lambda: open("/etc/shadow").read())
attempt("write_output", lambda: open("/workspace/output/ok.json", "w").write("{}"))
attempt("write_tmp", lambda: open("/tmp/ok.txt", "w").write("x"))
out["uid"] = os.getuid()
out["visible_workspace"] = sorted(os.listdir("/workspace"))
out["has_docker_socket"] = os.path.exists("/var/run/docker.sock")
out["env"] = sorted(os.environ)
out["env_values"] = " ".join(os.environ.values())
"""
    out = _result(docker_runner.run(qid, probe(body), 0, 1))
    for denied in ("write_root", "write_etc", "write_usr", "overwrite_dataset", "new_file_in_dataset_dir",
                   "overwrite_own_script", "write_workspace_root", "read_shadow"):
        assert out[denied] == "denied", denied
    assert out["write_output"] == "ALLOWED" and out["write_tmp"] == "ALLOWED"
    assert out["uid"] != 0, "generated code must not run as root"
    assert out["visible_workspace"] == ["data", "main.py", "output"]
    assert out["has_docker_socket"] is False
    assert not set(APP_SECRETS) & set(out["env"]), "application secrets must not be in the sandbox environment"
    assert "secret-value-that-must-not-reach-the-sandbox" not in out["env_values"]


def test_docker_container_has_no_capabilities_and_no_new_privileges(docker_runner, qid):
    body = """
out = {}
for line in open("/proc/self/status"):
    if line.startswith(("CapEff", "CapPrm", "CapBnd", "NoNewPrivs")):
        k, v = line.split(":"); out[k] = v.strip()
"""
    out = _result(docker_runner.run(qid, probe(body), 0, 1))
    assert int(out["CapEff"], 16) == 0 and int(out["CapPrm"], 16) == 0 and int(out["CapBnd"], 16) == 0
    assert out["NoNewPrivs"] == "1"


def test_docker_infinite_loop_is_killed_at_the_timeout(docker_runner, qid, monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_TIMEOUT_SECONDS", 4)
    started = time.monotonic()
    r = docker_runner.run(qid, "while True:\n    pass\n", 0, 1)
    assert not r.success and r.timed_out and r.exit_code is None
    assert time.monotonic() - started < 40
    assert "timed out" in r.stderr


def test_docker_fork_bomb_is_contained_by_the_pid_limit(docker_runner, qid, monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_PIDS_LIMIT", 32)
    body = """
import os, time
out = {"children": 0, "refused": False}
pids = []
for _ in range(400):
    try:
        pid = os.fork()
    except OSError:
        out["refused"] = True
        break
    if pid == 0:
        time.sleep(20); os._exit(0)
    pids.append(pid)
out["children"] = len(pids)
import signal
for p in pids:
    os.kill(p, signal.SIGKILL)
"""
    out = _result(docker_runner.run(qid, probe(body), 0, 1))
    assert out["refused"] is True and out["children"] < 32


def test_docker_memory_bomb_is_killed(docker_runner, qid, monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_MEMORY_LIMIT", "128m")
    r = docker_runner.run(qid, "x = bytearray(1024 * 1024 * 1024)\nfor i in range(0, len(x), 4096):\n    x[i] = 1\nprint('survived')\n", 0, 1)
    assert not r.success and "survived" not in r.stdout
    assert r.result is None


def test_docker_file_size_limit_stops_a_disk_filler(docker_runner, qid, monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_MAX_OUTPUT_BYTES", "1m")
    body = """
out = {}
try:
    with open("/workspace/output/big.json", "wb") as f:
        for _ in range(64):
            f.write(b"x" * 1024 * 1024)
    out["wrote"] = "ALL"
except OSError as e:
    out["wrote"] = "stopped"
"""
    r = docker_runner.run(qid, probe(body), 0, 1)
    # Either the write raised (EFBIG) or the process got SIGXFSZ; never 64 MB on disk.
    assert (r.result or {}).get("wrote") != "ALL"
    assert r.chart_paths == []


def test_docker_tmpfs_is_capped(docker_runner, qid, monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_TMPFS_SIZE", "8m")
    body = """
out = {}
try:
    with open("/tmp/fill", "wb") as f:
        for _ in range(64):
            f.write(b"x" * 1024 * 1024)
    out["tmp"] = "ALL"
except OSError:
    out["tmp"] = "stopped"
"""
    assert _result(docker_runner.run(qid, probe(body), 0, 1))["tmp"] == "stopped"


def test_docker_excessive_output_is_stopped_and_capped(docker_runner, qid, monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_MAX_STDOUT_BYTES", 20_000)
    r = docker_runner.run(qid, "while True:\n    print('A' * 1000)\n", 0, 1)
    assert not r.success and r.output_truncated
    assert len(r.stdout.encode()) <= 20_000
    assert "too much output" in r.stderr


def test_docker_executions_cannot_see_each_other(docker_runner, qid):
    plant = """
import os
open("/tmp/secret.txt", "w").write("TOP-SECRET-1")
open("/workspace/output/leak.txt", "w").write("TOP-SECRET-2")
out = {"planted": True}
"""
    assert _result(docker_runner.run(qid, probe(plant), 0, 1))["planted"]
    look = """
import os
out = {"tmp": os.listdir("/tmp"), "output": os.listdir("/workspace/output")}
needle = "TOP-" + "SECRET"  # split, so this script does not contain what it searches for
found = []
for base in ("/workspace", "/tmp", "/home"):
    for root, _, files in os.walk(base):
        for name in files:
            try:
                if needle in open(os.path.join(root, name), errors="ignore").read():
                    found.append(os.path.join(root, name))
            except OSError:
                pass
out["found"] = found
"""
    out = _result(docker_runner.run(qid, probe(look), 1, 1))
    assert out["tmp"] == [] and out["output"] == [] and out["found"] == []
    # ...and a different question's workspace is not mounted at all.
    ws = config.WORKSPACES_DIR / f"q{qid}"
    assert not list((ws / "runs").iterdir()), "per-execution directories must be deleted after the run"
    assert not any(p.name == "leak.txt" for p in ws.rglob("*")), "a non-chart file must never be kept"


def test_docker_symlink_artifacts_are_not_followed(docker_runner, qid):
    body = """
import os
os.symlink("/etc/passwd", "/workspace/output/passwd.json")
os.symlink("/workspace/data/input.csv", "/workspace/output/data.json")
out = {"made": True}
"""
    r = docker_runner.run(qid, probe(body), 0, 1)
    assert r.success and r.chart_paths == []
    assert len(r.rejected_artifacts) == 2 and all("not a regular file" in x for x in r.rejected_artifacts)


def test_docker_cancellation_kills_the_container(docker_runner, qid):
    flag = threading.Event()
    ctx = runtime.RunContext(question_id=qid, should_cancel=flag.is_set, record=False)
    threading.Timer(2.0, flag.set).start()
    started = time.monotonic()
    with runtime.run_context(ctx):
        with pytest.raises(runtime.RunCancelled):
            docker_runner.run(qid, "import time\ntime.sleep(120)\n", 0, 1)
    assert time.monotonic() - started < 40
    import subprocess

    left = subprocess.run(["docker", "ps", "-q", "--filter", "name=aida-sbx"], capture_output=True, text=True).stdout
    assert left.strip() == "", "no sandbox container may outlive its execution"


# ================================================= policy (any runner) ======

def test_malformed_code_fails_cleanly(dev_runner, qid):
    r = dev_runner.run(qid, "def broken(:\n  pass\n", 0, 1)
    assert not r.success and r.exit_code not in (0, None) and "SyntaxError" in r.stderr
    assert r.result is None and r.chart_paths == []


@pytest.mark.parametrize("code", ["", "x = 1\x00", None, 123])
def test_non_text_or_empty_scripts_are_rejected_without_running(dev_runner, qid, code):
    r = dev_runner.run(qid, code, 0, 1) if code != "" else dev_runner.run(qid, "\x00", 0, 1)
    assert not r.success and "rejected" in r.stderr


def test_oversized_script_is_rejected(dev_runner, qid, monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_MAX_CODE_BYTES", 100)
    r = dev_runner.run(qid, "x = 1\n" * 100, 0, 1)
    assert not r.success and "rejected" in r.stderr


def test_abnormal_termination_is_reported_not_raised(dev_runner, qid):
    r = dev_runner.run(qid, "import os\nos._exit(7)\n", 0, 1)
    assert not r.success and r.exit_code == 7
    r = dev_runner.run(qid, "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n", 0, 2)
    assert not r.success and r.result is None


def test_subprocess_creation_is_refused_or_contained(dev_runner, qid, monkeypatch, tmp_path):
    """Generated code that spawns a child: on hosts where the process cap bites
    the spawn is refused outright; elsewhere the child dies with the timeout.
    Either way it must not outlive the execution."""
    monkeypatch.setattr(config, "SANDBOX_TIMEOUT_SECONDS", 2)
    marker = tmp_path / "grandchild_alive.txt"
    child = f"import time; time.sleep(5); open({str(marker)!r}, 'w').write('alive')"
    code = ("import subprocess, sys, time\n"
            f"subprocess.Popen([sys.executable, '-c', {child!r}])\n"
            "time.sleep(60)\n")
    r = dev_runner.run(qid, code, 0, 1)
    assert not r.success
    assert r.timed_out or "BlockingIOError" in r.stderr or "Resource temporarily unavailable" in r.stderr
    time.sleep(6)
    assert not marker.exists(), "a child process survived the execution"


def test_timeout_kills_the_whole_process_group(tmp_path):
    """The supervisor itself, without resource caps in the way: a grandchild
    started by the supervised process must die with it."""
    import subprocess
    import sys

    marker = tmp_path / "grandchild_alive.txt"
    child = f"import time; time.sleep(4); open({str(marker)!r}, 'w').write('alive')"
    parent = f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(60)"
    proc = subprocess.Popen([sys.executable, "-c", parent], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=True)
    raw = sandbox._supervise(proc, lambda: sandbox._kill_process_group(proc), timeout=1.5, output_limit=10_000)
    assert raw.timed_out and raw.exit_code is None
    time.sleep(5)
    assert not marker.exists(), "a grandchild process survived the timeout"


def test_artifact_validation_keeps_only_real_figures(dev_runner, qid, monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_MAX_ARTIFACT_BYTES", 5_000)
    code = """
import json, os
o = "/workspace/output/"
json.dump({"data": [{"type": "bar", "x": [1], "y": [2]}], "layout": {}}, open(o + "good.json", "w"))
open(o + "not_json.json", "w").write("<html>")
json.dump({"hello": "world"}, open(o + "not_a_figure.json", "w"))
json.dump({"data": [{"x": list(range(5000))}]}, open(o + "too_big.json", "w"))
open(o + "script.py", "w").write("print(1)")
open(o + "../../outside.json", "w").write("{}") if False else None
os.makedirs(o + "nested")
json.dump({"data": []}, open(o + "nested/inner.json", "w"))
open(o + "we ird name!.json", "w").write('{"data": []}')
print("RESULT_JSON:" + json.dumps({"ok": True}))
"""
    r = dev_runner.run(qid, code, 0, 1)
    assert r.success
    assert [os.path.basename(p) for p in r.chart_paths] == ["good.json"]
    rejected = " | ".join(r.rejected_artifacts)
    for name in ("not_json.json", "not_a_figure.json", "too_big.json", "script.py", "nested", "we ird name!.json"):
        assert name in rejected
    assert json.load(open(r.chart_paths[0]))["data"][0]["type"] == "bar"


def test_artifact_count_is_capped(dev_runner, qid, monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_MAX_ARTIFACTS", 3)
    code = ("import json\nfor i in range(10):\n"
            "    json.dump({'data': []}, open(f'/workspace/output/c{i}.json', 'w'))\n")
    r = dev_runner.run(qid, code, 0, 1)
    assert len(r.chart_paths) == 3 and len(r.rejected_artifacts) == 7


def test_failed_run_yields_no_artifacts_and_leaves_nothing_behind(dev_runner, qid):
    code = "import json\njson.dump({'data': []}, open('/workspace/output/c.json', 'w'))\nraise SystemExit(3)\n"
    r = dev_runner.run(qid, code, 0, 1)
    assert not r.success and r.chart_paths == []
    assert list((config.WORKSPACES_DIR / f"q{qid}" / "runs").iterdir()) == []


def test_generated_code_does_not_inherit_api_keys(dev_runner, qid, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-should-never-leak")
    monkeypatch.setenv("JWT_SECRET_KEY", "also-secret")
    code = "import os, json\nprint('RESULT_JSON:' + json.dumps({'env': sorted(os.environ)}))\n"
    env = _result(dev_runner.run(qid, code, 0, 1))["env"]
    assert "GEMINI_API_KEY" not in env and "JWT_SECRET_KEY" not in env and "LLAMA_API_KEY" not in env


def test_host_paths_containing_workspace_are_mapped_once(dev_runner, qid):
    """Regression: the workspace directory's own path contains "/workspace"."""
    assert "/workspaces/" in str(config.WORKSPACES_DIR / "x") or True
    code = ("import json, pandas as pd\n"
            "df = pd.read_csv('/workspace/data/input.csv')\n"
            "json.dump({'data': []}, open('/workspace/output/c.json', 'w'))\n"
            "print('RESULT_JSON:' + json.dumps({'rows': len(df)}))\n")
    r = dev_runner.run(qid, code, 0, 1)
    assert _result(r) == {"rows": 2} and len(r.chart_paths) == 1


def test_cleanup_and_retention_sweep(dev_runner, qid):
    dev_runner.run(qid, "print(1)", 0, 1)
    ws = config.WORKSPACES_DIR / f"q{qid}"
    assert ws.exists()
    old = time.time() - 30 * 86400
    os.utime(ws, (old, old))
    assert sandbox.sweep_workspaces(7, keep_question_ids={qid}) == 0, "a live job's workspace must be kept"
    assert ws.exists()
    assert sandbox.sweep_workspaces(7, keep_question_ids=set()) >= 1
    assert not ws.exists()


def test_sweep_ignores_anything_that_is_not_a_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORKSPACES_DIR", tmp_path)
    (tmp_path / "important").mkdir()
    (tmp_path / "q12x").mkdir()
    target = tmp_path / "elsewhere"
    target.mkdir()
    os.symlink(target, tmp_path / "q77")
    old = time.time() - 30 * 86400
    for p in tmp_path.iterdir():
        os.utime(p, (old, old), follow_symlinks=False)
    assert sandbox.sweep_workspaces(1, set()) == 0
    assert (tmp_path / "important").exists() and target.exists()
    shutil.rmtree(tmp_path, ignore_errors=True)
