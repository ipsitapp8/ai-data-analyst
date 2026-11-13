"""The exact inference conditions of an evaluation run, recorded in its report:
base model revision and weight hash, adapter revision (file hashes), tokenizer
and chat template hash, system prompt and tool schema hashes, decoding
parameters, library versions and machine. Base and fine-tuned runs differ only
in `adapter`."""
from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

BASE_REPO = "mlx-community/Qwen3-4B-Instruct-2507-4bit"


def _sha(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record(adapter: str | None, decoding: dict) -> dict:
    from huggingface_hub import snapshot_download
    import mlx
    import mlx_lm

    from app.analyst import orchestrator, planner, tools

    snap = Path(snapshot_download(BASE_REPO, local_files_only=True))
    rec = {
        "base_model": {"repo": BASE_REPO, "revision": snap.name,
                       "weights_sha256": _sha(snap / "model.safetensors"),
                       "tokenizer_sha256": _sha(snap / "tokenizer.json"),
                       "chat_template_sha256": _sha(snap / "chat_template.jinja") if (snap / "chat_template.jinja").exists() else None},
        "adapter": None,
        "prompt": {"system_prompt_sha256": _text_sha(planner.SYSTEM + tools.catalogue()),
                   "narration_prompt_sha256": _text_sha(orchestrator.NARRATE_SYSTEM),
                   "tool_schema_sha256": _text_sha(repr(tools.TOOLS)),
                   "builder": "app.analyst.planner.build_prompt"},
        "decoding": decoding,
        "environment": {"mlx": getattr(mlx, "__version__", None) or _mlx_version(), "mlx_lm": mlx_lm.__version__,
                        "python": platform.python_version(), "machine": platform.machine(), "platform": platform.platform()},
    }
    if adapter and adapter != "none":
        a = ROOT / adapter if not Path(adapter).is_absolute() else Path(adapter)
        rec["adapter"] = {"path": str(a.relative_to(ROOT)) if a.is_relative_to(ROOT) else str(a),
                          "files_sha256": {p.name: _sha(p) for p in sorted(a.glob("*.safetensors")) + [a / "adapter_config.json"]
                                           if p.exists()}}
    return rec


def _mlx_version() -> str | None:
    try:
        from importlib.metadata import version
        return version("mlx")
    except Exception:  # noqa: BLE001
        return None
