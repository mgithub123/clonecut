"""Shared plumbing: subprocess wrappers, ffprobe, hashing, JSON cache."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import config


class ToolError(RuntimeError):
    """A external tool (ffmpeg/ffprobe) failed in a way the user needs to see."""


def require_binary(name: str) -> str:
    resolved = shutil.which(name)
    if not resolved:
        raise ToolError(
            f"'{name}' was not found on PATH. Install ffmpeg (it ships both ffmpeg and "
            f"ffprobe) or point FFMPEG_BIN / FFPROBE_BIN at your build."
        )
    return resolved


def run(cmd: list[str], *, capture_stderr: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command, raising ToolError with the tail of stderr on failure."""
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE if capture_stderr else None,
        text=True,
    )
    if check and proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-12:]
        raise ToolError(
            f"Command failed ({proc.returncode}): {' '.join(cmd[:6])} ...\n"
            + "\n".join("    " + line for line in tail)
        )
    return proc


def ffprobe_json(path: Path) -> dict[str, Any]:
    require_binary(config.FFPROBE)
    proc = run([
        config.FFPROBE, "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ])
    return json.loads(proc.stdout)


def parse_fraction(value: str | None, default: float = 0.0) -> float:
    if not value:
        return default
    if "/" in value:
        num, _, den = value.partition("/")
        try:
            den_f = float(den)
            return float(num) / den_f if den_f else default
        except ValueError:
            return default
    try:
        return float(value)
    except ValueError:
        return default


# --- hashing ----------------------------------------------------------------

_HASH_MEMO_PATH = config.CACHE_DIR / "hash-memo.json"


def _load_memo() -> dict[str, Any]:
    try:
        return json.loads(_HASH_MEMO_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def file_hash(path: Path) -> str:
    """SHA-256 of file contents, memoised on (path, size, mtime_ns).

    The memo is only a speed-up: a file whose size or mtime changed is re-hashed,
    so cache keys always follow content.
    """
    path = path.resolve()
    st = path.stat()
    key = str(path)
    stamp = [st.st_size, st.st_mtime_ns]

    memo = _load_memo()
    hit = memo.get(key)
    if hit and hit.get("stamp") == stamp:
        return hit["hash"]

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    value = digest.hexdigest()

    memo[key] = {"stamp": stamp, "hash": value}
    _HASH_MEMO_PATH.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(_HASH_MEMO_PATH, json.dumps(memo, indent=2))
    return value


# --- json helpers -----------------------------------------------------------

def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_json(path: Path, data: Any) -> None:
    _atomic_write(path, json.dumps(data, indent=2, ensure_ascii=False))


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


# --- analysis cache ---------------------------------------------------------

def cache_path(kind: str, file_hash_hex: str) -> Path:
    return config.CACHE_DIR / kind / f"{file_hash_hex}.json"


def cache_load(kind: str, file_hash_hex: str) -> dict[str, Any] | None:
    p = cache_path(kind, file_hash_hex)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    if data.get("_ingest_version") != config.INGEST_VERSION:
        return None
    return data


def cache_store(kind: str, file_hash_hex: str, data: dict[str, Any]) -> None:
    payload = dict(data)
    payload["_ingest_version"] = config.INGEST_VERSION
    write_json(cache_path(kind, file_hash_hex), payload)


def rel(path: Path) -> str:
    """Project-relative path when possible, absolute otherwise. Keeps EDLs portable."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(config.ROOT))
    except ValueError:
        return str(p)


def r3(x: float) -> float:
    return round(float(x), 3)
