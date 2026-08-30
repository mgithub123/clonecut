#!/usr/bin/env python3
"""Stage 6: a local desktop app for the whole pipeline.

Runs a small web server on 127.0.0.1 and opens a browser at it. The UI drives
the same command-line stages this project already has -- each action shells out
to `ingest.py`, `plan.py`, `render.py` or `log.py` and streams their output
back -- so the app can never drift from what the CLI actually does.

Stdlib only, on purpose: adding the app must not add an install step.

    uv run app.py                 # opens your browser
    uv run app.py --no-browser    # just serve
    uv run app.py --port 9000
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import uuid
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import config

WEBUI_DIR = config.ROOT / "webui"

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}

# Uploads are written straight to disk, so the name has to be inert.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
CHUNK = 512 * 1024


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------

class Job:
    """One background stage run, with its output captured line by line."""

    def __init__(self, kind: str, argv: list[str]) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.argv = argv
        self.lines: list[str] = []
        self.status = "running"          # running | ok | failed
        self.returncode: int | None = None
        self.result: dict = {}
        self.started = time.time()
        self.finished: float | None = None
        self._lock = threading.Lock()

    def emit(self, line: str) -> None:
        with self._lock:
            self.lines.append(line.rstrip("\n"))

    def snapshot(self, since: int = 0) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "kind": self.kind,
                "status": self.status,
                "returncode": self.returncode,
                "result": self.result,
                "lines": self.lines[since:],
                "total_lines": len(self.lines),
                "elapsed": round((self.finished or time.time()) - self.started, 1),
            }


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def start_job(kind: str, argv: list[str], *, on_done=None, stdin_text: str | None = None) -> Job:
    """Run a stage in a thread, streaming its stdout into the job."""
    job = Job(kind, argv)
    with JOBS_LOCK:
        JOBS[job.id] = job

    def run() -> None:
        try:
            job.emit("$ " + " ".join(_display(a) for a in argv))
            proc = subprocess.Popen(
                argv,
                cwd=str(config.ROOT),
                stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            if stdin_text is not None and proc.stdin:
                proc.stdin.write(stdin_text)
                proc.stdin.close()
            assert proc.stdout is not None
            for line in proc.stdout:
                job.emit(line)
            proc.wait()
            job.returncode = proc.returncode
            job.status = "ok" if proc.returncode == 0 else "failed"
            if on_done and job.status == "ok":
                job.result = on_done() or {}
        except Exception as exc:                      # noqa: BLE001 - shown to the user
            job.emit(f"app error: {exc}")
            job.emit(traceback.format_exc())
            job.status = "failed"
            job.returncode = job.returncode if job.returncode is not None else -1
        finally:
            job.finished = time.time()

    threading.Thread(target=run, daemon=True).start()
    return job


def _display(arg: str) -> str:
    """Shorten absolute paths inside the project when echoing a command."""
    try:
        p = Path(arg)
        if p.is_absolute() and p.is_relative_to(config.ROOT):
            return str(p.relative_to(config.ROOT))
    except (ValueError, OSError):
        pass
    return arg


def stage(script: str, *args: str) -> list[str]:
    """argv for one of the project's own stages, on this interpreter."""
    return [sys.executable, str(config.ROOT / script), *args]


# ---------------------------------------------------------------------------
# reading project state
# ---------------------------------------------------------------------------

def _media(directory: Path, suffixes: set[str]) -> list[dict]:
    if not directory.exists():
        return []
    out = []
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in suffixes and not p.name.startswith("."):
            out.append({
                "name": p.name,
                "path": f"{directory.name}/{p.name}",
                "size": p.stat().st_size,
            })
    return out


def _profiles() -> list[dict]:
    out = []
    for p in sorted(config.PROFILE_DIR.glob("*.json")):
        entry = {"name": p.stem, "path": f"profiles/{p.name}", "clips": [], "bpm": None,
                 "duration": None}
        try:
            data = json.loads(p.read_text())
            entry["clips"] = [c.get("path") for c in data.get("clips", [])]
            audio = data.get("audio") or {}
            entry["bpm"] = audio.get("bpm")
            entry["duration"] = audio.get("duration")
            entry["audio"] = audio.get("path")
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
        out.append(entry)
    return out


def _latest_meta() -> dict | None:
    metas = sorted(config.PLANS_DIR.glob("*-meta.json"))
    if not metas:
        return None
    try:
        data = json.loads(metas[-1].read_text())
    except (OSError, json.JSONDecodeError):
        return None
    stamp = data.get("stamp") or metas[-1].name.replace("-meta.json", "")
    prompt = config.PLANS_DIR / f"{stamp}-prompt.md"
    kf_dir = config.PLANS_DIR / f"{stamp}-keyframes"
    data["stamp"] = stamp
    data["prompt_exists"] = prompt.exists()
    data["prompt_chars"] = prompt.stat().st_size if prompt.exists() else 0
    data["keyframe_count"] = len(list(kf_dir.glob("*.jpg"))) if kf_dir.exists() else 0
    return data


def _plans() -> list[dict]:
    """Every EDL in plans/, newest first, with a short summary."""
    out = []
    for p in sorted(config.PLANS_DIR.glob("*.json"), reverse=True):
        if p.name.endswith("-meta.json"):
            continue
        stamp_match = re.match(r"(\d{8}-\d{6})-", p.stem)
        entry = {"path": f"plans/{p.name}", "name": p.stem, "variant": p.stem,
                 "stamp": stamp_match.group(1) if stamp_match else "",
                 "segments": None, "captions": None, "duration": None, "notes": ""}
        try:
            edl = json.loads(p.read_text())
            entry["variant"] = edl.get("variant_name", p.stem)
            entry["segments"] = len(edl.get("segments", []))
            entry["captions"] = len(edl.get("captions", []))
            entry["notes"] = edl.get("strategy_notes", "")
            entry["duration"] = round(sum(
                (s["out"] - s["in"]) / float(s.get("speed") or 1.0)
                for s in edl.get("segments", [])
            ), 2)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
        out.append(entry)
    return out


def _outputs() -> list[dict]:
    out = []
    for p in sorted(config.OUT_DIR.glob("*.mp4"), reverse=True):
        sheet = p.with_name(p.stem + "-sheet.png")
        out.append({
            "name": p.name,
            "path": f"out/{p.name}",
            "size": p.stat().st_size,
            "mtime": p.stat().st_mtime,
            "sheet": f"out/{sheet.name}" if sheet.exists() else None,
        })
    return out


def _logged() -> list[dict]:
    if not config.DB_PATH.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT v.id, v.variant_name, v.video_path, v.posted_at, v.hook_type,"
            "       v.hook_text, v.song_section, v.cut_count, v.duration,"
            "       (SELECT COUNT(*) FROM metrics m WHERE m.video_id = v.id) AS pulls,"
            "       (SELECT m.views FROM metrics m WHERE m.video_id = v.id"
            "         ORDER BY m.pulled_at DESC, m.id DESC LIMIT 1) AS views"
            "  FROM videos v ORDER BY v.id"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except sqlite3.Error:
        return []


def project_state() -> dict:
    config.ensure_dirs()
    ffmpeg = shutil.which(config.FFMPEG)
    ffprobe = shutil.which(config.FFPROBE)
    return {
        "root": str(config.ROOT),
        "ffmpeg": bool(ffmpeg and ffprobe),
        "ffmpeg_path": ffmpeg,
        "clips": _media(config.RAW_DIR, VIDEO_SUFFIXES),
        "tracks": _media(config.MUSIC_DIR, AUDIO_SUFFIXES),
        "profiles": _profiles(),
        "latest_plan": _latest_meta(),
        "plans": _plans(),
        "outputs": _outputs(),
        "logged": _logged(),
        "min_history": config.MIN_HISTORY_FOR_RETRIEVAL,
        "variants": config.PLAN_VARIANTS,
    }


# ---------------------------------------------------------------------------
# path safety
# ---------------------------------------------------------------------------

ALLOWED_DIRS = {
    "raw": config.RAW_DIR, "music": config.MUSIC_DIR, "profiles": config.PROFILE_DIR,
    "plans": config.PLANS_DIR, "out": config.OUT_DIR,
}


def safe_project_path(rel_path: str) -> Path:
    """Resolve a client-supplied 'dir/name' inside the project, or raise."""
    rel_path = unquote(rel_path or "").strip().lstrip("/")
    parts = Path(rel_path).parts
    if not parts or parts[0] not in ALLOWED_DIRS:
        raise ValueError(f"path must start with one of {sorted(ALLOWED_DIRS)}: {rel_path!r}")
    base = ALLOWED_DIRS[parts[0]]
    target = (base / Path(*parts[1:])).resolve()
    if not target.is_relative_to(base.resolve()):
        raise ValueError(f"path escapes {parts[0]}/: {rel_path!r}")
    return target


def safe_upload_name(name: str) -> str:
    name = Path(unquote(name or "")).name
    cleaned = "".join(c for c in name if c.isalnum() or c in "._- ()").strip()
    cleaned = cleaned.lstrip(".") or "upload"
    return cleaned[:120]


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "clonecut"

    def log_message(self, fmt, *args):            # quieter console
        if self.path.startswith("/api/job/"):
            return
        sys.stderr.write("  %s %s\n" % (self.command, self.path))

    def handle_one_request(self):
        # A browser closing a video stream mid-flight is routine, not an error.
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    # -- helpers ----------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, data, code: int = 200) -> None:
        self._send(code, json.dumps(data).encode(), "application/json; charset=utf-8")

    def _error(self, message: str, code: int = 400) -> None:
        self._json({"error": message}, code)

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_UPLOAD_BYTES:
            raise ValueError("upload too large")
        return self.rfile.read(length) if length else b""

    def _payload(self) -> dict:
        raw = self._body()
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _serve_file(self, path: Path, *, download: str | None = None) -> None:
        if not path.is_file():
            return self._error("not found", 404)
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        size = path.stat().st_size

        start, end = 0, size - 1
        partial = False
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes=") and not download and size:
            try:
                start_s, _, end_s = rng[6:].partition("-")
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else size - 1
                end = min(end, size - 1)
                if start > end or start >= size:
                    raise ValueError
                partial = True
            except ValueError:
                start, end, partial = 0, size - 1, False

        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        if self.command == "HEAD":
            return
        # Streamed in chunks so a 2 GB clip never lands in memory.
        with path.open("rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(CHUNK, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    # -- GET --------------------------------------------------------------
    def do_GET(self) -> None:                     # noqa: N802
        url = urlparse(self.path)
        route, query = url.path, parse_qs(url.query)
        try:
            if route in ("/", "/index.html"):
                return self._serve_file(WEBUI_DIR / "index.html")
            if route in ("/app.js", "/style.css"):
                return self._serve_file(WEBUI_DIR / route.lstrip("/"))

            if route == "/api/state":
                return self._json(project_state())

            if route.startswith("/api/job/"):
                job = JOBS.get(route.rsplit("/", 1)[-1])
                if not job:
                    return self._error("no such job", 404)
                since = int((query.get("since") or ["0"])[0])
                return self._json(job.snapshot(since))

            if route == "/api/prompt":
                stamp = (query.get("stamp") or [""])[0]
                path = config.PLANS_DIR / f"{Path(stamp).name}-prompt.md"
                if not path.is_file():
                    return self._error("no prompt for that stamp", 404)
                return self._json({"stamp": stamp, "text": path.read_text()})

            if route == "/api/keyframes.zip":
                stamp = Path((query.get("stamp") or [""])[0]).name
                kf_dir = config.PLANS_DIR / f"{stamp}-keyframes"
                if not kf_dir.is_dir():
                    return self._error("no keyframes for that stamp", 404)
                buf = BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                    for img in sorted(kf_dir.glob("*.jpg")):
                        z.write(img, img.name)
                return self._send(200, buf.getvalue(), "application/zip",
                                  {"Content-Disposition":
                                   f'attachment; filename="{stamp}-keyframes.zip"'})

            if route == "/api/report":
                argv = stage("log.py", "report")
                proc = subprocess.run(argv, cwd=str(config.ROOT), capture_output=True, text=True)
                return self._json({"text": (proc.stdout or "") + (proc.stderr or "")})

            if route == "/media":
                return self._serve_file(safe_project_path((query.get("path") or [""])[0]))

            return self._error("not found", 404)
        except (BrokenPipeError, ConnectionResetError):
            # The browser hung up - routine when <video> stops fetching.
            self.close_connection = True
        except ValueError as exc:
            return self._error(str(exc))
        except Exception as exc:                  # noqa: BLE001
            traceback.print_exc()
            return self._error(f"{type(exc).__name__}: {exc}", 500)

    # -- POST -------------------------------------------------------------
    def do_POST(self) -> None:                    # noqa: N802
        url = urlparse(self.path)
        route, query = url.path, parse_qs(url.query)
        try:
            if route == "/api/upload":
                return self._upload(query)
            if route == "/api/ingest":
                return self._ingest(self._payload())
            if route == "/api/plan/prompt":
                return self._plan_prompt(self._payload())
            if route == "/api/plan/ingest":
                return self._plan_ingest(self._payload())
            if route == "/api/render":
                return self._render(self._payload())
            if route == "/api/contact-sheet":
                return self._contact_sheet(self._payload())
            if route == "/api/log/post":
                return self._log_post(self._payload())
            if route == "/api/log/metrics":
                return self._log_metrics(self._payload())
            if route == "/api/delete":
                return self._delete(self._payload())
            if route == "/api/shutdown":
                threading.Timer(0.3, lambda: os._exit(0)).start()
                return self._json({"ok": True})
            return self._error("not found", 404)
        except (BrokenPipeError, ConnectionResetError):
            # The browser hung up - routine when <video> stops fetching.
            self.close_connection = True
        except ValueError as exc:
            return self._error(str(exc))
        except Exception as exc:                  # noqa: BLE001
            traceback.print_exc()
            return self._error(f"{type(exc).__name__}: {exc}", 500)

    # -- actions ----------------------------------------------------------
    def _upload(self, query: dict) -> None:
        kind = (query.get("dir") or [""])[0]
        if kind not in ("raw", "music"):
            return self._error("dir must be raw or music")
        name = safe_upload_name((query.get("name") or [""])[0])
        suffix = Path(name).suffix.lower()
        allowed = VIDEO_SUFFIXES if kind == "raw" else AUDIO_SUFFIXES
        if suffix not in allowed:
            return self._error(
                f"{name}: {kind}/ takes {', '.join(sorted(allowed))}"
            )
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self._error("empty upload")
        if length > MAX_UPLOAD_BYTES:
            return self._error(f"{name} is larger than the {MAX_UPLOAD_BYTES // (1024**3)} GB limit")

        target = ALLOWED_DIRS[kind] / name
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        written = 0
        try:
            with tmp.open("wb") as fh:
                while written < length:
                    chunk = self.rfile.read(min(CHUNK, length - written))
                    if not chunk:
                        break
                    fh.write(chunk)
                    written += len(chunk)
            if written != length:
                raise OSError(f"upload cut short at {written} of {length} bytes")
            tmp.replace(target)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            return self._error(str(exc), 500)
        return self._json({"ok": True, "path": f"{kind}/{name}", "size": written})

    def _ingest(self, body: dict) -> None:
        videos = [safe_project_path(v) for v in body.get("videos") or []]
        audio = body.get("audio")
        if not videos:
            return self._error("pick at least one clip")
        if not audio:
            return self._error("pick a music track")
        args = ["--video", *[str(v) for v in videos], "--audio", str(safe_project_path(audio))]
        if body.get("no_transcript"):
            args.append("--no-transcript")
        if body.get("force"):
            args.append("--force")
        job = start_job("ingest", stage("ingest.py", *args))
        return self._json({"job": job.id})

    def _plan_prompt(self, body: dict) -> None:
        profile = safe_project_path(body.get("profile") or "")
        args = ["prompt", "--profile", str(profile)]
        if (body.get("notes") or "").strip():
            args += ["--notes", body["notes"].strip()]
        if body.get("db"):
            args += ["--db", str(Path(body["db"]).name)]
        job = start_job("plan-prompt", stage("plan.py", *args),
                        on_done=lambda: {"meta": _latest_meta()})
        return self._json({"job": job.id})

    def _plan_ingest(self, body: dict) -> None:
        reply = body.get("reply") or ""
        if not reply.strip():
            return self._error("paste Claude's reply first")
        profile = safe_project_path(body.get("profile") or "")
        before = {p.name for p in config.PLANS_DIR.glob("*.json")}
        reply_path = config.PLANS_DIR / "last-reply.txt"
        reply_path.write_text(reply)

        def done() -> dict:
            new = sorted(p.name for p in config.PLANS_DIR.glob("*.json")
                         if p.name not in before and not p.name.endswith("-meta.json"))
            return {"new_plans": [f"plans/{n}" for n in new]}

        job = start_job("plan-ingest",
                        stage("plan.py", "ingest", str(reply_path), "--profile", str(profile)),
                        on_done=done)
        return self._json({"job": job.id})

    def _render(self, body: dict) -> None:
        edls = [str(safe_project_path(e)) for e in body.get("edls") or []]
        if not edls:
            return self._error("pick at least one plan to render")
        before = {p.name for p in config.OUT_DIR.glob("*.mp4")}

        def done() -> dict:
            new = sorted((p for p in config.OUT_DIR.glob("*.mp4") if p.name not in before),
                         key=lambda p: p.stat().st_mtime)
            return {"new_outputs": [f"out/{p.name}" for p in new]}

        job = start_job("render", stage("render.py", *edls), on_done=done)
        return self._json({"job": job.id})

    def _contact_sheet(self, body: dict) -> None:
        video = safe_project_path(body.get("video") or "")
        args = [str(video)]
        if body.get("edl"):
            args += ["--from-edl", str(safe_project_path(body["edl"]))]
        job = start_job("contact-sheet",
                        stage("tools/contact_sheet.py", *args),
                        on_done=lambda: {"outputs": _outputs()})
        return self._json({"job": job.id})

    def _log_post(self, body: dict) -> None:
        video = safe_project_path(body.get("video") or "")
        args = [str(video)]
        if body.get("posted_at"):
            args += ["--posted-at", str(body["posted_at"])]
        if body.get("caption"):
            args += ["--caption", str(body["caption"])]
        if body.get("hashtags"):
            args += ["--hashtags", str(body["hashtags"])]
        job = start_job("log-post", stage("log.py", "post", *args))
        return self._json({"job": job.id})

    def _log_metrics(self, body: dict) -> None:
        try:
            video_id = int(body.get("video_id"))
        except (TypeError, ValueError):
            return self._error("video_id must be a number")
        order = ["views", "watch_through_rate", "likes", "shares", "comments",
                 "follows", "saves"]
        values = body.get("values") or {}
        # log.py metrics prompts field by field; feed it one line each, blank to skip.
        stdin_text = "".join(f"{str(values.get(k, '')).strip()}\n" for k in order)
        args = ["metrics", str(video_id)]
        if body.get("pulled_at"):
            args += ["--pulled-at", str(body["pulled_at"])]
        job = start_job("log-metrics", stage("log.py", *args), stdin_text=stdin_text)
        return self._json({"job": job.id})

    def _delete(self, body: dict) -> None:
        target = safe_project_path(body.get("path") or "")
        if target.parent.resolve() not in (config.RAW_DIR.resolve(), config.MUSIC_DIR.resolve()):
            return self._error("only files in raw/ and music/ can be removed here")
        if not target.is_file():
            return self._error("not found", 404)
        target.unlink()
        return self._json({"ok": True})


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def free_port(preferred: int) -> int:
    for port in [preferred, *range(preferred + 1, preferred + 20)]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise SystemExit(f"no free port near {preferred}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 6: the desktop app.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if not WEBUI_DIR.is_dir():
        raise SystemExit(f"missing {WEBUI_DIR} - the app's HTML lives there")

    config.ensure_dirs()
    port = free_port(args.port)
    url = f"http://127.0.0.1:{port}/"

    if not shutil.which(config.FFMPEG):
        print("  ! ffmpeg is not on PATH - analysing and rendering will fail.")
        print("    Install it (brew install ffmpeg / apt install ffmpeg), then restart.\n")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"clonecut is running at  {url}")
    print("Leave this window open while you use it. Ctrl+C to quit.\n")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
