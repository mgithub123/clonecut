#!/usr/bin/env python3
"""Stage 1a - Stock footage.

Pulls extra clips off Pexels into ``raw/`` so there is more to edit with than
what has been shot. Nothing downstream knows or cares that a clip came from
here: ``ingest.py`` picks the files up like any other footage.

    uv run stock.py search "neon arcade at night" -n 6
    uv run stock.py search "rain on window" -n 4 --preview
    uv run stock.py credits

Stdlib only, on purpose - the app already takes that line, and adding footage
should not add an install step.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

import common
import config
from common import ToolError

SIDECAR_SUFFIX = ".stock.json"
SIGNUP_URL = "https://www.pexels.com/api/"
TIMEOUT = 60

# Pexels ids are digits, and the id is the last dash-separated part of the stem.
_ID_FROM_NAME = re.compile(rf"^{re.escape(config.STOCK_PREFIX)}.*-(\d+)$")


# ---------------------------------------------------------------------------
# key
# ---------------------------------------------------------------------------

def api_key() -> str:
    """The Pexels key, from the environment or a file beside the project.

    A missing key is the single most likely first-run failure, so it reports
    where to get one rather than surfacing a 401 from deeper in.
    """
    key = (os.environ.get("PEXELS_API_KEY") or "").strip()
    if key:
        return key
    if config.PEXELS_KEY_FILE.exists():
        key = config.PEXELS_KEY_FILE.read_text(encoding="utf-8").strip()
        if key:
            return key
    raise ToolError(
        "No Pexels API key. Get a free one at " + SIGNUP_URL + ", then either:\n"
        "    export PEXELS_API_KEY=your-key\n"
        f"    echo your-key > {common.rel(config.PEXELS_KEY_FILE)}"
    )


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def search(query: str, *, count: int, key: str, page: int = 1) -> list[dict[str, Any]]:
    """Ask Pexels for portrait video matching `query`."""
    params = urllib.parse.urlencode({
        "query": query,
        "orientation": config.STOCK_ORIENTATION,
        "size": config.STOCK_SIZE,
        # ask for at least a page's worth so there is slack when a result
        # carries no usable rendition; 80 is the Pexels ceiling.
        "per_page": min(80, max(count, config.STOCK_PER_PAGE)),
        "page": page,
    })
    request = urllib.request.Request(
        f"{config.PEXELS_API}?{params}",
        headers={"Authorization": key, "User-Agent": "clonecut/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise ToolError(
                f"Pexels rejected the API key (401). Check it at {SIGNUP_URL}."
            ) from exc
        if exc.code == 429:
            raise ToolError(
                "Pexels rate limit reached (429). The free tier allows 200 requests "
                "an hour; try again later."
            ) from exc
        raise ToolError(f"Pexels returned HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise ToolError(f"Could not reach Pexels: {exc.reason}") from exc

    return payload.get("videos", [])


def best_file(video: dict[str, Any]) -> dict[str, Any] | None:
    """The largest rendition no taller than STOCK_MAX_HEIGHT.

    Pexels lists several renditions per clip and the ordering is not reliable,
    so this picks rather than trusting the first. Everything above the canvas
    height is bytes that get scaled away, and if a clip offers nothing at or
    below it the smallest rendition is still better than skipping the clip.
    """
    files = [f for f in video.get("video_files", []) if f.get("link")]
    if not files:
        return None

    def height(f: dict[str, Any]) -> int:
        return int(f.get("height") or 0)

    usable = [f for f in files if height(f) <= config.STOCK_MAX_HEIGHT]
    if usable:
        return max(usable, key=height)
    return min(files, key=height)


# ---------------------------------------------------------------------------
# naming and dedup
# ---------------------------------------------------------------------------

def slugify(text: str, limit: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:limit].strip("-") or "clip"


def target_path(query: str, video_id: int, suffix: str) -> Path:
    return config.RAW_DIR / f"{config.STOCK_PREFIX}{slugify(query)}-{video_id}{suffix}"


def existing_ids() -> dict[int, Path]:
    """Pexels ids already sitting in raw/, read straight off the filenames.

    The id is in the name, so there is no separate state file to fall out of
    step with what is actually on disk - delete a clip and it is fetchable
    again, which is the behaviour you would expect.
    """
    found: dict[int, Path] = {}
    if not config.RAW_DIR.exists():
        return found
    for path in config.RAW_DIR.iterdir():
        if not path.is_file() or path.name.endswith(SIDECAR_SUFFIX):
            continue
        match = _ID_FROM_NAME.match(path.stem)
        if match:
            found.setdefault(int(match.group(1)), path)
    return found


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------

def download(url: str, dest: Path) -> int:
    """Stream to a .part file and rename once it is whole.

    An interrupted download must not leave something that looks like footage in
    raw/, because ingest.py would happily try to analyse it.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "clonecut/0.1"})
    written = 0
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response, part.open("wb") as fh:
            while chunk := response.read(1 << 20):
                fh.write(chunk)
                written += len(chunk)
    except Exception:
        part.unlink(missing_ok=True)
        raise
    os.replace(part, dest)
    return written


def write_sidecar(path: Path, video: dict[str, Any], chosen: dict[str, Any], query: str) -> Path:
    """Record where a clip came from.

    Pexels asks for credit when their footage is used commercially, and in a
    month this file is the only thing that still knows which clip was which.
    """
    sidecar = path.with_name(path.stem + SIDECAR_SUFFIX)
    common.write_json(sidecar, {
        "provider": "pexels",
        "id": video.get("id"),
        "author": video.get("user", {}).get("name"),
        "author_url": video.get("user", {}).get("url"),
        "page_url": video.get("url"),
        "query": query,
        "fetched_at": date.today().isoformat(),
        "width": chosen.get("width"),
        "height": chosen.get("height"),
        "fps": chosen.get("fps"),
        "duration": video.get("duration"),
        "license": "Pexels License - free for commercial use, credit appreciated",
        "download_url": chosen.get("link"),
    })
    return sidecar


def human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}GB"


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_search(args: argparse.Namespace) -> int:
    config.ensure_dirs()
    key = api_key()
    query = " ".join(args.query).strip()
    if not query:
        raise ToolError("give something to search for")

    videos = search(query, count=args.count, key=key, page=args.page)
    if not videos:
        print(f"no portrait video on Pexels for {query!r}")
        return 0

    already = existing_ids()
    print(f"{query!r}: {len(videos)} result(s) on page {args.page}")

    taken = 0
    skipped = 0
    for video in videos:
        if taken >= args.count:
            break
        video_id = video.get("id")
        chosen = best_file(video)
        if video_id is None or chosen is None:
            continue

        author = video.get("user", {}).get("name") or "unknown"
        shape = f"{chosen.get('width')}x{chosen.get('height')}"
        duration = video.get("duration")
        label = f"{shape}  {duration}s  by {author}"

        if video_id in already:
            print(f"  have    {label}  -> {common.rel(already[video_id])}")
            skipped += 1
            continue

        suffix = Path(urllib.parse.urlparse(chosen["link"]).path).suffix or ".mp4"
        if suffix not in {".mp4", ".mov", ".m4v", ".webm"}:
            suffix = ".mp4"
        dest = target_path(query, video_id, suffix)

        if args.preview:
            print(f"  would   {label}  -> {common.rel(dest)}")
            taken += 1
            continue

        print(f"  getting {label}  -> {common.rel(dest)}", flush=True)
        size = download(chosen["link"], dest)
        write_sidecar(dest, video, chosen, query)
        print(f"          {human_bytes(size)}")
        taken += 1

    verb = "would download" if args.preview else "downloaded"
    print(f"{verb} {taken}, already had {skipped}")
    if taken and not args.preview:
        print(f"\nnext:  uv run ingest.py --video 'raw/{config.STOCK_PREFIX}*' --audio music/<track>")
    return 0


def cmd_credits(args: argparse.Namespace) -> int:
    """Print the attribution line for stock footage on disk."""
    sidecars = sorted(config.RAW_DIR.glob(f"*{SIDECAR_SUFFIX}")) if config.RAW_DIR.exists() else []
    if not sidecars:
        print("no stock footage in raw/")
        return 0
    for path in sidecars:
        try:
            meta = common.read_json(path)
        except (OSError, json.JSONDecodeError):
            print(f"{path.name}: unreadable sidecar")
            continue
        clip = path.name[: -len(SIDECAR_SUFFIX)]
        print(f"{clip}\n    Video by {meta.get('author')} on Pexels - {meta.get('page_url')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch stock footage from Pexels into raw/.",
        epilog=f"Needs a free API key from {SIGNUP_URL}.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("search", help="search Pexels and download portrait clips")
    p.add_argument("query", nargs="+", help="what to look for, e.g. neon arcade at night")
    p.add_argument("-n", "--count", type=int, default=5, help="how many clips to keep (default 5)")
    p.add_argument("--page", type=int, default=1, help="result page, for digging past the obvious hits")
    p.add_argument("--preview", action="store_true", help="list what would be fetched, download nothing")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("credits", help="print attribution for the stock clips in raw/")
    p.set_defaults(func=cmd_credits)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
