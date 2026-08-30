#!/usr/bin/env python3
"""Stage 4 - Learn.

A local SQLite database of what was posted and how it did, plus the CLI to fill
it in and read it back.

    uv run log.py post out/fast-cut-hook-20260830-140027.mp4 --posted-at 2026-08-30
    uv run log.py metrics 1
    uv run log.py report

Metrics are typed in by hand from the TikTok analytics screen. Nothing here
scrapes anything.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import common
import config
from common import ToolError, rel

SCHEMA_VERSION = 2

# A group smaller than this is reported but flagged: with three videos you are
# looking at noise, and the report should say so rather than imply a finding.
MIN_GROUP = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id                 INTEGER PRIMARY KEY,
    video_path         TEXT NOT NULL UNIQUE,
    variant_name       TEXT NOT NULL,
    edl_path           TEXT,
    edl_json           TEXT NOT NULL,
    features_json      TEXT NOT NULL,
    rendered_at        TEXT,
    posted_at          TEXT,
    caption            TEXT,
    hashtags           TEXT,
    -- derived features, columned out so the report can group on them
    hook_type          TEXT,
    hook_text          TEXT,
    hook_position      TEXT,
    cut_count          INTEGER,
    avg_segment_length REAL,
    cuts_per_second    REAL,
    caption_count      INTEGER,
    caption_density    REAL,
    duration           REAL,
    audio_start        REAL,
    song_section       TEXT,
    song_section_index INTEGER,
    beat_synced        INTEGER,
    uses_speed_ramp    INTEGER,
    created_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics (
    id                 INTEGER PRIMARY KEY,
    video_id           INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    pulled_at          TEXT NOT NULL,
    views              INTEGER,
    watch_through_rate REAL,
    likes              INTEGER,
    shares             INTEGER,
    comments           INTEGER,
    follows            INTEGER,
    saves              INTEGER,
    created_at         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS metrics_video ON metrics(video_id, pulled_at);
"""

# v2 adds the context a video was planned from, so retrieval can ask "what did I
# do last time the material looked like this" rather than only "what did best".
MIGRATION_V2 = """
ALTER TABLE videos ADD COLUMN audio_source TEXT;
ALTER TABLE videos ADD COLUMN bpm REAL;
ALTER TABLE videos ADD COLUMN clips_json TEXT;
"""


# ---------------------------------------------------------------------------
# database
# ---------------------------------------------------------------------------

def connect(path: Path | None = None) -> sqlite3.Connection:
    path = Path(path) if path else config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    migrate(conn)
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    """Bring an existing database up to the current schema, in order.

    Each step is guarded by user_version, so an old database upgrades in place
    and a new one is created at the current version.
    """
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version < 1:
        conn.executescript(SCHEMA)
        version = 1
    if version < 2:
        conn.executescript(SCHEMA)          # no-ops on an existing database
        conn.executescript(MIGRATION_V2)
        _backfill_v2(conn)
        version = 2
    conn.execute(f"PRAGMA user_version = {version}")
    conn.commit()


def _backfill_v2(conn: sqlite3.Connection) -> None:
    """Fill the new context columns for rows written before they existed.

    Track and clips come straight out of the stored EDL. BPM needs the ingest
    profile, which may no longer be on disk - left NULL when it is not, and
    similarity treats a missing component as unavailable rather than as zero.
    """
    for row in conn.execute("SELECT id, edl_json FROM videos").fetchall():
        try:
            edl = json.loads(row["edl_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        source = edl.get("audio", {}).get("source")
        clips = sorted({s["clip"] for s in edl.get("segments", [])})
        profile = find_profile_for_audio(source) if source else None
        bpm = (profile or {}).get("audio", {}).get("bpm")
        conn.execute(
            "UPDATE videos SET audio_source = ?, clips_json = ?, bpm = ? WHERE id = ?",
            (source, json.dumps(clips), bpm, row["id"]),
        )


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# derived features that need more than the EDL
# ---------------------------------------------------------------------------

def classify_hook(text: str | None) -> str:
    """Bucket a hook caption into a type.

    Grouping the report by the literal hook text would give one video per group
    and tell you nothing; what you want to know is whether *questions* beat
    *claims*. This is a rough heuristic on purpose - it only has to be stable
    enough to group by.
    """
    if not text or not text.strip():
        return "none"
    t = text.strip().lower()
    first = t.split()[0] if t.split() else ""
    if t.endswith("?") or first in {"how", "why", "what", "who", "when", "where", "did", "can"}:
        return "question"
    if t.startswith("pov") or t.startswith("when you") or t.startswith("me when"):
        return "pov"
    if first and first[0].isdigit():
        return "number"
    if any(t.startswith(p) for p in ("no ", "not ", "never ", "don't", "dont ", "stop ")):
        return "negation"
    if first in {"listen", "watch", "wait", "turn", "look", "try", "play", "read", "go"}:
        return "command"
    return "statement"


def find_profile_for_audio(audio_source: str) -> dict[str, Any] | None:
    """Locate the ingest profile that describes this track, so the song section
    a variant used can be recorded without the user having to name it."""
    target = Path(audio_source).name
    for candidate in sorted(config.PROFILE_DIR.glob("*.json")):
        try:
            profile = common.read_json(candidate)
        except (OSError, json.JSONDecodeError):
            continue
        audio = profile.get("audio") or {}
        if Path(audio.get("path", "")).name == target:
            return profile
    return None


def resolve_song_section(edl_data: dict[str, Any],
                         profile: dict[str, Any] | None) -> tuple[str | None, int | None]:
    """Which section of the track the edit sits in, by its audio start."""
    if profile is None:
        profile = find_profile_for_audio(edl_data["audio"]["source"])
    if not profile:
        return None, None
    start = edl_data["audio"]["start"]
    for section in profile.get("audio", {}).get("sections", []):
        if section["start"] <= start < section["end"]:
            return section.get("label"), section.get("index")
    return None, None


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def read_sidecar(video: Path) -> dict[str, Any]:
    sidecar = video.with_suffix(".json")
    if not sidecar.exists():
        raise ToolError(
            f"No sidecar found at {rel(sidecar)}.\n"
            f"    render.py writes one next to every mp4 it produces; without it there "
            f"is no record of which EDL made this file. Re-render to get one."
        )
    data = common.read_json(sidecar)
    if "edl" not in data or "derived_features" not in data:
        raise ToolError(f"{rel(sidecar)} is not a render sidecar (no edl / derived_features).")
    return data


def record_video(conn: sqlite3.Connection, video: Path, *, posted_at: str,
                 caption: str | None, hashtags: str | None,
                 profile: dict[str, Any] | None = None) -> int:
    sidecar = read_sidecar(video)
    edl = sidecar["edl"]
    features = dict(sidecar["derived_features"])

    if profile is None:
        profile = find_profile_for_audio(edl["audio"]["source"])
    section_label, section_index = resolve_song_section(edl, profile)
    features["song_section"] = section_label
    features["song_section_index"] = section_index
    hook_text = features.get("hook_type")          # the raw hook caption text
    features["hook_type"] = classify_hook(hook_text)
    features["hook_text"] = hook_text

    row = {
        "video_path": rel(video),
        "variant_name": edl["variant_name"],
        "edl_path": sidecar.get("edl_path"),
        "edl_json": json.dumps(edl, ensure_ascii=False),
        "features_json": json.dumps(features, ensure_ascii=False),
        "rendered_at": sidecar.get("rendered_at"),
        "posted_at": posted_at,
        "caption": caption,
        "hashtags": hashtags,
        "hook_type": features["hook_type"],
        "hook_text": hook_text,
        "hook_position": features.get("hook_position"),
        "cut_count": features.get("cut_count"),
        "avg_segment_length": features.get("avg_segment_length"),
        "cuts_per_second": features.get("cuts_per_second"),
        "caption_count": features.get("caption_count"),
        "caption_density": features.get("caption_density"),
        "duration": features.get("duration"),
        "audio_start": features.get("audio_start"),
        "song_section": section_label,
        "song_section_index": section_index,
        "beat_synced": int(bool(features.get("beat_synced"))),
        "uses_speed_ramp": int(bool(features.get("uses_speed_ramp"))),
        # the context this edit was planned from, for similarity retrieval
        "audio_source": edl["audio"]["source"],
        "bpm": (profile or {}).get("audio", {}).get("bpm"),
        "clips_json": json.dumps(sorted({s["clip"] for s in edl["segments"]})),
        "created_at": now(),
    }

    existing = conn.execute(
        "SELECT id FROM videos WHERE video_path = ?", (row["video_path"],)
    ).fetchone()
    if existing:
        assignments = ", ".join(f"{k} = :{k}" for k in row if k != "video_path")
        conn.execute(f"UPDATE videos SET {assignments} WHERE video_path = :video_path", row)
        conn.commit()
        return existing["id"]

    columns = ", ".join(row)
    placeholders = ", ".join(f":{k}" for k in row)
    cur = conn.execute(f"INSERT INTO videos ({columns}) VALUES ({placeholders})", row)
    conn.commit()
    return int(cur.lastrowid)


def record_metrics(conn: sqlite3.Connection, video_id: int, values: dict[str, Any],
                   pulled_at: str) -> int:
    if not conn.execute("SELECT 1 FROM videos WHERE id = ?", (video_id,)).fetchone():
        raise ToolError(f"No video with id {video_id}. Run `log.py list` to see them.")
    row = {
        "video_id": video_id,
        "pulled_at": pulled_at,
        "created_at": now(),
        **{k: values.get(k) for k in
           ("views", "watch_through_rate", "likes", "shares", "comments", "follows", "saves")},
    }
    columns = ", ".join(row)
    placeholders = ", ".join(f":{k}" for k in row)
    cur = conn.execute(f"INSERT INTO metrics ({columns}) VALUES ({placeholders})", row)
    conn.commit()
    return int(cur.lastrowid)


# ---------------------------------------------------------------------------
# input parsing for the interactive prompt
# ---------------------------------------------------------------------------

def parse_count(raw: str) -> int | None:
    raw = raw.strip().replace(",", "").replace(" ", "")
    if not raw:
        return None
    multiplier = 1
    if raw and raw[-1].lower() in "km":
        multiplier = 1_000 if raw[-1].lower() == "k" else 1_000_000
        raw = raw[:-1]
    try:
        return int(round(float(raw) * multiplier))
    except ValueError:
        raise ValueError(f"{raw!r} is not a number")


def parse_rate(raw: str) -> float | None:
    """Accept 45%, 0.45 or 45 and land on 0.45.

    TikTok shows these as percentages, so a bare number above 1 is a percentage;
    below or equal to 1 it is already a fraction.
    """
    raw = raw.strip().replace(" ", "")
    if not raw:
        return None
    percent = raw.endswith("%")
    raw = raw.rstrip("%")
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{raw!r} is not a number")
    if percent or value > 1:
        value /= 100.0
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{value:.3f} is not a rate between 0 and 1")
    return value


PROMPTS = [
    ("views", "Views", parse_count),
    ("watch_through_rate", "Watch-through rate (e.g. 42% or 0.42)", parse_rate),
    ("likes", "Likes", parse_count),
    ("shares", "Shares", parse_count),
    ("comments", "Comments", parse_count),
    ("follows", "Follows gained", parse_count),
    ("saves", "Saves", parse_count),
]


def prompt_metrics(stream=None) -> dict[str, Any]:
    stream = stream or sys.stdin
    values: dict[str, Any] = {}
    print("Type the numbers from the TikTok analytics screen. Blank to skip a field.\n")
    for key, label, parse in PROMPTS:
        while True:
            print(f"  {label}: ", end="", flush=True)
            line = stream.readline()
            if not line:            # EOF: treat the rest as skipped
                print()
                return values
            if not getattr(stream, "isatty", lambda: False)():
                # Piped input is not echoed by the terminal, so echo it ourselves
                # and the transcript still reads like the session it was.
                print(line.rstrip("\n"))
            try:
                values[key] = parse(line)
            except ValueError as exc:
                print(f"    {exc}. Try again, or leave blank to skip.")
                continue
            break
    return values


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def latest_metrics(conn: sqlite3.Connection) -> dict[int, sqlite3.Row]:
    """The most recent pull per video - what a video finally did, not its first day.

    Picking the row by ORDER BY rather than by MAX() of each column separately:
    MAX(pulled_at) and MAX(id) can come from different rows when pulls are
    entered out of order (backfilling an earlier date is a normal thing to do),
    and then no row matches both and the video drops out of the report silently.
    """
    rows = conn.execute(
        "SELECT * FROM metrics m WHERE m.id = ("
        "  SELECT m2.id FROM metrics m2 WHERE m2.video_id = m.video_id"
        "  ORDER BY m2.pulled_at DESC, m2.id DESC LIMIT 1)"
    ).fetchall()
    return {r["video_id"]: r for r in rows}


def posted_videos(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM videos WHERE posted_at IS NOT NULL ORDER BY posted_at, id"
    ).fetchall()


def pace_bucket(avg_segment_length: float | None) -> str:
    if avg_segment_length is None:
        return "unknown"
    if avg_segment_length < 1.0:
        return "fast (<1s cuts)"
    if avg_segment_length < 2.0:
        return "medium (1-2s cuts)"
    return "slow (2s+ cuts)"


def _rate(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or not denominator:
        return None
    return numerator / denominator


def build_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """One row per posted video, joined to its latest metrics pull."""
    metrics = latest_metrics(conn)
    out = []
    for video in posted_videos(conn):
        m = metrics.get(video["id"])
        if m is None:
            continue
        out.append({
            "id": video["id"],
            "variant_name": video["variant_name"],
            "hook_type": video["hook_type"] or "none",
            "pace": pace_bucket(video["avg_segment_length"]),
            "song_section": video["song_section"] or "unknown",
            "beat_synced": "beat-synced" if video["beat_synced"] else "free",
            "views": m["views"],
            "watch_through_rate": m["watch_through_rate"],
            "share_rate": _rate(m["shares"], m["views"]),
            "like_rate": _rate(m["likes"], m["views"]),
            "pulled_at": m["pulled_at"],
        })
    return out


def summarise(rows: Iterable[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)

    out = []
    for name, members in groups.items():
        def mean(field: str) -> float | None:
            vals = [m[field] for m in members if m[field] is not None]
            return statistics.fmean(vals) if vals else None

        views = [m["views"] for m in members if m["views"] is not None]
        out.append({
            "group": name,
            "n": len(members),
            "median_views": statistics.median(views) if views else None,
            "watch_through": mean("watch_through_rate"),
            "share_rate": mean("share_rate"),
            "like_rate": mean("like_rate"),
        })
    out.sort(key=lambda g: (g["watch_through"] is None, -(g["watch_through"] or 0)))
    return out


# ---------------------------------------------------------------------------
# similarity retrieval
# ---------------------------------------------------------------------------
#
# The planner asks "what did I do last time the material looked like this",
# which is a different question from "what did best". A past video shot to a
# different song at a different tempo is weak evidence about this one however
# well it performed.
#
# Every component is a named number in 0..1 with a fixed weight, so a retrieval
# can be explained in the prompt rather than being a black box. A component that
# cannot be computed (no BPM recorded, no notes typed) is dropped and the
# remaining weights are renormalised, rather than being scored zero - which
# would silently punish older rows for missing data.

WEIGHTS = {
    "same_track": 0.40,     # the strongest signal: same song, same structure
    "tempo": 0.25,          # a similar BPM means similar cutting rhythms work
    "clips": 0.20,          # overlapping source footage
    "notes": 0.15,          # what you said you were going for this time
}

TEMPO_SPAN = 40.0           # BPM difference at which tempo similarity reaches 0
SIMILAR_ENOUGH = 0.25       # below this, retrieval is not really similarity

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "over", "under",
    "than", "then", "when", "where", "what", "how", "why", "its", "it's", "was",
    "are", "you", "your", "out", "one", "two", "not", "but", "all", "can", "has",
    "have", "will", "would", "video", "edit", "shot", "clip", "cut", "cuts",
}


def _tokens(*texts: str | None) -> set[str]:
    out: set[str] = set()
    for text in texts:
        if not text:
            continue
        word = ""
        for ch in text.lower():
            if ch.isalnum():
                word += ch
            else:
                if len(word) >= 3 and word not in _STOPWORDS:
                    out.add(word)
                word = ""
        if len(word) >= 3 and word not in _STOPWORDS:
            out.add(word)
    return out


def similarity(context: dict[str, Any], video: dict[str, Any]) -> tuple[float, list[str]]:
    """Score a past video against the current planning context.

    Returns the score in 0..1 and the human-readable reasons it scored.
    """
    parts: dict[str, float] = {}
    reasons: list[str] = []

    here = context.get("audio_source")
    there = video.get("audio_source")
    if here and there:
        same = Path(here).name == Path(there).name
        parts["same_track"] = 1.0 if same else 0.0
        if same:
            reasons.append("same track")

    a, b = context.get("bpm"), video.get("bpm")
    if a and b:
        closeness = max(0.0, 1.0 - abs(a - b) / TEMPO_SPAN)
        parts["tempo"] = closeness
        if closeness >= 0.85:
            reasons.append(f"similar tempo ({b:.0f} vs {a:.0f} BPM)")

    mine = {Path(c).name for c in context.get("clips") or []}
    theirs = {Path(c).name for c in video.get("clips") or []}
    if mine and theirs:
        overlap = len(mine & theirs) / len(mine | theirs)
        parts["clips"] = overlap
        if mine & theirs:
            reasons.append(f"shares {len(mine & theirs)} clip(s)")

    note_tokens = _tokens(context.get("notes"))
    if note_tokens:
        past = _tokens(video.get("strategy_notes"), video.get("hook_text"),
                       video.get("variant_name"))
        if past:
            shared = note_tokens & past
            parts["notes"] = len(shared) / len(note_tokens)
            if shared:
                reasons.append("matches your notes (" + ", ".join(sorted(shared)[:3]) + ")")

    available = {k: v for k, v in parts.items() if k in WEIGHTS}
    if not available:
        return 0.0, ["nothing comparable recorded"]
    total_weight = sum(WEIGHTS[k] for k in available)
    score = sum(WEIGHTS[k] * v for k, v in available.items()) / total_weight
    return score, reasons or ["different material"]


def build_context(profile: dict[str, Any] | None, notes: str | None) -> dict[str, Any]:
    """What the planner is working from right now, in the shape similarity wants."""
    audio = (profile or {}).get("audio", {})
    return {
        "audio_source": audio.get("path"),
        "bpm": audio.get("bpm"),
        "clips": [c["path"] for c in (profile or {}).get("clips", [])],
        "notes": notes,
    }


def candidates(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Every posted video with metrics, as a flat record for retrieval."""
    metrics = latest_metrics(conn)
    out = []
    for video in posted_videos(conn):
        m = metrics.get(video["id"])
        if m is None:
            continue
        try:
            edl = json.loads(video["edl_json"])
        except (json.JSONDecodeError, TypeError):
            edl = {}
        try:
            clips = json.loads(video["clips_json"]) if video["clips_json"] else []
        except (json.JSONDecodeError, TypeError):
            clips = []
        out.append({
            "id": video["id"],
            "variant_name": video["variant_name"],
            "strategy_notes": edl.get("strategy_notes"),
            "hook_type": video["hook_type"] or "none",
            "hook_text": video["hook_text"],
            "pace": pace_bucket(video["avg_segment_length"]),
            "cut_count": video["cut_count"],
            "avg_segment_length": video["avg_segment_length"],
            "duration": video["duration"],
            "song_section": video["song_section"] or "unknown",
            "beat_synced": bool(video["beat_synced"]),
            "audio_source": video["audio_source"],
            "bpm": video["bpm"],
            "clips": clips,
            "views": m["views"],
            "watch_through_rate": m["watch_through_rate"],
            "share_rate": _rate(m["shares"], m["views"]),
            "like_rate": _rate(m["likes"], m["views"]),
        })
    return out


def most_similar(conn: sqlite3.Connection, context: dict[str, Any],
                 limit: int) -> list[dict[str, Any]]:
    scored = []
    for video in candidates(conn):
        score, reasons = similarity(context, video)
        scored.append({**video, "similarity": score, "why": reasons})
    # Similarity first, then watch-through, so ties break toward what worked.
    scored.sort(key=lambda v: (-v["similarity"], -(v["watch_through_rate"] or 0)))
    return scored[:limit]


# ---------------------------------------------------------------------------
# retrieval for the planner
# ---------------------------------------------------------------------------

def history_for_prompt(limit: int = 5, db_path: Path | None = None,
                       profile: dict[str, Any] | None = None,
                       notes: str | None = None) -> tuple[str, int]:
    """What to tell the planner about past performance, and how many videos it
    is based on.

    Below the threshold this says so rather than dressing up a handful of data
    points as a finding.
    """
    try:
        conn = connect(db_path)
    except sqlite3.Error:
        return ("No performance history is available (the database could not be opened).", 0)

    with conn:
        rows = candidates(conn)
        context = build_context(profile, notes)
        picked = most_similar(conn, context, limit) if rows else []
        groups = {key: summarise(build_rows(conn), key) for key, _ in GROUPINGS} if rows else {}

    if len(rows) < config.MIN_HISTORY_FOR_RETRIEVAL:
        return (
            f"Only {len(rows)} posted video(s) have metrics logged, which is below the "
            f"{config.MIN_HISTORY_FOR_RETRIEVAL} needed for any of this to mean anything. "
            f"Treat it as no history: judge this edit on the footage and the music alone, "
            f"and do not claim any choice here is proven.",
            len(rows),
        )

    lines = [f"{len(rows)} posted videos have metrics logged.", ""]
    # Do not claim these are "similar" when nothing scored as similar - with a
    # new track and new footage the ranking falls back to raw performance, and
    # the model should be told that is what it is looking at.
    best = max((v["similarity"] for v in picked), default=0.0)
    if best >= SIMILAR_ENOUGH:
        lines.append("The closest past edits to the material you are planning from now:")
    else:
        lines.append("Nothing logged was made from material like this, so these are "
                     "simply the strongest so far - weak evidence for this edit:")
    lines.append("")
    for v in picked:
        lines.append(
            f"- {v['variant_name']}: {_pct(v['watch_through_rate'])} watch-through, "
            f"{_pct(v['share_rate'])} share rate, {_int(v['views'])} views"
        )
        lines.append(
            f"    {v['hook_type']} hook, {v['pace']}, {v['song_section']} section, "
            f"{'beat-synced' if v['beat_synced'] else 'not beat-synced'}, "
            f"{v['cut_count']} cuts averaging {v['avg_segment_length']:.2f}s"
        )
        if v.get("hook_text"):
            lines.append(f"    hook was: {v['hook_text']!r}")
        lines.append(f"    retrieved because: {', '.join(v['why'])}")

    lines += ["", f"Across all {len(rows)} videos, mean watch-through by group "
                  f"(number of videos in brackets):"]
    for key, title in GROUPINGS:
        parts = [f"{g['group']} {_pct(g['watch_through'])} [{g['n']}]" for g in groups.get(key, [])]
        if len(parts) > 1:
            lines.append(f"  {title[3:]}: " + ", ".join(parts))

    lines += ["", "These are small numbers over a handful of videos. Use them as a hint "
                  "about what to try, not a rule - and if the footage points somewhere "
                  "else, follow the footage and say so in strategy_notes."]
    return "\n".join(lines), len(rows)


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _int(value: Any) -> str:
    return "-" if value is None else f"{int(value):,}"


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_post(args: argparse.Namespace) -> int:
    video = Path(args.video)
    if not video.exists():
        raise ToolError(f"Video not found: {video}")
    try:
        posted = datetime.strptime(args.posted_at, "%Y-%m-%d").date()
    except ValueError:
        raise ToolError(f"--posted-at must look like 2026-08-30, got {args.posted_at!r}")

    with connect(args.db) as conn:
        video_id = record_video(
            conn, video,
            posted_at=posted.isoformat(),
            caption=args.caption,
            hashtags=args.hashtags,
        )
        row = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()

    print(f"logged video {video_id}: {row['variant_name']}  posted {row['posted_at']}")
    print(f"  {row['cut_count']} cuts, {row['avg_segment_length']:.2f}s average, "
          f"{row['duration']:.1f}s, {pace_bucket(row['avg_segment_length'])}")
    print(f"  hook: {row['hook_type']}"
          + (f" ({row['hook_text']!r})" if row["hook_text"] else ""))
    print(f"  song section: {row['song_section'] or 'unknown'}"
          + ("" if row["song_section"] else "  (no matching profile in profiles/)"))
    print(f"  beat-synced: {'yes' if row['beat_synced'] else 'no'}")
    if not row["caption"]:
        print("  note: no caption recorded - pass --caption to log what you posted with it")
    print(f"\nNext: uv run log.py metrics {video_id}")
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    pulled = args.pulled_at or date.today().isoformat()
    try:
        datetime.strptime(pulled, "%Y-%m-%d")
    except ValueError:
        raise ToolError(f"--pulled-at must look like 2026-08-30, got {pulled!r}")

    with connect(args.db) as conn:
        row = conn.execute("SELECT * FROM videos WHERE id = ?", (args.video_id,)).fetchone()
        if row is None:
            raise ToolError(f"No video with id {args.video_id}. Run `log.py list` to see them.")
        print(f"{row['variant_name']}  ({row['video_path']})  posted {row['posted_at']}")
        values = prompt_metrics()
        if not any(v is not None for v in values.values()):
            print("\nnothing entered, nothing saved")
            return 0
        record_metrics(conn, args.video_id, values, pulled)
        count = conn.execute(
            "SELECT COUNT(*) FROM metrics WHERE video_id = ?", (args.video_id,)
        ).fetchone()[0]

    print(f"\nsaved for {pulled}  ({count} pull(s) recorded for this video)")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    with connect(args.db) as conn:
        rows = conn.execute(
            "SELECT v.*, COUNT(m.id) AS pulls, MAX(m.pulled_at) AS last_pull "
            "FROM videos v LEFT JOIN metrics m ON m.video_id = v.id "
            "GROUP BY v.id ORDER BY v.posted_at IS NULL, v.posted_at, v.id"
        ).fetchall()

    if not rows:
        print("nothing logged yet - start with:  uv run log.py post out/<video>.mp4 "
              "--posted-at YYYY-MM-DD")
        return 0

    print(f"{'id':>3}  {'posted':<11} {'variant':<28} {'pulls':>5}  {'last pull':<11} hook")
    for r in rows:
        print(f"{r['id']:>3}  {(r['posted_at'] or '-'):<11} {r['variant_name'][:28]:<28} "
              f"{r['pulls']:>5}  {(r['last_pull'] or '-'):<11} {r['hook_type'] or '-'}")
    return 0


GROUPINGS = [
    ("hook_type", "By hook type"),
    ("pace", "By cut pace"),
    ("song_section", "By song section"),
    ("beat_synced", "By beat sync"),
]


def cmd_report(args: argparse.Namespace) -> int:
    with connect(args.db) as conn:
        rows = build_rows(conn)
        total_videos = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
        total_pulls = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]

    print("=" * 72)
    print("LUCKY DOG - PERFORMANCE REPORT")
    print("=" * 72)

    if not rows:
        print(f"\n{total_videos} video(s) logged, {total_pulls} metric pull(s).")
        print("Nothing to report yet: a video needs both a posted date and at least one")
        print("metrics pull before it can be counted.")
        print("\n  uv run log.py post out/<video>.mp4 --posted-at YYYY-MM-DD")
        print("  uv run log.py metrics <id>")
        return 0

    print(f"\n{len(rows)} posted video(s) with metrics, {total_pulls} pull(s) in total.")
    if len(rows) < config.MIN_HISTORY_FOR_RETRIEVAL:
        print(f"Below {config.MIN_HISTORY_FOR_RETRIEVAL} videos none of this is worth acting on,")
        print("and the planner is told to ignore it.")

    for key, title in GROUPINGS:
        groups = summarise(rows, key)
        if len(groups) < 2:
            continue
        print(f"\n{title}")
        print(f"  {'group':<20} {'n':>3}  {'med. views':>10}  {'watch-thru':>10}  "
              f"{'share':>7}  {'like':>7}")
        for g in groups:
            flag = "  (too few to trust)" if g["n"] < MIN_GROUP else ""
            print(f"  {g['group']:<20} {g['n']:>3}  {_int(g['median_views']):>10}  "
                  f"{_pct(g['watch_through']):>10}  {_pct(g['share_rate']):>7}  "
                  f"{_pct(g['like_rate']):>7}{flag}")

    print(f"\nGroups of fewer than {MIN_GROUP} are marked. This is plain averaging over a")
    print("small sample - a difference between two groups is a hint to test, not a finding.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 4: log what was posted and how it did.")
    parser.add_argument("--db", type=Path, default=None, help=f"default: {rel(config.DB_PATH)}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("post", help="mark a rendered video as published")
    p.add_argument("video", type=Path)
    p.add_argument("--posted-at", required=True, help="YYYY-MM-DD")
    p.add_argument("--caption", default=None, help="the caption you posted with it")
    p.add_argument("--hashtags", default=None, help="the hashtags you used")
    p.set_defaults(func=cmd_post)

    p = sub.add_parser("metrics", help="type in numbers from the TikTok analytics screen")
    p.add_argument("video_id", type=int)
    p.add_argument("--pulled-at", default=None, help="YYYY-MM-DD (default: today)")
    p.set_defaults(func=cmd_metrics)

    p = sub.add_parser("list", help="show logged videos and their ids")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("report", help="which features go with which outcomes")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ToolError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
