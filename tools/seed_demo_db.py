#!/usr/bin/env python3
"""Fill a throwaway database with synthetic posts and metrics, so `log.py report`
can be seen working without waiting weeks for real numbers.

    uv run tools/seed_demo_db.py --db demo.db
    uv run log.py --db demo.db report

Never point this at luckydog.db - the numbers are invented.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import log  # noqa: E402

# Two tracks, so similarity retrieval has something to discriminate on: a plan
# made against goodbye-party should pull back the goodbye-party edits, not
# whichever video happened to do best.
TRACKS = {
    "goodbye-party": {"source": "music/goodbye-party.wav", "bpm": 120.19,
                      "clips": ["raw/test-a.mp4", "raw/test-b.mp4"]},
    "back-room":     {"source": "music/back-room.wav", "bpm": 88.0,
                      "clips": ["raw/backroom-1.mp4", "raw/backroom-2.mp4"]},
}

# (variant, hook caption, avg segment length, song section, beat synced,
#  track, "true" underlying watch-through the fake metrics are drawn around)
RECIPES = [
    ("fast-cut-hook",    "why does this sound like that?", 0.8, "high", 1, "goodbye-party", 0.46),
    ("question-open",    "how did we record this?",        0.9, "high", 1, "goodbye-party", 0.44),
    ("question-slow",    "what happened to the demo?",     2.4, "mid",  0, "goodbye-party", 0.33),
    ("statement-fast",   "we recorded this in one take",   0.7, "high", 1, "goodbye-party", 0.39),
    ("statement-medium", "this is the second take",        1.5, "mid",  1, "goodbye-party", 0.31),
    ("statement-slow",   "the whole song is one loop",     2.6, "low",  0, "goodbye-party", 0.24),
    ("negation-fast",    "no click track",                 0.9, "high", 1, "goodbye-party", 0.41),
    ("pov-fast",         "pov you found the band early",   0.8, "high", 1, "goodbye-party", 0.43),
    ("backroom-slow",    "the whole song is one loop",     2.8, "low",  0, "back-room",     0.52),
    ("backroom-hold",    "we tracked this at 3am",         2.2, "mid",  0, "back-room",     0.49),
    ("backroom-quiet",   None,                             1.9, "low",  0, "back-room",     0.47),
    ("backroom-drums",   "listen to the room",             1.3, "high", 1, "back-room",     0.51),
]


def fake_sidecar(variant, hook, avg_len, section, synced, tmp: Path, audio_start: float,
                 track: dict) -> Path:
    cuts = max(2, round(12 / avg_len))
    segments = [{"clip": track["clips"][i % len(track["clips"])], "in": 0.0,
                 "out": round(avg_len, 3), "speed": 1.0,
                 "snap_to_beat": bool(synced), "reason": "seed"}
                for i in range(cuts)]
    captions = []
    if hook:
        captions.append({"text": hook, "start": 0.1, "end": 2.2,
                         "style": "hook", "position": "upper-third"})
    captions.append({"text": "Lucky Dog - out now", "start": 3.0, "end": 5.0,
                     "style": "body", "position": "lower-third"})
    edl = {
        "variant_name": variant,
        "strategy_notes": "seeded demo row",
        "target_duration": round(avg_len * cuts, 3),
        "audio": {"source": track["source"], "start": audio_start,
                  "fade_in": 0.0, "fade_out": 0.5},
        "segments": segments,
        "captions": captions,
    }
    lengths = [s["out"] - s["in"] for s in segments]
    duration = sum(lengths)
    features = {
        "duration": round(duration, 3),
        "cut_count": cuts,
        "avg_segment_length": round(duration / cuts, 3),
        "shortest_segment": round(min(lengths), 3),
        "longest_segment": round(max(lengths), 3),
        "cuts_per_second": round(cuts / duration, 3),
        "caption_count": len(captions),
        "caption_density": round(len(captions) / duration, 3),
        "beat_synced": bool(synced),
        "uses_speed_ramp": False,
        "hook_type": hook,
        "hook_position": "upper-third" if hook else None,
        "clips_used": sorted(set(track["clips"])),
        "audio_start": audio_start,
    }
    video = tmp / f"{variant}.mp4"
    video.write_bytes(b"")          # a placeholder; nothing reads the pixels
    video.with_suffix(".json").write_text(json.dumps({
        "video": video.name,
        "rendered_at": "2026-08-01T00:00:00+00:00",
        "edl_path": f"plans/{variant}.json",
        "edl": edl,
        "derived_features": features,
    }, indent=2))
    return video


SECTION_STARTS = {"low": 1.0, "mid": 24.0, "high": 9.0}


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed a demo database with invented numbers.")
    parser.add_argument("--db", type=Path, default=Path("demo.db"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.db.name == "luckydog.db":
        print("refusing to seed luckydog.db with invented numbers", file=sys.stderr)
        return 1
    args.db.unlink(missing_ok=True)

    rng = random.Random(args.seed)
    tmp = Path("cache") / "_demo_renders"
    tmp.mkdir(parents=True, exist_ok=True)

    conn = log.connect(args.db)
    posted = date(2026, 7, 1)
    for i, (variant, hook, avg_len, section, synced, track_name, true_wtr) in enumerate(RECIPES):
        track = TRACKS[track_name]
        video = fake_sidecar(variant, hook, avg_len, section, synced, tmp,
                             SECTION_STARTS[section], track)
        posted_at = posted + timedelta(days=i * 3)
        # A stand-in profile, so each seeded row carries its own track's BPM and
        # sections instead of borrowing whatever profile happens to be on disk.
        fake_profile = {"audio": {
            "path": track["source"], "bpm": track["bpm"],
            "sections": [{"start": 0.0, "end": 8.0, "label": "low", "index": 0},
                         {"start": 8.0, "end": 22.0, "label": "high", "index": 1},
                         {"start": 22.0, "end": 40.0, "label": "mid", "index": 2}],
        }}
        vid = log.record_video(conn, video, posted_at=posted_at.isoformat(),
                               caption=hook or "", hashtags="#luckydog",
                               profile=fake_profile)
        # Two pulls per video, a few days apart, so maturation is visible.
        base_views = int(rng.lognormvariate(8.6, 1.0))
        for day, growth in ((2, 1.0), (9, rng.uniform(1.6, 3.2))):
            views = int(base_views * growth)
            wtr = min(0.95, max(0.05, rng.gauss(true_wtr, 0.03)))
            log.record_metrics(conn, vid, {
                "views": views,
                "watch_through_rate": round(wtr, 4),
                "likes": int(views * rng.uniform(0.04, 0.11)),
                "shares": int(views * rng.uniform(0.002, 0.02) * (1.6 if synced else 1.0)),
                "comments": int(views * rng.uniform(0.001, 0.006)),
                "follows": int(views * rng.uniform(0.001, 0.008)),
                "saves": int(views * rng.uniform(0.003, 0.02)),
            }, (posted_at + timedelta(days=day)).isoformat())
    conn.close()

    print(f"seeded {len(RECIPES)} videos with 2 pulls each into {args.db}")
    print(f"\n  uv run log.py --db {args.db} list")
    print(f"  uv run log.py --db {args.db} report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
