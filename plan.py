#!/usr/bin/env python3
"""Stage 2 - Plan (manual mode).

The planner does not call an API. It assembles everything the model needs into a
prompt you paste into the Claude app, then reads the reply back, validates it,
snaps the cuts to the beat grid, and writes EDLs to plans/.

    uv run plan.py prompt --profile profiles/goodbye-party.json --notes "..."
    # paste plans/<stamp>-prompt.md into Claude, attach the keyframes,
    # save the reply to a file, then:
    uv run plan.py ingest reply.txt --profile profiles/goodbye-party.json

Snapping happens here, in code, rather than in the prompt: a model asked to do
arithmetic across a beat grid will drift, and a cut that is 40ms off the beat is
visible. The model is told to place cuts musically and approximately, and this
file makes them exact.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import common
import config
import schema
from common import ToolError, r3, rel
from schema import EDL, EdlError

STAMP_FMT = "%Y%m%d-%H%M%S"


# ---------------------------------------------------------------------------
# trimming the profile down to what a model can actually use
# ---------------------------------------------------------------------------

def downsample_curve(curve: dict[str, list[float]], points: int = 48) -> list[list[float]]:
    """Reduce a dense times/values curve to a handful of [time, value] pairs.

    The raw curves run to thousands of points; nobody reads that, and it crowds
    out the parts of the profile that carry real information.
    """
    times, values = curve.get("times", []), curve.get("values", [])
    if not times:
        return []
    if len(times) <= points:
        return [[r3(t), r3(v)] for t, v in zip(times, values)]
    step = len(times) / points
    out = []
    for i in range(points):
        lo, hi = int(i * step), max(int(i * step) + 1, int((i + 1) * step))
        window = values[lo:hi]
        out.append([r3(times[lo]), r3(sum(window) / len(window)) if window else 0.0])
    return out


def compact_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """The footage profile with raw arrays downsampled and paths kept."""
    clips = []
    for clip in profile["clips"]:
        motion = clip.get("motion", [])
        transcript = clip.get("transcript") or {}
        clips.append({
            "path": clip["path"],
            "duration": clip["duration"],
            "resolution": f"{clip['width']}x{clip['height']}",
            "fps": clip["fps"],
            "scene_cuts": clip.get("scene_cuts", []),
            "scenes": clip.get("scenes", []),
            # per-second motion is already compact and is the most useful signal
            "motion_per_second": [[m["t"], m["energy_norm"]] for m in motion],
            "busiest_seconds": [m["t"] for m in sorted(motion, key=lambda m: -m["energy"])[:6]],
            "speech_regions": clip.get("speech", []),
            "silence_regions": clip.get("silence", []),
            "transcript": {
                "text": transcript.get("text", ""),
                "segments": transcript.get("segments", []),
            },
            "keyframe_times": [k["t"] for k in clip.get("keyframes", [])],
        })

    audio = profile["audio"]
    beats = audio.get("beats", [])
    beat_note = beats if len(beats) <= 200 else (
        f"{len(beats)} beats every {audio.get('beat_interval')}s "
        f"from {beats[0]} to {beats[-1]}"
    )
    return {
        "clips": clips,
        "music": {
            "path": audio["path"],
            "duration": audio["duration"],
            "bpm": audio["bpm"],
            "beat_interval": audio.get("beat_interval"),
            "beats": beat_note,
            "sections": audio.get("sections", []),
            "energy_curve": downsample_curve(audio.get("rms_energy", {})),
            "onset_strength_curve": downsample_curve(audio.get("onset_strength", {})),
        },
    }


# ---------------------------------------------------------------------------
# choosing which keyframes to show
# ---------------------------------------------------------------------------

def select_keyframes(profile: dict[str, Any], budget: int) -> list[dict[str, Any]]:
    """Pick a spread of keyframes within a budget.

    Scene starts first (they show what each shot actually is), then the busiest
    moments, then an even fill. Within each tier the clips are taken in
    round-robin, so a long first clip cannot eat the whole budget and leave a
    later clip unseen.
    """
    per_clip: dict[str, list[list[float]]] = {}
    for clip in profile["clips"]:
        frames = clip.get("keyframes", [])
        if not frames:
            continue
        times = [k["t"] for k in frames]
        motion = {m["t"]: m["energy"] for m in clip.get("motion", [])}
        seen: set[float] = set()
        tiers: list[list[float]] = [[], [], []]

        def take(t: float, tier: int) -> None:
            if t not in seen:
                seen.add(t)
                tiers[tier].append(t)

        def nearest(t: float) -> float:
            return min(times, key=lambda k: abs(k - t))

        for scene in clip.get("scenes", []):
            take(nearest(scene["start"]), 0)
        for sec in sorted(motion, key=lambda s: -motion[s])[:4]:
            take(nearest(sec), 1)
        for t in times:
            take(t, 2)
        per_clip[clip["path"]] = tiers

    chosen: list[tuple[str, float]] = []
    for tier in range(3):
        queues = {path: list(tiers[tier]) for path, tiers in per_clip.items()}
        while len(chosen) < budget and any(queues.values()):
            for path in list(queues):
                if not queues[path] or len(chosen) >= budget:
                    continue
                chosen.append((path, queues[path].pop(0)))
    chosen.sort()

    lookup = {(c["path"], k["t"]): k["path"]
              for c in profile["clips"] for k in c.get("keyframes", [])}
    return [{"clip": path, "t": t, "image": lookup[(path, t)]}
            for path, t in chosen if (path, t) in lookup]


# ---------------------------------------------------------------------------
# history (Stage 4 fills this in; until then it says so)
# ---------------------------------------------------------------------------

def retrieve_history(limit: int = 5, db_path: Path | None = None,
                     profile: dict[str, Any] | None = None,
                     notes: str | None = None) -> tuple[str, int]:
    """Past EDLs and how they performed. Returns (text, logged_count).

    The profile and notes go through so retrieval can rank by similarity to what
    is being planned now, rather than just by what performed best.
    """
    try:
        import log
    except ImportError:
        return (
            "No performance history yet - the logging stage is not built. Judge this "
            "edit on the footage and the music alone, and do not claim any of these "
            "choices is proven.",
            0,
        )
    return log.history_for_prompt(limit, db_path, profile=profile, notes=notes)


# ---------------------------------------------------------------------------
# the prompt
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
# Edit plan request - Lucky Dog

You are planning short-form vertical videos (TikTok, 1080x1920) for a music
project called Lucky Dog. Your job is to decide the edit and nothing else: you
output a JSON Edit Decision List, and a separate renderer executes it. You never
see or write ffmpeg.

Produce **{variants} EDLs with genuinely different strategies** - not three
versions of one idea. Vary the hook, the pacing, and the section of the song
used. If two of your variants would cut to roughly the same rhythm from roughly
the same shots, replace one.

## Output format

Reply with a JSON array of {variants} EDL objects and nothing else - no prose
before or after, no commentary between them. Put it in a single ```json code
block.

{schema}

### Rules that will be checked

- `clip` must be one of the paths listed under "Footage" below, exactly as written.
- `in` and `out` are times **within the source clip**, and `out` must not exceed
  that clip's duration.
- `audio.start` is an offset into the music track. The edit's total length is the
  sum of the segment lengths (divided by speed), and the music must be long
  enough to cover it from `audio.start`.
- Caption `start`/`end` are times **in the finished video**, starting at 0.
- Unknown fields are rejected. So is a `variant_name` that is not kebab-case.

### About the beat grid

Set `snap_to_beat: true` on segments whose cut should land on the beat. **Do not
try to compute beat-aligned timestamps yourself** - give musically sensible
approximate boundaries and code will snap them to the exact beat afterwards.
Arithmetic across a beat grid is where models drift, and a cut 40ms off the beat
is visible.

## Footage

The keyframe images attached to this message are labelled with the clip and
timestamp they come from. `motion_per_second` is normalised 0-1 frame-difference
energy - high means movement, low means a near-static shot.

```json
{footage}
```

## Music

```json
{music}
```

## What has worked before

{history}

{notes}## Reminder

{variants} EDLs, genuinely different from each other, as a JSON array in one
```json block, no other text.
"""

NOTES_TEMPLATE = """## Format notes from me

{notes}

Treat this as a direction to try, not a rule. If it fights the footage, say so
in `strategy_notes` and do what the footage supports.

"""


def build_prompt(profile: dict[str, Any], *, variants: int, notes: str | None,
                 keyframes: list[dict[str, Any]],
                 db_path: Path | None = None) -> tuple[str, str, int]:
    compact = compact_profile(profile)
    history, logged = retrieve_history(db_path=db_path, profile=profile, notes=notes)

    footage = dict(compact)
    music = footage.pop("music")
    footage["keyframes_attached"] = [
        {"label": keyframe_label(i, kf), "clip": kf["clip"], "t": kf["t"]}
        for i, kf in enumerate(keyframes, 1)
    ]

    prompt = PROMPT_TEMPLATE.format(
        variants=variants,
        schema=schema.describe_schema(),
        footage=compact_numbers(json.dumps(footage, indent=2, ensure_ascii=False)),
        music=compact_numbers(json.dumps(music, indent=2, ensure_ascii=False)),
        history=history,
        notes=NOTES_TEMPLATE.format(notes=notes.strip()) if notes else "",
    )
    return prompt, history, logged


def keyframe_label(index: int, kf: dict[str, Any]) -> str:
    stem = Path(kf["clip"]).stem
    return f"{index:02d}_{stem}_t{kf['t']:g}s"


# ---------------------------------------------------------------------------
# reading the reply back
# ---------------------------------------------------------------------------

FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

_NUMERIC_ARRAY = re.compile(r"\[\s*\n\s*(-?[\d.eE+\-]+(?:,\s*\n?\s*-?[\d.eE+\-]+)*)\s*\n\s*\]")
_PAIR_ARRAY = re.compile(r"\[\s*\n\s*(\[[^\[\]]*\](?:,\s*\n\s*\[[^\[\]]*\])*)\s*\n\s*\]")


def compact_numbers(text: str) -> str:
    """Collapse arrays that hold only numbers onto one line.

    json.dumps(indent=2) puts a 65-beat grid on 65 lines, which is unreadable in
    a prompt you have to paste by hand and wastes tokens for no benefit.
    """
    prev = None
    while prev != text:
        prev = text
        text = _NUMERIC_ARRAY.sub(
            lambda m: "[" + ", ".join(part.strip() for part in m.group(1).split(",")) + "]",
            text,
        )
        text = _PAIR_ARRAY.sub(
            lambda m: "[" + " ".join(line.strip() for line in m.group(1).splitlines()) + "]",
            text,
        )
    return text


def extract_json_objects(text: str) -> list[Any]:
    """Pull JSON out of a pasted reply.

    Handles a fenced block, several fenced blocks, a bare array, a bare object,
    or objects with prose around them - all of which a chat reply can contain.
    """
    candidates: list[str] = [m.group(1).strip() for m in FENCE.finditer(text)]
    if not candidates:
        candidates = [text.strip()]

    found: list[Any] = []
    for chunk in candidates:
        try:
            found.append(json.loads(chunk))
            continue
        except json.JSONDecodeError:
            pass
        found.extend(_scan_objects(chunk))

    if not found:
        found = _scan_objects(text)

    flat: list[Any] = []
    for item in found:
        flat.extend(item) if isinstance(item, list) else flat.append(item)
    return flat


def _scan_objects(text: str) -> list[Any]:
    """Find complete JSON values by walking bracket depth, ignoring strings."""
    out, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            if depth == 0:
                start = i
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    out.append(json.loads(text[start:i + 1]))
                except json.JSONDecodeError:
                    pass
                start = None
            depth = max(0, depth)
    return out


# ---------------------------------------------------------------------------
# beat snapping - the reason the model is not asked to do arithmetic
# ---------------------------------------------------------------------------

MIN_SEGMENT = 0.15


def snap_edl_to_beats(edl: EDL, profile: dict[str, Any]) -> tuple[EDL, list[str]]:
    """Move every snap_to_beat boundary onto the nearest beat.

    A segment's position in the finished video is the running sum of the segment
    lengths before it, so the cut that matters musically is at
    `audio.start + timeline_position`. Each snapped segment's end is moved to the
    nearest beat in that music time, and its out point follows.
    """
    beats = profile.get("audio", {}).get("beats") or []
    report: list[str] = []
    if not beats:
        return edl, ["no beat grid in the profile; nothing was snapped"]

    durations = {c["path"]: c["duration"] for c in profile.get("clips", [])}
    data = json.loads(edl.to_json())

    def nearest(t: float) -> float:
        return min(beats, key=lambda b: abs(b - t))

    # Align the whole grid first: if the music starts off-beat, every downstream
    # snap inherits that offset.
    if any(s.snap_to_beat for s in edl.segments):
        original = data["audio"]["start"]
        aligned = nearest(original)
        if abs(aligned - original) > 1e-6:
            data["audio"]["start"] = r3(aligned)
            report.append(f"audio.start {original:.3f}s -> {aligned:.3f}s (nearest beat)")

    audio_start = data["audio"]["start"]
    timeline = 0.0
    for i, seg in enumerate(data["segments"]):
        out_dur = (seg["out"] - seg["in"]) / seg["speed"]
        if not seg.get("snap_to_beat"):
            timeline += out_dur
            continue

        desired_end = audio_start + timeline + out_dur
        target = nearest(desired_end)
        new_out_dur = target - (audio_start + timeline)
        new_out = seg["in"] + new_out_dur * seg["speed"]
        clip_duration = durations.get(seg["clip"])

        why = None
        if new_out_dur < MIN_SEGMENT:
            why = f"snapping would leave {new_out_dur:.3f}s"
        elif clip_duration is not None and new_out > clip_duration:
            # Try the beat before instead of running off the end of the clip.
            earlier = [b for b in beats if b < target]
            if earlier:
                alt = earlier[-1]
                alt_dur = alt - (audio_start + timeline)
                alt_out = seg["in"] + alt_dur * seg["speed"]
                if alt_dur >= MIN_SEGMENT and alt_out <= clip_duration:
                    target, new_out_dur, new_out = alt, alt_dur, alt_out
                else:
                    why = f"nearest beats run past the end of {seg['clip']}"
            else:
                why = f"nearest beat runs past the end of {seg['clip']}"

        if why:
            seg["snap_to_beat"] = False
            report.append(f"segment {i}: left unsnapped - {why}")
            timeline += out_dur
            continue

        if abs(new_out - seg["out"]) > 1e-6:
            report.append(
                f"segment {i}: out {seg['out']:.3f}s -> {r3(new_out):.3f}s "
                f"(cut lands on beat {target:.3f}s)"
            )
        seg["out"] = r3(new_out)
        timeline += new_out_dur

    data["target_duration"] = r3(timeline)
    return EDL.model_validate(data), report


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_prompt(args: argparse.Namespace) -> int:
    profile = common.read_json(args.profile)
    config.ensure_dirs()
    stamp = datetime.now().strftime(STAMP_FMT)

    keyframes = select_keyframes(profile, args.keyframes)
    prompt, history, logged = build_prompt(
        profile, variants=args.variants, notes=args.notes, keyframes=keyframes,
        db_path=args.db,
    )

    prompt_path = config.PLANS_DIR / f"{stamp}-prompt.md"
    prompt_path.write_text(prompt, encoding="utf-8")

    kf_dir = config.PLANS_DIR / f"{stamp}-keyframes"
    kf_dir.mkdir(parents=True, exist_ok=True)
    for i, kf in enumerate(keyframes, 1):
        src = Path(kf["image"])
        if src.exists():
            shutil.copy2(src, kf_dir / f"{keyframe_label(i, kf)}.jpg")

    common.write_json(config.PLANS_DIR / f"{stamp}-meta.json", {
        "stamp": stamp,
        "profile": rel(Path(args.profile)),
        "variants": args.variants,
        "notes": args.notes,
        "logged_videos": logged,
        "history_is_meaningful": logged >= config.MIN_HISTORY_FOR_RETRIEVAL,
        "keyframes": [{"label": keyframe_label(i, k), **k} for i, k in enumerate(keyframes, 1)],
    })

    print(f"prompt     {rel(prompt_path)}  ({len(prompt):,} characters)")
    print(f"keyframes  {rel(kf_dir)}  ({len(keyframes)} images)")
    if logged < config.MIN_HISTORY_FOR_RETRIEVAL:
        print(f"history    {logged} logged videos - below the {config.MIN_HISTORY_FOR_RETRIEVAL} "
              f"needed to be meaningful, and the prompt says so")
    print()
    print("Next:")
    print(f"  1. open {rel(prompt_path)} and paste the whole thing into Claude")
    print(f"  2. attach every image in {rel(kf_dir)}")
    print("  3. save the reply to a file, then:")
    print(f"     uv run plan.py ingest reply.txt --profile {rel(Path(args.profile))}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    text = sys.stdin.read() if str(args.reply) == "-" else Path(args.reply).read_text(encoding="utf-8")
    profile = common.read_json(args.profile)
    config.ensure_dirs()

    objects = extract_json_objects(text)
    if not objects:
        raise ToolError(
            "No JSON found in the reply. Paste Claude's whole message, including the "
            "```json block, or save just the JSON array to the file."
        )

    stamp = datetime.now().strftime(STAMP_FMT)
    written, failed = [], 0
    for i, obj in enumerate(objects):
        try:
            edl = EDL.model_validate(obj)
        except Exception as exc:
            failed += 1
            name = obj.get("variant_name", f"#{i}") if isinstance(obj, dict) else f"#{i}"
            print(f"\nREJECTED  variant {name}", file=sys.stderr)
            from pydantic import ValidationError
            if isinstance(exc, ValidationError):
                print(schema.format_validation_error(exc), file=sys.stderr)
            else:
                print(f"  {exc}", file=sys.stderr)
            continue

        snapped, report = snap_edl_to_beats(edl, profile)
        errors, warnings = schema.validate_media(snapped)

        path = config.PLANS_DIR / f"{stamp}-{snapped.variant_name}.json"
        snapped.write(path)
        written.append(path)

        print(f"\n{snapped.variant_name}  -> {rel(path)}")
        print(f"  {len(snapped.segments)} segments, {len(snapped.captions)} captions, "
              f"{snapped.duration:.2f}s"
              f"{', beat-synced' if snapped.beat_synced else ''}")
        print(f"  {snapped.strategy_notes}")
        for line in report:
            print(f"  snap: {line}")
        for e in errors:
            print(f"  ERROR:   {e}", file=sys.stderr)
        for w in warnings:
            print(f"  warning: {w}")
        if errors:
            print("  this EDL will not render until those are fixed", file=sys.stderr)

    print(f"\n{len(written)} plan(s) written, {failed} rejected")
    if written:
        print(f"render one with:\n  uv run render.py {rel(written[0])}")
    return 1 if failed and not written else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 2: plan edits (manual mode, no API calls).")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prompt", help="write the prompt to paste into Claude")
    p.add_argument("--profile", required=True, type=Path, help="a profile from ingest.py")
    p.add_argument("--notes", default=None, help="free-text trend notes: a format you want to try")
    p.add_argument("--variants", type=int, default=config.PLAN_VARIANTS)
    p.add_argument("--keyframes", type=int, default=config.PLAN_KEYFRAME_BUDGET)
    p.add_argument("--db", type=Path, default=None,
                   help=f"performance history to draw on (default: {rel(config.DB_PATH)})")
    p.set_defaults(func=cmd_prompt)

    p = sub.add_parser("ingest", help="read Claude's reply back and write EDLs")
    p.add_argument("reply", type=Path, help="file holding the reply, or - for stdin")
    p.add_argument("--profile", required=True, type=Path)
    p.set_defaults(func=cmd_ingest)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ToolError, EdlError) as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
