#!/usr/bin/env python3
"""Stage 7d - Montage.

Assembles several performed shots into one video, cutting between characters.

    uv run montage.py shots.json
    uv run montage.py shots.json --dry-run      # validate and show the cut plan

A shot list looks like this:

    {
      "song":  "goodbyparty 1.4",
      "mix":   "music/goodbyparty 1.4.wav",
      "vox":   "~/Desktop/LuckyDog-Harmony/goodbyparty 1.4 vox.wav",
      "shots": [
        {"rig": "doctor", "start": 41.2, "duration": 9,
         "background": "rainy-window", "rain": true, "tears": 6,
         "text": "this is our song. it's called goodbye party"},
        {"rig": "dog", "start": 50.2, "duration": 6}
      ]
    }

Two things it does that matter.

Cuts land on the beat. Shot boundaries are snapped to the song's beat grid from
the ingest profile, the same grid plan.py uses. A cut half a beat off reads as a
mistake even when nobody can say why.

Shots are validated against what each rig can actually do before anything renders.
These rigs differ sharply - one has no pupils, another has a single mouth drawing -
so a shot asking for something impossible should fail in a second, not after
twenty minutes of rendering.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

import common
import config
import perform
import voice
from common import ToolError

FPS = perform.FPS


def beat_grid(track: str) -> list[float]:
    prof = config.PROFILE_DIR / f"{track}.json"
    if not prof.exists():
        raise ToolError(f"no profile for {track!r} - run ingest.py first")
    return json.loads(prof.read_text()).get("audio", {}).get("beats") or []


def snap(t: float, beats: list[float], max_shift: float = 0.25) -> tuple[float, float]:
    """Nearest beat, if it is close enough to be the intended one."""
    if not beats:
        return t, 0.0
    near = min(beats, key=lambda b: abs(b - t))
    return (near, near - t) if abs(near - t) <= max_shift else (t, 0.0)


def load_shots(path: Path) -> dict:
    if not path.exists():
        raise ToolError(f"no such shot list: {path}")
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ToolError(f"{common.rel(path)} is not valid JSON: {exc}") from exc
    for key in ("song", "mix", "vox", "shots"):
        if key not in doc:
            raise ToolError(f"{common.rel(path)} is missing {key!r}")
    if not doc["shots"]:
        raise ToolError("shot list is empty")
    return doc


def validate(doc: dict) -> list[dict]:
    """Check every shot against its rig before rendering anything."""
    vox = Path(doc["vox"]).expanduser()
    if not vox.exists():
        raise ToolError(f"no vocal stem at {vox}")
    mix = Path(doc["mix"]).expanduser()
    if not mix.exists():
        raise ToolError(f"no mix at {mix}")

    beats = beat_grid(doc["song"])
    out = []
    for i, sh in enumerate(doc["shots"], 1):
        for key in ("rig", "start", "duration"):
            if key not in sh:
                raise ToolError(f"shot {i} is missing {key!r}")
        r = perform.load_rig(sh["rig"])
        caps = r["capabilities"]
        if not caps.get("lipsync"):
            raise ToolError(f"shot {i}: rig {sh['rig']!r} has no mouth library, so it "
                            f"cannot lip-sync. Generate one first (see face.py).")
        if sh.get("tears") and not caps.get("tears"):
            raise ToolError(f"shot {i}: rig {sh['rig']!r} does not support tears")
        if sh.get("gaze") and not caps.get("gaze"):
            raise ToolError(f"shot {i}: rig {sh['rig']!r} has no pupils, so gaze is impossible")
        if sh.get("pose") is not None:
            # Fail here rather than twenty minutes into a render. A pose can be
            # missing, or present but unusable, or usable but too foreshortened to
            # sing - three different answers, so say which.
            g = perform.pose_geometry(r, int(sh["pose"]))
            if not g["has_mouth"]:
                print(f"  shot {i}: pose {sh['pose']} shows no mouth at this angle, "
                      f"so none is drawn - a silent cutaway")
            elif not g["lipsync"]:
                print(f"  shot {i}: pose {sh['pose']} has the mouth at "
                      f"{g['foreshortening']:.0%} of head-on width - it will be held shut")
        if sh.get("gesture"):
            if sh.get("pose") is None:
                raise ToolError(f"shot {i}: gesture needs a pose - the hand plates are "
                                f"baked per view angle")
            if not caps.get("gesture"):
                raise ToolError(f"shot {i}: rig {sh['rig']!r} has no baked hand poses")
        start, shift = snap(float(sh["start"]), beats)
        dur = float(sh["duration"])
        end, _ = snap(start + dur, beats)
        out.append({**sh, "_index": i, "_start": start, "_duration": max(0.5, end - start),
                    "_shift": shift, "_rig": r})
    return out


def render(doc: dict, shots: list[dict], out_dir: Path, name: str) -> Path:
    vox = str(Path(doc["vox"]).expanduser())
    mix = str(Path(doc["mix"]).expanduser())
    all_frames: list[Path] = []

    for sh in shots:
        i, r = sh["_index"], sh["_rig"]
        start, dur = sh["_start"], sh["_duration"]
        n = int(round(dur * FPS))
        print(f"\nshot {i}: {sh['rig']}  {start:.2f}s +{dur:.2f}s  ({n} frames)")

        an = voice.analyse(vox, start, dur, doc["song"])
        an["_audio"] = vox
        print(f"  {len(an['onsets'])} syllables")

        track = perform.mouth_track(r, an, n)
        work = config.CACHE_DIR / f"montage-{name}-shot{i}"
        engine = perform.choose_engine(sh.get("engine", "auto"), r)
        if engine == "blender" and not (sh.get("tears") or sh.get("gesture")):
            import act
            frames = act.build_frames(r, an, track, work, blink=sh.get("blink", True),
                                      pose=(int(sh["pose"]) if sh.get("pose") is not None else None))
        else:
            frames = perform.build_frames(r, an, track, work,
                                          blink=sh.get("blink", True),
                                          tears=int(sh.get("tears", 0)),
                                          pose=(int(sh["pose"]) if sh.get("pose") is not None else None),
                                          gesture=bool(sh.get("gesture")))
        bg = None
        if sh.get("background"):
            bp = Path(sh["background"])
            if not bp.exists():
                for ext in (".jpg", ".png", ".jpeg"):
                    cand = perform.ASSETS / "backgrounds" / f"{sh['background']}{ext}"
                    if cand.exists():
                        bp = cand
                        break
            if not bp.exists():
                raise ToolError(f"shot {i}: no background {sh['background']!r}")
            bg = perform.fit_background(bp)

        comp = perform.composite(frames, bg=bg, rain=bool(sh.get("rain")),
                                 text=sh.get("text", ""),
                                 char_width=int(sh.get("char_width", 705)),
                                 char_top=int(sh.get("char_top", 430)),
                                 out_dir=config.CACHE_DIR / f"montage-{name}-comp{i}")
        all_frames.extend(comp)

    # Contiguous shots let the song play straight through; otherwise each shot
    # carries its own slice and the audio cuts with the picture.
    contiguous = all(abs(shots[k]["_start"] + shots[k]["_duration"] - shots[k + 1]["_start"]) < 0.05
                     for k in range(len(shots) - 1))
    total = len(all_frames) / FPS
    out = out_dir / f"{name}-{datetime.datetime.now():%Y%m%d-%H%M%S}.mp4"
    print(f"\nassembling {len(all_frames)} frames ({total:.2f}s), "
          f"audio {'continuous' if contiguous else 'cut per shot'}")

    if contiguous:
        perform.mux(all_frames, mix, shots[0]["_start"], total, out)
    else:
        seg_dir = config.CACHE_DIR / f"montage-{name}-audio"
        seg_dir.mkdir(parents=True, exist_ok=True)
        parts = []
        for sh in shots:
            seg = seg_dir / f"seg{sh['_index']:03d}.wav"
            common.run([config.FFMPEG, "-y", "-loglevel", "error",
                        "-ss", str(sh["_start"]), "-t", f"{sh['_duration']:.3f}",
                        "-i", mix, str(seg)])
            parts.append(seg)
        listing = seg_dir / "list.txt"
        listing.write_text("".join(f"file '{p.name}'\n" for p in parts))
        joined = seg_dir / "joined.wav"
        common.run([config.FFMPEG, "-y", "-loglevel", "error", "-f", "concat",
                    "-safe", "0", "-i", str(listing), str(joined)])
        perform.mux(all_frames, str(joined), 0.0, total, out)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("shots", help="shot list JSON")
    p.add_argument("--name", default=None, help="output stem (default: the shot list's name)")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--dry-run", action="store_true", help="validate and show the cut plan")
    a = p.parse_args(argv)

    path = Path(a.shots)
    doc = load_shots(path)
    shots = validate(doc)
    name = a.name or path.stem

    print(f"{len(shots)} shots, song {doc['song']!r}")
    total = 0.0
    for sh in shots:
        shift = sh["_shift"]
        note = f"  (snapped {shift:+.3f}s to the beat)" if abs(shift) > 0.001 else ""
        extras = [k for k in ("pose", "gesture", "background", "rain", "tears", "text") if sh.get(k)]
        print(f"  {sh['_index']}. {sh['rig']:8s} {sh['_start']:7.2f}s +{sh['_duration']:5.2f}s"
              f"{note}   {', '.join(extras)}")
        total += sh["_duration"]
    print(f"  total {total:.2f}s")

    if a.dry_run:
        return 0
    out_dir = Path(a.out_dir) if a.out_dir else config.OUT_DIR
    out = render(doc, shots, out_dir, name)
    common.write_json(out.with_suffix(".json"), {
        "video": out.name, "tool": "montage.py", "song": doc["song"],
        "shots": [{"rig": s["rig"], "start": common.r3(s["_start"]),
                   "duration": common.r3(s["_duration"]),
                   "beat_shift": common.r3(s["_shift"]),
                   "background": s.get("background"), "text": s.get("text", "")}
                  for s in shots],
        "duration": common.r3(total),
    })
    print("wrote", common.rel(out))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
