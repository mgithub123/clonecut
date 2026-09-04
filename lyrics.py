#!/usr/bin/env python3
"""Stage 7e - Burn timed lyrics onto a finished shot.

    uv run lyrics.py out/doc-regress-20260903-174900.mp4 \
        --vox "~/Desktop/LuckyDog-Harmony/goodbyparty 1.4 vox.wav" \
        --lyrics "music/goodbyparty 1.4.lyrics.txt" \
        --start 41.2 --duration 9

Two things it does that matter.

**It transcribes the vocal stem, not the mix.** The ingest profile already holds a
whisper transcript, but it was run on the full mix and is useless here: five words
across the nine seconds this shot covers, one of them held for twenty-nine. On the
isolated stem the same window comes back cleanly, every line, with word timings.

**It shows the written lyrics, not what whisper heard.** Whisper is used only for
*when*; the words come from the lyrics file. So "goodbye party" cannot arrive as
"good bye potty", and a line the singer slurs still appears as it was written.
Matching is by token overlap against every line in the file, which handles a chorus
that repeats a line four times - each sung instance finds its own copy.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

import caption_styles as cs
import common
import config
from common import ToolError

MODEL = "small"          # base mishears sung vowels; small is right for one window
PAD = 3.0                # transcribe a little either side so edge lines are whole


def _norm(s: str) -> list[str]:
    return re.findall(r"[a-z']+", s.lower())


def transcribe(vox: Path, start: float, dur: float, *, model: str = MODEL) -> list[dict]:
    """Whisper segments over one window of the stem, in song time. Cached."""
    key = f"{common.file_hash(vox)[:12]}-{start:.2f}-{dur:.2f}-{model}"
    cached = config.CACHE_DIR / f"lyrics-asr-{key}.json"
    if cached.exists():
        return json.loads(cached.read_text())

    config.ensure_dirs()
    clip = config.CACHE_DIR / f"lyrics-win-{key}.wav"
    t0 = max(0.0, start - PAD)
    common.run([config.FFMPEG, "-y", "-loglevel", "error", "-ss", str(t0),
                "-t", str(dur + 2 * PAD), "-i", str(vox),
                "-ac", "1", "-ar", "16000", str(clip)])
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:                       # pragma: no cover
        raise ToolError(f"faster-whisper is not installed: {exc}") from exc
    m = WhisperModel(model, device=config.WHISPER_DEVICE,
                     compute_type=config.WHISPER_COMPUTE_TYPE)
    segs, _ = m.transcribe(str(clip), word_timestamps=True, vad_filter=False)
    out = [{"start": s.start + t0, "end": s.end + t0, "text": s.text.strip()}
           for s in segs if s.text.strip()]
    common.write_json(cached, out)
    return out


def align(segments: list[dict], lyrics: Path, start: float, dur: float) -> list[dict]:
    """Whisper's timings, the lyrics file's words.

    Each heard segment takes the written line it overlaps most, scanning forward so
    a repeated chorus line consumes a new copy each time rather than the same one.
    A segment matching nothing is kept as heard - better a slightly wrong word than
    a gap where the singer clearly sang.
    """
    lines = [l.strip() for l in lyrics.read_text().splitlines() if l.strip()]
    toks = [_norm(l) for l in lines]
    used = 0
    out = []
    for seg in segments:
        if seg["end"] <= start or seg["start"] >= start + dur:
            continue
        heard = set(_norm(seg["text"]))
        best, best_score = None, 0.0
        for i in range(used, len(lines)):
            if not toks[i]:
                continue
            score = len(heard & set(toks[i])) / max(len(toks[i]), 1)
            if score > best_score:
                best, best_score = i, score
        text = lines[best] if best is not None and best_score >= 0.5 else seg["text"]
        if best is not None and best_score >= 0.5:
            used = best + 1
        out.append({"start": max(seg["start"], start), "end": min(seg["end"], start + dur),
                    "text": text, "heard": seg["text"], "score": round(best_score, 2)})

    # Hold each line until the next one starts. Whisper ends a segment when the
    # singing stops, which leaves half a second of nothing between lines - on
    # screen that reads as a flicker rather than as a phrase ending.
    for i, c in enumerate(out):
        c["end"] = out[i + 1]["start"] if i + 1 < len(out) else start + dur
    return out


def _wrap(draw, text: str, font, max_w: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def card(text: str, *, size: int, top: int) -> Image.Image:
    """One transparent full-canvas overlay holding a lyric line.

    White with a heavy black stroke: these sit over a moving rainy background, and
    a drop shadow alone disappears against the bright parts of the glass.
    """
    from PIL import ImageFont
    im = Image.new("RGBA", (cs.CANVAS_W, cs.CANVAS_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(cs.resolve_font(), size)
    lines = _wrap(d, text, font, int(cs.CANVAS_W * 0.88))
    lh = int(size * 1.22)
    y = top
    for ln in lines:
        w = d.textlength(ln, font=font)
        d.text(((cs.CANVAS_W - w) / 2, y), ln, font=font, fill=(255, 255, 255),
               stroke_width=max(4, size // 10), stroke_fill=(0, 0, 0))
        y += lh
    return im


def burn(video: Path, cues: list[dict], out: Path, *, start: float,
         size: int, top: int) -> Path:
    """Overlay one PNG per line, each enabled only for its own moment.

    A handful of overlays with time windows, rather than a frame sequence: the
    character video is left untouched and re-encoded once.
    """
    if not cues:
        raise ToolError("no lyric lines land inside this window")
    work = config.CACHE_DIR / f"lyrics-cards-{out.stem}"
    work.mkdir(parents=True, exist_ok=True)
    inputs, filters, last = [], [], "0:v"
    for i, c in enumerate(cues):
        p = work / f"card-{i:02d}.png"
        card(c["text"], size=size, top=top).save(p)
        inputs += ["-i", str(p)]
        s, e = c["start"] - start, c["end"] - start
        nxt = f"v{i}"
        filters.append(f"[{last}][{i + 1}:v]overlay=0:0:enable='between(t,{s:.3f},{e:.3f})'[{nxt}]")
        last = nxt
    cmd = [config.FFMPEG, "-y", "-loglevel", "error", "-i", str(video), *inputs,
           "-filter_complex", ";".join(filters), "-map", f"[{last}]", "-map", "0:a?",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           "-c:a", "copy", str(out)]
    common.run(cmd)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("video", help="a finished shot to write lyrics onto")
    p.add_argument("--vox", required=True, help="the vocal stem, not the mix")
    p.add_argument("--lyrics", required=True, help="the written lyrics, one line per line")
    p.add_argument("--start", type=float, required=True, help="where the shot starts in the song")
    p.add_argument("--duration", type=float, default=None,
                   help="shot length (default: the video's own)")
    p.add_argument("--size", type=int, default=58)
    p.add_argument("--top", type=int, default=cs.SAFE_Y0 + 20,
                   help=f"y of the first line (safe box starts at {cs.SAFE_Y0})")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    video = Path(a.video).expanduser()
    if not video.exists():
        raise ToolError(f"no such video: {video}")
    vox = Path(a.vox).expanduser()
    if not vox.exists():
        raise ToolError(f"no vocal stem at {vox}")
    lyr = Path(a.lyrics).expanduser()
    if not lyr.exists():
        raise ToolError(f"no lyrics at {lyr}")

    dur = a.duration
    if dur is None:
        probe = subprocess.run([config.FFPROBE, "-v", "error", "-show_entries",
                                "format=duration", "-of", "csv=p=0", str(video)],
                               capture_output=True, text=True)
        dur = float(probe.stdout.strip())

    print(f"transcribing the stem {a.start}s +{dur:.2f}s ({a.model})")
    segs = transcribe(vox, a.start, dur, model=a.model)
    cues = align(segs, lyr, a.start, dur)
    for c in cues:
        flag = "" if c["score"] >= 0.5 else "   <- no line matched, showing what was heard"
        print(f"  {c['start']:6.2f}-{c['end']:6.2f}  {c['text']}{flag}")

    out = Path(a.out) if a.out else config.OUT_DIR / f"{video.stem}-lyrics.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    burn(video, cues, out, start=a.start, size=a.size, top=a.top)
    common.write_json(out.with_suffix(".json"), {
        "video": out.name, "tool": "lyrics.py", "source": video.name,
        "start": a.start, "duration": common.r3(dur), "model": a.model,
        "lines": [{"start": common.r3(c["start"]), "end": common.r3(c["end"]),
                   "text": c["text"], "heard": c["heard"], "match": c["score"]}
                  for c in cues],
    })
    print("wrote", common.rel(out))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
