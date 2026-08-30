#!/usr/bin/env python3
"""Stage 3 - Render.

Takes an EDL and produces an mp4. Pure ffmpeg via subprocess, no API calls, no
decisions: every timing in the output comes from the EDL as written.

    uv run render.py examples/hand-written-reference.json
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import caption_styles as cs
import common
import config
import schema
from common import ToolError, rel
from schema import EDL, Caption, EdlError, Segment

# Output format. Short-form vertical, and a pixel format every phone can decode.
FPS = 30
VIDEO_CODEC = ["-c:v", "libx264", "-preset", "medium", "-crf", "20",
               "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1"]
AUDIO_CODEC = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]

# Crop bias, -1..1 on each axis: 0 centres the crop, negative favours the top
# or left of frame. Subjects usually sit slightly above centre, so a small
# negative Y is often better than dead centre on landscape source.
CROP_BIAS_X = 0.0
CROP_BIAS_Y = 0.0


# ---------------------------------------------------------------------------
# ffmpeg filter escaping
# ---------------------------------------------------------------------------

def q(value: str) -> str:
    """Quote a value for use inside a filter option.

    ffmpeg single-quoted strings cannot contain a quote, so a literal ' has to
    close the quote, escape itself, and reopen: 'foo'\\''bar'.
    """
    return "'" + str(value).replace("'", r"'\''") + "'"


# ---------------------------------------------------------------------------
# caption layout
# ---------------------------------------------------------------------------

def wrap_caption(text: str, style: dict, box_width: float) -> str:
    """Break a caption into lines that fit the available width.

    drawtext does not wrap, so this has to be decided up front. Glyph advance is
    estimated from the font size - exact metrics would mean a font-parsing
    dependency for a value that only decides where a line breaks.
    """
    if style["uppercase"]:
        text = text.upper()
    per_char = style["size"] * style["char_width_ratio"]
    max_chars = max(8, int(box_width / per_char))
    if len(text) <= max_chars:
        return text
    return "\n".join(textwrap.wrap(text, width=max_chars, break_long_words=True) or [text])


def caption_filters(cap: Caption, font: str, text_dir: Path, index: int) -> list[str]:
    """Build the drawtext filters for one caption: one per wrapped line.

    Handing drawtext a multi-line string centres the *block* but left-aligns the
    lines inside it, which reads as a mistake on a short-form video. Drawing each
    line as its own centred filter is the version-independent fix - ffmpeg only
    grew a text_align option in 7.0, and this has to work on whatever ffmpeg is
    already installed.
    """
    style = cs.get_style(cap.style.value)
    pos = cs.layout(cap.position.value)

    avail = cs.CANVAS_W - 2 * pos["margin_x"]
    box_width = avail * style["width_frac"]
    lines = wrap_caption(cap.text, style, box_width).split("\n")

    line_height = style["size"] * cs.LINE_HEIGHT_RATIO + style["line_spacing"]
    centre_y = cs.SAFE_Y0 + pos["anchor"] * cs.SAFE_H

    # The anchor positions the middle of the block, so a caption that wraps onto
    # three lines reaches further than a one-liner. Pull the block back inside
    # the safe area rather than letting it ride under the TikTok UI.
    block_half = (len(lines) - 1) / 2 * line_height + style["size"] * cs.LINE_HEIGHT_RATIO / 2
    if 2 * block_half <= cs.SAFE_H:
        centre_y = min(max(centre_y, cs.SAFE_Y0 + block_half), cs.SAFE_Y1 - block_half)
    else:
        centre_y = cs.SAFE_Y0 + cs.SAFE_H / 2   # too tall to fit; centre it

    enable = q(f"between(t,{cap.start:.3f},{cap.end:.3f})")

    filters = []
    for j, line in enumerate(lines):
        # Text goes in a file so no caption content has to survive filtergraph
        # escaping - quotes, colons and commas are all just bytes in a file.
        text_file = text_dir / f"cap_{index:03d}_{j:02d}.txt"
        text_file.write_text(line, encoding="utf-8")

        # Stack the lines around the anchor; text_h centres each one on its row.
        row_y = centre_y + (j - (len(lines) - 1) / 2) * line_height
        opts = [
            f"fontfile={q(font)}",
            f"textfile={q(str(text_file))}",
            "reload=0",
            # Without this, drawtext expands %{...} into ffmpeg metadata and
            # discards the whole line on a bare % ("Stray % near ''"), so a
            # caption reading "100% live" would silently render as nothing.
            "expansion=none",
            f"fontsize={style['size']}",
            f"fontcolor={q(style['color'])}",
            f"x={q('(w-text_w)/2')}",
            f"y={q(f'{row_y:.1f}-text_h/2')}",
            f"enable={enable}",
        ]
        if style["border_w"]:
            opts += [f"borderw={style['border_w']}", f"bordercolor={q(style['border_color'])}"]
        if style["shadow_x"] or style["shadow_y"]:
            opts += [f"shadowx={style['shadow_x']}", f"shadowy={style['shadow_y']}",
                     f"shadowcolor={q(style['shadow_color'])}"]
        if style["box"]:
            opts += ["box=1", f"boxcolor={q(style['box_color'])}",
                     f"boxborderw={style['box_padding']}"]
        filters.append("drawtext=" + ":".join(opts))
    return filters


# ---------------------------------------------------------------------------
# filtergraph
# ---------------------------------------------------------------------------

def segment_chain(index: int, seg: Segment) -> str:
    """Normalise one segment to the output canvas.

    scale with force_original_aspect_ratio=increase makes the source *cover*
    1080x1920, then crop takes the middle back out - crop-to-fill, no bars.
    """
    crop_x = f"(iw-ow)/2*(1+{CROP_BIAS_X})"
    crop_y = f"(ih-oh)/2*(1+{CROP_BIAS_Y})"
    steps = [
        f"scale={cs.CANVAS_W}:{cs.CANVAS_H}:force_original_aspect_ratio=increase",
        f"crop={cs.CANVAS_W}:{cs.CANVAS_H}:{q(crop_x)}:{q(crop_y)}",
        "setsar=1",
        # Drop the input's start time and apply the speed change in one go.
        f"setpts=(PTS-STARTPTS)/{seg.speed}",
        f"fps={FPS}",
        "format=yuv420p",
    ]
    return f"[{index}:v]" + ",".join(steps) + f"[v{index}]"


def build_command(edl: EDL, out_path: Path, text_dir: Path, font: str) -> list[str]:
    total = edl.duration
    args: list[str] = [config.FFMPEG, "-hide_banner", "-nostdin", "-y"]

    # One input per segment. Seeking before -i keeps ffmpeg from decoding the
    # whole clip, and a clip used twice simply appears as two inputs.
    for seg in edl.segments:
        args += ["-ss", f"{seg.in_:.6f}", "-t", f"{seg.source_duration:.6f}",
                 "-i", str(schema.resolve_media(seg.clip))]

    audio_index = len(edl.segments)
    args += ["-ss", f"{edl.audio.start:.6f}", "-t", f"{total:.6f}",
             "-i", str(schema.resolve_media(edl.audio.source))]

    chains = [segment_chain(i, seg) for i, seg in enumerate(edl.segments)]

    labels = "".join(f"[v{i}]" for i in range(len(edl.segments)))
    chains.append(f"{labels}concat=n={len(edl.segments)}:v=1:a=0[vcat]")

    # Captions chain one after another over the concatenated video.
    current = "vcat"
    step = 0
    for i, cap in enumerate(sorted(edl.captions, key=lambda c: c.start)):
        for filt in caption_filters(cap, font, text_dir, i):
            nxt = f"c{step}"
            chains.append(f"[{current}]{filt}[{nxt}]")
            current, step = nxt, step + 1
    chains.append(f"[{current}]null[vout]")

    # Audio is replaced entirely: only the music input reaches the output.
    asteps = ["asetpts=PTS-STARTPTS"]
    if edl.audio.fade_in > 0:
        asteps.append(f"afade=t=in:st=0:d={edl.audio.fade_in:.3f}")
    if edl.audio.fade_out > 0:
        start = max(0.0, total - edl.audio.fade_out)
        asteps.append(f"afade=t=out:st={start:.3f}:d={edl.audio.fade_out:.3f}")
    asteps.append("aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo")
    chains.append(f"[{audio_index}:a]" + ",".join(asteps) + "[aout]")

    args += ["-filter_complex", ";".join(chains)]
    args += ["-map", "[vout]", "-map", "[aout]"]
    args += VIDEO_CODEC + AUDIO_CODEC
    args += ["-r", str(FPS), "-t", f"{total:.6f}", "-movflags", "+faststart"]
    args += [str(out_path)]
    return args


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------

def render(edl_path: Path, *, out_dir: Path | None = None, dry_run: bool = False,
           verbose: bool = False) -> Path | None:
    edl = schema.load_edl(edl_path)

    # Loud, before anything is encoded.
    warnings = schema.require_valid_media(edl, source=str(edl_path))
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)

    font = cs.resolve_font()
    common.require_binary(config.FFMPEG)

    out_dir = out_dir or config.OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"{edl.variant_name}-{stamp}.mp4"

    # Caption text lives in files for the duration of the render, so no caption
    # content ever has to survive filtergraph escaping.
    with tempfile.TemporaryDirectory(prefix="clonecut-captions-") as tmp:
        text_dir = Path(tmp)
        cmd = build_command(edl, out_path, text_dir, font)

        if dry_run:
            print(" \\\n  ".join(shlex.quote(a) for a in cmd))
            return None

        print(f"rendering {edl.variant_name}: {len(edl.segments)} segments, "
              f"{len(edl.captions)} captions, {edl.duration:.2f}s")
        if verbose:
            print(" \\\n  ".join(shlex.quote(a) for a in cmd))

        started = time.monotonic()
        proc = subprocess.run(
            cmd[:1] + ["-v", "warning", "-stats"] + cmd[1:],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        elapsed = time.monotonic() - started

        if proc.returncode != 0:
            tail = "\n".join("    " + l for l in (proc.stderr or "").strip().splitlines()[-20:])
            raise ToolError(f"ffmpeg failed while rendering {edl_path}:\n{tail}")

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise ToolError(f"ffmpeg reported success but {out_path} is missing or empty.")

    sidecar = write_sidecar(out_path, edl_path, edl, cmd, elapsed, font)
    verify(out_path, edl)

    size_mb = out_path.stat().st_size / 1e6
    print(f"wrote {rel(out_path)}  ({size_mb:.1f} MB, {elapsed:.1f}s to render)")
    print(f"      {rel(sidecar)}")
    return out_path


def write_sidecar(out_path: Path, edl_path: Path, edl: EDL, cmd: list[str],
                  elapsed: float, font: str) -> Path:
    """Record what produced this file, so a render can always be traced back."""
    sidecar = out_path.with_suffix(".json")
    payload: dict[str, Any] = {
        "video": out_path.name,
        "rendered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "render_seconds": round(elapsed, 2),
        "edl_path": rel(Path(edl_path)),
        "edl": json.loads(edl.to_json()),
        "derived_features": edl.derived_features(),
        "render_settings": {
            "canvas": f"{cs.CANVAS_W}x{cs.CANVAS_H}",
            "fps": FPS,
            "crop_bias": [CROP_BIAS_X, CROP_BIAS_Y],
            "font": font,
            "video_codec": VIDEO_CODEC,
            "audio_codec": AUDIO_CODEC,
            "safe_area": {"top": cs.SAFE_TOP, "bottom": cs.SAFE_BOTTOM},
        },
        "sources": {
            ref: common.file_hash(schema.resolve_media(ref))
            for ref in sorted({s.clip for s in edl.segments} | {edl.audio.source})
        },
        "ffmpeg_command": cmd,
    }
    common.write_json(sidecar, payload)
    return sidecar


def verify(out_path: Path, edl: EDL) -> None:
    """Confirm the file we just wrote is actually what the EDL asked for."""
    info = common.ffprobe_json(out_path)
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    problems = []
    if video is None:
        problems.append("no video stream in the output")
    else:
        if (video.get("width"), video.get("height")) != (cs.CANVAS_W, cs.CANVAS_H):
            problems.append(f"expected {cs.CANVAS_W}x{cs.CANVAS_H}, got "
                            f"{video.get('width')}x{video.get('height')}")
    if audio is None:
        problems.append("no audio stream in the output")
    duration = float(info.get("format", {}).get("duration") or 0.0)
    if abs(duration - edl.duration) > 0.5:
        problems.append(f"expected about {edl.duration:.2f}s, got {duration:.2f}s")
    if problems:
        raise ToolError("the rendered file does not match the EDL:\n"
                        + "\n".join(f"  - {p}" for p in problems))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 3: render an EDL to an mp4.")
    parser.add_argument("edl", nargs="+", type=Path, help="EDL file(s) to render")
    parser.add_argument("--out-dir", type=Path, default=None, help=f"default: {rel(config.OUT_DIR)}")
    parser.add_argument("--dry-run", action="store_true", help="print the ffmpeg command and stop")
    parser.add_argument("--verbose", action="store_true", help="print the ffmpeg command too")
    args = parser.parse_args(argv)

    config.ensure_dirs()
    failed = 0
    for path in args.edl:
        try:
            render(path, out_dir=args.out_dir, dry_run=args.dry_run, verbose=args.verbose)
        except (EdlError, ToolError, cs.FontNotFound, KeyError) as exc:
            failed += 1
            print(f"\nerror: {exc}\n", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
