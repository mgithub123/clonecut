#!/usr/bin/env python3
"""Sample frames from a rendered video into one image, for checking captions
and framing without scrubbing a player.

    uv run tools/contact_sheet.py out/some-video.mp4
    uv run tools/contact_sheet.py out/some-video.mp4 --at 0.5 2.0 7.0
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common  # noqa: E402
import config  # noqa: E402
import schema  # noqa: E402

# TikTok's chrome, drawn over the frames so it is obvious when a caption would
# sit underneath it.
import caption_styles as cs  # noqa: E402


def frame_times(video: Path, count: int) -> list[float]:
    info = common.ffprobe_json(video)
    duration = float(info["format"]["duration"])
    step = duration / (count + 1)
    return [round(step * (i + 1), 3) for i in range(count)]


def build(video: Path, times: list[float], out: Path, cols: int, show_safe: bool) -> Path:
    with tempfile.TemporaryDirectory(prefix="clonecut-sheet-") as tmp:
        tmpdir = Path(tmp)
        for i, t in enumerate(times):
            common.run([
                config.FFMPEG, "-v", "error", "-nostdin", "-y",
                "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
                str(tmpdir / f"f{i:03d}.png"),
            ])

        thumb_w = 300
        thumb_h = round(thumb_w * cs.CANVAS_H / cs.CANVAS_W)
        scale_f = thumb_w / cs.CANVAS_W

        chains = []
        for i, t in enumerate(times):
            steps = [f"scale={thumb_w}:{thumb_h}"]
            if show_safe:
                # The bands TikTok's own UI occupies.
                top_h = round(cs.SAFE_TOP * scale_f)
                bot_y = round((cs.CANVAS_H - cs.SAFE_BOTTOM) * scale_f)
                bot_h = thumb_h - bot_y
                steps.append(f"drawbox=0:0:{thumb_w}:{top_h}:red@0.28:t=fill")
                steps.append(f"drawbox=0:{bot_y}:{thumb_w}:{bot_h}:red@0.28:t=fill")
            steps.append(
                f"drawtext=text='{t:.2f}s':fontsize=20:fontcolor=white:"
                f"box=1:boxcolor=black@0.7:boxborderw=5:x=6:y=6"
            )
            # Bake the grid gap into each tile; xstack has no padding option.
            steps.append(f"pad={thumb_w + 10}:{thumb_h + 10}:0:0:color=#1b1b1b")
            chains.append(f"[{i}:v]" + ",".join(steps) + f"[t{i}]")

        # tile() works on one stream over time; these are separate inputs, so the
        # grid is xstack at literal offsets. Every tile is the same size, so the
        # offsets are just arithmetic.
        gap = 10
        cell_w, cell_h = thumb_w + gap, thumb_h + gap
        layout = "|".join(
            f"{(i % cols) * cell_w}_{(i // cols) * cell_h}" for i in range(len(times))
        )
        labels = "".join(f"[t{i}]" for i in range(len(times)))
        if len(times) == 1:
            chains.append("[t0]null[out]")
        else:
            chains.append(
                f"{labels}xstack=inputs={len(times)}:layout={layout}:fill=#1b1b1b[out]"
            )

        cmd = [config.FFMPEG, "-v", "error", "-nostdin", "-y"]
        for i in range(len(times)):
            cmd += ["-i", str(tmpdir / f"f{i:03d}.png")]
        cmd += ["-filter_complex", ";".join(chains), "-map", "[out]",
                "-frames:v", "1", str(out)]
        common.run(cmd)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample frames from a video into a contact sheet.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--at", nargs="+", type=float, default=None,
                        help="explicit timestamps; default is evenly spaced")
    parser.add_argument("--count", type=int, default=6, help="how many frames when --at is not given")
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--from-edl", type=Path, default=None,
                        help="sample the midpoint of every caption in this EDL")
    parser.add_argument("--no-safe-area", action="store_true",
                        help="do not shade the region TikTok's UI covers")
    args = parser.parse_args(argv)

    if not args.video.exists():
        print(f"error: {args.video} not found", file=sys.stderr)
        return 1

    if args.from_edl:
        edl = schema.load_edl(args.from_edl)
        times = sorted((c.start + c.end) / 2 for c in edl.captions)
        if not times:
            times = frame_times(args.video, args.count)
    else:
        times = args.at or frame_times(args.video, args.count)

    out = args.out or args.video.with_name(args.video.stem + "-sheet.png")
    build(args.video, times, out, args.cols, not args.no_safe_area)
    print(f"wrote {common.rel(out)}  ({len(times)} frames at "
          f"{', '.join(f'{t:.2f}s' for t in times)})")
    print("red bands mark where TikTok's own UI sits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
