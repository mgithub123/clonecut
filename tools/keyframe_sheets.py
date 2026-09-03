#!/usr/bin/env python3
"""Pack a prompt's keyframes into a few labelled grids.

Some chat apps cap how many images you can attach at once - Gemini allows ten -
which is fewer than the keyframes a multi-clip prompt generates. Every tile
keeps the keyframe's own label, so the plan can still refer to a clip and a
timestamp exactly as it would with the images attached one by one.

    uv run tools/keyframe_sheets.py plans/<stamp>-keyframes
    uv run tools/keyframe_sheets.py plans/<stamp>-keyframes --max-sheets 4
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import caption_styles as cs  # noqa: E402
import config  # noqa: E402
from common import ToolError, rel  # noqa: E402

TILE_W = 480          # per tile; 3 across is still legible when downscaled
LABEL_H = 34


def esc(text: str) -> str:
    """Escape for ffmpeg drawtext."""
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’")


def label_tile(src: Path, dest: Path, text: str, font: str) -> None:
    """One keyframe, scaled, with its label burned into a bar underneath."""
    vf = (
        f"scale={TILE_W}:-2,"
        f"pad=iw:ih+{LABEL_H}:0:0:color=black,"
        f"drawtext=fontfile='{font}':text='{esc(text)}':"
        f"x=6:y=h-{LABEL_H - 8}:fontsize=19:fontcolor=white:expansion=none"
    )
    subprocess.run(
        [config.FFMPEG, "-v", "error", "-i", str(src), "-vf", vf, "-frames:v", "1", "-y", str(dest)],
        check=True,
    )


def build_sheet(tiles: list[Path], dest: Path, cols: int) -> None:
    rows = math.ceil(len(tiles) / cols)
    cmd = [config.FFMPEG, "-v", "error"]
    for t in tiles:
        cmd += ["-i", str(t)]

    parts, row_labels = [], []
    for r in range(rows):
        row = [i for i in range(r * cols, min((r + 1) * cols, len(tiles)))]
        streams = "".join(f"[{i}:v]" for i in row)
        if len(row) == 1:
            parts.append(f"{streams}copy[r{r}]")
        else:
            parts.append(f"{streams}hstack=inputs={len(row)}[r{r}]")
        row_labels.append(f"[r{r}]")

    if rows == 1:
        parts.append(f"{row_labels[0]}copy[out]")
    else:
        # short final rows would break vstack, so pad them to full width first
        last = len(tiles) - (rows - 1) * cols
        if last != cols:
            parts[-1] = parts[-1].replace(f"[r{rows - 1}]", "[rlast]")
            parts.append(f"[rlast]pad={TILE_W * cols}:ih:0:0:color=black[r{rows - 1}]")
        parts.append("".join(row_labels) + f"vstack=inputs={rows}[out]")

    cmd += ["-filter_complex", ";".join(parts), "-map", "[out]", "-frames:v", "1", "-y", str(dest)]
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pack prompt keyframes into a few labelled grids.")
    p.add_argument("keyframe_dir", type=Path)
    p.add_argument("--max-sheets", type=int, default=4,
                   help="how many images to end up with (default: 4)")
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args(argv)

    frames = sorted(args.keyframe_dir.glob("*.jpg")) + sorted(args.keyframe_dir.glob("*.png"))
    if not frames:
        print(f"error: no keyframes in {args.keyframe_dir}", file=sys.stderr)
        return 1

    try:
        font = cs.resolve_font()
    except cs.FontNotFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out_dir = args.out_dir or args.keyframe_dir.parent
    stem = args.keyframe_dir.name.replace("-keyframes", "")
    per_sheet = math.ceil(len(frames) / args.max_sheets)

    written = []
    with tempfile.TemporaryDirectory() as tmp:
        tiles = []
        for f in frames:
            dest = Path(tmp) / f"{f.stem}.png"
            label_tile(f, dest, f.stem, font)
            tiles.append(dest)

        for n, start in enumerate(range(0, len(tiles), per_sheet), 1):
            chunk = tiles[start:start + per_sheet]
            dest = out_dir / f"{stem}-sheet{n}.jpg"
            build_sheet(chunk, dest, args.cols)
            written.append((dest, len(chunk)))

    print(f"{len(frames)} keyframes -> {len(written)} sheet(s)")
    for dest, count in written:
        print(f"  {rel(dest)}  ({count} frames, {dest.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
