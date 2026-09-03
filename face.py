#!/usr/bin/env python3
"""Stage 7b - Face assets.

Two jobs, both about getting artwork into a state a compositor can use.

    uv run face.py sheet assets/rigs/doctor/reference/gemini-sheet-v2.png \\
        --out assets/rigs/doctor/mouths

`sheet` cuts a generated mouth chart into one transparent PNG per shape. The
generator is asked for a green background and one mouth per tile, but it also
draws its own captions and the odd decorative flourish, so per tile we keep only
the LARGEST connected shape and throw the rest away. Colours are then snapped to
the rig's exact palette, because a generator will hand back #333 where the rig
uses #000 and the difference shows the moment it sits next to real rig artwork.

Why generated mouths at all: none of the three Lucky Dog rigs has a usable mouth
chart. The dog has 20 drawings but they cluster at two extremes with nothing in
between; the doctor has 3 (its three view angles) and the robot has 1. Without
this step none of them can lip-sync.

`unmul` is the other half. Harmony renders artwork over white, so every edge pixel
is a blend of the black outline and the background. Composited straight onto a dark
scene that blend reads as a pale halo tracing the whole silhouette. Since we know
the background was white, the blend inverts exactly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

import common
import config
from common import ToolError

# Shape names in reading order across the generated sheet.
SHEET_ORDER = ("CLOSED", "SMALL", "OO", "EE", "AH", "OH", "WIDE", "FV")

# The Lucky Dog rig palette. Generated art gets snapped to exactly these.
RIG_PALETTE = np.array([[0, 0, 0],          # outline and mouth interior
                        [241, 241, 239],    # teeth
                        [214, 117, 136]])   # tongue

MIN_TILE_PX = 20_000        # a real tile is far bigger than any stray green speck
TILE_INSET = 8              # trim the tile border before looking for the shape
ROW_BUCKET = 200            # row grouping when sorting tiles into reading order


def _green(rgb: np.ndarray, *, loose: bool = False) -> np.ndarray:
    """Chroma-key mask. `loose` widens it for trimming inside a tile."""
    if loose:
        return (rgb[:, :, 1] > 140) & (rgb[:, :, 0] < 170) & (rgb[:, :, 2] < 170)
    return (rgb[:, :, 1] > 150) & (rgb[:, :, 0] < 150) & (rgb[:, :, 2] < 150)


def sheet_tiles(sheet: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of the green tiles, in reading order."""
    lab, n = ndimage.label(_green(sheet))
    boxes = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(ys) < MIN_TILE_PX:
            continue
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
    boxes.sort(key=lambda b: (b[1] // ROW_BUCKET, b[0]))
    return boxes


def extract_shape(tile: np.ndarray) -> Image.Image | None:
    """One tile -> a tight transparent PNG of just the mouth.

    Keeping only the largest connected component is what drops the generator's
    captions and decorations without needing to know they are there.
    """
    keep = ndimage.binary_opening(~_green(tile, loose=True), iterations=2)
    lab, k = ndimage.label(keep)
    if k == 0:
        return None
    sizes = ndimage.sum(keep, lab, range(1, k + 1))
    big = ndimage.binary_fill_holes(lab == (int(np.argmax(sizes)) + 1))
    ys, xs = np.where(big)
    if not len(ys):
        return None
    sub = tile[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    mask = big[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    dist = np.linalg.norm(sub[:, :, None, :] - RIG_PALETTE[None, None, :, :], axis=3)
    snapped = RIG_PALETTE[np.argmin(dist, axis=2)].astype(np.uint8)
    return Image.fromarray(np.dstack([snapped, (mask * 255).astype(np.uint8)]))


def split_sheet(path: Path, out_dir: Path, names=SHEET_ORDER) -> list[Path]:
    sheet = np.array(Image.open(path).convert("RGB")).astype(int)
    boxes = sheet_tiles(sheet)
    if len(boxes) != len(names):
        raise ToolError(f"found {len(boxes)} tiles in {common.rel(path)}, expected {len(names)} "
                        f"({', '.join(names)}) - is the background solid green?")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, (x0, y0, x1, y1) in zip(names, boxes):
        tile = sheet[y0 + TILE_INSET:y1 - TILE_INSET, x0 + TILE_INSET:x1 - TILE_INSET]
        img = extract_shape(tile)
        if img is None:
            raise ToolError(f"tile {name} in {common.rel(path)} contained no shape")
        dest = out_dir / f"{name}.png"
        img.save(dest)
        written.append(dest)
        print(f"  {name:7s} {img.width}x{img.height}  aspect {img.width / img.height:.2f}")
    return written


def unmul(im: Image.Image, *, thresh: int = 736, ring: int = 3) -> Image.Image:
    """Art drawn on white -> straight RGBA, with the white divided back out.

    An edge pixel is a blend of the black outline and the white page, so
    alpha = 1 - luminance/255 recovers the coverage and the colour un-premultiplies
    exactly. Only applied in a narrow ring around the true background (found by
    connectivity, so enclosed whites like teeth and shirts are left alone).
    """
    from post import bg_mask                      # connectivity-based, protects interior whites
    rgb = np.array(im.convert("RGB")).astype(np.float64)
    out = np.array(im.convert("RGBA"))
    bg = bg_mask(rgb.astype(int), thresh)
    near = ndimage.binary_dilation(bg, iterations=ring) & ~bg
    lum = rgb.mean(axis=2)
    alpha = np.ones(lum.shape)
    alpha[near] = np.clip(1.0 - lum[near] / 255.0, 0.0, 1.0)
    alpha[bg] = 0.0
    colour = rgb.copy()
    m = near & (alpha > 0.02)
    if m.any():
        a = alpha[m][:, None]
        colour[m] = np.clip((rgb[m] - 255.0 * (1.0 - a)) / a, 0, 255)
    out[..., :3] = colour.astype(np.uint8)
    out[..., 3] = (alpha * 255).astype(np.uint8)
    return Image.fromarray(out)


def cmd_sheet(a: argparse.Namespace) -> int:
    src = Path(a.sheet)
    if not src.exists():
        raise ToolError(f"no such sheet: {a.sheet}")
    out = Path(a.out) if a.out else config.ROOT / "assets" / "mouths"
    print(f"splitting {common.rel(src)} -> {common.rel(out)}")
    written = split_sheet(src, out, tuple(a.names.split(",")) if a.names else SHEET_ORDER)
    print(f"wrote {len(written)} shapes")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sheet", help="split a generated mouth chart into transparent PNGs")
    s.add_argument("sheet")
    s.add_argument("--out", help=f"default: {common.rel(config.ROOT / 'assets/mouths')}")
    s.add_argument("--names", help=f"comma-separated, in sheet order (default: {','.join(SHEET_ORDER)})")
    s.set_defaults(func=cmd_sheet)
    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
