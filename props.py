"""Stage 8p - Props a character holds.

    uv run props.py draw guitar --out cache/props/guitar.png     # look at it

A prop is an image with a place on the rig, recorded in rig.json["props"]:

    "props": {"guitar": {"at": [x, y], "width": w, "angle": deg, "strum": [x, y],
                         "note": "..."}}

`at` is the top-left of the prop on the render canvas before rotation, `width`
its drawn width, `angle` a rotation about its centre (degrees, positive is
anticlockwise), and `strum` the point on the strings the playing hand moves
about. Everything is in render-resolution pixels, the same space as the face
and the hands.

The guitar here is drawn in code, in the rig's flat style - black outline,
flat fills, no shading - as a stand-in. The real one comes from the Gemini app
through `gen.py prompt --job props`, and replaces this file the moment it is
accepted under assets/rigs/<rig>/gen/props/GUITAR.png.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

import common
import config
from common import ToolError

OUTLINE = (0, 0, 0, 255)
BODY = (150, 96, 60, 255)         # a warm brown, from nothing in the rig; the paste will fix it
BODY_DARK = (118, 72, 44, 255)
NECK = (92, 62, 42, 255)
FRET = (214, 200, 170, 255)
HOLE = (28, 20, 16, 255)
STRING = (222, 222, 222, 255)
PICKGUARD = (40, 40, 44, 255)
STROKE = 7                        # px at 3840 render width, matches the rig's line weight
SS = 3                            # supersample for clean curves


def guitar(width: int) -> Image.Image:
    """An acoustic guitar lying horizontally, neck to the left, transparent around."""
    W = width * SS
    H = int(width * 0.42) * SS
    s = STROKE * SS
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    # body: two lobes on the right 45% of the width
    bx0 = int(W * 0.52)
    cy = H // 2
    lower = [bx0 + int(W * 0.20), cy - int(H * 0.46), W - s, cy + int(H * 0.46)]
    upper = [bx0, cy - int(H * 0.36), bx0 + int(W * 0.30), cy + int(H * 0.36)]
    for box in (lower, upper):
        d.ellipse(box, fill=BODY, outline=OUTLINE, width=s)
    # waist: paint over the join, then redraw outlines where they show
    d.rectangle([lower[0] + s * 2, cy - int(H * 0.30), upper[2] - s * 2, cy + int(H * 0.30)], fill=BODY)
    # neck
    nx1 = bx0 + s
    ny0, ny1 = cy - int(H * 0.09), cy + int(H * 0.09)
    d.rectangle([int(W * 0.10), ny0, nx1, ny1], fill=NECK, outline=OUTLINE, width=s)
    for k in range(9):
        x = int(W * 0.14) + k * int(W * 0.04)
        d.line([(x, ny0 + s), (x, ny1 - s)], fill=FRET, width=max(SS, s // 2))
    # headstock
    d.rounded_rectangle([s, cy - int(H * 0.14), int(W * 0.11), cy + int(H * 0.14)],
                        radius=int(H * 0.06), fill=NECK, outline=OUTLINE, width=s)
    # sound hole and pickguard
    hx = bx0 + int(W * 0.22)
    d.ellipse([hx - int(H * 0.13), cy - int(H * 0.13), hx + int(H * 0.13), cy + int(H * 0.13)],
              fill=HOLE, outline=OUTLINE, width=s)
    d.ellipse([hx + int(H * 0.02), cy + int(H * 0.06), hx + int(H * 0.30), cy + int(H * 0.36)], fill=PICKGUARD)
    # bridge
    brx = lower[0] + int((lower[2] - lower[0]) * 0.72)
    d.rounded_rectangle([brx - int(W * 0.02), cy - int(H * 0.12), brx + int(W * 0.02), cy + int(H * 0.12)],
                        radius=s, fill=BODY_DARK, outline=OUTLINE, width=s)
    # strings
    for k in range(6):
        y = cy - int(H * 0.07) + k * int(H * 0.028)
        d.line([(int(W * 0.03), y), (brx, y)], fill=STRING, width=max(SS, s // 3))
    return im.resize((W // SS, H // SS), Image.LANCZOS)


def place(prop: dict) -> tuple[Image.Image, tuple[int, int]]:
    """The prop image rotated as recorded, and where to paste it (render px)."""
    kind = prop.get("kind", "guitar")
    src = prop.get("file")
    im = Image.open(config.ROOT / src).convert("RGBA") if src else guitar(int(prop["width"]))
    if im.width != int(prop["width"]):
        im = im.resize((int(prop["width"]), int(im.height * prop["width"] / im.width)), Image.LANCZOS)
    angle = float(prop.get("angle", 0))
    rot = im.rotate(angle, resample=Image.BICUBIC, expand=True)
    x, y = prop["at"]
    # keep the un-rotated top-left where it was recorded: offset by the growth
    return rot, (int(x - (rot.width - im.width) / 2), int(y - (rot.height - im.height) / 2))


def cmd_draw(a) -> int:
    im = guitar(a.width)
    out = Path(a.out) if a.out else config.CACHE_DIR / "props" / f"{a.prop}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)
    print(f"wrote {common.rel(out)} {im.width}x{im.height}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    dr = sub.add_parser("draw", help="draw a stand-in prop")
    dr.add_argument("prop", choices=("guitar",))
    dr.add_argument("--width", type=int, default=900)
    dr.add_argument("--out", default=None)
    dr.set_defaults(func=cmd_draw)
    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
