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
from PIL import Image, ImageDraw
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


def derive(src_dir: Path, out_dir: Path, spec: dict[str, tuple[int, float, float]]) -> list[Path]:
    """Build a vowel set by scaling a few base drawings non-uniformly.

    For a rig whose mouth chart is mostly drawn at an angle, only the head-on
    drawings are usable, and there are rarely more than three. Scaling a symmetric
    mouth keeps it symmetric, so a narrow tall version of the open mouth reads as
    OO and a wide flat one reads as EE - without any new artwork.

    Not a substitute for drawn vowels: the lips do not change shape, only the
    proportions of the opening. It is the difference between a mouth that moves
    and a mouth that appears to form words.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, (src, sx, sy) in spec.items():
        p = Path(src_dir) / f"D{src}.png"
        if not p.exists():
            raise ToolError(f"no source drawing {src} at {common.rel(p)} - extract with rig.py mouths")
        im = Image.open(p).convert("RGBA")
        w, h = max(1, int(im.width * sx)), max(1, int(im.height * sy))
        dest = Path(out_dir) / f"{name}.png"
        im.resize((w, h), Image.LANCZOS).save(dest)
        print(f"  {name:7s} <- drawing {src:<3} {im.width}x{im.height} -> {w}x{h}")
        written.append(dest)
    return written


def symmetry(png: Path) -> float:
    """How head-on a mouth is drawn: 1.0 is perfectly symmetric.

    The deciding measurement when picking base shapes - a mouth drawn with a lean
    reads as the head being turned, and cutting between several leaning mouths
    makes it appear to swing side to side.
    """
    a = np.array(Image.open(png).convert("RGBA"))
    al = a[..., 3] > 128
    rgb = a[..., :3].astype(int)
    opening = ((rgb.sum(axis=2) < 200) |
               ((rgb[:, :, 0] > 150) & (rgb[:, :, 1] < 150) & (rgb[:, :, 2] > 100))) & al
    ys, xs = np.where(opening)
    if not len(xs):
        return 0.0
    cx = (xs.min() + xs.max()) / 2
    left, right = int((xs < cx).sum()), int((xs >= cx).sum())
    return round(min(left, right) / max(left, right), 3) if max(left, right) else 0.0


def cmd_derive(a: argparse.Namespace) -> int:
    spec = {}
    for part in a.spec.split(","):
        name, rest = part.split("=")
        src, sx, sy = rest.split(":")
        spec[name.strip()] = (int(src), float(sx), float(sy))
    made = derive(Path(a.src), Path(a.out), spec)
    print(f"wrote {len(made)} shapes to {common.rel(Path(a.out))}")
    return 0


def cmd_symmetry(a: argparse.Namespace) -> int:
    rows = [(symmetry(p), p.stem) for p in sorted(Path(a.dir).glob("*.png"))]
    print(f"{'symm':>6}  drawing      (1.0 = head-on; below ~0.9 the mouth leans)")
    for v, n in sorted(rows, reverse=True):
        print(f"{v:>6.2f}  {n}")
    return 0


BLACK=(0,0,0,255); TEETH=(241,241,239,255); TONGUE=(214,117,136,255); SS=4

def _lens(d, box, pinch, fill=255):
    """Opening outline: an ellipse blended toward a lens as `pinch` rises."""
    x0,y0,x1,y1 = box; w,h = x1-x0, y1-y0
    cy=(y0+y1)/2
    pts_top, pts_bot = [], []
    for i in range(65):
        t=i/64
        x=x0+w*t
        # ellipse half-height at t, flattened toward straight lines at the corners
        e=(1-(2*t-1)**2)**0.5
        k=e**(1+2*pinch)
        pts_top.append((x, cy-h/2*k)); pts_bot.append((x, cy+h/2*k))
    d.polygon(pts_top+pts_bot[::-1], fill=fill)

def mouth(w,h,*,pinch=0.35,teeth_top=True,teeth_bottom=False,tongue=True,
          closed=False,stroke=None):
    stroke=(stroke or max(3,int(h*0.13)))*SS
    W,H=w*SS,h*SS; pad=stroke*2
    canvas=(W+pad*2, H+pad*2)
    if closed:                                  # a smile line, not an opening
        im=Image.new("RGBA",canvas,(0,0,0,0)); d=ImageDraw.Draw(im)
        d.arc([pad, pad-H, pad+W, pad+H], 20, 160, fill=BLACK, width=stroke)
        return im.resize((im.width//SS,im.height//SS), Image.LANCZOS)

    shape=Image.new("L",canvas,0); ds=ImageDraw.Draw(shape)
    _lens(ds,[pad,pad,pad+W,pad+H],pinch)
    inner=np.array(shape)>128
    ys,xs=np.where(inner)
    ix0,ix1,iy0,iy1=xs.min(),xs.max(),ys.min(),ys.max()
    ih,iw=iy1-iy0,ix1-ix0

    fill=Image.new("RGBA",canvas,(0,0,0,0)); df=ImageDraw.Draw(fill)
    df.bitmap((0,0),shape,fill=BLACK)
    if tongue and ih>16*SS:
        tw,th=int(iw*0.66),int(ih*0.52)
        cx=(ix0+ix1)//2
        df.ellipse([cx-tw//2, iy1-th, cx+tw//2, iy1+th//3], fill=TONGUE)
    def teeth(y0,y1,n):
        span=int(iw*0.60); x0=(ix0+ix1)//2-span//2
        gap=max(2*SS,span//34); tw=(span-gap*(n-1))//n
        for k in range(n):
            x=x0+k*(tw+gap)
            df.rounded_rectangle([x,y0,x+tw,y1],radius=max(2,tw//7),fill=TEETH)
    if teeth_top and ih>12*SS: teeth(iy0, iy0+int(ih*0.36), 4)
    if teeth_bottom and ih>26*SS: teeth(iy1-int(ih*0.30), iy1, 4)

    # clip every interior fill to the opening, then stroke the outline on top
    from scipy import ndimage
    keep=ndimage.binary_erosion(inner, iterations=max(1,stroke//2))
    a=np.array(fill); a[...,3]=np.where(keep, a[...,3], 0)
    out=Image.fromarray(a)
    ring=ImageDraw.Draw(out)
    edge=Image.new("L",canvas,0)
    _lens(ImageDraw.Draw(edge),[pad,pad,pad+W,pad+H],pinch)
    em=np.array(edge)>128
    outline=em & ~ndimage.binary_erosion(em, iterations=stroke)
    b=np.array(out); b[outline]=BLACK
    return Image.fromarray(b).resize(((W+pad*2)//SS,(H+pad*2)//SS), Image.LANCZOS)


DRAWN_SPEC = {
    "CLOSED": dict(w=168, h=26, pinch=0.30, teeth_top=False, teeth_bottom=False, tongue=False, closed=True),
    "SMALL":  dict(w=150, h=40, pinch=0.45, teeth_top=True,  teeth_bottom=False, tongue=True),
    "OO":     dict(w= 86, h=76, pinch=0.05, teeth_top=False, teeth_bottom=False, tongue=True),
    "FV":     dict(w=162, h=44, pinch=0.55, teeth_top=True,  teeth_bottom=False, tongue=False),
    "EE":     dict(w=172, h=54, pinch=0.65, teeth_top=True,  teeth_bottom=True,  tongue=False),
    "OH":     dict(w=118, h=94, pinch=0.10, teeth_top=True,  teeth_bottom=False, tongue=True),
    "AH":     dict(w=160, h=92, pinch=0.30, teeth_top=True,  teeth_bottom=False, tongue=True),
    "WIDE":   dict(w=176, h=112, pinch=0.40, teeth_top=True, teeth_bottom=True,  tongue=True),
}


def cmd_draw(a: argparse.Namespace) -> int:
    """Draw a vowel set from scratch. For a rig with nothing to derive from."""
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, kw in DRAWN_SPEC.items():
        im = mouth(**kw)
        bb = im.getbbox()
        if bb:
            im = im.crop(bb)
        im.save(out / f"{name}.png")
        print(f"  {name:7s} {im.width}x{im.height}")
    print(f"wrote {len(DRAWN_SPEC)} shapes to {common.rel(out)}")
    return 0


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

    d = sub.add_parser("derive", help="build a vowel set by scaling a few base drawings")
    d.add_argument("src", help="directory of extracted drawings (D1.png, D11.png, ...)")
    d.add_argument("--out", required=True)
    d.add_argument("--spec", required=True,
                   help="NAME=drawing:xscale:yscale, comma separated")
    d.set_defaults(func=cmd_derive)

    w = sub.add_parser("draw", help="draw a vowel set from scratch (no source art needed)")
    w.add_argument("--out", required=True)
    w.set_defaults(func=cmd_draw)

    y = sub.add_parser("symmetry", help="rank drawings by how head-on they are")
    y.add_argument("dir")
    y.set_defaults(func=cmd_symmetry)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
