#!/usr/bin/env python3
"""Stage 8g - Image generation in manual mode.

The only image model available is the Gemini app: a membership, no API credits.
So this does what plan.py does with the Claude app - writes a bundle you paste
in, and reads the result back.

    uv run gen.py prompt --rig robot --job mouths     # writes gen/<stamp>-robot-mouths/
    uv run gen.py ingest gen/<stamp>-robot-mouths downloaded.png

`prompt` writes prompt.md, the reference images to attach (the rig's head, the
current shapes if it has any, and a blank grid template at the target size so
the model lays the sheet out where face.py expects it), and meta.json recording
what was sent and the SHA-256 of every reference.

`ingest` splits the returned sheet with the face.py sheet code, un-premultiplies
cells drawn on white, and rejects a sheet whose cell count is wrong, whose cells
are empty, or whose line weight or palette is far from the rig's own art - both
measured, and the numbers printed either way. Accepted shapes go under
assets/rigs/<rig>/gen/<job>/ with a sidecar naming the bundle they came from.
Generated art is not regenerable, so it lives in assets/, in git.

The prompt text is built from the rig config, not typed: the character, the
reference, the requested cells, and the hard rules, all in one place so it can
be tuned.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

import common
import config
import face
from common import ToolError

GEN_DIR = config.ROOT / "gen"
GREEN = (40, 220, 40)           # what face._green() keys on
GUTTER = 24
MIN_PALETTE_FRACTION = 0.70     # of opaque pixels within PALETTE_TOL of a rig colour
PALETTE_TOL = 48
STROKE_RATIO = (0.5, 2.0)       # accepted line weight relative to the source plate
DARK = 90                       # below this a pixel is outline

# Each job: the cells in reading order, what to draw in each, the cell size in
# plate pixels at the rig's render resolution, and the columns of the grid.
JOBS = {
    "mouths": {
        "cells": {
            "CLOSED": "mouth closed, lips together, at rest",
            "SMALL": "mouth slightly open, a narrow gap",
            "OO": "lips pushed forward in a small round O",
            "FV": "upper teeth resting on the lower lip",
            "EE": "wide flat smile showing the teeth, as in 'see'",
            "OH": "mouth open in a tall oval, as in 'go'",
            "AH": "mouth open wide, jaw dropped, as in 'father'",
            "WIDE": "mouth open as wide as it goes, singing a held note",
        },
        "cell": (3.0, 2.4), "cols": 4,
        "what": "the mouth only, on its own, no face around it",
    },
    "brows": {
        "cells": {
            "RAISED": "both eyebrows raised high, surprised",
            "NEUTRAL": "eyebrows at rest",
            "FURROWED": "eyebrows pulled down and together, angry or intent",
            "ASYMMETRIC": "one eyebrow raised, the other level, sceptical",
        },
        "cell": (3.2, 1.6), "cols": 2,
        "what": "the pair of eyebrows only, both in one cell, no eyes or face",
    },
    "eyes": {
        "cells": {
            "OPEN": "both eyes fully open",
            "HALF": "both eyes half closed, lids halfway down",
            "SHUT": "both eyes shut, a closed line each",
            "PUPILS": "both eyes open with a clear dark pupil in each, looking straight ahead",
        },
        "cell": (3.2, 1.8), "cols": 2,
        "what": "the pair of eyes only, both in one cell, no brows or face",
    },
    "expressions": {
        "cells": {
            "HAPPY": "a happy face, eyes bright, big smile",
            "SAD": "a sad face, brows up in the middle, mouth turned down",
            "ANGRY": "an angry face, brows down, teeth bared",
            "SURPRISED": "a surprised face, eyes wide, mouth a round O",
        },
        "cell": (6.0, 6.0), "cols": 2,
        "what": "the whole head, front view, same size and framing as the reference",
    },
}

RULES = [
    "Flat colour only. No shading, no gradients, no highlights, no texture.",
    "The same black outline weight as the reference, everywhere.",
    "Front view, straight on, symmetric, level.",
    "Use only the colours in the reference. No new colours.",
    "Exactly one item per green cell, centred in it, nothing touching the cell edge.",
    "Nothing outside the grid. No captions, labels, numbers, arrows or decoration.",
    "Keep the green cell backgrounds exactly as they are; draw on top of them.",
    "Return the whole sheet as one image at the size it was given.",
]


# ------------------------------------------------------------------ rig context

def load_rig(name: str) -> dict:
    p = config.ROOT / "assets" / "rigs" / name / "rig.json"
    if not p.exists():
        raise ToolError(f"no rig {name!r}")
    r = json.loads(p.read_text())
    r["_dir"] = p.parent
    return r


def head_reference(r: dict) -> Image.Image:
    """The rig's head, front-on, from the tracked head plate at its crop box."""
    res = r["render"]["resolution"][0]
    p = r["_dir"] / "plates" / f"plate-head-{res}.png"
    if not p.exists():
        raise ToolError(f"rig {r['name']!r} has no tracked head plate ({common.rel(p)})")
    im = Image.open(p).convert("RGBA")
    if np.asarray(im)[..., 3].min() >= 250:
        im = face.unmul(im)
    crop = r["crop"]
    im = im.crop(tuple(crop))
    bb = im.getbbox()
    if bb:
        im = im.crop(bb)
    canvas = Image.new("RGB", im.size, (255, 255, 255))
    canvas.paste(im, (0, 0), im)
    return canvas


def palette_of(r: dict) -> list[tuple[int, int, int]]:
    """The colours the rig actually uses at its face, from the head reference."""
    a = np.asarray(head_reference(r)).reshape(-1, 3)
    q = (a // 16) * 16 + 8
    vals, counts = np.unique(q, axis=0, return_counts=True)
    keep = [tuple(int(x) for x in v) for v, c in zip(vals, counts) if c >= 0.005 * len(a)]
    return keep


def current_shapes(r: dict, job: str) -> Image.Image | None:
    """A contact of the shapes the rig already has for this job, if any."""
    d = r["_dir"] / ("mouths" if job == "mouths" else f"gen/{job}")
    files = sorted(d.glob("*.png")) if d.exists() else []
    if not files:
        return None
    ims = [Image.open(f).convert("RGBA") for f in files]
    w = sum(i.width for i in ims) + 20 * (len(ims) + 1)
    h = max(i.height for i in ims) + 40
    sheet = Image.new("RGB", (w, h), (255, 255, 255))
    x = 20
    for f, im in zip(files, ims):
        sheet.paste(im, (x, 20), im)
        x += im.width + 20
    return sheet


def template(r: dict, job: str) -> tuple[Image.Image, list[tuple[int, int, int, int]]]:
    """A blank grid of green cells at the target size, in reading order."""
    spec = JOBS[job]
    fw = int(r["face"]["width"])
    cw, ch = int(fw * spec["cell"][0]), int(fw * spec["cell"][1])
    cw, ch = max(cw, 200), max(ch, 160)
    cols = spec["cols"]
    n = len(spec["cells"])
    rows = (n + cols - 1) // cols
    W = cols * cw + (cols + 1) * GUTTER
    H = rows * ch + (rows + 1) * GUTTER
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    boxes = []
    for k in range(n):
        cx, cy = k % cols, k // cols
        x0 = GUTTER + cx * (cw + GUTTER)
        y0 = GUTTER + cy * (ch + GUTTER)
        d.rectangle([x0, y0, x0 + cw - 1, y0 + ch - 1], fill=GREEN)
        boxes.append((x0, y0, x0 + cw - 1, y0 + ch - 1))
    return im, boxes


# ------------------------------------------------------------------ prompt

def prompt_text(r: dict, job: str, refs: list[str]) -> str:
    spec = JOBS[job]
    cells = list(spec["cells"].items())
    lines = [
        f"# {r['name']}: {job} sheet",
        "",
        f"You are drawing new parts for an existing cartoon character, {r['name']!r}, "
        f"in exactly its own style. The first attached image is the character's head, "
        f"front view. Match it precisely: the same line weight, the same flat colours, "
        f"the same proportions.",
    ]
    if r.get("eye_note"):
        lines.append(f"About the eyes: {r['eye_note']}")
    if r.get("mouth_source", {}).get("note"):
        lines.append(f"About the mouth: {r['mouth_source']['note']}")
    if len(refs) > 2:
        lines.append(f"The second attached image shows the shapes the character already has "
                     f"for this job; keep the style, do not copy the shapes.")
    lines += [
        "",
        f"The last attached image is a grid template of {len(cells)} green cells. Draw "
        f"{spec['what']} - one per cell, in this order, reading left to right and top to bottom:",
        "",
    ]
    for i, (name, what) in enumerate(cells, 1):
        lines.append(f"{i}. **{name}** - {what}")
    lines += ["", "Rules:", ""]
    lines += [f"- {rule}" for rule in RULES]
    lines += [
        "",
        "Attach, in this order: " + ", ".join(refs) + ".",
        "Download the returned image as PNG and run:",
        "",
        "    uv run gen.py ingest <this folder> <downloaded.png>",
    ]
    return "\n".join(lines) + "\n"


def write_bundle(name: str, job: str, out_root: Path = GEN_DIR) -> Path:
    if job not in JOBS:
        raise ToolError(f"no job {job!r}; have {', '.join(JOBS)}")
    r = load_rig(name)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = out_root / f"{stamp}-{name}-{job}"
    out.mkdir(parents=True, exist_ok=True)
    refs = []
    head = head_reference(r)
    head.save(out / "ref-head.png")
    refs.append("ref-head.png")
    cur = current_shapes(r, job)
    if cur is not None:
        cur.save(out / "ref-current.png")
        refs.append("ref-current.png")
    tmpl, boxes = template(r, job)
    tmpl.save(out / "template.png")
    refs.append("template.png")
    text = prompt_text(r, job, refs)
    (out / "prompt.md").write_text(text)
    common.write_json(out / "meta.json", {
        "rig": name, "job": job, "stamp": stamp, "cells": list(JOBS[job]["cells"]),
        "template_size": list(tmpl.size), "template_boxes": boxes,
        "references": {f: common.file_hash(out / f) for f in refs},
        "prompt_sha256": common.file_hash(out / "prompt.md"),
        "palette": palette_of(r),
    })
    return out


# ------------------------------------------------------------------ ingest

def stroke_width(im: Image.Image) -> float:
    """Median outline thickness in pixels: twice the median distance from an
    outline pixel to the nearest non-outline pixel, over the outline's spine."""
    a = np.asarray(im.convert("RGBA"))
    dark = (a[..., :3].max(axis=2) < DARK) & (a[..., 3] > 128)
    if dark.sum() < 50:
        return 0.0
    dt = ndimage.distance_transform_edt(dark)
    # the spine: outline pixels at a local maximum of distance. A filled dark
    # region - a mouth cavity - has a short, deep spine; the outline's is long
    # and shallow, so the lower quartile is the line weight.
    mx = ndimage.maximum_filter(dt, size=3)
    spine = dark & (dt >= mx - 0.01) & (dt > 0.5)
    if spine.sum() < 10:
        return float(2 * np.percentile(dt[dark], 25))
    return float(2 * np.percentile(dt[spine], 25))


def palette_fraction(im: Image.Image, palette: list) -> float:
    a = np.asarray(im.convert("RGBA"))
    op = a[a[..., 3] > 128][:, :3].astype(int)
    if not len(op):
        return 0.0
    pal = np.array(palette, dtype=int)
    d = np.abs(op[:, None, :] - pal[None, :, :]).max(axis=2).min(axis=1)
    return float((d <= PALETTE_TOL).mean())


def ingest(folder: Path, sheet: Path) -> list[Path]:
    meta_p = folder / "meta.json"
    if not meta_p.exists():
        raise ToolError(f"{common.rel(folder)} is not a gen bundle (no meta.json)")
    meta = json.loads(meta_p.read_text())
    if not sheet.exists():
        raise ToolError(f"no such sheet: {sheet}")
    r = load_rig(meta["rig"])
    names = tuple(meta["cells"])
    work = folder / "ingest"
    work.mkdir(exist_ok=True)
    print(f"splitting {common.rel(sheet)}: expecting {len(names)} cells")
    palette = [tuple(p) for p in meta["palette"]]
    # raises on a wrong count or an empty cell; colours snap to this rig's own palette
    written = face.split_sheet(sheet, work, names, palette=np.array(palette))

    ref = head_reference(r)
    ref_stroke = stroke_width(ref)
    problems = []
    for p in written:
        im = Image.open(p).convert("RGBA")
        a = np.asarray(im)
        # a cell drawn on white instead of the green: a halo of white edge pixels
        edge = a[..., 3] > 128
        white = (a[..., :3].min(axis=2) > 235) & edge
        if white.sum() > 0.5 * edge.sum():
            im = face.unmul(im)
            im.save(p)
        sw = stroke_width(im)
        pf = palette_fraction(im, palette)
        ratio = sw / ref_stroke if ref_stroke else 0
        ok = STROKE_RATIO[0] <= ratio <= STROKE_RATIO[1] and pf >= MIN_PALETTE_FRACTION
        print(f"  {p.stem:10s} {im.width}x{im.height}  stroke {sw:.1f}px vs {ref_stroke:.1f} "
              f"(x{ratio:.2f})  palette {pf:.0%}  {'ok' if ok else 'REJECT'}")
        if not ok:
            problems.append(p.stem)
    if problems:
        raise ToolError(f"sheet rejected: {', '.join(problems)} are off the rig's line weight or palette "
                        f"(accepted stroke x{STROKE_RATIO[0]}-{STROKE_RATIO[1]}, palette >= "
                        f"{MIN_PALETTE_FRACTION:.0%}). The cells are in {common.rel(work)} to look at.")
    dest = r["_dir"] / "gen" / meta["job"]
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for p in written:
        target = dest / p.name
        Image.open(p).save(target)
        common.write_json(target.with_suffix(".json"), {
            "bundle": folder.name, "sheet_sha256": common.file_hash(sheet),
            "prompt_sha256": meta["prompt_sha256"], "cell": p.stem, "job": meta["job"],
        })
        out.append(target)
    print(f"accepted {len(out)} shapes -> {common.rel(dest)}")
    return out


# ------------------------------------------------------------------ CLI

def cmd_prompt(a) -> int:
    out = write_bundle(a.rig, a.job)
    print(f"wrote {common.rel(out)}")
    print(f"  paste prompt.md into the Gemini app and attach the images it lists,")
    print(f"  then: uv run gen.py ingest {common.rel(out)} <downloaded.png>")
    return 0


def cmd_ingest(a) -> int:
    ingest(Path(a.folder), Path(a.sheet))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("prompt", help="write a bundle to paste into the Gemini app")
    s.add_argument("--rig", required=True)
    s.add_argument("--job", required=True, choices=sorted(JOBS))
    s.set_defaults(func=cmd_prompt)
    i = sub.add_parser("ingest", help="split and check a returned sheet")
    i.add_argument("folder")
    i.add_argument("sheet")
    i.set_defaults(func=cmd_ingest)
    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
