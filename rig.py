#!/usr/bin/env python3
"""Stage 7c - Rig preparation.

Harmony scene files are XML, and Harmony ships a headless renderer. Together that
means a character rig can be driven entirely from a script: no GUI, no clicking,
216 frames at 4K in under twenty seconds.

    uv run rig.py info   ~/Desktop/LuckyDog-Harmony/LD_DOCTOR_RIG_MA
    uv run rig.py plate  ~/Desktop/LuckyDog-Harmony/LD_DOCTOR_RIG_MA --out /tmp/head \\
        --keep 16,18,19,21,22,23,24,26,27,28,30 --resolution 3840x2160
    uv run rig.py measure ~/Desktop/LuckyDog-Harmony/LD_DOCTOR_RIG_MA

Every rig in this project is a *turnaround* - a model sheet where the character
rotates through view angles over the first few frames and then holds. That is the
wrong starting point for a performance, so `plate` freezes the pegs to frame 1
before rendering. It also fills exposure gaps: these rigs leave holes where a
layer is simply not exposed on some frames, which renders as body parts blinking
out of existence. Neither problem is visible until you render, and both look like
corruption rather than the authoring choices they are.

`plate` renders a chosen subset of elements against white, so a head can be
composited over a body and moved independently.

Nothing here writes to the rig you point it at - the scene is copied first.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np

import common
import config
from common import ToolError

HARMONY = ("/Applications/Toon Boom Harmony 27 Premium/Harmony 27 Premium.app"
           "/Contents/MacOS/Harmony Premium")

_COL0 = r'<column type="0"[^>]*>(?:(?!</column>).)*?</column>'
_SEQ = r'<elementSeq exposures="([^"]*)" val="([^"]*)" id="(\d+)"'


def find_scene(rig_dir: Path, name: str | None = None) -> Path:
    """Pick the scene deterministically: an explicit name, else the one matching
    the folder, else the shortest stem. Rigs ship several variants side by side
    and glob order is not intent."""
    rig_dir = Path(rig_dir)
    scenes = [p for p in rig_dir.glob("*.xstage")
              if "BACKUP" not in p.name and not p.name.endswith("~")]
    if not scenes:
        raise ToolError(f"no .xstage in {common.rel(rig_dir)}")
    if name:
        for p in scenes:
            if p.stem == name or p.name == name:
                return p
        raise ToolError(f"no scene {name!r} in {common.rel(rig_dir)}; "
                        f"have {', '.join(sorted(p.stem for p in scenes))}")
    for p in scenes:
        if p.stem == rig_dir.name:
            return p
    return sorted(scenes, key=lambda p: (len(p.stem), p.stem))[0]


def frames_of(spec: str) -> set[int]:
    """Expand a Harmony exposure spec like '1-5,7,10-12' into frame numbers."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-") and not part.startswith("-"):
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


class Scene:
    """One .xstage, held as text. Every edit is a regex rewrite of that text."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise ToolError(f"no scene at {self.path}")
        self.xml = self.path.read_text(encoding="utf-8", errors="replace")
        table = self.path.parent / "scene.elementTable"
        self.elements = dict(re.findall(r'<element id="(\d+)" elementName="([^"]*)"',
                                        table.read_text())) if table.exists() else {}

    # ---- inspection -------------------------------------------------------
    @property
    def resolution(self) -> tuple[int, int]:
        m = re.search(r'<resolution [^>]*size="(\d+),(\d+)"', self.xml)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    @property
    def length(self) -> int:
        m = re.search(r'nbframes="(\d+)"', self.xml)
        return int(m.group(1)) if m else 0

    def animated_pegs(self) -> int:
        return sum(1 for m in re.finditer(r"<path3D[^>]*>.*?</path3D>", self.xml, re.S)
                   if len(re.findall(r"<pt val=", m.group(0))) > 1)

    def drawing_counts(self) -> dict[str, int]:
        """How many .tvg files each element actually has on disk."""
        out = {}
        edir = self.path.parent / "elements"
        for eid, name in self.elements.items():
            d = edir / name
            out[name] = len(list(d.glob("*.tvg"))) if d.is_dir() else 0
        return out

    def exposure_gaps(self, nframes: int) -> dict[str, list[int]]:
        """Frames where a layer that otherwise runs to the end is missing."""
        gaps = {}
        for col in re.findall(_COL0, self.xml, re.S):
            seqs = re.findall(_SEQ, col)
            if not seqs:
                continue
            covered: set[int] = set()
            for exp, _v, _i in seqs:
                covered |= frames_of(exp)
            if not covered or max(covered) != nframes:
                continue                      # guides / unused layers: leave alone
            missing = [f for f in range(1, nframes + 1) if f not in covered]
            if missing:
                gaps[self.elements.get(seqs[0][2], f"id{seqs[0][2]}")] = missing
        return gaps

    # ---- edits ------------------------------------------------------------
    def freeze_pegs(self) -> tuple[int, int]:
        """Collapse all peg animation to its frame-1 value. Kills the turnaround."""
        n_path = n_curve = 0

        def fix_path(m):
            nonlocal n_path
            block = m.group(0)
            pts = re.findall(r'<pt val="[^"]*" lockedInTime="[0-9.]+"/>', block)
            if len(pts) <= 1:
                return block
            keep = next((q for q in pts if re.search(r'lockedInTime="1"', q)), pts[0])
            n_path += 1
            inner = re.search(r"(<points>)(.*?)(</points>)", block, re.S)
            return block[:inner.start(2)] + "\n       " + keep + "\n      " + block[inner.end(2):]

        self.xml = re.sub(r"<path3D[^>]*>.*?</path3D>", fix_path, self.xml, flags=re.S)

        def fix_curve(m):
            nonlocal n_curve
            head, body, tail = m.group(1), m.group(2), m.group(3)
            pm = re.search(r'(<points version="2">)(.*?)(</points>)', body, re.S)
            if not pm:
                return m.group(0)
            pts = re.findall(r'<pt constSeg="[a-z]*" x="([^"]*)" yLocal="([^"]*)" y="([^"]*)"/>',
                             pm.group(2))
            if len(pts) <= 1:
                return m.group(0)
            chosen = next(((yl, y) for x, yl, y in pts if 1 in frames_of(x)),
                          (pts[0][1], pts[0][2]))
            n_curve += 1
            new = ('\n      <pt constSeg="false" x="1" yLocal="%s" y="%s"/>\n     ' % chosen)
            return head + body[:pm.start(2)] + new + body[pm.end(2):] + tail

        self.xml = re.sub(r'(<column type="3"[^>]*>)(.*?)(</column>)', fix_curve,
                          self.xml, flags=re.S)
        return n_path, n_curve

    def fill_exposure_gaps(self, nframes: int) -> dict[str, int]:
        """Forward-fill holes in any layer that already reaches the last frame.

        Only touches layers that run to the end, so guides and deliberately
        short reference layers are left as the rig author made them.
        """
        filled: dict[str, int] = {}

        def repl(m):
            col = m.group(0)
            seqs = re.findall(_SEQ, col)
            if not seqs:
                return col
            eid = seqs[0][2]
            f2v: dict[int, str] = {}
            for exp, val, _ in seqs:
                for f in frames_of(exp):
                    f2v[f] = val
            if not f2v or max(f2v) != nframes:
                return col
            missing = [f for f in range(1, nframes + 1) if f not in f2v]
            if not missing:
                return col
            last = None
            for f in range(1, nframes + 1):
                if f in f2v:
                    last = f2v[f]
                elif last is not None:
                    f2v[f] = last
            first = f2v[min(f2v)]
            for f in range(1, nframes + 1):
                f2v.setdefault(f, first)
            filled[self.elements.get(eid, f"id{eid}")] = len(missing)
            return re.sub(r"(\s*<elementSeq[^>]*/>)+", _emit(f2v, eid, nframes), col, count=1)

        self.xml = re.sub(_COL0, repl, self.xml, flags=re.S)
        return filled

    def keep_only(self, keep: set[str]) -> None:
        """Blank every drawing column whose element id is not in `keep`."""
        def repl(m):
            col = m.group(0)
            seqs = re.findall(_SEQ, col)
            if not seqs:
                return col
            if seqs[0][2] not in keep:
                return re.sub(r"(\s*<elementSeq[^>]*/>)+", "\n    ", col, count=1)
            return col
        self.xml = re.sub(_COL0, repl, self.xml, flags=re.S)

    def hold_frame_one(self, nframes: int) -> None:
        """Every kept layer shows its frame-1 drawing for the whole scene."""
        def repl(m):
            col = m.group(0)
            seqs = re.findall(_SEQ, col)
            if not seqs:
                return col
            eid = seqs[0][2]
            f2v: dict[int, str] = {}
            for exp, val, _ in seqs:
                for f in frames_of(exp):
                    f2v[f] = val
            if not f2v:
                return col
            val = f2v.get(1, f2v[min(f2v)])
            body = '\n     <elementSeq exposures="1-%d" val="%s" id="%s"/>\n    ' % (nframes, val, eid)
            return re.sub(r"(\s*<elementSeq[^>]*/>)+", body, col, count=1)
        self.xml = re.sub(_COL0, repl, self.xml, flags=re.S)

    def set_resolution(self, w: int, h: int) -> None:
        self.xml = re.sub(r'(<resolution name=")[^"]*(" size=")\d+,\d+(")',
                          rf'\g<1>clonecut_{w}x{h}\g<2>{w},{h}\g<3>', self.xml, count=1)

    def set_length(self, n: int) -> None:
        self.xml = re.sub(r'(nbframes=")\d+(")', rf'\g<1>{n}\g<2>', self.xml)
        self.xml = re.sub(r'(<scene [^>]*stopFrame=")\d+(")', rf'\g<1>{n}\g<2>', self.xml)

    def set_write_prefix(self, prefix: str) -> None:
        self.xml = re.sub(r'(<drawingName val=")[^"]*(")', rf'\g<1>{prefix}\g<2>', self.xml)

    def save(self, path: Path | None = None) -> Path:
        dest = Path(path or self.path)
        dest.write_text(self.xml, encoding="utf-8")
        return dest


def _emit(f2v: dict[int, str], eid: str, n: int) -> str:
    """Frame->drawing map back into compact <elementSeq> runs."""
    out, start, cur = [], 1, f2v[1]
    for f in range(2, n + 2):
        v = f2v.get(f)
        if f > n or v != cur:
            out.append((start, f - 1, cur))
            if f <= n:
                start, cur = f, v
    return "\n" + "".join(
        '     <elementSeq exposures="%s" val="%s" id="%s"/>\n'
        % (f"{a}-{b}" if b > a else f"{a}", v, eid) for a, b, v in out) + "    "


def render(scene: Path, timeout: int = 900) -> list[Path]:
    """Run Harmony headless on a scene and return the frames it wrote."""
    if not Path(HARMONY).exists():
        raise ToolError(f"Harmony not found at {HARMONY}")
    frames = scene.parent / "frames"
    frames.mkdir(exist_ok=True)
    before = set(frames.glob("*.tga"))
    common.run([HARMONY, "-batch", str(scene)])
    made = sorted(set(frames.glob("*.tga")) - before) or sorted(frames.glob("*.tga"))
    if not made:
        raise ToolError(f"Harmony rendered nothing for {common.rel(scene)}")
    return made


def plate(rig_dir: Path, out_dir: Path, *, keep: set[str] | None = None,
          resolution: tuple[int, int] | None = None, nframes: int = 1,
          freeze: bool = True, hold: bool = True,
          scene_name: str | None = None) -> list[Path]:
    """Copy a rig, prepare it, render, and return the frames.

    The copy matters: these edits are destructive and the rigs are irreplaceable.
    """
    rig_dir = Path(rig_dir)
    chosen = find_scene(rig_dir, scene_name)
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(rig_dir, out_dir, ignore=shutil.ignore_patterns("frames", "*.BACKUP*"))
    (out_dir / "frames").mkdir(exist_ok=True)

    scene = Scene(out_dir / chosen.name)
    if freeze:
        p, c = scene.freeze_pegs()
        print(f"  froze {p} motion paths, {c} value curves")
    if keep:
        scene.keep_only(keep)
        print(f"  kept {len(keep)} elements")
    if hold:
        scene.hold_frame_one(nframes)
    scene.set_length(nframes)
    if resolution:
        scene.set_resolution(*resolution)
        print(f"  resolution {resolution[0]}x{resolution[1]}")
    scene.set_write_prefix("frames/plate-")
    scene.save()
    made = render(scene.path)
    print(f"  rendered {len(made)} frame(s)")
    return made


# ---------------------------------------------------------------- measuring

def measure(plate_path: Path) -> dict:
    """Derive rig geometry from a rendered plate, rather than by hand.

    Everything here was found by colour and difference analysis when this was done
    manually for the doctor, so it repeats for any rig.
    """
    from PIL import Image
    from scipy import ndimage
    rgb = np.array(Image.open(plate_path).convert("RGB")).astype(int)
    solid = rgb.sum(axis=2) < 720
    ys, xs = np.where(solid)
    if not len(ys):
        raise ToolError(f"{common.rel(plate_path)} looks empty")
    out = {"character_bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]}

    # eyes: the two largest non-grey blobs in the upper half of the character
    top = ys.min() + (ys.max() - ys.min()) // 2
    sat = (rgb.max(axis=2) - rgb.min(axis=2)) > 40
    bright = rgb.mean(axis=2) > 150
    cand = sat & bright
    cand[top:, :] = False
    lab, n = ndimage.label(cand)
    blobs = []
    for i in range(1, n + 1):
        by, bx = np.where(lab == i)
        if len(by) < 300:
            continue
        blobs.append((len(by), [int(bx.min()), int(by.min()), int(bx.max()), int(by.max())]))
    blobs.sort(reverse=True)
    if len(blobs) >= 2:
        out["eyes"] = sorted([b[1] for b in blobs[:2]], key=lambda b: b[0])
    return out


def mouth_library(rig_dir: Path, out_dir: Path, element: str, mapping: dict[str, int],
                  *, resolution: tuple[int, int] = (3840, 2160),
                  scene_name: str | None = None) -> list[Path]:
    """Extract named mouth shapes from a rig that already has a mouth chart.

    Renders the character once with the mouth element blanked, then once per
    wanted drawing, and keeps the difference. That isolates the mouth without
    needing to know where on the face it sits - the same trick that found the
    mouth box in the first place.

    A rig with a real chart does not need generated mouths; this is the path for
    the dog, which has twenty.
    """
    from PIL import Image
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = sorted({int(v) for v in mapping.values()})

    base_dir = Path("/tmp") / f"mouthlib-{rig_dir.name}-base"
    plate(rig_dir, base_dir, keep=None, resolution=resolution, nframes=1,
          scene_name=scene_name)
    # blank the mouth element and re-render to get the face without it
    chosen = find_scene(base_dir, scene_name)
    sc = Scene(chosen)
    def blank(m):
        col = m.group(0)
        seqs = re.findall(_SEQ, col)
        if seqs and seqs[0][2] == element:
            return re.sub(r"(\s*<elementSeq[^>]*/>)+", "\n    ", col, count=1)
        return col
    sc.xml = re.sub(_COL0, blank, sc.xml, flags=re.S)
    sc.set_write_prefix("frames/nomouth-")
    sc.save()
    render(sc.path)
    nom = sorted(base_dir.glob("frames/nomouth-*.tga"))[0]
    base = np.array(Image.open(nom).convert("RGB")).astype(int)

    # now one frame per wanted drawing
    work = Path("/tmp") / f"mouthlib-{rig_dir.name}-shapes"
    plate(rig_dir, work, keep=None, resolution=resolution, nframes=len(wanted),
          scene_name=scene_name)
    sc2 = Scene(find_scene(work, scene_name))
    body = "\n" + "".join(
        '     <elementSeq exposures="%d" val="%d" id="%s"/>\n' % (i, d, element)
        for i, d in enumerate(wanted, 1)) + "    "
    def setmouth(m):
        col = m.group(0)
        seqs = re.findall(_SEQ, col)
        if seqs and seqs[0][2] == element:
            return re.sub(r"(\s*<elementSeq[^>]*/>)+", body, col, count=1)
        return col
    sc2.xml = re.sub(_COL0, setmouth, sc2.xml, flags=re.S)
    sc2.set_write_prefix("frames/shape-")
    sc2.save()
    render(sc2.path)

    by_drawing = {}
    for i, d in enumerate(wanted, 1):
        f = work / "frames" / f"shape-{i:04d}.tga"
        if not f.exists():
            raise ToolError(f"expected {common.rel(f)}")
        rgb = np.array(Image.open(f).convert("RGB")).astype(int)
        mask = np.abs(rgb - base).sum(axis=2) > 18
        ys, xs = np.where(mask)
        if not len(ys):
            raise ToolError(f"drawing {d} of element {element} looks identical to no mouth")
        sub = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        km = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        by_drawing[d] = np.dstack([sub.astype(np.uint8), (km * 255).astype(np.uint8)])

    written = []
    for name, d in mapping.items():
        dest = out_dir / f"{name}.png"
        Image.fromarray(by_drawing[int(d)]).save(dest)
        h, w = by_drawing[int(d)].shape[:2]
        print(f"  {name:7s} <- drawing {d:<3} {w}x{h}")
        written.append(dest)
    return written


# ------------------------------------------------------------------- CLI

def cmd_info(a) -> int:
    rig = Path(a.rig)
    s = Scene(find_scene(rig, a.scene))
    print(f"{common.rel(s.path)}")
    print(f"  resolution {s.resolution[0]}x{s.resolution[1]}   frames {s.length}")
    peg = s.animated_pegs()
    print(f"  animated peg paths: {peg}" + ("   <- turnaround, needs freezing" if peg > 5 else ""))
    counts = {k: v for k, v in s.drawing_counts().items() if v}
    print(f"  elements with drawings: {len(counts)}")
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1])[:12]:
        print(f"     {n:3d}  {name}")
    gaps = s.exposure_gaps(s.length)
    if gaps:
        print(f"  exposure gaps in {len(gaps)} layers (would render as parts blinking out):")
        for name, miss in sorted(gaps.items(), key=lambda kv: -len(kv[1]))[:6]:
            print(f"     {name}: {len(miss)} frames missing")
    return 0


def cmd_plate(a) -> int:
    res = None
    if a.resolution:
        w, h = a.resolution.lower().split("x")
        res = (int(w), int(h))
    keep = set(a.keep.split(",")) if a.keep else None
    made = plate(Path(a.rig), Path(a.out), keep=keep, resolution=res,
                 nframes=a.frames, freeze=not a.no_freeze, scene_name=a.scene)
    for p in made[:4]:
        print(f"  {common.rel(p)}")
    return 0


def cmd_mouths(a) -> int:
    mapping = {}
    for pair in a.map.split(","):
        k, v = pair.split("=")
        mapping[k.strip()] = int(v)
    w, h = a.resolution.lower().split("x")
    made = mouth_library(Path(a.rig), Path(a.out), a.element, mapping,
                         resolution=(int(w), int(h)), scene_name=a.scene)
    print(f"wrote {len(made)} shapes to {common.rel(Path(a.out))}")
    return 0


def cmd_measure(a) -> int:
    if a.plate:
        src = Path(a.plate)
    else:
        made = plate(Path(a.rig), Path(a.out or "/tmp/rig-measure"),
                     resolution=(3840, 2160), nframes=1)
        src = made[0]
    import json
    print(json.dumps(measure(src), indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    i = sub.add_parser("info", help="what a rig contains and what is wrong with it")
    i.add_argument("rig")
    i.add_argument("--scene", help="scene stem, when a rig holds several")
    i.set_defaults(func=cmd_info)

    pl = sub.add_parser("plate", help="render a prepared subset of a rig")
    pl.add_argument("rig")
    pl.add_argument("--out", required=True, help="working copy directory")
    pl.add_argument("--keep", help="comma-separated element ids to render")
    pl.add_argument("--resolution", help="e.g. 3840x2160")
    pl.add_argument("--frames", type=int, default=1)
    pl.add_argument("--no-freeze", action="store_true", help="keep the turnaround animation")
    pl.add_argument("--scene", help="scene stem, when a rig holds several")
    pl.set_defaults(func=cmd_plate)

    ml = sub.add_parser("mouths", help="extract a mouth library from a rig that has one")
    ml.add_argument("rig")
    ml.add_argument("--element", required=True, help="element id of the mouth layer")
    ml.add_argument("--out", required=True)
    ml.add_argument("--map", required=True,
                    help="SHAPE=drawing pairs, e.g. CLOSED=16,SMALL=17,WIDE=11")
    ml.add_argument("--resolution", default="3840x2160")
    ml.add_argument("--scene")
    ml.set_defaults(func=cmd_mouths)

    me = sub.add_parser("measure", help="derive geometry from a rendered plate")
    me.add_argument("rig", nargs="?")
    me.add_argument("--plate", help="measure an existing plate instead of rendering")
    me.add_argument("--out")
    me.set_defaults(func=cmd_measure)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
