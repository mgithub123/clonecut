#!/usr/bin/env python3
"""Stage 7c - Rig preparation.

Harmony scene files are XML, and Harmony ships a headless renderer. Together that
means a character rig can be driven entirely from a script: no GUI, no clicking,
216 frames at 4K in under twenty seconds.

    uv run rig.py info   ~/Desktop/LuckyDog-Harmony/LD_DOCTOR_RIG_MA
    uv run rig.py plate  ~/Desktop/LuckyDog-Harmony/LD_DOCTOR_RIG_MA --out cache/head \\
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
import os
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
        txt = table.read_text() if table.exists() else ""
        self.elements = dict(re.findall(r'<element id="(\d+)" elementName="([^"]*)"', txt))
        # elementFolder, not elementName, is where the drawings actually live. The robot
        # has two elements both named Hand: ids 35 and 44, folders Hand and Hand.44, with
        # 10 and 9 poses. Reading by name gave element 44 the other hand's drawing list
        # and asked Harmony for a pose it does not have.
        self.folders = dict(re.findall(
            r'<element id="(\d+)"[^>]*elementFolder="([^"]*)"', txt))

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
            d = edir / self.folders.get(eid, name)
            out[name] = len(list(d.glob("*.tvg"))) if d.is_dir() else 0
        return out

    def drawing_counts_by_id(self) -> dict[str, int]:
        """Same as drawing_counts but keyed by element id.

        drawing_counts() keys by name, and the robot has two elements both called
        Hand (ids 35 and 44) - one silently overwrites the other, under-reporting
        its hand poses by half.
        """
        out = {}
        edir = self.path.parent / "elements"
        for eid, name in self.elements.items():
            d = edir / self.folders.get(eid, name)
            out[eid] = len(list(d.glob("*.tvg"))) if d.is_dir() else 0
        return out

    def drawing_ids_by_id(self) -> dict[str, list[str]]:
        """The actual drawing names each element has, e.g. ["1","2","D7"].

        Taken from the .tvg filenames (Name-<drawing>.tvg) rather than assumed to
        be 1..n: the dog's mouths are numbered with gaps, and an element's frame-1
        exposure is often not drawing 1.
        """
        out = {}
        edir = self.path.parent / "elements"
        for eid, name in self.elements.items():
            folder = self.folders.get(eid, name)
            d = edir / folder
            if not d.is_dir():
                out[eid] = []
                continue
            vals = []
            for f in sorted(d.glob("*.tvg")):
                stem = f.stem
                for pre in (name + "-", folder + "-"):
                    if stem.startswith(pre):
                        stem = stem[len(pre):]
                        break
                else:
                    stem = stem.split("-", 1)[1] if "-" in stem else stem
                vals.append(stem)
            out[eid] = vals
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

    def expose_drawing(self, eid: str, val: str, nframes: int) -> None:
        """Hold one named drawing of one element for the whole scene.

        hold_frame_one() pins whatever an element happens to show on frame 1. To
        bake a hand library the drawing has to be chosen, not inherited.
        """
        def repl(m):
            col = m.group(0)
            seqs = re.findall(_SEQ, col)
            if not seqs or seqs[0][2] != eid:
                return col
            body = '\n     <elementSeq exposures="1-%d" val="%s" id="%s"/>\n    ' % (
                nframes, val, eid)
            return re.sub(r"(\s*<elementSeq[^>]*/>)+", body, col, count=1)
        self.xml = re.sub(_COL0, repl, self.xml, flags=re.S)

    def set_resolution(self, w: int, h: int) -> None:
        self.xml = re.sub(r'(<resolution name=")[^"]*(" size=")\d+,\d+(")',
                          rf'\g<1>clonecut_{w}x{h}\g<2>{w},{h}\g<3>', self.xml, count=1)

    def set_length(self, n: int) -> None:
        self.xml = re.sub(r'(nbframes=")\d+(")', rf'\g<1>{n}\g<2>', self.xml)
        self.xml = re.sub(r'(<scene [^>]*stopFrame=")\d+(")', rf'\g<1>{n}\g<2>', self.xml)

    def set_write_format(self, fmt: str = "TGA4") -> None:
        """TGA4 is 32-bit with alpha. The default TGA is 24-bit, so anything the
        rig does not cover comes out black - which is invisible on a rig that
        renders over a white colour card and fatal on one that does not."""
        self.xml = re.sub(r'(<drawingType val=")[^"]*(")', rf'\g<1>{fmt}\g<2>', self.xml)

    def set_write_prefix(self, prefix: str) -> None:
        self.xml = re.sub(r'(<drawingName val=")[^"]*(")', rf'\g<1>{prefix}\g<2>', self.xml)

    def drop_colour_cards(self) -> int:
        """Unwire every COLOR_CARD, so what the rig does not cover is transparent.

        The dog's and doctor's scenes each hold a white Colour-Card filling the
        frame; the robot's does not. That single difference is why only the robot
        baked with a usable matte: TGA4 gives the render an alpha channel, but a
        colour card fills it, so every element came back opaque with a full-canvas
        bounding box and nothing to crop to.

        A colour card is a source, not a pass-through, so unlike a cutter it only
        needs its outgoing links dropped - there is nothing to rewire in its place.
        """
        cards = set(re.findall(r'<module type="COLOR_CARD" name="([^"]*)"', self.xml))
        if not cards:
            return 0
        dropped = 0

        def fix(block: re.Match) -> str:
            nonlocal dropped
            body = block.group(0)
            links = re.findall(r'<link\s[^/]*/>', body)
            keep = [l for l in links
                    if re.search(r'out="([^"]*)"', l).group(1) not in cards]
            if len(keep) == len(links):
                return body
            dropped += len(links) - len(keep)
            first = re.search(r"<link\s", body).start()
            head = body[:first]
            tail = body[body.rindex("/>") + 2:]
            pad = re.search(r"\n(\s*)<link\s", body)
            joiner = "\n" + (pad.group(1) if pad else "           ")
            return head + joiner.join(l.strip() for l in keep).lstrip() + tail

        self.xml = re.sub(r"<linkedlist>.*?</linkedlist>", fix, self.xml, flags=re.S)
        return dropped

    def bypass_cutters(self) -> int:
        """Rewire every CUTTER out of the node graph, so each element renders its raw art.

        A Cutter takes the image on port 0 and a matte on port 1. Baking one element
        at a time blanks every other element, which empties those mattes - and a
        Cutter with no matte outputs nothing, so ten of the robot's elements came
        back blank: its eyelids, teeth, decals and shadows are all cut by a matte
        drawn on a different element.

        Bypassing is the right answer rather than a workaround. The point of baking
        element-wise is to recompose in Python later, and the matte elements are
        baked too, so the masking is reconstructible - whereas a Cutter applied
        during the bake is one more decision frozen into the plate.

        Names repeat between groups ("Cutter" appears in several), so each
        <linkedlist> is resolved on its own.
        """
        cutters = set(re.findall(r'<module type="CUTTER" name="([^"]*)"', self.xml))
        if not cutters:
            return 0
        removed = 0

        def fix(block: re.Match) -> str:
            nonlocal removed
            body = block.group(0)
            links = re.findall(r'<link\b[^/]*/>', body)
            parsed = [{k: v for k, v in re.findall(r'(\w+)="([^"]*)"', l)} for l in links]
            here = {l["in"] for l in parsed if l.get("in") in cutters}
            if not here:
                return body
            # each cutter's image source, i.e. what it should be replaced by
            image = {c: None for c in here}
            for l in parsed:
                if l.get("in") in here and l.get("inport", "0") == "0":
                    image[l["in"]] = l.get("out")

            def resolve(name, seen=()):
                while name in here and name not in seen:
                    seen = seen + (name,)
                    name = image.get(name)
                return name

            out, indent = [], re.search(r"\n(\s*)<link", body)
            pad = indent.group(1) if indent else "           "
            for l in parsed:
                if l.get("in") in here:
                    continue                      # an input to a cutter: goes away with it
                src = resolve(l.get("out"))
                if src is None:
                    continue                      # cutter had no image input; nothing to pass on
                attrs = f'out="{src}" in="{l["in"]}"'
                if "outport" in l and l["out"] not in here:
                    attrs += f' outport="{l["outport"]}"'
                if "inport" in l:
                    attrs += f' inport="{l["inport"]}"'
                out.append(f"{pad}<link {attrs}/>")
            removed += len(here)
            first = re.search(r"<link\s", body).start()   # not "<linkedlist>", which shares the prefix
            head = body[:first]
            tail = body[body.rindex("/>") + 2:]
            return head + "\n".join(out).lstrip() + tail

        self.xml = re.sub(r"<linkedlist>.*?</linkedlist>", fix, self.xml, flags=re.S)
        return removed

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
    """Run Harmony headless on a scene and return the frames it wrote.

    CLONECUT_NO_HARMONY=1 makes this raise instead of rendering. The licence is a
    trial expiring 2026-10-03, so "does this still work without Harmony?" needs to
    be answerable now, while there is still time to bake whatever the answer says
    is missing - not discovered on the day it stops opening.
    """
    if os.environ.get("CLONECUT_NO_HARMONY"):
        raise ToolError(
            f"CLONECUT_NO_HARMONY is set and something asked Harmony to render "
            f"{common.rel(scene)}.\n"
            f"  Whatever this render was for is not baked yet - bake it while the "
            f"licence lasts.")
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
          freeze: bool = True, hold: bool = True, alpha: bool = True,
          colour_card: bool = False, scene_name: str | None = None) -> list[Path]:
    """Copy a rig, prepare it, render, and return the frames.

    The copy matters: these edits are destructive and the rigs are irreplaceable.
    """
    rig_dir = Path(rig_dir)
    chosen = find_scene(rig_dir, scene_name)
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(rig_dir, out_dir, ignore=shutil.ignore_patterns("frames", "*.BACKUP*"))
    for p in out_dir.rglob("*"):            # masters are read-only; the copy must not be
        p.chmod(p.stat().st_mode | 0o200)
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
    if not colour_card and scene.drop_colour_cards():
        # A white card behind the rig fills the alpha channel, so the plate comes
        # back opaque and only face.unmul() can recover a matte from it. Dropping
        # it gives a true one. The already-tracked dog and doctor plates were made
        # with the card in place; they still load correctly, because
        # perform._matted() decides by looking at the alpha rather than assuming.
        print("  dropped the colour card")
    if alpha:
        # Without this the Write node emits 24-bit TGA and the plate arrives
        # composited onto the scene's background colour. That is survivable when
        # the background is white - face.unmul() inverts the blend - but the robot's
        # scene composites on black, where the same inversion makes every pixel
        # opaque and the whole character disappears into it.
        scene.set_write_format("TGA4")
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

    base_dir = config.CACHE_DIR / f"mouthlib-{rig_dir.name}-base"
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
    work = config.CACHE_DIR / f"mouthlib-{rig_dir.name}-shapes"
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


def bake_drawings(work: Path, scene_name: str, out_dir: Path, targets: list[tuple[str, str]],
                  resolution: tuple[int, int]) -> dict:
    """Render every drawing of every multi-drawing element, one per frame.

    The element pass captures whatever each element happens to expose on frame 1.
    That loses the parts of a rig that are libraries rather than layers: the dog's
    twenty mouths and twenty-four pockets, the robot's twenty hand poses. Those are
    the only gesture and articulation vocabulary these rigs have, and once the
    licence lapses they cannot be got back.

    Drawings are saved tight-cropped with their offset recorded, so a pose can be
    put back exactly where the rig had it without storing 33MP of transparency.
    """
    from PIL import Image
    if not targets:
        return {}
    sc = Scene(work / scene_name)
    p, c = sc.freeze_pegs()
    cut = sc.bypass_cutters()
    card = sc.drop_colour_cards()

    frame_of: dict[tuple[str, str], int] = {t: i for i, t in enumerate(targets, 1)}
    by_el: dict[str, list[tuple[int, str]]] = {}
    for (eid, val), fr in frame_of.items():
        by_el.setdefault(eid, []).append((fr, val))

    def expose(m):
        col = m.group(0)
        seqs = re.findall(_SEQ, col)
        if not seqs:
            return col
        eid = seqs[0][2]
        if eid not in by_el:
            return re.sub(r"(\s*<elementSeq[^>]*/>)+", "\n    ", col, count=1)
        body = "\n" + "".join(
            '     <elementSeq exposures="%d" val="%s" id="%s"/>\n' % (fr, val, eid)
            for fr, val in sorted(by_el[eid])) + "    "
        return re.sub(r"(\s*<elementSeq[^>]*/>)+", body, col, count=1)

    sc.xml = re.sub(_COL0, expose, sc.xml, flags=re.S)
    sc.set_length(len(targets)); sc.set_resolution(*resolution)
    sc.set_write_format("TGA4")
    sc.set_write_prefix("frames/dw-")
    sc.save()
    print(f"  {len(targets)} drawings across {len(by_el)} elements, "
          f"{cut} cutters bypassed, {card} colour cards dropped, "
          f"{resolution[0]}x{resolution[1]}")
    render(sc.path)

    out_dir.mkdir(parents=True, exist_ok=True)
    got: dict = {}
    for (eid, val), fr in sorted(frame_of.items(), key=lambda kv: kv[1]):
        src_f = work / "frames" / f"dw-{fr:04d}.tga"
        if not src_f.exists():
            continue
        arr = np.array(Image.open(src_f).convert("RGBA"))
        ys, xs = np.where(arr[..., 3] > 8)
        rec = got.setdefault(eid, {})
        if not len(ys):
            rec[val] = None                    # a genuinely blank drawing; recorded, not silently dropped
            continue
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", val)
        dest = out_dir / f"{eid}-{safe}.png"
        Image.fromarray(arr[y0:y1 + 1, x0:x1 + 1]).save(dest)
        rec[val] = {"file": dest.name, "bbox": [x0, y0, x1, y1]}
    return got


def bake_poses(rig_dir: Path, out_dir: Path, cfg: dict, *,
               resolution: tuple[int, int] = (3840, 2160),
               scene_name: str | None = None, hands: bool = True) -> dict:
    """Render every turnaround view angle as the head/body plate pair the
    front-on path already uses, plus the geometry needed to drive it.

    The point is that a pose is not a special kind of picture. build_frames()
    wants a head plate, a body plate and absolute coordinates for the mouth and
    eyes; give it those for a three-quarter view and a turned character sings
    through exactly the same code. Treating a pose as a flat full-figure image
    would have meant a second compositing path and hand-measured geometry per
    angle - which is what put the dog's mouth on its nose the first time.

    Pegs animate over time, so pose p only exists on frame p and a subset cannot
    be moved onto frame 1. But rendering the turnaround *unfrozen* with a chosen
    subset yields that subset at every pose in one pass, so four passes cover a
    whole rig.

    Geometry is measured, never typed in: the mouth pass is the mouth element
    alone, so its bounding box at each pose gives centre_x, lip_line and width
    directly, and the eye pass gives the eye boxes the same way.
    """
    from PIL import Image
    from scipy import ndimage

    rig_dir = Path(rig_dir); out_dir = Path(out_dir)
    chosen = find_scene(rig_dir, scene_name)
    probe = Scene(chosen)
    nframes = probe.length
    if not probe.animated_pegs():
        raise ToolError(
            f"{common.rel(chosen)} has no animated pegs, so it holds no view "
            f"angles - its turnaround was frozen away. Bake poses from the "
            f"master scene instead.")

    work = config.CACHE_DIR / f"poses-{rig_dir.name}"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(rig_dir, work, ignore=shutil.ignore_patterns("frames", "*.BACKUP*", "*.bak-*"))
    for p in work.rglob("*"):
        p.chmod(p.stat().st_mode | 0o200)
    (work / "frames").mkdir(exist_ok=True)
    pristine = Scene(chosen).xml
    layers = cfg["layers"]

    def run(keep: set[str], tag: str, drawing: tuple[str, str] | None = None) -> list[Path]:
        sc = Scene(work / chosen.name)
        sc.xml = pristine
        sc.drop_colour_cards()
        sc.keep_only(keep)
        if drawing:
            sc.expose_drawing(drawing[0], drawing[1], nframes)
        sc.set_length(nframes)
        sc.set_resolution(*resolution)
        sc.set_write_format("TGA4")
        sc.set_write_prefix(f"frames/{tag}-")
        sc.save()
        for old in (work / "frames").glob(f"{tag}-*.tga"):
            old.unlink()
        render(sc.path)
        return sorted((work / "frames").glob(f"{tag}-*.tga"))

    def solid(path: Path):
        a = np.array(Image.open(path).convert("RGBA"))
        return a, a[..., 3] > 8

    def box(mask):
        ys, xs = np.where(mask)
        if not len(ys):
            return None
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    ids = lambda key: {str(i) for i in layers[key]}
    print(f"  {nframes} turnaround frames, {resolution[0]}x{resolution[1]}")

    head_f = run(ids("head"), "head")
    body_f = run(ids("body"), "body")

    # A view angle, not a frame: the turnaround rotates and then holds, so most
    # frames repeat. Identity is the head and body together - the head alone is
    # equal on frames where only an arm moved.
    seen: dict[tuple[str, str], int] = {}
    frame_pose: dict[int, int] = {}
    for k, (hf, bf) in enumerate(zip(head_f, body_f), 1):
        key = (common.file_hash(hf), common.file_hash(bf))
        if key not in seen:
            seen[key] = len(seen) + 1
        frame_pose[k] = seen[key]
    first_frame = {}
    for fr, pose in frame_pose.items():
        first_frame.setdefault(pose, fr)
    print(f"  {nframes} frames -> {len(first_frame)} distinct poses")

    # Measure by difference against the head plate, never by rendering a part on
    # its own. Alone, the mouth is cut to nothing - its Cutter's matte lives on an
    # element that a subset render blanks - and bypassing the cutter instead would
    # hand back the raw art, which for the robot is a 591px strip of teeth behind a
    # 221px slot. head-plus-mouth minus head is exactly the pixels the rig shows.
    mouth_ids = {str(layers["mouth"])} | {str(i) for i in layers.get("mouth_family", [])}
    withmouth_f = run(ids("head") | mouth_ids, "withmouth")
    noeye_f = run(ids("head") - {str(layers["eyes"])}, "noeye")

    def diff(a_path: Path, b_path: Path):
        a = np.array(Image.open(a_path).convert("RGBA")).astype(int)
        b = np.array(Image.open(b_path).convert("RGBA")).astype(int)
        d = np.abs(a - b)
        return (d[..., :3].sum(axis=2) > 18) | (d[..., 3] > 8)

    out_dir.mkdir(parents=True, exist_ok=True)
    poses: dict[str, dict] = {}
    for pose, fr in sorted(first_frame.items()):
        rec: dict = {"pose": pose, "frame": fr}
        for tag, files in (("head", head_f), ("body", body_f)):
            arr, m = solid(files[fr - 1])
            dest = out_dir / f"{tag}-{pose:02d}.png"
            Image.fromarray(arr).save(dest)              # full canvas: build_frames composites absolutely
            rec[tag] = dest.name
            rec[f"{tag}_bbox"] = box(m)

        mm = diff(withmouth_f[fr - 1], head_f[fr - 1])
        mb = box(mm)
        rec["mouth_bbox"] = mb
        if mb:
            rec["centre_x"] = (mb[0] + mb[2]) // 2
            rec["lip_line"] = mb[1]
            rec["anchor_y"] = mb[3]
            rec["mouth_width"] = mb[2] - mb[0] + 1
            hb = rec["head_bbox"]
            rec["max_mouth_height"] = max(8, (hb[3] - mb[1]) if hb else mb[3] - mb[1])

        em = diff(head_f[fr - 1], noeye_f[fr - 1])
        lab, n = ndimage.label(em)
        boxes = []
        for i in range(1, n + 1):
            b = box(lab == i)
            if b and (b[2] - b[0]) * (b[3] - b[1]) > 200:
                boxes.append(b)
        rec["eyes"] = sorted(boxes)[:2]

        hb = rec["head_bbox"]
        rec["head_pivot"] = [(hb[0] + hb[2]) // 2, hb[3]] if hb else None
        poses[str(pose)] = rec

    if hands and layers.get("hands"):
        counts = probe.drawing_ids_by_id()
        nh = run(ids("body") - {str(i) for g in layers["hands"] for i in g}, "nohand")
        for pose, fr in sorted(first_frame.items()):
            arr, _ = solid(nh[fr - 1])
            d = out_dir / f"nohand-{pose:02d}.png"
            Image.fromarray(arr).save(d)
            poses[str(pose)]["body_nohands"] = d.name
        for gi, group in enumerate(layers["hands"]):
            lead = str(group[0])
            for val in counts.get(lead, []):
                safe = re.sub(r"[^A-Za-z0-9_.-]", "_", val)
                fs = run({str(i) for i in group}, f"h{gi}x{safe}", drawing=(lead, val))
                for pose, fr in sorted(first_frame.items()):
                    arr, m = solid(fs[fr - 1])
                    b = box(m)
                    ent = poses[str(pose)].setdefault("hands", {}).setdefault(str(gi), {})
                    if not b:
                        ent[val] = None
                        continue
                    dest = out_dir / f"hand{gi}-{safe}-{pose:02d}.png"
                    Image.fromarray(arr[b[1]:b[3] + 1, b[0]:b[2] + 1]).save(dest)
                    ent[val] = {"file": dest.name, "at": [b[0], b[1]]}
            print(f"  hand group {gi}: {len(counts.get(lead, []))} poses")

    # A rig has no profile mouth. As the head turns the visible mouth narrows to a
    # sliver - the robot's goes from 112px head-on to 10px in profile - and scaling
    # a front-drawn shape into that reads as a smudge, not singing. Record it per
    # pose rather than let a shot silently produce one.
    # The tail of a turnaround is not always a pose. The doctor's last frames render
    # a collar, a skirt and boots with no head at all - the rig part-disassembled
    # after the rotation ends. A pose with no head or no body is not a view angle.
    for p in poses.values():
        p["usable"] = bool(p.get("head_bbox") and p.get("body_bbox"))
    widest = max((p.get("mouth_width") or 0) for p in poses.values() if p["usable"]) or 1
    for p in poses.values():
        w = p.get("mouth_width") or 0
        p["mouth_foreshortening"] = round(w / widest, 3)
        p["lipsync"] = bool(p["usable"] and w >= 0.35 * widest)
    good = [p for p in poses.values() if p["usable"]]
    can = sum(1 for p in good if p["lipsync"])
    print(f"  {len(good)}/{len(poses)} poses usable, {can} of those can lip-sync "
          f"(mouth >= 35% of head-on width)")

    manifest = {"rig": rig_dir.name, "scene": chosen.name,
                "resolution": list(resolution), "frames": nframes,
                "frame_pose": {str(k): v for k, v in frame_pose.items()},
                "poses": poses}
    common.write_json(out_dir / "manifest.json", manifest)
    shutil.rmtree(work, ignore_errors=True)
    return manifest


def bake(rig_dir: Path, out_dir: Path, *, resolution=(3840,2160),
         scene_name: str | None = None, include_all: bool = True,
         turnaround: bool = True) -> dict:
    """Render every element of a rig, alone, on the full registered canvas.

    Elements rather than compositions. A head plate, a body plate, a mouth-in-head
    plate and an eyelid state are all just subsets of the elements; bake the parts
    and any of them can be recomposed in Python afterwards, at any z-order, with
    any later fix. Bake a composition and you have banked whatever was wrong with
    it at the time - which is exactly what happened to the dog's head plate, which
    silently contained six mouth-family elements.

    One Harmony invocation, not one per element: element i is exposed only on
    frame i and everything else is blanked, so an N-frame render yields N plates.
    """
    from PIL import Image
    rig_dir = Path(rig_dir); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    chosen = find_scene(rig_dir, scene_name)
    probe = Scene(chosen)
    counts = probe.drawing_counts_by_id()
    ids = [i for i, n in counts.items() if n]          # elements that actually draw
    if not ids:
        raise ToolError(f"no drawable elements in {common.rel(chosen)}")

    work = config.CACHE_DIR / f"bake-{rig_dir.name}"
    if work.exists(): shutil.rmtree(work)
    shutil.copytree(rig_dir, work, ignore=shutil.ignore_patterns("frames","*.BACKUP*","*.bak-*"))
    for p in work.rglob("*"):               # the masters are read-only; the working copy must not be
        p.chmod(p.stat().st_mode | 0o200)
    (work/"frames").mkdir(exist_ok=True)
    sc = Scene(work / chosen.name)
    p, c = sc.freeze_pegs()
    cut = sc.bypass_cutters()
    card = sc.drop_colour_cards()
    n = len(ids)

    def one_per_frame(m):
        col = m.group(0)
        seqs = re.findall(_SEQ, col)
        if not seqs: return col
        eid = seqs[0][2]
        if eid not in ids:
            return re.sub(r"(\s*<elementSeq[^>]*/>)+", "\n    ", col, count=1)
        f2v = {}
        for e, v, _ in seqs:
            for fr in frames_of(e): f2v[fr] = v
        val = f2v.get(1, f2v[min(f2v)])
        frame = ids.index(eid) + 1
        body = '\n     <elementSeq exposures="%d" val="%s" id="%s"/>\n    ' % (frame, val, eid)
        return re.sub(r"(\s*<elementSeq[^>]*/>)+", body, col, count=1)

    sc.xml = re.sub(_COL0, one_per_frame, sc.xml, flags=re.S)
    sc.set_length(n); sc.set_resolution(*resolution)
    sc.set_write_format("TGA4")
    sc.set_write_prefix("frames/el-")
    sc.save()
    print(f"  {n} elements, pegs frozen ({p} paths, {c} curves), "
          f"{cut} cutters bypassed, {card} colour cards dropped, "
          f"{resolution[0]}x{resolution[1]}")
    render(sc.path)

    manifest = {"rig": rig_dir.name, "scene": chosen.name,
                "resolution": list(resolution),
                "cropped": True,
                "crop_note": "element plates, library drawings and turnaround poses are cropped to their bbox on the resolution-sized canvas; paste at bbox[:2] to restore. _ALL.png and _ALL_NOCUT.png are full-canvas.",
                "scene_sha256": common.file_hash(chosen),
                "cutters_bypassed": cut,
                "colour_cards_dropped": card,
                "elements": {}}
    for k, eid in enumerate(ids, 1):
        src = work/"frames"/f"el-{k:04d}.tga"
        if not src.exists(): continue
        im = Image.open(src)
        arr = np.array(im.convert("RGBA"))
        solid = arr[..., 3] > 8 if arr.shape[2] == 4 else (arr[..., :3].sum(axis=2) < 720)
        ys, xs = np.where(solid)
        name = probe.elements.get(eid, f"id{eid}")
        dest = out_dir / f"{eid}-{name}.png"
        box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(ys) else None
        # Saved cropped to its own art, with the offset recorded. A 7680x4320 canvas
        # per element is 33MP of mostly nothing; the bbox makes the crop lossless for
        # recompositing - paste at bbox[:2] and you have the original canvas back.
        rgba = arr if arr.shape[2] == 4 else np.dstack([arr, np.full(arr.shape[:2], 255, np.uint8)])
        Image.fromarray(rgba[box[1]:box[3] + 1, box[0]:box[2] + 1] if box else rgba).save(dest)
        manifest["elements"][eid] = {
            "name": name, "drawings": counts[eid], "file": dest.name,
            "folder": probe.folders.get(eid, name), "bbox": box,
        }
    # every drawing of every element that has more than one - the rigs' pose and
    # articulation libraries, which the frame-1 element pass above cannot see
    multi = [(eid, v) for eid, vals in probe.drawing_ids_by_id().items()
             if len(vals) > 1 for v in vals]
    if multi:
        got = bake_drawings(work, chosen.name, out_dir / "drawings", multi, resolution)
        for eid, rec in got.items():
            manifest["elements"].setdefault(eid, {"name": probe.elements.get(eid, f"id{eid}")})
            manifest["elements"][eid]["library"] = rec
        manifest["drawings_baked"] = sum(len(r) for r in got.values())

    if include_all:
        # Two ground truths. _ALL is the rig as authored - what a recomposite should
        # look like. _ALL_NOCUT is the same with cutters bypassed, which is the one
        # the element plates can actually be checked against, since they were baked
        # that way too. Without it, every masked element reads as a mismatch.
        for tag, bypass in (("all", False), ("allnc", True)):
            sc2 = Scene(work / chosen.name)
            sc2.xml = Scene(chosen).xml
            sc2.freeze_pegs()
            sc2.drop_colour_cards()
            if bypass:
                sc2.bypass_cutters()
            sc2.hold_frame_one(1); sc2.set_length(1)
            sc2.set_resolution(*resolution); sc2.set_write_format("TGA4")
            sc2.set_write_prefix(f"frames/{tag}-")
            sc2.save(); render(sc2.path)
            got_p = sorted((work/"frames").glob(f"{tag}-*.tga"))
            if got_p:
                dest = out_dir / ("_ALL.png" if not bypass else "_ALL_NOCUT.png")
                Image.open(got_p[0]).convert("RGBA").save(dest)
                manifest["ground_truth" if not bypass else "ground_truth_nocut"] = dest.name

    if turnaround:
        # The view angles. Every one of these rigs is a turnaround first and a
        # character second, and the poses are the only way to cut to a three-quarter
        # view later. freeze_pegs() has been throwing them away all session.
        sc3 = Scene(work / chosen.name)
        sc3.xml = Scene(chosen).xml
        sc3.drop_colour_cards()
        nposes = sc3.length
        sc3.set_resolution(*resolution); sc3.set_write_format("TGA4")
        sc3.set_write_prefix("frames/turn-")
        sc3.save(); render(sc3.path)
        tdir = out_dir / "turnaround"; tdir.mkdir(parents=True, exist_ok=True)
        poses = []
        for f in sorted((work/"frames").glob("turn-*.tga")):
            arr = np.array(Image.open(f).convert("RGBA"))
            ys, xs = np.where(arr[..., 3] > 8)
            if not len(ys):
                continue                       # held/blank frames at the tail
            k = int(f.stem.split("-")[-1])
            d = tdir / f"pose-{k:04d}.png"
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            Image.fromarray(arr[y0:y1 + 1, x0:x1 + 1]).save(d)
            poses.append({"frame": k, "file": d.name, "bbox": [x0, y0, x1, y1]})
        manifest["turnaround"] = {"frames": nposes, "poses": poses}
        print(f"  {len(poses)} turnaround poses")

    common.write_json(out_dir/"manifest.json", manifest)
    shutil.rmtree(work, ignore_errors=True)
    return manifest


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


def cmd_bake(a) -> int:
    w, h = a.resolution.lower().split("x")
    m = bake(Path(a.rig), Path(a.out), resolution=(int(w), int(h)), scene_name=a.scene)
    print(f"baked {len(m['elements'])} elements -> {common.rel(Path(a.out))}")
    return 0


def cmd_poses(a) -> int:
    import json as _json
    w, h = a.resolution.lower().split("x")
    cfg = _json.loads((config.ROOT / "assets" / "rigs" / a.name / "rig.json").read_text())
    m = bake_poses(Path(a.rig), Path(a.out), cfg, resolution=(int(w), int(h)),
                   scene_name=a.scene, hands=not a.no_hands)
    print(f"baked {len(m['poses'])} poses -> {common.rel(Path(a.out))}")
    return 0


def cmd_measure(a) -> int:
    if a.plate:
        src = Path(a.plate)
    else:
        made = plate(Path(a.rig), Path(a.out or config.CACHE_DIR / "rig-measure"),
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

    bk = sub.add_parser("bake", help="render every element alone (survives losing Harmony)")
    bk.add_argument("rig")
    bk.add_argument("--out", required=True)
    bk.add_argument("--resolution", default="3840x2160")
    bk.add_argument("--scene")
    bk.set_defaults(func=cmd_bake)

    q = sub.add_parser("poses", help="bake every turnaround view angle as head/body plates")
    q.add_argument("rig", help="rig directory holding the scene with the turnaround")
    q.add_argument("--name", required=True, help="rig name under assets/rigs")
    q.add_argument("--out", required=True)
    q.add_argument("--scene", default=None)
    q.add_argument("--resolution", default="3840x2160")
    q.add_argument("--no-hands", action="store_true", help="skip the hand library passes")
    q.set_defaults(func=cmd_poses)

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
