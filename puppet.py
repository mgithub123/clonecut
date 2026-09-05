#!/usr/bin/env python3
"""Stage 8a - The layered puppet.

A rig is a folder of PNG plates plus rig.json. This module turns the baked plates
into a puppet: a tree of bones, each with a pivot and a parent, and a stack of
layers in draw order, each attached to a bone. Everything in it is measured from
the plates and recorded in rig.json under "puppet", so a rig compiled from any
source - Harmony today, cut-up drawings or generated images later - animates the
same way. The format is written down in RIGS.md.

    uv run puppet.py build  doctor          # measure, write rig.json["puppet"]
    uv run puppet.py verify doctor          # recomposite vs the ground truths
    uv run puppet.py tree   doctor          # print bones and layers
    uv run puppet.py render doctor          # Blender, at rest, checked against the truth

Three measurements, and why each is measured rather than typed in:

Draw order comes from the rig's own _ALL_NOCUT.png. For every pair of plates that
overlap, the pixels where their colours differ say which one the rig drew on top.
That gives a partial order, which is sorted; a naive element-id order is 12-28%
wrong against the truth, the measured one 1.5-4.5%.

Pivots come from where a part's art meets its parent's: the overlap of the two
masks. An arm pivots where it joins the body, an ear where it joins the head. The
Harmony pegs carry authored pivots in field units, but converting those needs the
camera, and a rig from another source has no pegs at all.

Mattes come from the difference between the two ground truths. _ALL.png is the rig
as authored, with its cutters; _ALL_NOCUT.png has them bypassed. Where a layer is
visible in the recomposite but not in the authored image, it was cut, and the
baked plate whose shape best explains the cut region is its matte.

Parenting is the one thing taken from the scene: the peg hierarchy in the tracked
.xstage. That is Harmony-only and is said so in the output; a rig without a scene
falls back to overlap geometry, which is right for limbs and wrong for hair.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import common
import config
from common import ToolError

ASSETS = config.ROOT / "assets"


def _idkey(eid: str) -> tuple[int, int]:
    """Sort key for layer ids: element number, then instance ('36#2')."""
    base, _, inst = str(eid).partition("#")
    return (int(base), int(inst or 0))
SOLID = 200            # alpha above this counts as opaque for measurements
TELL = 20              # two plates must differ by this much to say who is on top
MATCH = 12             # a plate pixel within this of the truth "explains" it
MIN_MATTE_FRAC = 0.10  # below this the cut is deformer and edge noise, not a matte
MIN_MATTE_FIT = 0.75   # the best-fitting plate must explain this much of the cut
HIDDEN_FRAC = 0.95     # a layer cut this far is a drawing the rig hides at rest


# ------------------------------------------------------------------ loading

def rig_path(name: str) -> Path:
    p = ASSETS / "rigs" / name / "rig.json"
    if not p.exists():
        have = sorted(d.name for d in (ASSETS / "rigs").iterdir() if (d / "rig.json").exists())
        raise ToolError(f"no rig {name!r}; have {', '.join(have) or 'none'}")
    return p


def load_rig(name: str) -> dict:
    p = rig_path(name)
    r = json.loads(p.read_text())
    r["_dir"] = p.parent
    return r


def manifest(r: dict) -> dict:
    p = r["_dir"] / "plates" / "manifest.json"
    if not p.exists():
        raise ToolError(f"rig {r['name']!r} has no baked plates ({common.rel(p)})")
    return json.loads(p.read_text())


class Plate:
    """One baked image at its offset on the rig canvas, loaded lazily."""

    def __init__(self, path: Path, bbox: list[int]):
        self.path, self.bbox = Path(path), [int(v) for v in bbox]
        self._arr = None

    @property
    def arr(self) -> np.ndarray:
        if self._arr is None:
            self._arr = np.asarray(Image.open(self.path).convert("RGBA")).astype(np.int16)
        return self._arr

    @property
    def image(self) -> Image.Image:
        return Image.fromarray(self.arr.astype(np.uint8))

    def mask(self, canvas: tuple[int, int], solid: int = SOLID) -> np.ndarray:
        m = np.zeros((canvas[1], canvas[0]), bool)
        x0, y0, x1, y1 = self.bbox
        a = self.arr[..., 3]
        m[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = a > solid
        return m


def split_instances(plate: Plate, n: int) -> list[tuple[list[int], Plate]]:
    """The plate's n largest blobs, left to right, each as a cropped plate.

    Returns nothing unless the plate falls into exactly n substantial blobs, so
    a pair of legs drawn touching stays one layer rather than a wrong split.
    """
    from scipy import ndimage
    a = plate.arr
    lab, k = ndimage.label(ndimage.binary_dilation(a[..., 3] > 8, iterations=3))
    if k < n:
        return []
    sizes = sorted(((int((lab == i).sum()), i) for i in range(1, k + 1)), reverse=True)
    big = [i for s, i in sizes if s >= 0.05 * sizes[0][0]]
    if len(big) != n:
        return []
    out = []
    for i in big:
        ys, xs = np.where((lab == i) & (a[..., 3] > 8))
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        sub = Plate(plate.path, [plate.bbox[0] + x0, plate.bbox[1] + y0,
                                 plate.bbox[0] + x1, plate.bbox[1] + y1])
        cut = a[y0:y1 + 1, x0:x1 + 1].copy()
        cut[..., 3] = np.where((lab[y0:y1 + 1, x0:x1 + 1] == i), cut[..., 3], 0)
        sub._arr = cut
        out.append((sub.bbox, sub))
    out.sort(key=lambda t: t[0][0])
    return out


def plates_of(r: dict, man: dict) -> dict[str, Plate]:
    """Every element that has art, as a plate. Reference layers have no bbox."""
    d = r["_dir"] / "plates"
    out = {}
    for eid, e in man["elements"].items():
        if e.get("bbox") and (d / e["file"]).exists():
            out[eid] = Plate(d / e["file"], e["bbox"])
    if not out:
        raise ToolError(f"rig {r['name']!r} has no element plates with art")
    return out


# ------------------------------------------------------------------ scene hierarchy

class _Scope:
    def __init__(self, name, parent=None):
        self.name, self.parent = name, parent
        self.modules: dict[str, str] = {}          # name -> type
        self.groups: dict[str, "_Scope"] = {}
        self.links: list[dict] = []
        self.read_col: dict[str, str] = {}         # READ name -> column name


def _parse_scene(xml: str) -> _Scope:
    """The node graph of a .xstage as nested scopes, links kept per scope."""
    root = _Scope("")
    cur = root
    tag = re.compile(r'<(group name="([^"]*)"[^>]*>|/group>|module type="([A-Z_]+)" name="([^"]*)"[^>]*>'
                     r'|link [^>]*/>|element col="([^"]*)")')
    last_read = None
    for m in tag.finditer(xml):
        t = m.group(1)
        if t.startswith("group name"):
            g = _Scope(m.group(2), cur)
            cur.groups[m.group(2)] = g
            cur = g
        elif t == "/group>":
            cur = cur.parent or root
        elif t.startswith("module"):
            cur.modules[m.group(4)] = m.group(3)
            last_read = m.group(4) if m.group(3) == "READ" else None
        elif t.startswith("link"):
            cur.links.append({k: v for k, v in re.findall(r'(\w+)="([^"]*)"', t)})
        elif t.startswith("element col") and last_read:
            cur.read_col[last_read] = m.group(5)
    return root


def scene_hierarchy(xstage: Path) -> dict:
    """Pegs, their parents, and which element each READ hangs from.

    Walks upstream from every READ and every PEG through deformation groups and
    Multi-Port boundaries until it reaches a PEG. Names repeat between groups, so
    a node is identified by its scope path.
    """
    xml = xstage.read_text(encoding="utf-8", errors="replace")
    root = _parse_scene(xml)
    col_id = dict(re.findall(r'<column type="0" name="([^"]*)"[^>]*id="(\d+)"', xml))

    def key(scope: _Scope, name: str) -> str:
        parts = []
        s = scope
        while s is not None and s.name:
            parts.append(s.name)
            s = s.parent
        return "/".join(list(reversed(parts)) + [name]) if parts else name

    def upstream(scope: _Scope, name: str, port: str = "0"):
        """The (scope, node) feeding `name`'s input port, crossing group walls."""
        for l in scope.links:
            if l.get("in") == name and l.get("inport", "0") == port:
                src = l["out"]
                if src == "Multi-Port-In" and scope.parent is not None:
                    return upstream(scope.parent, scope.name, l.get("outport", "0"))
                if src in scope.groups:
                    g = scope.groups[src]
                    for gl in g.links:
                        if gl.get("in") == "Multi-Port-Out" and gl.get("inport", "0") == l.get("outport", "0"):
                            return g, gl["out"]
                    return None
                return scope, src
        return None

    def nearest_peg(scope: _Scope, name: str, seen: set):
        node = upstream(scope, name)
        while node is not None:
            s, n = node
            k = key(s, n)
            if k in seen:
                return None
            seen.add(k)
            if s.modules.get(n) == "PEG":
                return k
            node = upstream(s, n)
        return None

    pegs: dict[str, dict] = {}
    reads: dict[str, dict] = {}

    def walk(scope: _Scope):
        for name, typ in scope.modules.items():
            k = key(scope, name)
            if typ == "PEG":
                pegs[k] = {"name": name, "parent": nearest_peg(scope, name, {k})}
            elif typ == "READ":
                eid = col_id.get(scope.read_col.get(name, ""), None)
                reads[k] = {"name": name, "element": eid, "peg": nearest_peg(scope, name, {k})}
        for g in scope.groups.values():
            walk(g)

    walk(root)
    return {"pegs": pegs, "reads": reads}


# ------------------------------------------------------------------ measurements

def measure_order(plates: dict[str, Plate], truth: np.ndarray) -> tuple[list[str], dict]:
    """Draw order, bottom first, by peeling the ground truth from the top.

    The layer on top is the one whose opaque pixels the truth still shows. Take
    it, claim its pixels, and ask again: the next layer's own pixels are now
    judged only where nothing above it has been drawn. A pairwise test cannot do
    this - where a third layer covers part of an overlap it votes for the wrong
    side - and it put the doctor's hair highlight under the hair.
    """
    H, W = truth.shape[:2]
    avail = np.ones((H, W), bool)
    remaining = sorted(plates, key=_idkey)
    peeled, scores = [], {}
    while remaining:
        best = None
        for eid in remaining:
            pl = plates[eid]
            x0, y0, x1, y1 = pl.bbox
            a = pl.arr
            solid = a[..., 3] > SOLID
            cand = solid & avail[y0:y0 + a.shape[0], x0:x0 + a.shape[1]]
            n = int(cand.sum())
            if n == 0:
                frac = -1.0                   # fully hidden: nothing left to judge it by
            else:
                G = truth[y0:y0 + a.shape[0], x0:x0 + a.shape[1]]
                ok = (np.abs(G[..., :3] - a[..., :3]).max(axis=2) < MATCH) & cand
                frac = float(ok.sum()) / n
            if best is None or frac > best[0] or (frac == best[0] and _idkey(eid) > _idkey(best[1])):
                best = (frac, eid)
        frac, eid = best
        scores[eid] = round(frac, 3)
        peeled.append(eid)
        remaining.remove(eid)
        pl = plates[eid]
        x0, y0 = pl.bbox[:2]
        a = pl.arr
        avail[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] &= ~(a[..., 3] > SOLID)
    order = peeled[::-1]
    return order, {"visible_fraction": scores,
                   "hidden": [e for e in peeled if scores[e] < 0]}


def composite(layers: list[dict], plates: dict[str, Plate], canvas: tuple[int, int],
              drawings: dict[str, str] | None = None, plate_dir: Path | None = None,
              mattes: bool = True) -> Image.Image:
    """Draw the layers in order. `drawings` picks a library drawing per element."""
    out = Image.new("RGBA", canvas, (0, 0, 0, 0))
    chosen = drawings or {}
    for lay in layers:
        eid = lay["id"]
        if mattes and lay.get("hidden_at_rest") and eid not in chosen:
            continue
        pl = plates.get(eid)
        if eid in chosen and lay.get("library") and chosen[eid] in lay["library"]:
            ent = lay["library"][chosen[eid]]
            if ent is None:
                continue
            pl = Plate(plate_dir / ent["file"], ent["bbox"])
        if pl is None:
            continue
        im = pl.image
        if mattes and lay.get("matte"):
            mt = plates.get(lay["matte"]["layer"])
            if mt is not None:
                m = mt.mask(canvas, 8)
                if lay["matte"]["inverted"]:
                    m = ~m
                x0, y0 = pl.bbox[0], pl.bbox[1]
                sub = m[y0:y0 + im.height, x0:x0 + im.width]
                a = np.asarray(im).copy()
                a[..., 3] = np.where(sub, a[..., 3], 0)
                im = Image.fromarray(a)
        if mattes and lay.get("opacity") is not None and lay["opacity"] < 1.0:
            a = np.asarray(im).copy()
            a[..., 3] = (a[..., 3] * lay["opacity"]).astype(np.uint8)
            im = Image.fromarray(a)
        out.alpha_composite(im, (pl.bbox[0], pl.bbox[1]))
    return out


def compare(a: Image.Image, b: Image.Image, tol: int = 16) -> dict:
    A = np.asarray(a.convert("RGBA")).astype(np.int16)
    B = np.asarray(b.convert("RGBA")).astype(np.int16)
    cov = (A[..., 3] > 8) | (B[..., 3] > 8)
    d = (np.abs(A - B).max(axis=2) > tol) & cov
    return {"covered_px": int(cov.sum()), "differing_px": int(d.sum()),
            "differing_pct": round(100 * float(d.sum()) / max(1, int(cov.sum())), 2),
            "coverage_mismatch_px": int(((A[..., 3] > 128) != (B[..., 3] > 128)).sum())}


def measure_mattes(layers: list[dict], plates: dict[str, Plate], canvas: tuple[int, int],
                   authored: np.ndarray, nocut: np.ndarray,
                   see_through: list[np.ndarray] | None = None) -> dict[str, dict]:
    """Which baked plate cuts each layer, found by fit against the authored truth.

    Top layer first. A layer's visible region is what the already-matted layers
    above it leave uncovered; its cut region is the part of that where the
    authored truth no longer shows it. The plate (or its complement) whose mask
    keeps the surviving part and drops the cut part is the matte. Going top-down
    matters: the doctor's raw lower lid covers most of the jaw, and fitting the
    jaw before the lid was cut back to the eye found a matte for a sliver.
    """
    masks = {eid: p.mask(canvas, 8) for eid, p in plates.items()}
    changed = (np.abs(authored - nocut).max(axis=2) > MATCH)
    # Under a see-through layer every pixel changes between the two truths, and
    # that is opacity, not cutting: the robot's whole face sits under its dome.
    for m in see_through or []:
        changed &= ~m
    # A black lid over black hair cannot be seen either way, so it says nothing
    # about its matte. Only pixels where the layer differs from what lies under
    # it are observable, and only those are fitted.
    observable: dict[str, np.ndarray] = {}
    below = Image.new("RGBA", canvas, (0, 0, 0, 0))
    for lay in layers:
        pl = plates[lay["id"]]
        x0, y0 = pl.bbox[:2]
        a = pl.arr
        h, w = a.shape[:2]
        B = np.asarray(below).astype(np.int16)[y0:y0 + h, x0:x0 + w]
        vis = (B[..., 3] <= SOLID) | (np.abs(a[..., :3] - B[..., :3]).max(axis=2) > TELL)
        m = np.zeros((canvas[1], canvas[0]), bool)
        m[y0:y0 + h, x0:x0 + w] = vis
        observable[lay["id"]] = m
        below.alpha_composite(pl.image, (x0, y0))
    covered = np.zeros((canvas[1], canvas[0]), bool)
    out = {}
    for lay in reversed(layers):
        eid = lay["id"]
        solid = plates[eid].mask(canvas)
        own = solid & ~covered
        matted = solid
        obs = own & observable[eid]
        if obs.sum() >= 100:
            cut = obs & changed
            uncut = obs & ~changed
            frac = cut.sum() / obs.sum()
            if frac >= HIDDEN_FRAC:
                # not a matte: a drawing the rig keeps for later - a closed lid,
                # a hand behind a hand - and hides at rest
                out[eid] = {"hidden_at_rest": True, "cut_fraction": round(float(frac), 3),
                            "method": "this layer's observable pixels are all gone in _ALL"}
                matted = np.zeros_like(solid)
            elif frac >= MIN_MATTE_FRAC:
                best = None
                for oid, m in masks.items():
                    if oid == eid:
                        continue
                    for inv in (False, True):
                        mm = ~m if inv else m
                        # a matte keeps what is inside it: the cut region must fall
                        # outside, and the part that survived must fall inside. A
                        # plate that never touches the layer explains the cut for
                        # free, so both halves are required.
                        explained = (cut & ~mm).sum() / cut.sum()
                        kept = (uncut & mm).sum() / max(1, uncut.sum())
                        score = min(explained, kept)
                        if best is None or score > best[0]:
                            best = (score, oid, inv, float(explained), float(kept))
                if best and best[0] >= MIN_MATTE_FIT:
                    out[eid] = {"layer": best[1], "inverted": best[2],
                                "cut_fraction": round(float(frac), 3),
                                "fit": round(float(best[0]), 3),
                                "method": "plate whose mask best explains the region this layer "
                                          "loses between _ALL_NOCUT and _ALL, fitted top layer first"}
                    mm = ~masks[best[1]] if best[2] else masks[best[1]]
                    matted = solid & mm
        covered |= matted
    return out


def measure_pivot(child: np.ndarray, parent: np.ndarray,
                  near: list[int] | None = None) -> tuple[list[int], str]:
    """Where a part meets its parent: the proximal end of their overlap.

    A part has a free end and a joined end. The free end is the point of its art
    farthest from the parent's centre - the hand of an arm, the foot of a leg,
    the tip of an ear. The joint is at the other end, so the pivot is the centre
    of the quarter of the overlap with the parent's art that lies farthest from
    the free end. That finds the shoulder on an arm that also runs down the side
    of the chest, the hip on a leg that a stitches overlay covers, and the knee
    rather than the ankle on a shin that a boot hangs from too. With no overlap,
    the point of the child's art nearest the parent's centre; with no parent,
    the bottom-centre.
    """
    both = child & parent
    cy, cx = np.where(child)
    if not len(cy):
        return [0, 0], "no art"
    if both.sum() >= 20:
        py, px = np.where(parent)
        pcx, pcy = px.mean(), py.mean()
        far = int(np.argmax((cx - pcx) ** 2 + (cy - pcy) ** 2))
        fx, fy = cx[far], cy[far]
        ys, xs = np.where(both)
        d = (xs - fx) ** 2 + (ys - fy) ** 2
        k = max(20, int(len(d) * 0.25))
        pick = np.argsort(-d)[:k]
        return [int(round(xs[pick].mean())), int(round(ys[pick].mean()))], \
            "centre of the quarter of the overlap with parent art farthest from the part's free end"
    if parent.any():
        py, px = np.where(parent)
        pcx, pcy = px.mean(), py.mean()
        k = int(np.argmin((cx - pcx) ** 2 + (cy - pcy) ** 2))
        return [int(cx[k]), int(cy[k])], "nearest point of art to the parent's centre (no overlap)"
    return [int((cx.min() + cx.max()) // 2), int(cy.max())], "bottom-centre of art (root)"


def measure_opacity(layers: list[dict], plates: dict[str, Plate], canvas: tuple[int, int],
                    authored: np.ndarray) -> dict[str, float]:
    """Layers the rig draws see-through, found by fitting a blend to the truth.

    A plate bakes opaque even when the rig composites it at partial opacity -
    the robot's glass dome. Over the pixels where the layer sits on something
    opaque, the authored truth is alpha*layer + (1-alpha)*below; the least-squares
    alpha says how see-through it was drawn.
    """
    out = {}
    below = Image.new("RGBA", canvas, (0, 0, 0, 0))
    for lay in layers:
        pl = plates[lay["id"]]
        x0, y0 = pl.bbox[:2]
        a = pl.arr
        h, w = a.shape[:2]
        B = np.asarray(below).astype(np.int16)[y0:y0 + h, x0:x0 + w]
        T = authored[y0:y0 + h, x0:x0 + w]
        m = (a[..., 3] > SOLID) & (B[..., 3] > SOLID) & (T[..., 3] > SOLID)
        if m.sum() >= 2000:
            L = a[m][:, :3].astype(np.float64)
            Bm = B[m][:, :3].astype(np.float64)
            Tm = T[m][:, :3].astype(np.float64)
            # per-pixel alpha where the layer and what is under it differ enough
            # to say anything; the median survives a gradient and a few cut pixels
            diff = L - Bm
            tell = np.abs(diff).max(axis=1) > 40
            if tell.sum() >= 1000:
                per = ((Tm - Bm) * diff).sum(axis=1)[tell] / (diff ** 2).sum(axis=1)[tell]
                alpha = float(np.clip(np.median(per), 0.0, 1.0))
                pred = alpha * L + (1 - alpha) * Bm
                err = np.abs(pred - Tm).max(axis=1)[tell]
                err_opaque = np.abs(L - Tm).max(axis=1)[tell]
                tol = 2 * MATCH
                # below 0.1 the layer is simply not there, which is an order or
                # matte question, not an opacity
                # a real see-through layer is wrong almost everywhere when drawn
                # opaque, and right most places when blended
                if 0.1 <= alpha < 0.9 and (err < tol).mean() > 0.6 and (err_opaque < tol).mean() < 0.25:
                    out[lay["id"]] = round(alpha, 3)
        below.alpha_composite(pl.image, (x0, y0))
    return out


# ------------------------------------------------------------------ build

def build(name: str, *, write: bool = True) -> dict:
    r = load_rig(name)
    man = manifest(r)
    canvas = tuple(man["resolution"])
    d = r["_dir"] / "plates"
    plates = plates_of(r, man)
    nocut = np.asarray(Image.open(d / man["ground_truth_nocut"]).convert("RGBA")).astype(np.int16)
    authored = np.asarray(Image.open(d / man["ground_truth"]).convert("RGBA")).astype(np.int16)
    if nocut.shape[1::-1] != canvas:
        raise ToolError(f"{man['ground_truth_nocut']} is {nocut.shape[1::-1]}, canvas is {canvas}")

    print(f"{name}: {len(plates)} plates on a {canvas[0]}x{canvas[1]} canvas")
    instance_of: dict[str, tuple] = {}
    order = None  # measured after instances are split below

    # parenting: the scene's peg hierarchy when there is one
    hier, source = None, "overlap geometry"
    scene = r["_dir"] / "scene" / man["scene"]
    if scene.exists():
        hier = scene_hierarchy(scene)
        source = f"peg hierarchy of {common.rel(scene)} (Harmony)"
    bones: dict[str, dict] = {}
    layer_bone: dict[str, str] = {}
    instances: collections.Counter = collections.Counter()
    read_pegs: dict[str, list[str]] = collections.defaultdict(list)
    if hier:
        for rk, rd in hier["reads"].items():
            eid = rd["element"]
            if eid in plates and rd["peg"]:
                instances[eid] += 1
                read_pegs[eid].append(rd["peg"])
        # An element the scene draws twice - both arms, both legs, both ears - bakes
        # as one plate holding both. Where the plate falls into as many separate
        # blobs as the scene has instances, each blob becomes its own layer on its
        # own peg, so the limbs can move apart. Blobs are matched to pegs left to
        # right in the order the scene lists them, which is an assumption the
        # scene's transforms could contradict; it is recorded as such.
        def natural(k: str):
            return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", k)]

        for eid, n in list(instances.items()):
            if n < 2:
                continue
            # Harmony names a duplicated peg with a _1 suffix, and a rig author
            # duplicates a whole limb at once, so sorting the pegs naturally puts
            # every element's copies on the same side: Glove_P with Upper_Arm_P,
            # Glove_P_1 with Upper_Arm_P_1.
            read_pegs[eid].sort(key=natural)
            blobs = split_instances(plates[eid], n)
            if not blobs:
                continue
            for k, (bbox, sub) in enumerate(blobs):
                key_ = f"{eid}#{k + 1}"
                plates[key_] = sub
                layer_bone[key_] = read_pegs[eid][k]
                instance_of[key_] = (eid, k + 1, bbox)
            del plates[eid]
        for eid, pegs_ in read_pegs.items():
            if eid in plates:
                layer_bone.setdefault(eid, pegs_[0])
        used = set(layer_bone.values())
        # keep every peg on the path from a used peg to the root
        keep = set()
        for pk in used:
            while pk and pk not in keep:
                keep.add(pk)
                pk = hier["pegs"].get(pk, {}).get("parent")
        for pk in keep:
            bones[pk] = {"name": hier["pegs"][pk]["name"], "parent": hier["pegs"][pk]["parent"]}
        # collapse pass-through pegs: no layer, exactly one child
        children = collections.defaultdict(list)
        for pk, b in bones.items():
            children[b["parent"]].append(pk)
        for pk in list(bones):
            if pk not in used and len(children[pk]) == 1 and bones[pk]["parent"]:
                child = children[pk][0]
                bones[child]["parent"] = bones[pk]["parent"]
                children[bones[pk]["parent"]].remove(pk)
                children[bones[pk]["parent"]].append(child)
                del bones[pk]
    for eid in plates:
        if eid not in layer_bone:
            # no peg: hang it from the layer it overlaps most, else the root
            m = plates[eid].mask(canvas, 8)
            best = max(((int((m & plates[o].mask(canvas, 8)).sum()), o)
                        for o in plates if o != eid and o in layer_bone), default=(0, None))
            layer_bone[eid] = layer_bone[best[1]] if best[0] > 0 and best[1] else "root"
    if "root" in layer_bone.values() or not bones:
        bones.setdefault("root", {"name": "root", "parent": None})
        for b in bones.values():
            if b["parent"] is None and b["name"] != "root":
                b["parent"] = "root"
    print(f"  {len(bones)} bones from {source}")
    order, ostat = measure_order(plates, nocut)
    print(f"  draw order by peeling the truth; {len(ostat['hidden'])} layer(s) fully hidden at rest"
          + (f", {len(instance_of)} instance layers split" if instance_of else ""))

    # pivots: each bone's art is the union of its subtree's layers
    subtree_layers: dict[str, set] = collections.defaultdict(set)
    for eid, bk in layer_bone.items():
        pk = bk
        while pk:
            subtree_layers[pk].add(eid)
            pk = bones.get(pk, {}).get("parent")
    masks = {eid: plates[eid].mask(canvas) for eid in plates}
    # An element drawn several times whose plate did not split - a shadow laid
    # over the whole torso from three pegs - is an overlay, not a limb. It would
    # put a leg's free end on the far side of the chest, so it says nothing
    # about where joints are.
    overlay = {eid for eid, n in instances.items() if n > 1 and eid in plates}

    def union(ids) -> np.ndarray:
        m = np.zeros((canvas[1], canvas[0]), bool)
        for i in ids:
            if i not in overlay:
                m |= masks[i]
        return m

    def depth(pk):
        n, p = 0, bones[pk]["parent"]
        while p in bones:
            n, p = n + 1, bones[p]["parent"]
        return n

    own_layers = {pk: {e for e, k in layer_bone.items() if k == pk} for pk in bones}
    all_ids = set(plates)

    def parent_art_levels(pk):
        """What a bone's art is measured against, most trustworthy first: the
        parent's own layers, then each ancestor's own layers, then everything
        else in the rig. Siblings are not set apart: a leg next to its twin
        finds the hip anyway, because the joint is taken at the end of the
        overlap farthest from the leg's free end, and the twin only touches it
        at the feet."""
        mine = subtree_layers[pk]
        anc, p = [], bones[pk]["parent"]
        while p in bones:
            anc.append(p)
            p = bones[p]["parent"]
        levels = [own_layers[a] for a in anc]
        levels.append(all_ids - mine)
        return [l - mine for l in levels if l - mine]

    for pk in sorted(bones, key=depth):
        b = bones[pk]
        own = union(subtree_layers[pk])
        parent = b["parent"]
        # nearness is to the parent art's own centre, never to the parent's pivot:
        # a pivot inherited from the root's bottom-centre would drag every joint
        # down the leg toward the floor
        near = None
        b["pivot"], b["pivot_method"] = None, None
        for level in parent_art_levels(pk):
            par = union(level)
            if (own & par).sum() >= 20:
                b["pivot"], b["pivot_method"] = measure_pivot(own, par, near)
                break
        if b["pivot"] is None:
            levels = parent_art_levels(pk)
            par = union(levels[0]) if levels else np.zeros_like(own)
            b["pivot"], b["pivot_method"] = measure_pivot(own, par, near)
        b["layers"] = sorted(own_layers[pk], key=_idkey)

    layers = []
    for z, eid in enumerate(order):
        base = instance_of[eid][0] if eid in instance_of else eid
        e = man["elements"][base]
        lay = {"id": eid, "name": e["name"], "file": f"plates/{e['file']}", "bbox": e["bbox"],
               "bone": layer_bone[eid], "z": z}
        if eid in instance_of:
            _, k, bbox = instance_of[eid]
            lay.update({"element": base, "instance": k, "name": f"{e['name']}#{k}", "bbox": bbox,
                        "crop": [bbox[0] - e["bbox"][0], bbox[1] - e["bbox"][1],
                                 bbox[2] - e["bbox"][0], bbox[3] - e["bbox"][1]],
                        "crop_note": "this blob of the shared plate, in plate pixels; the scene "
                                     "draws the element more than once and the plate holds every copy"})
        if e.get("library") and eid not in instance_of:
            lay["library"] = {k: ({"file": f"plates/drawings/{v['file']}", "bbox": v["bbox"]} if v else None)
                              for k, v in e["library"].items()}
        if instances[base] > 1 and eid not in instance_of:
            lay["instances"] = instances[base]
            lay["instances_note"] = ("the scene draws this element more than once but the plate "
                                     "does not fall into that many blobs, so the copies move together")
        layers.append(lay)
    opac = measure_opacity(layers, plates, canvas, authored)
    for lay in layers:
        if lay["id"] in opac:
            lay["opacity"] = opac[lay["id"]]
            lay["opacity_method"] = "least-squares blend of the plate over what lies below it, fitted to _ALL"
    if opac:
        print(f"  {len(opac)} see-through layer(s): " + ", ".join(
            f"{next(l['name'] for l in layers if l['id'] == k)} {v:.2f}" for k, v in opac.items()))
    mattes = measure_mattes(layers, plates, canvas, authored, nocut,
                            see_through=[plates[k].mask(canvas, 8) for k in opac])
    hidden = 0
    for lay in layers:
        rec = mattes.get(lay["id"])
        if not rec:
            continue
        if rec.get("hidden_at_rest"):
            lay["hidden_at_rest"] = True
            lay["hidden_note"] = rec["method"]
            hidden += 1
        else:
            lay["matte"] = rec
    print(f"  {len(mattes) - hidden} mattes reconstructed, {hidden} layer(s) hidden at rest")

    # angles and hands, as discrete sets a frame can pick from
    angles = None
    pm = r["_dir"] / "poses" / "manifest.json"
    if pm.exists():
        pd = json.loads(pm.read_text())
        angles = {"source": "poses/manifest.json", "resolution": pd["resolution"],
                  "usable": sorted(int(k) for k, v in pd["poses"].items() if v["usable"]),
                  "default": pd.get("default", 1),
                  "note": "whole-figure head/body plate pairs per view angle, at the poses "
                          "resolution; a frame at an angle other than the default swaps the "
                          "entire layer stack for that pair"}
    hands = {}
    for group in (r.get("layers", {}).get("hands") or []):
        lead = str(group[0])
        lib = man["elements"].get(lead, {}).get("library") or {}
        if lib:
            hands[lead] = {"drawings": sorted(lib), "with": [str(i) for i in group[1:]]}

    # Follow-through: which bones trail their parents, by how many frames. A
    # per-bone number so it can be tuned; the defaults come from the bone's name.
    import motion
    lag = {}
    for pk, b in bones.items():
        low = b["name"].lower()
        for pat, frames in motion.LAG_DEFAULT_FRAMES.items():
            if pat in low:
                lag[pk] = frames
                break

    # Gaze: where the pupils are and what they look like. A rig with pupil
    # layers moves those; one without gets a pupil drawn in the eye's own
    # outline colour at a size measured from its eye box.
    gaze = None
    eye_boxes = [[int(v * canvas[0] / r["render"]["resolution"][0]) for v in box]
                 for box in r.get("eyes", [])]
    pupil_layers = [l["id"] for l in layers if "pupil" in l["name"].lower()]
    if pupil_layers:
        gaze = {"source": "layer", "layers": pupil_layers,
                "note": "the rig has pupil drawings; darts move their bones"}
    elif eye_boxes:
        head_plate = r["_dir"] / "plates" / f"plate-head-{r['render']['resolution'][0]}.png"
        colour = [0, 0, 0]
        if head_plate.exists():
            hp = np.asarray(Image.open(head_plate).convert("RGBA")).astype(int)
            x0, y0, x1, y1 = r["eyes"][0]
            pad = 6
            sub = hp[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
            op = sub[sub[..., 3] > 128][:, :3]
            if len(op):
                colour = [int(v) for v in op[np.argmin(op.sum(axis=1))]]
        gaze = {"source": "drawn",
                "pupils": [{"centre": [(b[0] + b[2]) // 2, (b[1] + b[3]) // 2],
                            "radius": max(4, int(0.22 * (b[3] - b[1])))} for b in eye_boxes],
                "colour": colour,
                "note": "no pupil drawing in the rig; a disc is drawn in the eye's darkest "
                        "colour at 22% of the eye box height, canvas pixels"}

    puppet = {
        "generated_by": "puppet.py build - measured from plates/, do not edit by hand",
        "canvas": list(canvas),
        "ground_truth": {"nocut": f"plates/{man['ground_truth_nocut']}",
                         "authored": f"plates/{man['ground_truth']}"},
        "parenting_source": source,
        "bones": bones, "layers": layers, "angles": angles, "hands": hands,
        "lag": lag, "gaze": gaze,
    }
    # verify before writing, and record the numbers next to the data
    lay_plates = {l["id"]: plates[l["id"]] for l in layers}
    rest = composite(layers, lay_plates, canvas, mattes=False)
    vs_nocut = compare(rest, Image.fromarray(nocut.astype(np.uint8)))
    matted = composite(layers, lay_plates, canvas, mattes=True)
    vs_auth = compare(matted, Image.fromarray(authored.astype(np.uint8)))
    puppet["measured"] = {"recomposite_vs_nocut": vs_nocut, "matted_vs_authored": vs_auth}
    print(f"  recomposite vs _ALL_NOCUT: {vs_nocut['differing_pct']}% of covered pixels differ")
    print(f"  with mattes vs _ALL:       {vs_auth['differing_pct']}% of covered pixels differ")
    if write:
        r2 = json.loads(rig_path(name).read_text())
        r2["puppet"] = puppet
        common.write_json(rig_path(name), r2)
        print(f"  wrote puppet block to {common.rel(rig_path(name))}")
    return puppet


def load_puppet(name: str) -> tuple[dict, dict, dict[str, Plate]]:
    r = load_rig(name)
    pp = r.get("puppet")
    if not pp:
        raise ToolError(f"rig {name!r} has no puppet block - run: uv run puppet.py build {name}")
    plates = {}
    for l in pp["layers"]:
        pl = Plate(r["_dir"] / l["file"], l["bbox"])
        if l.get("crop"):
            full = Plate(r["_dir"] / l["file"], [0, 0, 0, 0]).arr
            c = l["crop"]
            pl._arr = full[c[1]:c[3] + 1, c[0]:c[2] + 1].copy()
        plates[l["id"]] = pl
    return r, pp, plates


def verify(name: str) -> dict:
    r, pp, plates = load_puppet(name)
    canvas = tuple(pp["canvas"])
    out = {}
    for tag, key, mattes in (("nocut", "nocut", False), ("authored", "authored", True)):
        truth = Image.open(r["_dir"] / pp["ground_truth"][key])
        im = composite(pp["layers"], plates, canvas, mattes=mattes, plate_dir=r["_dir"] / "plates")
        out[tag] = compare(im, truth)
        dest = config.CACHE_DIR / f"puppet-{name}-{tag}"
        dest.mkdir(parents=True, exist_ok=True)
        im.save(dest / "recomposite.png")
        A = np.asarray(im.convert("RGBA")).astype(np.int16)
        B = np.asarray(truth.convert("RGBA")).astype(np.int16)
        Image.fromarray(np.clip(np.abs(A - B).max(axis=2) * 4, 0, 255).astype(np.uint8)).save(dest / "diff.png")
        print(f"  {tag:9s} {out[tag]['differing_pct']:5.2f}% differ, "
              f"{out[tag]['coverage_mismatch_px']} px coverage mismatch  -> {common.rel(dest)}")
    return out


# ------------------------------------------------------------------ Blender

def blender_job(name: str, out_dir: Path, *, frames: int = 1, crop: list[int] | None = None,
                drawings: dict[str, str] | None = None, animation: dict | None = None) -> Path:
    """Write the job blenderpuppet.py builds a scene from."""
    r, pp, plates = load_puppet(name)
    chosen = drawings or {}
    layers = []
    for lay in pp["layers"]:
        if lay.get("hidden_at_rest") and lay["id"] not in chosen:
            continue
        file, bbox = lay["file"], lay["bbox"]
        if lay["id"] in chosen and lay.get("library"):
            ent = lay["library"].get(chosen[lay["id"]])
            if ent is None:
                continue
            file, bbox = ent["file"], ent["bbox"]
        entry = {"id": lay["id"], "name": lay["name"], "file": str(r["_dir"] / file),
                 "bbox": bbox, "bone": lay["bone"], "z": lay["z"]}
        if lay.get("crop"):
            with Image.open(r["_dir"] / file) as im:
                entry["plate_size"] = list(im.size)
            entry["crop"] = lay["crop"]
        if lay.get("opacity") is not None:
            entry["opacity"] = lay["opacity"]
        if lay.get("matte"):
            m = next(l for l in pp["layers"] if l["id"] == lay["matte"]["layer"])
            mbox = list(m["bbox"])
            if m.get("crop"):
                # an instance matte is one blob of a shared plate; Blender samples
                # the whole plate, so give it the plate's own canvas box
                with Image.open(r["_dir"] / m["file"]) as im:
                    pw, ph = im.size
                mbox = [m["bbox"][0] - m["crop"][0], m["bbox"][1] - m["crop"][1], 0, 0]
                mbox[2], mbox[3] = mbox[0] + pw - 1, mbox[1] + ph - 1
            entry["matte"] = {"file": str(r["_dir"] / m["file"]), "bbox": mbox,
                              "inverted": lay["matte"]["inverted"]}
        layers.append(entry)
    if crop is None:
        xs = [l["bbox"][0] for l in layers] + [l["bbox"][2] for l in layers]
        ys = [l["bbox"][1] for l in layers] + [l["bbox"][3] for l in layers]
        crop = [min(xs), min(ys), max(xs) + 1, max(ys) + 1]
    job = {"rig": name, "out_dir": str(out_dir), "frames": frames, "crop": crop,
           "canvas": pp["canvas"], "bones": pp["bones"], "layers": layers,
           "animation": animation or {}, "samples": 8, "filter_size": 1.0}
    out_dir.mkdir(parents=True, exist_ok=True)
    job_path = out_dir / "job.json"
    common.write_json(job_path, job)
    return job_path


def render(name: str, out_dir: Path | None = None, *, frames: int = 1,
           animation: dict | None = None, drawings: dict | None = None,
           scale: float = 0.5) -> dict:
    """Build the puppet in Blender and render it, at rest by default."""
    import shutil
    out_dir = out_dir or config.CACHE_DIR / f"puppet-{name}-blender"
    for old in out_dir.glob("rgba-*.png"):
        old.unlink()
    job_path = blender_job(name, out_dir, frames=frames, animation=animation, drawings=drawings)
    job = json.loads(job_path.read_text())
    job["scale"] = scale
    common.write_json(job_path, job)
    exe = None
    for c in (config.BLENDER,):
        if c and (shutil.which(c) or Path(c).exists()):
            exe = c
    if not exe:
        raise ToolError(f"no Blender at {config.BLENDER!r}; brew install --cask blender, or BLENDER_BIN=...")
    script = config.ROOT / "blenderpuppet.py"
    proc = common.run([exe, "-b", "--factory-startup", "-noaudio", "-P", str(script), "--", str(job_path)],
                      check=False)
    tail = "\n".join((proc.stdout or "").splitlines()[-20:])
    if proc.returncode != 0 or "[puppet] rendered" not in (proc.stdout or ""):
        raise ToolError(f"Blender failed ({proc.returncode}):\n{tail}\n{(proc.stderr or '')[-1500:]}")
    written = sorted(out_dir.glob("rgba-*.png"))
    info = json.loads((out_dir / "render.json").read_text())
    info["frames_written"] = [str(p) for p in written]
    return info


# ------------------------------------------------------------------ CLI

def cmd_build(a) -> int:
    for name in a.rig:
        build(name)
    return 0


def cmd_verify(a) -> int:
    for name in a.rig:
        print(name)
        verify(name)
    return 0


def cmd_tree(a) -> int:
    r, pp, _ = load_puppet(a.rig)
    kids = collections.defaultdict(list)
    for k, b in pp["bones"].items():
        kids[b["parent"]].append(k)
    by_bone = collections.defaultdict(list)
    for lay in pp["layers"]:
        by_bone[lay["bone"]].append(lay)

    def show(k, depth):
        b = pp["bones"][k]
        print("  " * depth + f"{b['name']}  pivot {b['pivot']}  ({b['pivot_method']})")
        for lay in by_bone[k]:
            extra = []
            if lay.get("library"):
                extra.append(f"{len(lay['library'])} drawings")
            if lay.get("matte"):
                extra.append(f"matte {lay['matte']['layer']}{' inv' if lay['matte']['inverted'] else ''}")
            if lay.get("instances"):
                extra.append(f"x{lay['instances']}")
            print("  " * depth + f"  - z{lay['z']:2d} {lay['name']}" + (f"  [{', '.join(extra)}]" if extra else ""))
        for c in sorted(kids[k]):
            show(c, depth + 1)

    roots = [k for k, b in pp["bones"].items() if not b["parent"] or b["parent"] not in pp["bones"]]
    for k in roots:
        show(k, 0)
    print(f"parenting: {pp['parenting_source']}")
    if pp.get("angles"):
        print(f"angles: {pp['angles']['usable']}")
    for lead, h in (pp.get("hands") or {}).items():
        print(f"hands {lead}: {', '.join(h['drawings'])}")
    return 0


def cmd_render(a) -> int:
    r, pp, plates = load_puppet(a.rig)
    info = render(a.rig, frames=a.frames, scale=a.scale)
    print(f"  Blender {info['blender']}: {info['frames']} frames at "
          f"{info['resolution'][0]}x{info['resolution'][1]} in {info['render_seconds']}s")
    first = Image.open(info["frames_written"][0])
    truth = Image.open(r["_dir"] / pp["ground_truth"]["authored"]).convert("RGBA")
    job = json.loads((Path(info["frames_written"][0]).parent / "job.json").read_text())
    c = job["crop"]
    ref = truth.crop((c[0], c[1], c[2], c[3]))
    if first.size != ref.size:
        ref = ref.resize(first.size, Image.LANCZOS)
    cmp = compare(first, ref)
    print(f"  rest frame vs _ALL (authored, scaled {a.scale}): {cmp['differing_pct']}% differ, "
          f"{cmp['coverage_mismatch_px']} px coverage mismatch")
    pil = composite(pp["layers"], plates, tuple(pp["canvas"]), mattes=True,
                    plate_dir=r["_dir"] / "plates").crop((c[0], c[1], c[2], c[3]))
    if first.size != pil.size:
        pil = pil.resize(first.size, Image.LANCZOS)
    cmp2 = compare(first, pil)
    print(f"  rest frame vs the Pillow recomposite of the same layers: {cmp2['differing_pct']}% differ, "
          f"{cmp2['coverage_mismatch_px']} px coverage mismatch")
    print(f"  frames in {common.rel(Path(info['frames_written'][0]).parent)}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="measure the puppet and write rig.json['puppet']")
    b.add_argument("rig", nargs="+")
    b.set_defaults(func=cmd_build)
    v = sub.add_parser("verify", help="recomposite the puppet against both ground truths")
    v.add_argument("rig", nargs="+")
    v.set_defaults(func=cmd_verify)
    t = sub.add_parser("tree", help="print bones, pivots and layers")
    t.add_argument("rig")
    t.set_defaults(func=cmd_tree)
    rd = sub.add_parser("render", help="build the puppet in Blender and render it at rest")
    rd.add_argument("rig")
    rd.add_argument("--frames", type=int, default=1)
    rd.add_argument("--scale", type=float, default=0.5, help="render scale of the plate canvas")
    rd.set_defaults(func=cmd_render)
    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
