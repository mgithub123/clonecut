"""Stage 8b - Acting: a performance on the puppet's bones.

Turns the voice analysis and the mouth track into per-frame tracks on the
armature, and renders them through Blender. The principles come from motion.py;
this file decides which bone gets which curve:

  body     beat hits (anticipation, drop, overshoot, settle), breathing,
           squash and stretch on the hit, the phrase-end sink
  head     the same a frame late and a little larger, plus slow drift, the
           phrase-end tilt, and the counter-motion of a settle
  ears, hair, hands, tail   follow-through: the parent's motion, late and
           damped, by the per-bone lag in rig.json
  eyes     blink by scaling the eye layers about their own centre; pupils
           dart toward a head turn a few frames before it happens
  mouth    one sprite per shape, at the face anchor, shown by the mouth track
  turns    the head steps through the baked view angles; a turned frame shows
           that angle's head/body plates in place of the layer stack

Output is what perform.build_frames() produces: RGBA frames cropped to the
rig's crop box at render resolution, so perform.composite() and mux() finish
them exactly as before.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import common
import config
import motion
import puppet
from common import ToolError

FPS = motion.FPS


# ------------------------------------------------------------------ bones

def _subtree(bones: dict, key: str) -> set:
    out, stack = set(), [key]
    while stack:
        k = stack.pop()
        out.add(k)
        stack += [c for c, b in bones.items() if b["parent"] == k]
    return out


def _lca(bones: dict, keys: list[str]) -> str | None:
    """Lowest common ancestor of some bones."""
    def chain(k):
        out = []
        while k in bones:
            out.append(k)
            k = bones[k]["parent"]
        return out
    chains = [chain(k) for k in keys if k in bones]
    if not chains:
        return None
    common_ = set(chains[0])
    for c in chains[1:]:
        common_ &= set(c)
    for k in chains[0]:
        if k in common_:
            return k
    return None


def cast(r: dict, pp: dict) -> dict:
    """Which bone plays which part, found from the rig's layer lists."""
    bones = pp["bones"]
    layer_bone = {l["id"]: l["bone"] for l in pp["layers"]}
    base = lambda lid: lid.split("#")[0]
    head_ids = {str(i) for i in r["layers"]["head"]}
    body_ids = {str(i) for i in r["layers"]["body"]}
    head_bones = [b for lid, b in layer_bone.items() if base(lid) in head_ids]
    body_bones = [b for lid, b in layer_bone.items() if base(lid) in body_ids]
    root = next(k for k, b in bones.items() if not b["parent"] or b["parent"] not in bones)
    head = _lca(bones, head_bones) or root
    body = _lca(bones, body_bones + [head]) or root
    if body == head:
        body = bones[head]["parent"] or root
    eye_ids = [lid for lid in layer_bone if base(lid) == str(r["layers"].get("eyes"))]
    hands = [layer_bone[lid] for lid in layer_bone
             if base(lid) in {str(g[0]) for g in r["layers"].get("hands", [])}]
    return {"root": root, "body": body, "head": head, "eyes": eye_ids, "hands": hands}


# ------------------------------------------------------------------ tracks

def tracks(r: dict, pp: dict, an: dict, track: list[str], n: int, *,
           blink: bool, poses: list[int]) -> dict:
    """Every per-frame curve the shot needs, in render-resolution pixels."""
    parts = cast(r, pp)
    beats = (an.get("beat") or {}).get("beats") or []
    hit_times = beats[::2] or [i / 2 for i in range(int(an["duration"] * 2))]
    hits = [int(round(t * FPS)) for t in hit_times if 0 <= t < an["duration"]]
    onsets = [int(round(t * FPS)) for t in an["onsets"]]
    import perform
    ends = perform.phrase_ends(track)

    B = motion.BEAT
    body_beat = motion.impulse(n, hits, pre=B["anticipation_frames"], pre_amp=B["anticipation_px"],
                               amp=B["hit_px"], post_amp=B["overshoot_px"], settle=B["settle_frames"])
    head_beat = B["head_scale"] * motion.lag(body_beat, B["head_delay_frames"], damp=1.0)
    sx, sy = motion.squash_stretch(n, hits)
    body_dy = body_beat + motion.breath(n, motion.BREATH["body_px"]) \
        + motion.settle(n, ends, onsets, amp=motion.SETTLE["body_px"])
    D = motion.DRIFT
    head_dy = head_beat + motion.breath(n, motion.BREATH["head_px"]) \
        + motion.breath(n, D["head_dy_px"], D["head_dy_period_s"], D["head_dy_phase"]) \
        + motion.settle(n, ends, onsets, amp=motion.SETTLE["head_px"])
    head_dx = motion.breath(n, D["head_dx_px"], D["head_dx_period_s"])
    head_rot = motion.breath(n, D["head_rot_deg"], D["head_rot_period_s"], D["head_rot_phase"]) \
        - 0.2 * body_beat + motion.settle(n, ends, onsets, amp=motion.SETTLE["head_tilt_deg"])
    out = {
        "parts": parts, "hits": hits, "ends": ends, "onsets": onsets,
        "body_dy": body_dy, "body_sx": sx, "body_sy": sy,
        "head_dy": head_dy, "head_dx": head_dx, "head_rot": head_rot,
        "blink": motion.blink_track(n) if blink and r["capabilities"].get("blink") else np.ones(n),
        "gaze_dx": motion.gaze_track(n, poses, pp["angles"]["usable"] if pp.get("angles") else [1]),
        "zoom": motion.camera_push(n),
    }
    # follow-through, per lagging bone: the delta from what it would do on time
    lagged = {}
    chain_head = body_dy + head_dy
    # a bone above the head or the body in the tree is not a child that trails;
    # the dog's Head_Collar_Master_P matches "collar" but carries the whole head
    def ancestors(k):
        out, k = set(), pp["bones"].get(k, {}).get("parent")
        while k in pp["bones"]:
            out.add(k)
            k = pp["bones"][k]["parent"]
        return out
    fixed = ancestors(parts["head"]) | ancestors(parts["body"]) | {parts["head"], parts["body"]}
    for key, frames in (pp.get("lag") or {}).items():
        if key not in pp["bones"] or key in fixed:
            continue
        under_head = key in _subtree(pp["bones"], parts["head"])
        chain = chain_head if under_head else body_dy
        rot = head_rot if under_head else np.zeros(n)
        lagged[key] = {"dy": motion.lag(chain, frames) - chain,
                       "rot": motion.lag(rot, frames) - rot}
    out["lagged"] = lagged
    return out


# ------------------------------------------------------------------ the job

def _pupil_png(radius: int, colour: list[int], dest: Path) -> Path:
    d = radius * 2 + 2
    im = Image.new("RGBA", (d, d), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse([1, 1, d - 2, d - 2], fill=tuple(colour) + (255,))
    im.save(dest)
    return dest


def build_job(r: dict, pp: dict, an: dict, track: list[str], out_dir: Path, *,
              blink: bool = True, pose: int | None = None,
              turns: list[tuple[int, int]] | None = None) -> Path:
    import perform
    n = len(track)
    canvas = tuple(pp["canvas"])
    S = canvas[0] / r["render"]["resolution"][0]       # render px -> canvas px
    default_pose = (pp.get("angles") or {}).get("default", 1)
    usable = (pp.get("angles") or {}).get("usable") or [default_pose]
    if pose is not None and pose not in usable:
        raise ToolError(f"rig {r['name']!r} has no usable pose {pose}; usable are {usable}")
    if pose is not None:
        poses = [pose] * n
    else:
        poses = motion.turn_track(n, turns or [], usable, default_pose)
    tr = tracks(r, pp, an, track, n, blink=blink, poses=poses)
    parts = tr["parts"]
    bones = dict(pp["bones"])

    # blink bones: one per eye layer, pivoting at the layer's own centre, so a
    # blink squashes the eye in place; pupils hang from them and carry the dart
    extra_bones, rebind, sprites = {}, {}, []
    z_top = max(l["z"] for l in pp["layers"]) + 1
    for lid in parts["eyes"]:
        lay = next(l for l in pp["layers"] if l["id"] == lid)
        bx0, by0, bx1, by1 = lay["bbox"]
        key = f"blink:{lid}"
        extra_bones[key] = {"name": key, "parent": lay["bone"],
                            "pivot": [(bx0 + bx1) // 2, (by0 + by1) // 2]}
        rebind[lid] = key
    gaze = pp.get("gaze") or {}
    pupil_bones = []
    if gaze.get("source") == "drawn" and parts["eyes"]:
        out_dir.mkdir(parents=True, exist_ok=True)
        png = _pupil_png(gaze["pupils"][0]["radius"], gaze["colour"], out_dir / "pupil.png")
        for i, pu in enumerate(gaze["pupils"]):
            cx, cy = pu["centre"]
            rad = pu["radius"]
            # the blink bone of the eye this pupil sits in
            host = next((f"blink:{lid}" for lid in parts["eyes"]
                         if _inside(next(l for l in pp["layers"] if l["id"] == lid)["bbox"], cx, cy)),
                        f"blink:{parts['eyes'][0]}")
            key = f"pupil:{i}"
            extra_bones[key] = {"name": key, "parent": host, "pivot": [cx, cy]}
            pupil_bones.append(key)
            sprites.append({"name": key, "file": str(png), "bbox": [cx - rad, cy - rad, cx + rad, cy + rad],
                            "bone": key, "z": z_top, "visible": [p == default_pose for p in poses]})
    elif gaze.get("source") == "layer":
        pupil_bones = [next(l["bone"] for l in pp["layers"] if l["id"] == lid) for lid in gaze["layers"]]
    bones.update(extra_bones)

    # mouth sprites, per pose the shot visits, per shape
    lib = perform.mouth_library(r)
    pct = r["mouth_width_pct"]
    for p in sorted(set(poses)):
        g = perform.pose_geometry(r, p) if p != default_pose or (r["_dir"] / "poses" / "manifest.json").exists() and p != 1 else None
        if p == default_pose:
            fa, has_mouth = r["face"], True
        else:
            fa, has_mouth = g["face"], g["has_mouth"]
        if not has_mouth:
            continue
        for name, shape in lib.items():
            tw = int(fa["width"] * pct.get(name, 0.5))
            sc = min(tw / shape.width, fa["max_mouth_height"] / shape.height)
            w, h = max(1, int(shape.width * sc)), max(1, int(shape.height * sc))
            top = fa["anchor_y"] - h if fa.get("anchor", "top") == "bottom" else fa["lip_line"]
            left = fa["centre_x"] - w // 2
            bbox = [int(left * S), int(top * S), int((left + w) * S) - 1, int((top + h) * S) - 1]
            sprites.append({"name": f"mouth:{p}:{name}", "file": str(r["_dir"] / "mouths" / f"{name}.png"),
                            "bbox": bbox, "bone": parts["head"], "z": z_top + 1,
                            "visible": [poses[f] == p and track[f] == name for f in range(n)]})

    # a turned frame shows the pose's own head/body plates instead of the layers
    hide = [p != default_pose for p in poses]
    if any(hide):
        tab = perform.pose_table(r)
        res = tab["resolution"]
        full = [0, 0, canvas[0] - 1, canvas[1] - 1]
        for p in sorted(set(poses) - {default_pose}):
            rec = tab["poses"][str(p)]
            vis = [q == p for q in poses]
            sprites.append({"name": f"pose:{p}:body", "file": str(r["_dir"] / "poses" / rec["body"]),
                            "bbox": full, "bone": parts["body"], "z": z_top - 0.5, "visible": vis})
            sprites.append({"name": f"pose:{p}:head", "file": str(r["_dir"] / "poses" / rec["head"]),
                            "bbox": full, "bone": parts["head"], "z": z_top - 0.25, "visible": vis})

    # bone tracks, every frame, in canvas units (Blender +Y is up)
    anim = {}
    fr = range(1, n + 1)
    anim[parts["body"]] = {
        "location": [[f, 0.0, -float(tr["body_dy"][f - 1]) * S] for f in fr],
        "scale": [[f, float(tr["body_sx"][f - 1]), float(tr["body_sy"][f - 1])] for f in fr],
    }
    anim[parts["head"]] = {
        "location": [[f, float(tr["head_dx"][f - 1]) * S, -float(tr["head_dy"][f - 1]) * S] for f in fr],
        "rotation": [[f, float(tr["head_rot"][f - 1])] for f in fr],
    }
    for key, lg in tr["lagged"].items():
        if key in (parts["body"], parts["head"]):
            continue
        anim[key] = {"location": [[f, 0.0, -float(lg["dy"][f - 1]) * S] for f in fr],
                     "rotation": [[f, float(lg["rot"][f - 1])] for f in fr]}
    for lid in parts["eyes"]:
        anim[f"blink:{lid}"] = {"scale": [[f, 1.0, float(tr["blink"][f - 1])] for f in fr]}
    for key in pupil_bones:
        anim[key] = {"location": [[f, float(tr["gaze_dx"][f - 1]) * S, 0.0] for f in fr]}

    crop = [int(v * S) for v in r["crop"]]
    job_path = puppet.blender_job(r["name"], out_dir, frames=n, crop=crop, animation=anim)
    job = json.loads(job_path.read_text())
    job["bones"] = bones
    # the rig's own mouth drawings stay out: the mouth sprites replace them, as
    # the head plate excluded them in Stage 7 (the dog's includes a tongue that
    # hangs to its chest)
    mouth_ids = {str(r["layers"].get("mouth"))} | {str(i) for i in r["layers"].get("mouth_family", [])}
    job["layers"] = [lay for lay in job["layers"] if lay["id"].split("#")[0] not in mouth_ids]
    for lay in job["layers"]:
        if lay["id"] in rebind:
            lay["bone"] = rebind[lay["id"]]
    job["sprites"] = sprites
    job["hide_layers"] = hide if any(hide) else None
    job["camera_zoom"] = [[f, float(tr["zoom"][f - 1])] for f in fr]
    job["scale"] = 1.0 / S
    job["acting"] = {"hits": tr["hits"], "phrase_ends": tr["ends"], "poses": poses,
                     "parts": parts, "constants": {k: getattr(motion, k) for k in
                                                   ("BEAT", "SQUASH", "BREATH", "DRIFT", "SETTLE",
                                                    "BLINK", "TURN", "CAMERA", "LAG_DAMP")}}
    common.write_json(job_path, job)
    return job_path


def _inside(bbox, x, y) -> bool:
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def build_frames(r: dict, an: dict, track: list[str], out_dir: Path, *,
                 blink: bool = True, pose: int | None = None,
                 turns: list[tuple[int, int]] | None = None) -> list[Path]:
    """The Blender-path twin of perform.build_frames(): same inputs, same output."""
    _, pp, _ = puppet.load_puppet(r["name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("rgba-*.png"):
        old.unlink()
    job_path = build_job(r, pp, an, track, out_dir, blink=blink, pose=pose, turns=turns)
    import shutil
    exe = config.BLENDER if (shutil.which(config.BLENDER) or Path(config.BLENDER).exists()) else None
    if not exe:
        raise ToolError(f"no Blender at {config.BLENDER!r}; brew install --cask blender, or BLENDER_BIN=...")
    proc = common.run([exe, "-b", "--factory-startup", "-noaudio", "-P",
                       str(config.ROOT / "blenderpuppet.py"), "--", str(job_path)], check=False)
    if proc.returncode != 0 or "[puppet] rendered" not in (proc.stdout or ""):
        tail = "\n".join((proc.stdout or "").splitlines()[-20:])
        raise ToolError(f"Blender failed ({proc.returncode}):\n{tail}\n{(proc.stderr or '')[-1500:]}")
    frames = sorted(out_dir.glob("rgba-*.png"))
    if len(frames) != len(track):
        raise ToolError(f"expected {len(track)} frames, Blender wrote {len(frames)}")
    info = json.loads((out_dir / "render.json").read_text())
    print(f"  Blender {info['blender']}: {info['frames']} frames at "
          f"{info['resolution'][0]}x{info['resolution'][1]} in {info['render_seconds']}s")
    return frames


def auto_turns(track: list[str], usable: list[int], start: int, min_gap: int = 8) -> list[tuple[int, int]]:
    """A turn schedule from the phrasing: at every second long breath, turn one
    usable pose over and come back on the following one."""
    import perform
    ends = perform.phrase_ends(track, run=min_gap)
    order = sorted(usable)
    if start not in order or len(order) < 2:
        return []
    i = order.index(start)
    nxt = order[i + 1] if i + 1 < len(order) else order[i - 1]
    sched, away = [], False
    for k, e in enumerate(ends):
        if k % 2 == 0 and not away:
            sched.append((e, nxt))
            away = True
        elif away:
            sched.append((e, start))
            away = False
    return sched
