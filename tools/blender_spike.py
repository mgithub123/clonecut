#!/usr/bin/env python3
"""Stage 8z - The Blender test.

Decides, by measurement rather than argument, whether the puppet engine should be
written in Pillow (what perform.build_frames() is today) or built in Blender.

    uv run tools/blender_spike.py                 # doctor, 24 frames, cache/spike/
    uv run tools/blender_spike.py --rig doctor --frames 24 --out cache/spike

The same file runs on both sides of the fence. Driven from the project venv it
writes a job file, launches Blender headless with itself as the script, and then
finishes the frames through perform.composite() and perform.mux() - the real
finishing path, so the output is a real vertical mp4 and not a Blender preview.
Inside Blender it reads the job file, builds two image planes from the doctor's
baked head and body plates, a two-bone armature (head parented to body), keyframes
a one-second beat bob with Blender's own easing, and renders with Eevee to PNG with
alpha. Nothing on the Blender side imports numpy or Pillow, because Blender's
Python has neither.

What is measured, and printed: wall time for the render, and how far the first
frame - the rest pose - is from the two-plate composite perform.py makes from the
same plates. That number is what says whether Blender can replace the compositor
without changing what the character looks like.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

JOB_NAME = "job.json"
FRAME_STEM = "rgba-"
FPS = 24


def _fcurves(action):
    """Every F-curve of an action, across the flat (4.x) and layered (5.x) APIs."""
    if hasattr(action, "fcurves"):
        return list(action.fcurves)
    out = []
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                out.extend(bag.fcurves)
    return out


# =============================================================== inside Blender

def _blender_side(job_path: Path) -> None:
    import math

    import bpy

    job = json.loads(job_path.read_text())
    out_dir = Path(job["out_dir"])
    n = int(job["frames"])
    crop = job["crop"]                       # [x0, y0, x1, y1] in plate pixels
    cw, ch = crop[2] - crop[0], crop[3] - crop[1]

    # A clean scene. Pixel space is the world: 1 Blender unit = 1 plate pixel,
    # image y runs down so world Y = -pixel y. The camera looks down -Z, so a
    # plane with a larger Z sits in front.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.frame_start, scene.frame_end = 1, n
    scene.render.resolution_x, scene.render.resolution_y = cw, ch
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.filter_size = float(job.get("filter_size", 1.0))
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 15
    scene.render.filepath = str(out_dir / FRAME_STEM)

    # Eevee's identifier moved between releases; take whichever this build has.
    engines = [e.identifier for e in
               bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
    eevee = next((e for e in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE") if e in engines), None)
    if eevee is None:
        raise RuntimeError(f"no Eevee in this Blender; engines are {engines}")
    scene.render.engine = eevee
    scene.eevee.taa_render_samples = int(job.get("samples", 8))

    # The art must come out the colour it went in. AgX/Filmic would grade it.
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    # Orthographic camera framing exactly the rig's crop box.
    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam_data.sensor_fit = "VERTICAL" if ch >= cw else "HORIZONTAL"
    cam_data.ortho_scale = float(max(cw, ch))
    cam_data.clip_start, cam_data.clip_end = 1.0, 10_000.0
    cam = bpy.data.objects.new("cam", cam_data)
    cam.location = ((crop[0] + crop[2]) / 2.0, -(crop[1] + crop[3]) / 2.0, 1000.0)
    cam.rotation_euler = (0.0, 0.0, 0.0)
    scene.collection.objects.link(cam)
    scene.camera = cam

    def image_plane(name: str, png: str, bbox: list[int], z: float) -> bpy.types.Object:
        """One plate as a textured, alpha-blended, unlit plane at its pixel offset."""
        x0, y0, x1, y1 = bbox
        verts = [(x0, -y0, z), (x1, -y0, z), (x1, -y1, z), (x0, -y1, z)]
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
        uv = mesh.uv_layers.new(name="UVMap")
        for loop, co in zip(mesh.loops, [(0, 1), (1, 1), (1, 0), (0, 0)]):
            uv.data[loop.index].uv = co
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        scene.collection.objects.link(obj)

        img = bpy.data.images.load(png)
        img.alpha_mode = "STRAIGHT"
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        # Blended, not dithered: a cutout wants clean sorted transparency, not
        # hashed noise that needs many samples to settle.
        for attr, val in (("surface_render_method", "BLENDED"), ("blend_method", "BLEND")):
            try:
                setattr(mat, attr, val)
            except (AttributeError, TypeError):
                pass
        nt = mat.node_tree
        for node in list(nt.nodes):
            nt.nodes.remove(node)
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        tex.interpolation = job.get("interpolation", "Linear")
        tex.extension = "CLIP"
        emit = nt.nodes.new("ShaderNodeEmission")
        emit.inputs["Strength"].default_value = 1.0
        clear = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader")
        outp = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
        nt.links.new(tex.outputs["Alpha"], mix.inputs["Fac"])
        nt.links.new(clear.outputs[0], mix.inputs[1])
        nt.links.new(emit.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], outp.inputs["Surface"])
        obj.data.materials.append(mat)
        return obj

    body = image_plane("body", job["body"]["file"], job["body"]["bbox"], 0.0)
    head = image_plane("head", job["head"]["file"], job["head"]["bbox"], 1.0)

    # Two bones. Body pivots at the bottom-centre of its art, head at the rig's
    # measured head pivot, head parented to body so the tilt rides the bob.
    arm_data = bpy.data.armatures.new("puppet")
    arm = bpy.data.objects.new("puppet", arm_data)
    scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    bx, by = job["body"]["pivot"]
    hx, hy = job["head"]["pivot"]
    eb_body = arm_data.edit_bones.new("body")
    eb_body.head, eb_body.tail = (bx, -by, 0.0), (bx, -by + 300.0, 0.0)
    eb_head = arm_data.edit_bones.new("head")
    eb_head.head, eb_head.tail = (hx, -hy, 0.0), (hx, -hy + 200.0, 0.0)
    eb_head.parent = eb_body
    eb_head.use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")

    for obj, bone in ((body, "body"), (head, "head")):
        vg = obj.vertex_groups.new(name=bone)
        vg.add(list(range(len(obj.data.vertices))), 1.0, "REPLACE")
        mod = obj.modifiers.new("puppet", "ARMATURE")
        mod.object = arm

    # A one-second beat bob with Blender's easing, and a counter-tilt on the head.
    # Bone-local Y runs along the bone, which points straight up, so Y is the bob
    # axis; bone-local Z is the camera axis, so Z is the in-plane tilt.
    pb_body = arm.pose.bones["body"]
    pb_head = arm.pose.bones["head"]
    pb_head.rotation_mode = "XYZ"
    bob = job.get("bob", [[1, 0.0], [7, -10.0], [13, 0.0], [n, 0.0]])
    tilt = job.get("tilt", [[1, 0.0], [7, 2.5], [13, 0.0], [n, 0.0]])
    for frame, dy in bob:
        pb_body.location = (0.0, float(dy), 0.0)
        pb_body.keyframe_insert("location", frame=int(frame))
    for frame, deg in tilt:
        pb_head.rotation_euler = (0.0, 0.0, math.radians(float(deg)))
        pb_head.keyframe_insert("rotation_euler", frame=int(frame))
    for fc in _fcurves(arm.animation_data.action):
        for kp in fc.keyframe_points:
            kp.interpolation = "BACK"           # the overshoot a beat hit wants
            kp.easing = "EASE_IN_OUT"
            kp.back = 1.4

    out_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / "spike.blend"))

    t0 = time.perf_counter()
    bpy.ops.render.render(animation=True, write_still=False)
    wall = time.perf_counter() - t0
    (out_dir / "render.json").write_text(json.dumps({
        "blender": bpy.app.version_string, "engine": eevee,
        "frames": n, "resolution": [cw, ch], "render_seconds": round(wall, 2),
        "samples": scene.eevee.taa_render_samples,
        "filter_size": scene.render.filter_size,
    }, indent=2))
    print(f"[spike] rendered {n} frames at {cw}x{ch} in {wall:.1f}s "
          f"with {eevee} on Blender {bpy.app.version_string}")


# ============================================================ project side

def _alpha_bbox(im) -> list[int]:
    bb = im.getbbox()
    if not bb:
        raise RuntimeError("plate is empty")
    return [int(v) for v in bb]


def _rest_reference(head, body, crop) -> "Image.Image":
    """What perform.build_frames() draws at rest, minus motion and mouth."""
    from PIL import Image
    canvas = Image.new("RGBA", head.size, (0, 0, 0, 0))
    canvas.alpha_composite(body, (0, 0))
    canvas.alpha_composite(head, (0, 0))
    return canvas.crop(tuple(crop))


def _compare(a, b) -> dict:
    import numpy as np
    A = np.asarray(a.convert("RGBA")).astype(int)
    B = np.asarray(b.convert("RGBA")).astype(int)
    if A.shape != B.shape:
        return {"shape_a": list(A.shape), "shape_b": list(B.shape), "match": False}
    cover = (A[..., 3] > 0) | (B[..., 3] > 0)
    d = np.abs(A - B)
    rgb = d[..., :3].max(axis=2)
    alpha = d[..., 3]
    # Straight-alpha RGB is meaningless where nothing is drawn, so RGB is scored
    # only where both sides are substantially opaque.
    solid = (A[..., 3] > 200) & (B[..., 3] > 200)
    return {
        "pixels_covered": int(cover.sum()),
        "rgb_mean_abs_diff_solid": round(float(rgb[solid].mean()), 3) if solid.any() else None,
        "rgb_pct_over_8_solid": round(100 * float((rgb[solid] > 8).mean()), 3) if solid.any() else None,
        "alpha_mean_abs_diff": round(float(alpha[cover].mean()), 3) if cover.any() else None,
        "alpha_pct_over_16": round(100 * float((alpha[cover] > 16).mean()), 3) if cover.any() else None,
        "coverage_mismatch_px": int(((A[..., 3] > 128) != (B[..., 3] > 128)).sum()),
    }


def _find_blender() -> str | None:
    import shutil

    import config
    cand = [os.environ.get("BLENDER_BIN"), config.BLENDER]
    for c in cand:
        if c and (shutil.which(c) or Path(c).exists()):
            return c
    return None


def _run_blender(job_path: Path) -> dict:
    """Prefer the bpy module (uv sync stays the only setup step); else the binary."""
    try:
        import bpy  # noqa: F401
        mode = f"bpy module {bpy.app.version_string}"
        _blender_side(job_path)
        return {"mode": mode}
    except ImportError:
        pass
    import common
    import config
    exe = _find_blender()
    if not exe:
        raise common.ToolError(
            f"no Blender: 'bpy' is not importable and {config.BLENDER!r} is not on PATH "
            f"or under /Applications. brew install --cask blender, or BLENDER_BIN=...")
    cmd = [exe, "-b", "--factory-startup", "-noaudio", "-P", str(Path(__file__).resolve()),
           "--", str(job_path)]
    t0 = time.perf_counter()
    proc = common.run(cmd, check=False)
    wall = time.perf_counter() - t0
    tail = "\n".join((proc.stdout or "").splitlines()[-25:])
    if proc.returncode != 0 or "[spike] rendered" not in (proc.stdout or ""):
        raise common.ToolError(f"Blender failed ({proc.returncode}):\n{tail}\n{proc.stderr[-2000:]}")
    return {"mode": f"binary {exe}", "process_seconds": round(wall, 2),
            "log_tail": tail}


def main(argv=None) -> int:
    import argparse
    import datetime

    import numpy as np
    from PIL import Image

    import common
    import config
    import perform

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rig", default="doctor")
    p.add_argument("--frames", type=int, default=FPS)
    p.add_argument("--out", default=None, help=f"default {common.rel(config.CACHE_DIR / 'spike')}")
    p.add_argument("--audio", default="~/Desktop/LuckyDog-Harmony/goodbyparty 1.4 vox.wav")
    p.add_argument("--start", type=float, default=41.2)
    p.add_argument("--background", default="rainy-window")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--filter-size", type=float, default=1.0)
    p.add_argument("--interpolation", default="Linear", choices=("Linear", "Closest"))
    p.add_argument("--render-only", action="store_true")
    a = p.parse_args(argv)

    config.ensure_dirs()
    out_dir = Path(a.out) if a.out else config.CACHE_DIR / "spike"
    out_dir.mkdir(parents=True, exist_ok=True)
    r = perform.load_rig(a.rig)
    head, body = perform.plates(r)
    hb, bb = _alpha_bbox(head), _alpha_bbox(body)
    # The plates perform.py composites are full-canvas; Blender gets each cropped
    # to its art with the offset recorded, the way manifest.json records element
    # plates, so the planes are no bigger than they need to be.
    hp, bp = out_dir / "plane-head.png", out_dir / "plane-body.png"
    head.crop(tuple(hb)).save(hp)
    body.crop(tuple(bb)).save(bp)
    job = {
        "rig": a.rig, "out_dir": str(out_dir), "frames": a.frames, "crop": list(r["crop"]),
        "samples": a.samples, "filter_size": a.filter_size, "interpolation": a.interpolation,
        "head": {"file": str(hp), "bbox": hb, "pivot": list(r["face"]["head_pivot"])},
        "body": {"file": str(bp), "bbox": bb, "pivot": [(bb[0] + bb[2]) / 2.0, float(bb[3])]},
    }
    job_path = out_dir / JOB_NAME
    common.write_json(job_path, job)
    for old in out_dir.glob(f"{FRAME_STEM}*.png"):
        old.unlink()

    print(f"rendering {a.frames} frames of {a.rig} in Blender -> {common.rel(out_dir)}")
    launch = _run_blender(job_path)
    frames = sorted(out_dir.glob(f"{FRAME_STEM}*.png"))
    if len(frames) != a.frames:
        raise common.ToolError(f"expected {a.frames} frames, Blender wrote {len(frames)}")
    render = json.loads((out_dir / "render.json").read_text())
    print(f"  {launch['mode']}: {render['frames']} frames at "
          f"{render['resolution'][0]}x{render['resolution'][1]} in {render['render_seconds']}s "
          f"render, {launch.get('process_seconds', render['render_seconds'])}s process")

    ref = _rest_reference(head, body, r["crop"])
    ref_path = out_dir / "reference-rest.png"
    ref.save(ref_path)
    first = Image.open(frames[0])
    cmp = _compare(first, ref)
    print("  rest frame vs perform's two-plate composite:")
    for k, v in cmp.items():
        print(f"    {k:28s} {v}")
    if cmp.get("coverage_mismatch_px") is not None:
        diff = np.abs(np.asarray(first.convert("RGBA")).astype(int)
                      - np.asarray(ref.convert("RGBA")).astype(int)).max(axis=2)
        Image.fromarray(np.clip(diff * 4, 0, 255).astype(np.uint8)).save(out_dir / "diff-rest.png")

    result = {"tool": "tools/blender_spike.py", "rig": a.rig, "launch": launch,
              "render": render, "rest_frame_vs_reference": cmp,
              "head_bbox": hb, "body_bbox": bb, "crop": list(r["crop"]),
              "plates": {k: common.file_hash(r["_dir"] / "plates" / f"plate-{k}-{r['render']['resolution'][0]}.png")
                         for k in ("head", "body")}}
    if a.render_only:
        common.write_json(out_dir / "spike.json", result)
        return 0

    # The real finishing path: background, rain, caption, mux with the stem.
    audio = Path(a.audio).expanduser()
    if not audio.exists():
        raise common.ToolError(f"no such audio: {audio}")
    bg_path = None
    for ext in (".jpg", ".png", ".jpeg"):
        cand = perform.ASSETS / "backgrounds" / f"{a.background}{ext}"
        if cand.exists():
            bg_path = cand
            break
    if bg_path is None:
        raise common.ToolError(f"no background {a.background!r}")
    dur = a.frames / FPS
    comp = perform.composite(frames, bg=perform.fit_background(bg_path), rain=True,
                             text="blender spike", char_width=705, char_top=430,
                             out_dir=out_dir / "comp")
    out = config.OUT_DIR / f"spike-{a.rig}-{datetime.datetime.now():%Y%m%d-%H%M%S}.mp4"
    perform.mux(comp, str(audio), a.start, dur, out, fade=False)
    result.update({"video": out.name, "audio": str(audio), "start": a.start,
                   "duration": dur, "background": a.background})
    common.write_json(out.with_suffix(".json"), result)
    common.write_json(out_dir / "spike.json", result)
    print("wrote", common.rel(out))
    return 0


if __name__ == "__main__":
    # Inside Blender (`blender -b -P this.py -- job.json`) bpy imports and the job
    # file follows "--". From the project venv there is no "--", so it is the driver.
    _job = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else None
    try:
        import bpy  # noqa: F401
    except ImportError:
        bpy = None
    if bpy is not None and _job:
        _blender_side(Path(_job))
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from common import ToolError
        try:
            sys.exit(main())
        except ToolError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
