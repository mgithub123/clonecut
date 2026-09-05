"""Stage 8a - The puppet as a Blender scene.

Runs inside Blender only:

    blender -b --factory-startup -noaudio -P blenderpuppet.py -- job.json

puppet.py writes the job: the layers in draw order with their plate files and
offsets, the bones with pivots and parents, the crop to frame, and any animation.
This file has no numpy and no Pillow, because Blender's Python has neither; it
builds planes, an armature and an orthographic camera in plate-pixel units
(1 unit = 1 pixel, image y runs down so world Y = -y), keyframes what the job
asks for, and renders with Eevee to PNG with alpha.

Each layer is one plane, textured with its plate and bound by vertex group to
its bone, so the armature moves it with real parenting and Blender's easing.
Draw order is a small Z step per layer. A layer with a matte gets the matte
plate's alpha multiplied into its own through a second image node, sampled in
canvas space - the same masking the Pillow compositor does.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import bpy

Z_STEP = 0.5          # canvas units between consecutive layers
FPS = 24


def _fcurves(action):
    if hasattr(action, "fcurves"):
        return list(action.fcurves)
    out = []
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                out.extend(bag.fcurves)
    return out


def _eevee_id() -> str:
    engines = [e.identifier for e in
               bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
    for e in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        if e in engines:
            return e
    raise RuntimeError(f"no Eevee in this Blender; engines are {engines}")


def _material(name: str, img, matte=None, canvas=None, bbox=None, opacity=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
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
    tex.interpolation = "Linear"
    tex.extension = "CLIP"
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 1.0
    clear = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    outp = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
    alpha_out = tex.outputs["Alpha"]
    if matte:
        # the matte plate sampled at this plane's canvas position: UV of this
        # plane -> canvas pixel -> UV of the matte plate
        mimg = bpy.data.images.load(matte["file"])
        mimg.alpha_mode = "STRAIGHT"
        mtex = nt.nodes.new("ShaderNodeTexImage")
        mtex.image = mimg
        mtex.interpolation = "Linear"
        mtex.extension = "CLIP"
        uv = nt.nodes.new("ShaderNodeUVMap")
        mapping = nt.nodes.new("ShaderNodeMapping")
        mapping.vector_type = "POINT"
        mx0, my0, mx1, my1 = matte["bbox"]
        mw, mh = mx1 - mx0 + 1, my1 - my0 + 1
        x0, y0, x1, y1 = bbox
        w, h = x1 - x0 + 1, y1 - y0 + 1
        # u' = (x0 + u*w - mx0) / mw ; v measured from the top of each plate
        mapping.inputs["Scale"].default_value = (w / mw, h / mh, 1.0)
        # v runs 0 at the bottom of each plate, so the offset is measured from there
        mapping.inputs["Location"].default_value = ((x0 - mx0) / mw, (my1 - y1) / mh, 0.0)
        nt.links.new(uv.outputs["UV"], mapping.inputs["Vector"])
        nt.links.new(mapping.outputs["Vector"], mtex.inputs["Vector"])
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        nt.links.new(tex.outputs["Alpha"], mul.inputs[0])
        if matte.get("inverted"):
            inv = nt.nodes.new("ShaderNodeMath")
            inv.operation = "SUBTRACT"
            inv.inputs[0].default_value = 1.0
            nt.links.new(mtex.outputs["Alpha"], inv.inputs[1])
            nt.links.new(inv.outputs[0], mul.inputs[1])
        else:
            nt.links.new(mtex.outputs["Alpha"], mul.inputs[1])
        alpha_out = mul.outputs[0]
    if opacity is not None and opacity < 1.0:
        op = nt.nodes.new("ShaderNodeMath")
        op.operation = "MULTIPLY"
        op.inputs[1].default_value = float(opacity)
        nt.links.new(alpha_out, op.inputs[0])
        alpha_out = op.outputs[0]
    nt.links.new(alpha_out, mix.inputs["Fac"])
    nt.links.new(clear.outputs[0], mix.inputs[1])
    nt.links.new(emit.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], outp.inputs["Surface"])
    return mat


def build(job: dict) -> dict:
    scene = bpy.context.scene
    out_dir = Path(job["out_dir"])
    n = int(job.get("frames", 1))
    crop = job["crop"]
    scale = float(job.get("scale", 1.0))
    cw, ch = crop[2] - crop[0], crop[3] - crop[1]
    rw, rh = max(8, int(round(cw * scale))), max(8, int(round(ch * scale)))

    scene.render.fps = FPS
    scene.frame_start, scene.frame_end = 1, n
    scene.render.resolution_x, scene.render.resolution_y = rw, rh
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.filter_size = float(job.get("filter_size", 1.0))
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 15
    scene.render.filepath = str(out_dir / "rgba-")
    eevee = _eevee_id()
    scene.render.engine = eevee
    scene.eevee.taa_render_samples = int(job.get("samples", 8))
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"

    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam_data.sensor_fit = "VERTICAL" if ch >= cw else "HORIZONTAL"
    cam_data.ortho_scale = float(max(cw, ch))
    cam_data.clip_start, cam_data.clip_end = 1.0, 100_000.0
    cam = bpy.data.objects.new("cam", cam_data)
    cam.location = ((crop[0] + crop[2]) / 2.0, -(crop[1] + crop[3]) / 2.0, 5000.0)
    scene.collection.objects.link(cam)
    scene.camera = cam

    # armature first, so planes can bind to it
    bones = job["bones"]
    arm_data = bpy.data.armatures.new("puppet")
    arm = bpy.data.objects.new("puppet", arm_data)
    scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    eb = {}
    for key, b in bones.items():
        px, py = b["pivot"]
        e = arm_data.edit_bones.new(key)
        e.head = (px, -py, 0.0)
        e.tail = (px, -py + 60.0, 0.0)
        eb[key] = e
    for key, b in bones.items():
        if b.get("parent") in eb:
            eb[key].parent = eb[b["parent"]]
            eb[key].use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")

    for lay in job["layers"]:
        x0, y0, x1, y1 = lay["bbox"]
        z = lay["z"] * Z_STEP
        # Eevee sorts blended surfaces by object origin, not per pixel, so every
        # plane gets its own origin at its own depth; with all origins at the
        # world origin the stack came out in arbitrary order.
        cx, cy = (x0 + x1 + 1) / 2.0, -(y0 + y1 + 1) / 2.0
        verts = [(x0 - cx, -y0 - cy, 0.0), (x1 + 1 - cx, -y0 - cy, 0.0),
                 (x1 + 1 - cx, -(y1 + 1) - cy, 0.0), (x0 - cx, -(y1 + 1) - cy, 0.0)]
        mesh = bpy.data.meshes.new(lay["name"])
        mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
        uv = mesh.uv_layers.new(name="UVMap")
        u0, v0, u1, v1 = 0.0, 0.0, 1.0, 1.0
        if lay.get("crop") and lay.get("plate_size"):
            pw, ph = lay["plate_size"]
            cx0, cy0, cx1, cy1 = lay["crop"]
            u0, u1 = cx0 / pw, (cx1 + 1) / pw
            v1, v0 = 1.0 - cy0 / ph, 1.0 - (cy1 + 1) / ph
        for loop, co in zip(mesh.loops, [(u0, v1), (u1, v1), (u1, v0), (u0, v0)]):
            uv.data[loop.index].uv = co
        mesh.update()
        obj = bpy.data.objects.new(f"{lay['id']}-{lay['name']}", mesh)
        obj.location = (cx, cy, z)
        scene.collection.objects.link(obj)
        img = bpy.data.images.load(lay["file"])
        img.alpha_mode = "STRAIGHT"
        obj.data.materials.append(_material(obj.name, img, lay.get("matte"), job.get("canvas"), lay["bbox"], lay.get("opacity")))
        bone = lay["bone"] if lay["bone"] in bones else None
        if bone:
            vg = obj.vertex_groups.new(name=bone)
            vg.add(list(range(4)), 1.0, "REPLACE")
            mod = obj.modifiers.new("puppet", "ARMATURE")
            mod.object = arm

    # animation: {bone: {"location": [[frame, dx, dy]], "rotation": [[frame, deg]]}}
    for key, tracks in (job.get("animation") or {}).items():
        if key not in arm.pose.bones:
            continue
        pb = arm.pose.bones[key]
        pb.rotation_mode = "XYZ"
        for frame, dx, dy in tracks.get("location", []):
            pb.location = (float(dx), float(dy), 0.0)
            pb.keyframe_insert("location", frame=int(frame))
        for frame, deg in tracks.get("rotation", []):
            pb.rotation_euler = (0.0, 0.0, math.radians(float(deg)))
            pb.keyframe_insert("rotation_euler", frame=int(frame))
    ease = job.get("easing")
    if ease and arm.animation_data and arm.animation_data.action:
        for fc in _fcurves(arm.animation_data.action):
            for kp in fc.keyframe_points:
                kp.interpolation = ease.get("interpolation", "BEZIER")
                kp.easing = ease.get("easing", "AUTO")
                if "back" in ease:
                    kp.back = float(ease["back"])

    out_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / "puppet.blend"))
    t0 = time.perf_counter()
    bpy.ops.render.render(animation=True, write_still=False)
    wall = time.perf_counter() - t0
    info = {"blender": bpy.app.version_string, "engine": eevee, "frames": n,
            "resolution": [rw, rh], "scale": scale, "render_seconds": round(wall, 2),
            "layers": len(job["layers"]), "bones": len(bones)}
    (out_dir / "render.json").write_text(json.dumps(info, indent=2))
    print(f"[puppet] rendered {n} frames at {rw}x{rh} in {wall:.1f}s, "
          f"{len(job['layers'])} layers on {len(bones)} bones, Blender {bpy.app.version_string}")
    return info


if __name__ == "__main__":
    bpy.ops.wm.read_factory_settings(use_empty=True)
    build(json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text()))
