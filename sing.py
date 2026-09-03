#!/usr/bin/env python3
"""sing.py - cheap cutout lip-sync for a flat character PNG.

Takes a transparent character image, a horizontal lip line and a chin line (or
--auto-mouth to find the teeth), and an audio segment. The part of the face below the
lip line drops with the vocal envelope; the gap is painted as a dark mouth interior.
Adds a head bob and blinks. Output: 1080x1920 mp4 with the audio, plus a JSON sidecar.

Runs with the system python3 (needs Pillow + numpy; the uv env does not have Pillow):

    python3 sing.py --face raw/MADS-FACE.png --audio "music/days like this 1.3.mp3" \
        --start 63.5 --duration 9 --auto-mouth --text "days like this" --name doc-sing

Measure --lip-y / --chin-y / --mouth-x0 / --mouth-x1 in source-image pixels if
--auto-mouth picks the wrong thing.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, wave, datetime
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import caption_styles as cs
import common
import config
from common import ToolError

W, H, FPS = cs.CANVAS_W, cs.CANVAS_H, 30
FONTP = "/System/Library/Fonts/HelveticaNeue.ttc"

def font(sz):
    idx = 0
    for i in range(14):
        try:
            if ImageFont.truetype(FONTP, 40, index=i).getname()[1] == "Regular": idx = i; break
        except Exception: break
    return ImageFont.truetype(FONTP, sz, index=idx)

def envelope(audio, start, dur, tmp):
    band = tmp / "sing_band.wav"
    subprocess.run([config.FFMPEG,"-y","-loglevel","error","-ss",str(start),"-t",str(dur),"-i",audio,
                    "-af","highpass=f=250,lowpass=f=3500","-ac","1","-ar","16000",str(band)], check=True)
    w = wave.open(str(band)); sr = w.getframerate()
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    n = int(len(x) / sr * FPS)
    env = np.array([np.sqrt(np.mean(x[int(i*sr/FPS):int((i+1)*sr/FPS)]**2) + 1e-9) for i in range(n)])
    env = np.clip(env / (np.percentile(env, 97) + 1e-9), 0, 1)
    env = np.where(env < 0.12, 0, env)                       # gate the quiet bits shut
    out = np.zeros_like(env); v = 0.0
    for i, e in enumerate(env):
        v = e if e > v else v * 0.72 + e * 0.28              # fast attack, slow release
        out[i] = v
    return out, n

def auto_mouth(face: Image.Image):
    """Find the teeth (near-white, opaque) and the chin (lowest opaque pixel under them)."""
    a = np.array(face); rgb = a[..., :3].astype(int); al = a[..., 3]
    white = (al > 200) & (rgb.min(axis=2) > 200)
    ys, xs = np.where(white)
    if len(ys) < 100: raise ToolError("auto-mouth: no white teeth region found; pass --lip-y etc.")
    # take the largest white cluster by rows around the median
    y0, y1 = int(np.percentile(ys, 2)), int(np.percentile(ys, 98))
    x0, x1 = int(np.percentile(xs, 2)), int(np.percentile(xs, 98))
    cx = (x0 + x1) // 2
    col = np.where(al[:, cx] > 0)[0]; chin = int(col.max())
    return y0, chin, x0, x1

def jaw_mask(face: Image.Image, lip, chin, mx0, mx1, skin):
    """Opaque skin-coloured region connected to the mouth, below the lip line, plus outlines.
    Hair strands hanging beside the jaw are excluded because they are not skin-coloured."""
    from PIL import ImageFilter
    a = np.array(face); rgb = a[..., :3].astype(int); al = a[..., 3]
    sk = np.array(skin[:3]); near = (np.abs(rgb - sk).sum(axis=2) < 90) & (al > 200)
    white = (al > 200) & (rgb.min(axis=2) > 200)
    cand = near | white
    cand[:lip - 4, :] = False; cand[chin + 12:, :] = False
    cand[:, :max(mx0 - 600, 0)] = False; cand[:, mx1 + 600:] = False
    seed = np.zeros_like(cand); cy = min(lip + 8, chin - 1); seed[cy, (mx0 + mx1)//2] = True
    if not cand[cy, (mx0 + mx1)//2]:
        ys, xs = np.where(cand[cy:cy+40, mx0:mx1]); seed[cy + ys[0], mx0 + xs[0]] = True
    reg = Image.fromarray((seed * 255).astype(np.uint8)); candim = Image.fromarray((cand * 255).astype(np.uint8))
    from PIL import ImageChops
    prev = None
    for _ in range(400):
        grown = ImageChops.multiply(reg.filter(ImageFilter.MaxFilter(9)), candim)
        if prev is not None and ImageChops.difference(grown, prev).getbbox() is None: break
        prev = grown; reg = grown
    reg = reg.filter(ImageFilter.MaxFilter(31))            # take the black outlines with it
    m = np.array(reg) > 0; m[:lip - 4, :] = False; m[chin + 14:, :] = False
    return Image.fromarray((m * 255).astype(np.uint8))

def auto_eyes(face: Image.Image, lip_y):
    """Cream/pale eye whites above the mouth: returns list of boxes, may be empty."""
    a = np.array(face); rgb = a[..., :3].astype(int); al = a[..., 3]
    pale = (al > 200) & (rgb[..., 0] > 200) & (rgb[..., 1] > 170) & (rgb[..., 2] > 120) & (rgb[..., 2] < 210)
    pale[lip_y - 40:, :] = False
    ys, xs = np.where(pale)
    if len(ys) < 100: return []
    mid = int(np.median(xs)); boxes = []
    for sel in (xs < mid, xs >= mid):
        if sel.sum() < 50: continue
        boxes.append((int(xs[sel].min()), int(ys[sel].min()), int(xs[sel].max()), int(ys[sel].max())))
    return boxes

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--face", required=True); p.add_argument("--audio", required=True)
    p.add_argument("--start", type=float, required=True); p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--auto-mouth", action="store_true")
    p.add_argument("--lip-y", type=int); p.add_argument("--chin-y", type=int)
    p.add_argument("--mouth-x0", type=int); p.add_argument("--mouth-x1", type=int)
    p.add_argument("--open-max", type=int, default=0, help="max jaw drop in source px (default: 12%% of face height)")
    p.add_argument("--text", default=""); p.add_argument("--text-y", type=int, default=1500)
    p.add_argument("--bg", default="#121214"); p.add_argument("--face-width", type=int, default=760)
    p.add_argument("--face-top", type=int, default=480)
    p.add_argument("--name", default="sing"); p.add_argument("--out-dir", default=None)
    a = p.parse_args(argv)

    out_dir = Path(a.out_dir) if a.out_dir else config.OUT_DIR; out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / ".sing_tmp"; tmp.mkdir(exist_ok=True)
    face = Image.open(a.face).convert("RGBA"); bb = face.getbbox(); face = face.crop(bb)
    fw, fh = face.size
    if a.auto_mouth:
        lip, chin, mx0, mx1 = auto_mouth(face)
    else:
        if None in (a.lip_y, a.chin_y, a.mouth_x0, a.mouth_x1): raise ToolError("pass --auto-mouth or all four mouth coords")
        lip, chin, mx0, mx1 = a.lip_y - bb[1], a.chin_y - bb[1], a.mouth_x0 - bb[0], a.mouth_x1 - bb[0]
    eyes = auto_eyes(face, lip)
    open_max = a.open_max or int(fh * 0.12)
    # skin colour: sample just above the mouth
    skin = tuple(int(v) for v in np.array(face)[max(lip - 60, 0), (mx0 + mx1)//2][:3]) + (255,)
    jm = jaw_mask(face, lip, chin, mx0, mx1, skin)
    jaw = Image.new("RGBA", face.size, (0, 0, 0, 0)); jaw.paste(face, (0, 0), jm)
    env, n = envelope(a.audio, a.start, a.duration, tmp)
    scale = a.face_width / fw; fnt = font(44)
    rng = np.random.default_rng(3); blink = set(); t = 40
    while t < n: blink.update([t, t+1, t+2]); t += int(rng.integers(70, 130))
    out = out_dir / f"{a.name}-{datetime.datetime.now():%Y%m%d-%H%M%S}.mp4"
    proc = subprocess.Popen([config.FFMPEG,"-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24","-s",f"{W}x{H}","-r",str(FPS),"-i","-",
                             "-ss",str(a.start),"-t",str(a.duration),"-i",a.audio,"-map","0:v","-map","1:a",
                             "-c:v","libx264","-crf","20","-pix_fmt","yuv420p","-c:a","aac","-b:a","192k","-shortest",str(out)], stdin=subprocess.PIPE)
    for k in range(n):
        drop = int(env[k] * open_max); bob = int(np.sin(k / FPS * 2 * np.pi * 1.1) * 6)
        im = Image.new("RGBA", (fw, fh + open_max + 10), (0, 0, 0, 0))
        im.paste(face, (0, 0), face)
        if drop > 2:
            d = ImageDraw.Draw(im)
            d.rounded_rectangle((mx0 + 8, lip - 6, mx1 - 8, lip + drop + 30), radius=36, fill=(30, 10, 14, 255), outline=(0, 0, 0, 255), width=8)
        im.paste(jaw, (0, drop), jaw)
        if k in blink and eyes:
            d = ImageDraw.Draw(im)
            for x0, y0, x1, y1 in eyes: d.ellipse((x0 - 2, y0 - 2, x1 + 2, y1 + 4), fill=skin)
        small = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
        bg = Image.new("RGB", (W, H), a.bg); bg.paste(small, ((W - small.width) // 2, a.face_top + bob), small)
        if a.text:
            d = ImageDraw.Draw(bg); tw = d.textlength(a.text, font=fnt); d.text(((W - tw) / 2, a.text_y), a.text, font=fnt, fill=(235, 235, 235))
        proc.stdin.write(bg.tobytes())
    proc.stdin.close(); proc.wait()
    side = {"video": out.name, "tool": "sing.py", "face": a.face, "audio": a.audio, "start": a.start, "duration": a.duration,
            "mouth": {"lip_y": lip + bb[1], "chin_y": chin + bb[1], "x0": mx0 + bb[0], "x1": mx1 + bb[0], "open_max": open_max},
            "eyes": [[e[0] + bb[0], e[1] + bb[1], e[2] + bb[0], e[3] + bb[1]] for e in eyes], "text": a.text,
            "env_mean": float(env.mean()), "frames": n}
    common.write_json(out.with_suffix(".json"), side)
    print("wrote", out, "| mouth", side["mouth"], "| eyes", len(eyes), "| env mean", round(side["env_mean"], 3))
    return 0

if __name__ == "__main__": sys.exit(main())
