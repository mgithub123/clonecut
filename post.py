#!/usr/bin/env python3
"""post.py - turn a Harmony frame render into a finished 1080x1920 post.

Takes the .tga/.png sequence a Harmony Write node produces (character on white),
knocks out the white background without punching holes in white eyes or teeth,
composites the character on a flat ground, burns one line of text in the style the
research validated, and muxes the song.

Runs with the SYSTEM python3 (needs Pillow, numpy, scipy - the uv env has none of them):

    python3 post.py --frames ~/Desktop/LuckyDog-Harmony/Luckydog_DOG/frames \\
        --audio "music/days like this 1.3.mp3" --start 63.5 \\
        --text "this is our song. it's called days like this" --name lucky-day01

One post = one --text change. Everything else stays fixed across the run, which is the
whole point: same character, same setup, a different line each day.
"""
from __future__ import annotations
import argparse, datetime, json, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

W, H = 1080, 1920
FONTP = "/System/Library/Fonts/HelveticaNeue.ttc"


def font(sz: int):
    idx = 0
    for i in range(14):
        try:
            if ImageFont.truetype(FONTP, 40, index=i).getname()[1] == "Regular":
                idx = i
                break
        except Exception:
            break
    return ImageFont.truetype(FONTP, sz, index=idx)


def bg_mask(rgb: np.ndarray, thresh: int = 736) -> np.ndarray:
    """True where the pixel is background: near-white AND connected to the border.

    The connectivity test is what protects the eyes and teeth - they are white too,
    but enclosed by the character's black outline, so they never touch the border.
    """
    whiteish = rgb.sum(axis=2) >= thresh
    lab, _ = ndimage.label(whiteish)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    border.discard(0)
    return np.isin(lab, list(border))


def matte(rgb: np.ndarray, erode: int = 2, feather: float = 0.8) -> np.ndarray:
    """Alpha for the character. The render is drawn over white, so the outermost ring of
    the silhouette is a white blend; keeping it leaves a bright halo on a dark ground.
    Eroding the solid mask a couple of pixels cuts that ring, then a small blur puts the
    anti-aliasing back."""
    solid = ~bg_mask(rgb)
    if erode:
        solid = ndimage.binary_erosion(solid, iterations=erode, border_value=1)
    a = ndimage.gaussian_filter(solid.astype(np.float32), sigma=feather)
    return np.clip(a * 255, 0, 255).astype(np.uint8)


def char_bbox(files, sample=8, thresh=736):
    x0 = y0 = 10 ** 9
    x1 = y1 = -1
    for f in files[::sample]:
        a = np.array(Image.open(f).convert("RGB")).astype(int)
        solid = ~bg_mask(a, thresh)
        ys, xs = np.where(solid)
        if not len(ys):
            continue
        x0, x1 = min(x0, xs.min()), max(x1, xs.max())
        y0, y1 = min(y0, ys.min()), max(y1, ys.max())
    return int(x0), int(y0), int(x1), int(y1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--frames", required=True, help="directory of rendered frames")
    p.add_argument("--glob", default="final-*.tga")
    p.add_argument("--audio", required=True)
    p.add_argument("--start", type=float, required=True, help="offset into the track, seconds")
    p.add_argument("--fps", type=float, default=24.0, help="frame rate the scene was rendered at")
    p.add_argument("--text", default="")
    p.add_argument("--text-size", type=int, default=44)
    p.add_argument("--text-y", type=int, default=1560)
    p.add_argument("--bg", default="#121214")
    p.add_argument("--char-width", type=int, default=720, help="character width in the 1080px output")
    p.add_argument("--char-top", type=int, default=430, help="y of the top of the character")
    p.add_argument("--erode", type=int, default=2, help="pixels to shave off the silhouette to kill the white fringe")
    p.add_argument("--feather", type=float, default=0.8)
    p.add_argument("--name", default="post")
    p.add_argument("--out-dir", default="out")
    a = p.parse_args(argv)

    root = Path(__file__).resolve().parent
    out_dir = root / a.out_dir
    out_dir.mkdir(exist_ok=True)
    files = sorted(Path(a.frames).glob(a.glob))
    if not files:
        raise SystemExit(f"no frames matched {a.glob} in {a.frames}")

    x0, y0, x1, y1 = char_bbox(files)
    cw, ch = x1 - x0, y1 - y0
    scale = a.char_width / cw
    dur = len(files) / a.fps
    fnt = font(a.text_size)
    print(f"{len(files)} frames, bbox {cw}x{ch}, scale {scale:.3f}, {dur:.2f}s")

    out = out_dir / f"{a.name}-{datetime.datetime.now():%Y%m%d-%H%M%S}.mp4"
    proc = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", f"{a.fps}", "-i", "-",
        "-ss", str(a.start), "-t", f"{dur:.3f}", "-i", a.audio,
        "-map", "0:v", "-map", "1:a",
        "-af", f"afade=t=out:st={max(dur - 0.5, 0):.2f}:d=0.5",
        "-c:v", "libx264", "-crf", "19", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)], stdin=subprocess.PIPE)

    for f in files:
        im = Image.open(f).convert("RGB")
        arr = np.array(im).astype(int)
        rgba = im.convert("RGBA")
        rgba.putalpha(Image.fromarray(matte(arr, a.erode, a.feather)))
        char = rgba.crop((x0, y0, x1, y1))
        char = char.resize((int(cw * scale), int(ch * scale)), Image.LANCZOS)
        canvas = Image.new("RGB", (W, H), a.bg)
        canvas.paste(char, ((W - char.width) // 2, a.char_top), char)
        if a.text:
            d = ImageDraw.Draw(canvas)
            tw = d.textlength(a.text, font=fnt)
            d.text(((W - tw) / 2, a.text_y), a.text, font=fnt, fill=(235, 235, 235))
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    proc.wait()

    side = {"video": out.name, "tool": "post.py", "frames": str(a.frames), "n_frames": len(files),
            "audio": a.audio, "start": a.start, "fps": a.fps, "text": a.text,
            "char_bbox": [x0, y0, x1, y1], "char_width": a.char_width, "bg": a.bg,
            "erode": a.erode, "feather": a.feather}
    out.with_suffix(".json").write_text(json.dumps(side, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
