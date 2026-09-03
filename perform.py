#!/usr/bin/env python3
"""Stage 7 - Perform.

Renders one shot: a Harmony character rig lip-synced to a vocal stem, with blinks,
tears, beat-locked body motion, breathing and a camera push, over an optional
background.

    uv run perform.py --rig doctor --audio "vox.wav" --start 41.2 --duration 9 \\
        --track "goodbyparty 1.4" --background rainy-window --rain \\
        --text "this is our song. it's called goodbye party"

The rig supplies geometry (assets/rigs/<name>/rig.json), voice.py supplies the
performance (syllable onsets, vowels, beat grid), rig.py supplies the plates.
Nothing about the clip is baked in.

Two things worth knowing about how the mouth is driven. The shape comes from the
*vowel* being sung, not from loudness - loudness only decides open versus shut and
picks between a loud and a quiet variant of the same vowel. And the whole mouth
track runs two frames ahead of the audio: measured lag from the RMS window is about
32 ms, and viewers perceive sound slightly late, so animators place mouths early.
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

import caption_styles as cs
import common
import config
import face
import lipsync
import rig as rigmod
import voice
from common import ToolError

FPS = 24
GATE = 0.14          # envelope below this and the mouth is shut
AUDIBLE = 0.055      # still sounding, even if below the gate - holds a decaying vowel
MIN_HOLD = 3         # frames; below this the mouth strobes
LEAD = 2             # frames the mouth runs ahead of the audio
VOWEL_HOLD = 8       # frames a vowel keeps its shape while still audible
BLINK_CURVE = (0.62, 0.18, 0.0, 0.38, 0.82)
ASSETS = config.ROOT / "assets"


# ------------------------------------------------------------------ rig + plates

def load_rig(name: str) -> dict:
    p = ASSETS / "rigs" / name / "rig.json"
    if not p.exists():
        have = sorted(d.name for d in (ASSETS / "rigs").iterdir() if (d / "rig.json").exists())
        raise ToolError(f"no rig {name!r}; have {', '.join(have) or 'none'}")
    r = json.loads(p.read_text())
    r["_dir"] = p.parent
    r["scene"] = Path(r["scene"]).expanduser()
    return r


def _matted(path) -> Image.Image:
    """A plate as RGBA, however it was rendered.

    Plates written as TGA4 already carry alpha and must be used as they are.
    Older ones are 24-bit, composited onto the scene background, and need
    face.unmul() to recover the matte. Telling them apart by looking is safer than
    by filename: unmul on a plate that already has alpha turns it opaque.
    """
    im = Image.open(path).convert("RGBA")
    if np.asarray(im)[..., 3].min() < 250:
        return im
    return face.unmul(im)


def plates(r: dict) -> tuple[Image.Image, Image.Image]:
    """Head (no mouth) and body plates.

    Tracked plates under assets/rigs/<name>/plates/ come first, and are keyed on
    nothing but the rig, the part and the width. Harmony is on a trial licence, so
    these files outlive the ability to make them: once baked they are the rig.

    The previous version hashed the scene file to build its cache key, which meant
    every render located and read the .xstage even on a cache hit - a hard
    dependency on Harmony's scene being present for work that needed none of it.
    """
    config.ensure_dirs()
    res = tuple(r["render"]["resolution"])
    out = []
    for part in ("head", "body"):
        tracked = r["_dir"] / "plates" / f"plate-{part}-{res[0]}.png"
        if tracked.exists():
            out.append(_matted(tracked))
            continue
        legacy = sorted(config.CACHE_DIR.glob(f"plate-{r['name']}-{part}-*-{res[0]}.png"))
        if legacy:
            out.append(_matted(legacy[0]))
            continue
        if os.environ.get("CLONECUT_NO_HARMONY"):
            raise ToolError(
                f"no tracked {part} plate for rig {r['name']} at {res[0]}px "
                f"({common.rel(tracked)}), and CLONECUT_NO_HARMONY is set.\n"
                f"  bake it while the licence lasts: uv run rig.py plate <scene> "
                f"--out {common.rel(tracked.parent)}")
        ids = {str(i) for i in r["layers"][part]}
        work = config.CACHE_DIR / f"rigwork-{r['name']}-{part}"
        made = rigmod.plate(r["scene"], work, keep=ids, resolution=res, nframes=1)
        tracked.parent.mkdir(parents=True, exist_ok=True)
        Image.open(made[0]).convert("RGBA").save(tracked)
        out.append(face.unmul(Image.open(tracked)))
    return out[0], out[1]


def mouth_library(r: dict) -> dict[str, Image.Image]:
    d = r["_dir"] / "mouths"
    lib = {p.stem: Image.open(p).convert("RGBA") for p in d.glob("*.png")}
    missing = [s for s in r["mouth_ladder"] if s not in lib]
    if missing:
        raise ToolError(f"rig {r['name']} is missing mouth shapes: {', '.join(missing)}\n"
                        f"  generate them, then: uv run face.py sheet <sheet.png> --out {common.rel(d)}")
    return lib


# ------------------------------------------------------------------ tracks

def mouth_track(r: dict, an: dict, n: int) -> list[str]:
    """One mouth shape per frame."""
    env = lipsync.envelope(str(an["_audio"]), an["start"], an["duration"], FPS)
    env = np.resize(env, n)
    loud = np.where(env < GATE, 0.0, (env - GATE) / (1 - GATE))

    per_frame = ["OH"] * n
    of = [int(round(t * FPS)) for t in an["onsets"]]
    for i, s in enumerate(of):
        e = of[i + 1] if i + 1 < len(of) else n
        for f in range(max(0, s), min(n, e)):
            per_frame[f] = an["vowels"][i]

    vmap = r["vowel_map"]

    def shape(f: int) -> str:
        rule = vmap.get(per_frame[f], per_frame[f])
        if isinstance(rule, str):
            return rule
        return rule["loud"] if loud[f] > rule["at"] else rule["quiet"]

    track = ["CLOSED" if env[f] < GATE else shape(f) for f in range(n)]

    # Some vowels keep their shape while still audible, so a decaying note does not
    # snap shut mid-word. Scoped deliberately: applied to every vowel it leaves the
    # mouth open around 90% of the time and the performance stops reading.
    held = set(r.get("vowel_hold", ["OO"]))
    for i, s in enumerate(of):
        if s >= n:
            break
        if an["vowels"][i] not in held:
            continue
        nxt = of[i + 1] if i + 1 < len(of) else n
        sh = shape(min(s, n - 1))
        for f in range(s, min(s + VOWEL_HOLD, nxt, n)):
            if env[f] >= AUDIBLE:
                track[f] = sh
            else:
                break

    # minimum hold, claimed forward so onsets are never pushed late
    i = 1
    while i < n:
        if track[i] != track[i - 1]:
            track[i:i + MIN_HOLD] = [track[i]] * len(track[i:i + MIN_HOLD])
            i += MIN_HOLD
        else:
            i += 1
    return track[LEAD:] + [track[-1]] * LEAD


def blink_track(n: int, every: float = 2.5) -> np.ndarray:
    b = np.ones(n)
    step = max(1, int(every * FPS))
    for s in range(step, n, step):
        for k, v in enumerate(BLINK_CURVE):
            if s + k < n:
                b[s + k] = v
    return b


def phrase_ends(track: list[str], run: int = 5) -> list[int]:
    """Frames where the mouth shuts for a while - a breath, i.e. end of a line."""
    out, i, n = [], 0, len(track)
    while i < n:
        if track[i] == "CLOSED":
            j = i
            while j < n and track[j] == "CLOSED":
                j += 1
            if j - i >= run:
                out.append(i)
            i = j
        else:
            i += 1
    return out


# ------------------------------------------------------------------ compositing

def _pulse(t: float, times, dur: float = 0.42) -> float:
    v = 0.0
    for b in times:
        d = t - b
        if 0 <= d < dur:
            x = d / dur
            v = max(v, float(np.sin(np.pi * x) * (1 - x) ** 0.6))
    return v


def _settle(t: float, ends: list[int], dur: float = 0.95) -> float:
    v = 0.0
    for f in ends:
        d = t - f / FPS
        if 0 <= d < dur:
            x = d / dur
            v += float(np.sin(np.pi * x) * (1 - x) ** 0.5)
    return v


def build_frames(r: dict, an: dict, track: list[str], out_dir: Path,
                 *, blink: bool = True, tears: int = 0) -> list[Path]:
    head, body = plates(r)
    lib = mouth_library(r)
    n = len(track)
    fa = r["face"]
    pivot = tuple(fa["head_pivot"])
    eyes = [tuple(e) for e in r["eyes"]]
    pct = r["mouth_width_pct"]
    crop = tuple(r["crop"])
    bl = blink_track(n) if (blink and r["capabilities"].get("blink")) else np.ones(n)
    ends = phrase_ends(track)

    beats = (an.get("beat") or {}).get("beats") or []
    pulse_times = beats[::2] or [i / 2 for i in range(int(an["duration"] * 2))]

    tear_events = []
    if tears and r["capabilities"].get("tears"):
        tp = r["tears"]["paths"]
        y0, y1 = r["tears"]["y"]
        for k in range(tears):
            start = int(n * (0.15 + 0.7 * k / max(1, tears)))
            xs = tp[k % len(tp)]
            tear_events.append((start, xs[0], xs[1], y0, y1))

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for f in range(n):
        t = f / FPS
        p = _pulse(t, pulse_times)
        st = _settle(t, ends)
        breath = float(np.sin(2 * np.pi * t / 4.2))
        body_dy = 2.2 * p + 1.6 * breath
        head_dy = 4.0 * _pulse(t - 0.05, pulse_times) + 1.6 * breath \
            + 1.1 * float(np.sin(2 * np.pi * t / 3.1 + 0.7)) + 3.2 * st
        head_dx = 1.8 * float(np.sin(2 * np.pi * t / 5.3))
        head_rot = 1.25 * float(np.sin(2 * np.pi * t / 6.1 + 1.2)) - 0.9 * p + 1.7 * st

        canvas = Image.new("RGBA", head.size, (0, 0, 0, 0))
        canvas.alpha_composite(body, (0, int(round(body_dy))))

        h = head.copy()
        shape = lib[track[f]]
        tw = int(fa["width"] * pct.get(track[f], 0.5))
        sc = min(tw / shape.width, fa["max_mouth_height"] / shape.height)
        m = shape.resize((max(1, int(shape.width * sc)), max(1, int(shape.height * sc))),
                         Image.LANCZOS)
        # Which edge of the mouth stays put as it opens. The doctor's mouth hangs
        # from the upper lip and grows downward; the dog's sits on a fixed lower
        # lip and grows upward. Top-anchoring the dog rides the mouth over its nose.
        if fa.get("anchor", "top") == "bottom":
            paste_y = fa["anchor_y"] - m.height
        else:
            paste_y = fa["lip_line"]
        h.paste(m, (fa["centre_x"] - m.width // 2, paste_y), m)

        if bl[f] < 0.999:
            a = np.array(h)
            for (x0, y0, x1, y1) in eyes:
                patch = a[y0:y1 + 1, x0:x1 + 1].copy()
                hh = y1 - y0 + 1
                nh = max(0, int(round(hh * bl[f])))
                a[y0:y1 + 1, x0:x1 + 1, :3] = 0
                a[y0:y1 + 1, x0:x1 + 1, 3] = 255
                if nh:
                    sq = np.array(Image.fromarray(patch).resize((patch.shape[1], nh), Image.LANCZOS))
                    top = y0 + (hh - nh) // 2
                    a[top:top + nh, x0:x1 + 1] = sq
            h = Image.fromarray(a)

        if tear_events:
            d = ImageDraw.Draw(h)
            for (s, xt, xb, ty0, ty1) in tear_events:
                k = f - s
                if not (0 <= k < 22):
                    continue
                if k < 4:
                    prog, rad = 0.0, 0.35 + 0.65 * (k + 1) / 4
                else:
                    prog = (k - 3) / 18
                    prog = prog * prog * 0.35 + prog * 0.65
                    rad = 1.0
                x = xt + (xb - xt) * prog
                y = ty0 + (ty1 - ty0) * prog
                w2, h2 = 11 * rad, 15 * rad
                d.ellipse([x - w2 / 2, y - h2 / 2, x + w2 / 2, y + h2 / 2],
                          fill=(191, 217, 236, 255), outline=(0, 0, 0, 255),
                          width=max(1, int(2 * rad)))

        h = h.rotate(head_rot, resample=Image.BICUBIC, center=pivot)
        canvas.alpha_composite(h, (int(round(head_dx)), int(round(head_dy))))

        zoom = 1.0 + 0.045 * (f / max(1, n - 1))
        if zoom != 1.0:
            w, hgt = canvas.size
            cx, cy = pivot[0], 600
            big = canvas.resize((int(w * zoom), int(hgt * zoom)), Image.LANCZOS)
            ox, oy = int(cx * zoom - cx), int(cy * zoom - cy)
            canvas = big.crop((ox, oy, ox + w, oy + hgt))

        dest = out_dir / f"rgba-{f + 1:04d}.png"
        canvas.crop(crop).save(dest)
        written.append(dest)
    return written


# ------------------------------------------------------------------ background

def rain_layer(bg: Image.Image, f: int, seed: int = 7, n: int = 130) -> Image.Image:
    """Falling rain, masked to the glass so it never crosses the window frame."""
    W, H = bg.size
    a = np.array(bg.convert("RGB")).astype(int)
    lum = a.mean(axis=2)
    glass = ndimage.binary_erosion(ndimage.binary_opening(lum > 42, iterations=2), iterations=3)
    ys, xs = np.where(glass)
    if not len(ys):
        return Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gm = Image.fromarray((glass * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))
    rng = np.random.default_rng(seed)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    sx = rng.uniform(x0, x1, n); sy = rng.uniform(y0 - 600, y1, n)
    ln = rng.uniform(34, 90, n); sp = rng.uniform(30, 58, n)
    op = rng.uniform(120, 210, n); wd = rng.choice([1, 1, 2], n)
    lay = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(lay)
    span = (y1 - y0) + 700
    for i in range(n):
        y = (sy[i] + sp[i] * f) % span + y0 - 600
        x = x0 + ((sx[i] + 0.13 * (y - sy[i]) - x0) % max(1, x1 - x0))
        d.line([(x, y), (x - 0.13 * ln[i], y + ln[i])], fill=int(op[i]), width=int(wd[i]))
    lay = lay.filter(ImageFilter.GaussianBlur(0.6))
    lay = Image.fromarray((np.array(lay).astype(float) * np.array(gm).astype(float) / 255).astype(np.uint8))
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.paste(Image.new("RGB", (W, H), (158, 176, 190)), (0, 0), lay)
    return out


def fit_background(path: Path) -> Image.Image:
    bg = Image.open(path).convert("RGB")
    s = max(cs.CANVAS_W / bg.width, cs.CANVAS_H / bg.height)
    bg = bg.resize((int(bg.width * s + 0.5), int(bg.height * s + 0.5)), Image.LANCZOS)
    ox, oy = (bg.width - cs.CANVAS_W) // 2, (bg.height - cs.CANVAS_H) // 2
    return bg.crop((ox, oy, ox + cs.CANVAS_W, oy + cs.CANVAS_H))


# ------------------------------------------------------------------ finish

def composite(frames: list[Path], *, bg: Image.Image | None, rain: bool, text: str,
              char_width: int, char_top: int, out_dir: Path) -> list[Path]:
    """Character frames -> finished 1080x1920 RGB frames. No audio, no encode.

    Split out from finish() so a montage can concatenate several shots and lay the
    song over the result once, instead of muxing each shot separately.
    """
    W, H = cs.CANVAS_W, cs.CANVAS_H
    x0 = y0 = 10 ** 9
    x1 = y1 = -1
    for p in frames[::4]:
        al = np.array(Image.open(p))[..., 3]
        ys, xs = np.where(al > 16)
        x0, x1 = min(x0, xs.min()), max(x1, xs.max())
        y0, y1 = min(y0, ys.min()), max(y1, ys.max())
    cw, ch = x1 - x0, y1 - y0
    scale = char_width / cw

    # The caption goes under the character but must stay inside the safe box -
    # below SAFE_Y1 is where the platform draws its own UI. If the character is
    # tall enough that both cannot hold, say so rather than quietly overlapping.
    char_bottom = char_top + int(ch * scale)
    text_y = char_bottom + 40
    limit = cs.SAFE_Y1 - 60
    if text_y > limit:
        fits = int((limit - 40 - char_top) / (ch / cw))
        print(f"  note: character ends at y{char_bottom}, leaving no room for the caption "
              f"inside the safe box (y{cs.SAFE_Y0}-{cs.SAFE_Y1}).")
        print(f"        caption clamped to y{limit}; --char-width {fits} would let both fit.")
        text_y = limit
    fnt = _font(44)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    n = len(frames)
    for i, p in enumerate(frames):
        if bg is not None:
            z = 1.0 + 0.018 * (i / max(1, n - 1))
            bw, bh = int(W * z), int(H * z)
            canvas = bg.resize((bw, bh), Image.LANCZOS).crop(
                ((bw - W) // 2, (bh - H) // 2, (bw - W) // 2 + W, (bh - H) // 2 + H)).convert("RGBA")
            if rain:
                canvas.alpha_composite(rain_layer(bg, i))
        else:
            canvas = Image.new("RGBA", (W, H), (0x12, 0x12, 0x14, 255))
        im = Image.open(p).crop((x0, y0, x1, y1)).resize(
            (int(cw * scale), int(ch * scale)), Image.LANCZOS)
        canvas.alpha_composite(im, ((W - im.width) // 2, char_top))
        rgbc = canvas.convert("RGB")
        if text:
            d = ImageDraw.Draw(rgbc)
            tw = d.textlength(text, font=fnt)
            d.text(((W - tw) / 2, text_y), text, font=fnt, fill=(235, 235, 235))
        dest = out_dir / f"comp-{i + 1:05d}.png"
        rgbc.save(dest)
        written.append(dest)
    return written


def mux(frames: list[Path], audio: str, start: float, dur: float,
        out: Path, *, fade: bool = True) -> Path:
    """Finished frames + a slice of the song -> mp4."""
    out.parent.mkdir(parents=True, exist_ok=True)
    af = [] if not fade else ["-af", f"afade=t=out:st={max(dur - 0.5, 0):.2f}:d=0.5"]
    proc = subprocess.Popen(
        [config.FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{cs.CANVAS_W}x{cs.CANVAS_H}", "-r", str(FPS), "-i", "-",
         "-ss", str(start), "-t", f"{dur:.3f}", "-i", audio,
         "-map", "0:v", "-map", "1:a", *af,
         "-c:v", "libx264", "-crf", "19", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)], stdin=subprocess.PIPE)
    for p in frames:
        proc.stdin.write(Image.open(p).convert("RGB").tobytes())
    proc.stdin.close()
    proc.wait()
    return out


def _font(sz: int):
    from PIL import ImageFont
    return ImageFont.truetype(cs.resolve_font(), sz)


# ------------------------------------------------------------------ CLI

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rig", required=True)
    p.add_argument("--audio", required=True, help="vocal stem, not the full mix")
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--track", help="profile name, for the beat grid")
    p.add_argument("--mix", help="audio to lay under the video (default: --audio)")
    p.add_argument("--background", help="name under assets/backgrounds, or a path")
    p.add_argument("--rain", action="store_true")
    p.add_argument("--text", default="")
    p.add_argument("--tears", type=int, default=0)
    p.add_argument("--no-blink", action="store_true")
    p.add_argument("--char-width", type=int, default=850)
    p.add_argument("--char-top", type=int, default=480)
    p.add_argument("--name", default="shot")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--frames-only", action="store_true",
                   help="write finished frames and stop, for montage.py")
    a = p.parse_args(argv)

    r = load_rig(a.rig)
    if not r["capabilities"].get("lipsync"):
        raise ToolError(f"rig {a.rig!r} cannot lip-sync - it has no mouth library")
    if not Path(a.audio).exists():
        raise ToolError(f"no such audio: {a.audio}")

    print(f"analysing {common.rel(Path(a.audio))} {a.start}s +{a.duration}s")
    an = voice.analyse(a.audio, a.start, a.duration, a.track)
    an["_audio"] = a.audio
    n = int(round(a.duration * FPS))
    print(f"  {len(an['onsets'])} syllables, vowels "
          f"{dict(__import__('collections').Counter(an['vowels']).most_common())}")

    track = mouth_track(r, an, n)
    runs = [len(list(g)) for _, g in itertools.groupby(track)]
    print(f"  {len(runs) - 1} mouth changes, mean hold {sum(runs) / len(runs):.1f}f, "
          f"shut {track.count('CLOSED')}/{n}")

    work = config.CACHE_DIR / f"perform-{a.rig}-{a.name}"
    print("rendering character frames")
    frames = build_frames(r, an, track, work, blink=not a.no_blink, tears=a.tears)

    bg = None
    if a.background:
        bp = Path(a.background)
        if not bp.exists():
            for ext in (".jpg", ".png", ".jpeg"):
                cand = ASSETS / "backgrounds" / f"{a.background}{ext}"
                if cand.exists():
                    bp = cand
                    break
        if not bp.exists():
            raise ToolError(f"no background {a.background!r}")
        bg = fit_background(bp)

    out_dir = Path(a.out_dir) if a.out_dir else config.OUT_DIR
    comp = composite(frames, bg=bg, rain=a.rain, text=a.text,
                     char_width=a.char_width, char_top=a.char_top,
                     out_dir=config.CACHE_DIR / f"comp-{a.rig}-{a.name}")
    if a.frames_only:
        print(f"wrote {len(comp)} frames to {common.rel(comp[0].parent)}")
        return 0
    out = mux(comp, a.mix or a.audio, a.start, a.duration,
              out_dir / f"{a.name}-{datetime.datetime.now():%Y%m%d-%H%M%S}.mp4")

    common.write_json(out.with_suffix(".json"), {
        "video": out.name, "tool": "perform.py", "rig": a.rig,
        "audio": common.rel(Path(a.audio)), "mix": a.mix, "start": common.r3(a.start),
        "duration": common.r3(a.duration), "track": a.track,
        "background": a.background, "rain": a.rain, "text": a.text,
        "tears": a.tears, "blink": not a.no_blink,
        "frames": n, "onsets": an["onsets"], "vowels": an["vowels"],
        "mouth_changes": len(runs) - 1, "shut_frames": track.count("CLOSED"),
        "char_width": a.char_width, "char_top": a.char_top,
    })
    print("wrote", common.rel(out))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
