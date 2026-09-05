#!/usr/bin/env python3
"""Stage 8p - A character playing along, from one drawing.

    uv run play.py --figure raw/doc-guitar.png --mix "music/goodbyparty 1.4.wav" \\
        --start 51.0 --duration 8 --background rainy-window --rain --name doc-riff

Not every shot needs the rig. A single full-figure drawing - the doctor with a
guitar, drawn once, mouth and pose baked in - can play an instrumental if two
things move: the whole figure on the beat, and the strumming hand on every note.

Strums come from onsets in the mix window, because there is no guitar stem for
this version of the song. The hand is cut out of the drawing along an ellipse
around its box, the hole behind it is filled with the colour of the ring around
it - flat art makes that exact - and the cut moves down on a hit and eases back
up, alternating down and up strokes. The figure bobs with motion.impulse on the
same onsets, small, and squashes a fraction of a percent about its feet.

Output goes through perform.composite() and perform.mux(), so the shot is a real
1080x1920 mp4 with the same background, rain and caption path as every other.
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import common
import config
import motion
import perform
from common import ToolError

FPS = motion.FPS
STRUM = dict(
    travel_px_pct=1.4,      # hand travel per strum, as % of the figure's height
    down_frames=2,          # frames to hit the string
    back_frames=4,          # frames to ease back
    min_gap_s=0.12,         # onsets closer than this are one strum
    hand_pad=1.25,          # the cut ellipse is this much bigger than the hand box
)
# Sized to the Stage 7 render that read right: about 1% of the figure's height
# per beat. Rocking the whole drawing about its feet and sliding it sideways was
# tried and read as a sheet of card on a stick; flat art moves in small amounts
# or in cuts, not in sweeps.
BOB = dict(hit_px_pct=0.6, accent_px_pct=0.6)   # whole-figure bob per hit, % of height; the first take's
ROCK = dict(deg=0.0, sway_pct=0.0, bar_beats=4)  # off; kept as knobs - it read as card on a stick
# The drawings are a strum cycle: the first is hand up, the last is hand down,
# any between are inbetweens. On each hit the cycle runs down through them and
# back up, the down stroke faster than the return, then holds the first drawing.
CYCLE = dict(down_frames_per_key=1, up_frames_per_key=2)
ACCENT = dict(frames=3)         # unused by the cycle; kept for the accent mode


def onsets(mix: Path, start: float, dur: float) -> list[float]:
    """Note onsets in the window, seconds from its start."""
    import librosa
    y, sr = librosa.load(str(mix), sr=22050, mono=True, offset=start, duration=dur)
    env = librosa.onset.onset_strength(y=y, sr=sr)
    fr = librosa.onset.onset_detect(onset_envelope=env, sr=sr, backtrack=False, units="time",
                                    pre_max=3, post_max=3, pre_avg=6, post_avg=6, delta=0.12, wait=2)
    out, last = [], -1.0
    for t in fr:
        if t - last >= STRUM["min_gap_s"]:
            out.append(float(t))
            last = t
    return out


def find_hand(fig: Image.Image, guess: tuple[int, int, int, int] | None) -> tuple[int, int, int, int]:
    """The strumming hand's box. Given explicitly, or the darkest sizeable blob
    in the lower-middle of the figure that is not black outline."""
    if guess:
        return guess
    from scipy import ndimage
    a = np.asarray(fig.convert("RGBA")).astype(int)
    al = a[..., 3] > 128
    rgb = a[..., :3]
    dark = al & (rgb.max(axis=2) < 110) & (rgb.max(axis=2) > 35) & (rgb.max(axis=2) - rgb.min(axis=2) < 30)
    ys, xs = np.where(al)
    y_mid = (ys.min() + ys.max()) / 2
    lab, k = ndimage.label(ndimage.binary_opening(dark, iterations=2))
    best = None
    for i in range(1, k + 1):
        by, bx = np.where(lab == i)
        if len(by) < 400 or by.mean() < y_mid * 0.8:
            continue
        # a hand is compact: the boots are wide, the sleeves long
        h, w = by.max() - by.min(), bx.max() - bx.min()
        if max(h, w) / max(1, min(h, w)) > 2.2:
            continue
        score = len(by)
        if best is None or score > best[0]:
            best = (score, [int(bx.min()), int(by.min()), int(bx.max()), int(by.max())])
    if not best:
        raise ToolError("could not find the strumming hand; pass --hand x0,y0,x1,y1")
    return tuple(best[1])


def cut_hand(fig: Image.Image, box: tuple[int, int, int, int]) -> tuple[Image.Image, Image.Image, tuple[int, int]]:
    """(figure with the hand removed and the hole filled, the hand cut, its offset)."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx, ry = (x1 - x0) / 2 * STRUM["hand_pad"], (y1 - y0) / 2 * STRUM["hand_pad"]
    mask = Image.new("L", fig.size, 0)
    ImageDraw.Draw(mask).ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    m = np.asarray(mask) > 0
    a = np.asarray(fig.convert("RGBA")).copy()
    hand = a.copy()
    hand[..., 3] = np.where(m, hand[..., 3], 0)
    # fill: the median colour of the opaque ring just outside the ellipse
    ring = (np.asarray(mask.filter(ImageFilter.MaxFilter(15))) > 0) & ~m & (a[..., 3] > 128)
    fill = np.median(a[ring][:, :3], axis=0).astype(np.uint8) if ring.any() else np.array([128, 128, 128], np.uint8)
    body = a.copy()
    inside = m & (a[..., 3] > 0)
    body[inside, :3] = fill
    hb = Image.fromarray(hand).getbbox()
    hand_im = Image.fromarray(hand).crop(hb)
    return Image.fromarray(body), hand_im, (hb[0], hb[1])


def frames_of(fig: Image.Image, hits: list[int], n: int, hand_box, out_dir: Path,
              keys: list[Image.Image] | None = None, accents: list[int] | None = None,
              beat_frames: float = 9.0) -> list[Path]:
    """Frames of the figure playing. With `keys` - two or more drawings of the
    strum, hand up and hand down - each onset switches to the next drawing and
    holds it, down then up; with one drawing, the hand is cut out and moved."""
    H = fig.getbbox()[3] - fig.getbbox()[1]
    if keys and len(keys) > 1:
        return _frames_keyed(keys, hits, n, out_dir, H, accents=accents, beat_frames=beat_frames)
    body, hand, (hx, hy) = cut_hand(fig, hand_box)
    travel = STRUM["travel_px_pct"] / 100 * H
    strum = np.zeros(n)
    for k, h in enumerate(hits):
        sign = 1.0 if k % 2 == 0 else -1.0            # down, then up
        for j in range(STRUM["down_frames"]):
            f = h + j
            if 0 <= f < n:
                strum[f] = sign * travel * (j + 1) / STRUM["down_frames"]
        for j in range(STRUM["back_frames"]):
            f = h + STRUM["down_frames"] + j
            if 0 <= f < n:
                strum[f] = sign * travel * (1 - motion.ease_out(np.array((j + 1) / STRUM["back_frames"])))
    B = motion.BEAT
    bob = motion.impulse(n, hits, pre=B["anticipation_frames"], pre_amp=B["anticipation_px"] * 0.5,
                         amp=BOB["hit_px_pct"] / 100 * H, post_amp=B["overshoot_px"] * 0.5,
                         settle=B["settle_frames"])
    sx, sy = motion.squash_stretch(n, hits)
    out_dir.mkdir(parents=True, exist_ok=True)
    W, Hc = fig.size
    written = []
    for f in range(n):
        canvas = Image.new("RGBA", (W, Hc), (0, 0, 0, 0))
        canvas.alpha_composite(body, (0, int(round(bob[f]))))
        canvas.alpha_composite(hand, (hx, hy + int(round(bob[f] + strum[f]))))
        if abs(sy[f] - 1) > 1e-4:
            # squash about the feet: scale, then re-anchor the bottom edge
            bb = canvas.getbbox()
            sc = canvas.resize((max(1, int(W * sx[f])), max(1, int(Hc * sy[f]))), Image.LANCZOS)
            re = Image.new("RGBA", (W, Hc), (0, 0, 0, 0))
            re.alpha_composite(sc, (int((W - sc.width) / 2), int(bb[3] - sc.height * bb[3] / Hc)))
            canvas = re
        dest = out_dir / f"rgba-{f + 1:04d}.png"
        canvas.save(dest)
        written.append(dest)
    return written


def _frames_keyed(keys: list[Image.Image], hits: list[int], n: int, out_dir: Path, H: int,
                  accents: list[int] | None = None, beat_frames: float = 9.0) -> list[Path]:
    """Two drawings flipped on every hit read as a flicker, and a figure that only
    moves on the hit reads as a sticker. So: the first drawing is the base, the
    second is the accent, shown for a few frames on the strong hits; underneath,
    the whole figure bobs on every beat, rocks about its feet through the bar,
    sways across it, and squashes on the hit."""
    accents = accents if accents is not None else hits[::2]
    which = np.zeros(n, int)
    if len(keys) == 2:
        # two keys: toggle on each hit and hold - the first take, which read best
        state, hitset = 0, set(hits)
        for f in range(n):
            if f in hitset:
                state ^= 1
            which[f] = state
    else:
        order = list(range(1, len(keys))) + list(range(len(keys) - 2, -1, -1))   # down through the keys, back up
        per = [CYCLE["down_frames_per_key"]] * (len(keys) - 1) + [CYCLE["up_frames_per_key"]] * (len(keys) - 1)
        for h in hits:
            f = h
            for key, hold in zip(order, per):
                for _ in range(hold):
                    if 0 <= f < n:
                        which[f] = key
                    f += 1
    B = motion.BEAT
    bob = motion.impulse(n, hits, pre=B["anticipation_frames"], pre_amp=B["anticipation_px"],
                         amp=BOB["hit_px_pct"] / 100 * H, post_amp=B["overshoot_px"],
                         settle=B["settle_frames"])
    bob += motion.impulse(n, accents, pre=B["anticipation_frames"], pre_amp=B["anticipation_px"],
                          amp=(BOB["accent_px_pct"] - BOB["hit_px_pct"]) / 100 * H,
                          post_amp=B["overshoot_px"], settle=B["settle_frames"])
    sx, sy = motion.squash_stretch(n, hits)
    bar = ROCK["bar_beats"] * beat_frames
    t = np.arange(n)
    rock = ROCK["deg"] * np.sin(2 * np.pi * t / bar)
    sway = ROCK["sway_pct"] / 100 * keys[0].width * np.sin(2 * np.pi * t / bar + np.pi / 2)
    out_dir.mkdir(parents=True, exist_ok=True)
    W, Hc = keys[0].size
    bb0 = keys[0].getbbox()
    feet = ((bb0[0] + bb0[2]) / 2, bb0[3])
    written = []
    for f in range(n):
        canvas = Image.new("RGBA", (W, Hc), (0, 0, 0, 0))
        canvas.alpha_composite(keys[which[f]], (0, 0))
        if abs(sy[f] - 1) > 1e-4:
            sc = canvas.resize((max(1, int(W * sx[f])), max(1, int(Hc * sy[f]))), Image.LANCZOS)
            re = Image.new("RGBA", (W, Hc), (0, 0, 0, 0))
            re.alpha_composite(sc, (int((W - sc.width) / 2), int(feet[1] - sc.height * feet[1] / Hc)))
            canvas = re
        canvas = canvas.rotate(rock[f], resample=Image.BICUBIC, center=feet)
        moved = Image.new("RGBA", (W, Hc), (0, 0, 0, 0))
        moved.alpha_composite(canvas, (int(round(sway[f])), int(round(bob[f]))))
        dest = out_dir / f"rgba-{f + 1:04d}.png"
        moved.save(dest)
        written.append(dest)
    return written


# --- parts: one drawing, cut so its pieces move on their own -----------------
PARTS = dict(
    hand_travel_pct=1.5,     # strum travel as % of figure height, across the strings
    hand_down_frames=2,      # frames to hit the strings
    hand_back_frames=5,      # frames to come back up
    hand_sleeve_px=48,       # how much of the sleeve travels with the glove, so the cuff never opens
    head_nod_pct=0.45,       # head drops this % of figure height on a beat
    head_nod_deg=1.6,        # and tips forward this much, about the neck
    head_accent_scale=1.7,   # both, on an accented beat
    head_delay_frames=1,     # the head follows the body
    body_bob_pct=0.6,        # whole-body bob per beat, % of height
    string_dir=(0.41, 0.91), # unit vector across the strings, screen x right, y down
)


def _fill_hole(a: np.ndarray, m: np.ndarray, grow: int = 9) -> np.ndarray:
    """Paint the region `m` of a straight-alpha RGBA array with the median colour
    of the opaque ring just outside it. Flat art makes this exact."""
    from scipy import ndimage
    ring = ndimage.binary_dilation(m, iterations=grow) & ~m & (a[..., 3] > 128)
    out = a.copy()
    if ring.any():
        fill = np.median(a[ring][:, :3], axis=0).astype(np.uint8)
        out[m & (a[..., 3] > 0), :3] = fill
    return out


def _layer(a: np.ndarray, m: np.ndarray) -> tuple[Image.Image, tuple[int, int]]:
    lay = a.copy()
    lay[..., 3] = np.where(m, lay[..., 3], 0)
    im = Image.fromarray(lay)
    bb = im.getbbox()
    return im.crop(bb), (bb[0], bb[1])


def cut_parts(fig: Image.Image) -> dict:
    """Head, strumming glove and the rest, from one drawing, by colour and place.

    The head is everything above the collar - the collar's top is the first row
    where the coat colour appears. The strumming glove is the largest compact
    dark blob in the middle band of the figure, with a strip of sleeve above it
    so the cuff never opens as it moves. Holes are filled with the colour of
    their surroundings.
    """
    from scipy import ndimage
    a = np.asarray(fig.convert("RGBA")).astype(np.int16)
    al = a[..., 3] > 128
    rgb = a[..., :3]
    ys, xs = np.where(al)
    top, bot = ys.min(), ys.max()
    H = bot - top
    # the coat: the most common saturated colour in the middle of the figure
    mid = al.copy(); mid[: top + H // 4] = False; mid[top + 3 * H // 4:] = False
    q = (rgb // 8) * 8
    vals, counts = np.unique(q[mid & ((rgb.max(axis=2) - rgb.min(axis=2)) > 12)], axis=0, return_counts=True)
    coat = vals[np.argmax(counts)]
    is_coat = al & (np.abs(rgb - coat).max(axis=2) < 20)
    rows = np.where(is_coat.sum(axis=1) > 40)[0]
    neck_y = int(rows.min()) - 8 if len(rows) else top + H // 4
    head = al.copy(); head[neck_y:] = False
    head = ndimage.binary_dilation(head, iterations=3) & al
    # the strumming glove
    dark = al & (rgb.max(axis=2) < 110) & (rgb.max(axis=2) > 35) & ((rgb.max(axis=2) - rgb.min(axis=2)) < 30)
    lab, k = ndimage.label(ndimage.binary_opening(dark, iterations=2))
    best = None
    for i in range(1, k + 1):
        by, bx = np.where(lab == i)
        cy = by.mean()
        if len(by) < 2000 or not (top + 0.3 * H < cy < top + 0.68 * H):
            continue
        if best is None or len(by) > best[0]:
            best = (len(by), i, [int(bx.min()), int(by.min()), int(bx.max()), int(by.max())])
    if not best:
        raise ToolError("could not find the strumming glove; the drawing needs a dark glove mid-figure")
    _, gi, gb = best
    glove = ndimage.binary_dilation(lab == gi, iterations=6) & al
    sleeve = np.zeros_like(glove)
    sleeve[max(0, gb[1] - PARTS["hand_sleeve_px"]):gb[1] + 10, gb[0]:gb[2] + 1] = True
    hand = glove | (sleeve & al & ~head)
    body = a.astype(np.uint8)
    body = _fill_hole(body, hand)
    body = _fill_hole(body, head, grow=6)
    body[..., 3] = np.where(head, 0, body[..., 3])          # nothing sits behind the head
    hand_im, hand_at = _layer(a.astype(np.uint8), hand)
    head_im, head_at = _layer(a.astype(np.uint8), head)
    return {"body": Image.fromarray(body), "hand": (hand_im, hand_at), "head": (head_im, head_at),
            "neck": (int(xs[ys == neck_y].mean()) if (ys == neck_y).any() else int(xs.mean()), neck_y),
            "glove_box": gb, "height": int(H)}


def _frames_parts(fig: Image.Image, hits: list[int], accents: list[int], n: int, out_dir: Path) -> list[Path]:
    parts = cut_parts(fig)
    H = parts["height"]
    P = PARTS
    B = motion.BEAT
    acc = set(accents)
    # strum: down across the strings on the beat, ease back
    travel = P["hand_travel_pct"] / 100 * H
    strum = np.zeros(n)
    for h in hits:
        for j in range(P["hand_down_frames"]):
            f = h + j
            if 0 <= f < n:
                strum[f] = travel * motion.ease_in_out(np.array((j + 1) / P["hand_down_frames"]))
        for j in range(P["hand_back_frames"]):
            f = h + P["hand_down_frames"] + j
            if 0 <= f < n:
                strum[f] = travel * (1 - motion.ease_out(np.array((j + 1) / P["hand_back_frames"])))
    dx, dy = P["string_dir"]
    bob = motion.impulse(n, hits, pre=B["anticipation_frames"], pre_amp=B["anticipation_px"],
                         amp=P["body_bob_pct"] / 100 * H, post_amp=B["overshoot_px"], settle=B["settle_frames"])
    nod = np.zeros(n); tilt = np.zeros(n)
    for h in hits:
        k = P["head_accent_scale"] if h in acc else 1.0
        for j in range(8):
            f = h + P["head_delay_frames"] + j
            if 0 <= f < n:
                e = float(np.sin(np.pi * (j + 0.5) / 8))
                nod[f] += k * P["head_nod_pct"] / 100 * H * e
                tilt[f] += k * P["head_nod_deg"] * e
    sx, sy = motion.squash_stretch(n, hits)
    out_dir.mkdir(parents=True, exist_ok=True)
    W, Hc = fig.size
    body = parts["body"]; hand_im, (hx, hy) = parts["hand"]; head_im, (ex, ey) = parts["head"]
    nx, ny = parts["neck"]
    written = []
    for f in range(n):
        canvas = Image.new("RGBA", (W, Hc), (0, 0, 0, 0))
        canvas.alpha_composite(body, (0, 0))
        canvas.alpha_composite(hand_im, (hx + int(round(dx * strum[f])), hy + int(round(dy * strum[f]))))
        head = Image.new("RGBA", (W, Hc), (0, 0, 0, 0))
        head.alpha_composite(head_im, (ex, ey))
        head = head.rotate(-tilt[f], resample=Image.BICUBIC, center=(nx, ny))
        canvas.alpha_composite(head, (0, int(round(nod[f]))))
        if abs(sy[f] - 1) > 1e-4:
            bb = canvas.getbbox()
            sc = canvas.resize((max(1, int(W * sx[f])), max(1, int(Hc * sy[f]))), Image.LANCZOS)
            re = Image.new("RGBA", (W, Hc), (0, 0, 0, 0))
            re.alpha_composite(sc, (int((W - sc.width) / 2), int(bb[3] - sc.height * bb[3] / Hc)))
            canvas = re
        moved = Image.new("RGBA", (W, Hc), (0, 0, 0, 0))
        moved.alpha_composite(canvas, (0, int(round(bob[f]))))
        dest = out_dir / f"rgba-{f + 1:04d}.png"
        moved.save(dest)
        written.append(dest)
    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--figure", required=True, nargs="+",
                   help="a full-figure drawing, or two or more drawings of the strum (hand up, hand "
                        "down) on the same canvas; transparent or on white")
    p.add_argument("--mix", required=True)
    p.add_argument("--track", default=None, help="profile name, for the beat grid (strum every beat)")
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--hand", default=None, help="x0,y0,x1,y1 of the strumming hand in the drawing")
    p.add_argument("--parts", action="store_true",
                   help="cut the first drawing into head, strumming glove and body, and move them")
    p.add_argument("--background", default=None)
    p.add_argument("--rain", action="store_true")
    p.add_argument("--text", default="")
    p.add_argument("--char-width", type=int, default=820)
    p.add_argument("--char-top", type=int, default=260)
    p.add_argument("--name", default="play")
    a = p.parse_args(argv)

    keys = []
    for fp in a.figure:
        fig_p = Path(fp)
        if not fig_p.exists():
            raise ToolError(f"no such figure: {fp}")
        im = Image.open(fig_p).convert("RGBA")
        if np.asarray(im)[..., 3].min() >= 250:
            import face
            im = face.unmul(im)
        keys.append(im)
    if len({k.size for k in keys}) != 1:
        raise ToolError("strum drawings must share one canvas size, so they register")
    fig = keys[0]
    hand_box = None
    if len(keys) == 1 and not a.parts:
        fig = fig.crop(fig.getbbox())
        keys = [fig]
        hand_box = find_hand(fig, tuple(int(v) for v in a.hand.split(",")) if a.hand else None)
        print(f"figure {fig.width}x{fig.height}, strumming hand at {hand_box}")
    else:
        print(f"{len(keys)} strum drawings at {fig.width}x{fig.height}")

    n = int(round(a.duration * FPS))
    onset_t = onsets(Path(a.mix), a.start, a.duration)
    accents = sorted({int(round(t * FPS)) for t in onset_t if t * FPS < n})
    hits, beat_frames = accents, 9.0
    if a.track:
        import voice
        an = voice.analyse(str(a.mix), a.start, a.duration, a.track)
        beats = (an.get("beat") or {}).get("beats") or []
        if beats:
            hits = sorted({int(round(t * FPS)) for t in beats if 0 <= t * FPS < n})
            beat_frames = FPS * float(np.median(np.diff(beats))) if len(beats) > 1 else 9.0
            # accents: the beats the mix's onsets land on, or every second beat
            near = {h for h in hits if any(abs(h - o) <= 2 for o in accents)}
            accents = sorted(near) if len(near) >= len(hits) // 4 else hits[::2]
    print(f"  {len(hits)} strums on the beat, {len(accents)} accents from the mix's onsets")

    work = config.CACHE_DIR / f"play-{a.name}"
    if a.parts:
        frames = _frames_parts(keys[0], hits, accents, n, work)
    else:
        frames = frames_of(fig, hits, n, hand_box, work, keys=keys, accents=accents, beat_frames=beat_frames)
    bg = None
    if a.background:
        bp = Path(a.background)
        if not bp.exists():
            for ext in (".jpg", ".png", ".jpeg"):
                cand = perform.ASSETS / "backgrounds" / f"{a.background}{ext}"
                if cand.exists():
                    bp = cand
        if not bp.exists():
            raise ToolError(f"no background {a.background!r}")
        bg = perform.fit_background(bp)
    comp = perform.composite(frames, bg=bg, rain=a.rain, text=a.text, char_width=a.char_width,
                             char_top=a.char_top, out_dir=config.CACHE_DIR / f"play-comp-{a.name}")
    out = config.OUT_DIR / f"{a.name}-{datetime.datetime.now():%Y%m%d-%H%M%S}.mp4"
    perform.mux(comp, a.mix, a.start, a.duration, out)
    common.write_json(out.with_suffix(".json"), {
        "video": out.name, "tool": "play.py", "figures": [common.rel(Path(f)) for f in a.figure],
        "figure_sha256": [common.file_hash(Path(f)) for f in a.figure], "mix": a.mix, "start": common.r3(a.start),
        "duration": common.r3(a.duration), "background": a.background, "rain": a.rain,
        "text": a.text, "hand": (list(hand_box) if hand_box else None), "strums": hits, "accents": accents,
        "rock": ROCK, "accent": ACCENT, "parts": (PARTS if a.parts else None), "strum": STRUM, "bob": BOB,
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
