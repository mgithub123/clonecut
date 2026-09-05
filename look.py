"""Stage 8d - The look: lighting and background, keyable over a shot.

A cutout on a photo is evenly lit, and evenly lit reads as a sticker. This is
the finishing path's lighting: a grade on the background, a directional light
on the character, the rain's shadow drifting down the face, and a vignette
over the lot. Every dial can change over the shot, so the light can move with
the song - cold and hard through the verse, warm as the hook lands.

A look is a JSON file under assets/looks/, or a name that finds one there:

    uv run perform.py ... --look window-cold-to-warm

Each dial is a number (or an RGB triple) or a list of [seconds, value] keys,
interpolated linearly and held past the ends. Seconds are from the start of
the shot. The dials, with their rest values, are in DIALS below; a look that
sets none of them is the picture untouched, pixel for pixel.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

import common
import config
from common import ToolError

LOOKS_DIR = config.ROOT / "assets" / "looks"

# The dials and what they mean at rest. Multipliers are 1.0 for "leave alone".
DIALS = {
    "saturation": 1.0,          # background colour, 0 grey .. 1 as shot .. >1 richer
    "brightness": 1.0,          # background level
    "tint": [1.0, 1.0, 1.0],    # background RGB multipliers: cold is [0.85, 0.92, 1.1]
    "vignette": 0.0,            # 0 none .. 1 black corners, over the finished frame
    "light_side": 0.0,          # -1 lit from the left edge, +1 from the right, 0 flat
    "light_gain": 1.0,          # multiplier on the character's lit side
    "shadow_gain": 1.0,         # multiplier on the character's far side
    "light_warmth": [1.0, 1.0, 1.0],  # RGB multipliers on the lit side only
    "rain_shadow": 0.0,         # 0 none .. 1 the rain's streaks fully darken the face
}
LIGHT_RAMP = (0.2, 0.85)        # where across the character the light falls off, 0..1
VIGNETTE_START = 0.35           # distance from centre (in half-frames) where it begins
VIGNETTE_CENTRE_Y = 0.45        # the vignette's centre, as a fraction of height
RAIN_BLUR_PX = 3                # the rain is out of focus by the time it lands on him


# ------------------------------------------------------------------ loading

def load(name_or_path: str | Path) -> dict:
    """A look by name under assets/looks/, or a path. Unknown dials are refused."""
    p = Path(name_or_path)
    if not p.exists():
        cand = LOOKS_DIR / f"{name_or_path}.json"
        if not cand.exists():
            have = sorted(q.stem for q in LOOKS_DIR.glob("*.json"))
            raise ToolError(f"no look {name_or_path!r}; have {', '.join(have) or 'none'}")
        p = cand
    data = common.read_json(p)
    bad = sorted(set(data) - set(DIALS) - {"name", "note"})
    if bad:
        raise ToolError(f"{common.rel(p)} sets dials that do not exist: {', '.join(bad)}")
    data["_path"] = str(p)
    return data


def at(look: dict | None, t: float) -> dict:
    """The dials at time t, keys interpolated linearly and held past the ends."""
    out = {k: (list(v) if isinstance(v, list) else v) for k, v in DIALS.items()}
    if not look:
        return out
    for k, v in look.items():
        if k not in DIALS:
            continue
        out[k] = _value(v, t, DIALS[k])
    return out


def _value(v, t: float, rest):
    vec = isinstance(rest, list)
    keyed = isinstance(v, list) and v and isinstance(v[0], list) and len(v[0]) == 2 \
        and isinstance(v[0][0], (int, float)) and (vec == isinstance(v[0][1], list))
    if not keyed:
        return list(v) if vec else float(v)
    keys = sorted(v, key=lambda kv: kv[0])
    if t <= keys[0][0]:
        return list(keys[0][1]) if vec else float(keys[0][1])
    if t >= keys[-1][0]:
        return list(keys[-1][1]) if vec else float(keys[-1][1])
    for (t0, a), (t1, b) in zip(keys[:-1], keys[1:]):
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
            if vec:
                return [float(x + (y - x) * f) for x, y in zip(a, b)]
            return float(a + (b - a) * f)
    return list(keys[-1][1]) if vec else float(keys[-1][1])


def is_flat(d: dict) -> bool:
    return all(np.allclose(d[k], DIALS[k]) for k in DIALS)


# ------------------------------------------------------------------ applying

def grade(bg: Image.Image, d: dict) -> Image.Image:
    """Saturation, brightness and tint on the background."""
    if np.isclose(d["saturation"], 1) and np.isclose(d["brightness"], 1) and np.allclose(d["tint"], 1):
        return bg
    a = np.asarray(bg.convert("RGB")).astype(np.float32)
    if not np.isclose(d["saturation"], 1):
        lum = a @ np.array([0.299, 0.587, 0.114], np.float32)
        a = lum[..., None] + (a - lum[..., None]) * d["saturation"]
    a = a * d["brightness"] * np.array(d["tint"], np.float32)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def light(char: Image.Image, d: dict, rain_mask: np.ndarray | None = None) -> Image.Image:
    """A directional light across the character, and the rain's shadow on it.

    The light is a ramp across the character's width from the lit edge to the
    far edge: gain and warmth at one end, shadow at the other. It runs across
    the whole figure, which is what a window beside him does.
    """
    flat = np.isclose(d["light_side"], 0) or (np.isclose(d["light_gain"], 1) and np.isclose(d["shadow_gain"], 1)
                                              and np.allclose(d["light_warmth"], 1))
    if flat and (rain_mask is None or np.isclose(d["rain_shadow"], 0)):
        return char
    a = np.asarray(char.convert("RGBA")).astype(np.float32)
    rgb = a[..., :3]
    if not flat:
        # x runs from the lit edge: column 0 when lit from the left, the last
        # column when lit from the right
        x = np.linspace(0.0, 1.0, char.width, dtype=np.float32)
        if d["light_side"] > 0:
            x = 1.0 - x
        ramp = np.clip((x - LIGHT_RAMP[0]) / (LIGHT_RAMP[1] - LIGHT_RAMP[0]), 0, 1)
        # ramp is 0 at the lit edge, 1 at the far edge; scale by how far the side is set
        side = abs(float(d["light_side"]))
        gain = d["light_gain"] + (d["shadow_gain"] - d["light_gain"]) * ramp
        gain = 1.0 + (gain - 1.0) * side
        warmth = np.array(d["light_warmth"], np.float32)
        warm = 1.0 + (warmth - 1.0)[None, :] * ((1.0 - ramp) * side)[:, None]     # (w, 3)
        rgb = rgb * gain[None, :, None] * warm[None, :, :]
    if rain_mask is not None and not np.isclose(d["rain_shadow"], 0):
        m = rain_mask.astype(np.float32)
        rgb = rgb * (1.0 - d["rain_shadow"] * m)[..., None]
    a[..., :3] = np.clip(rgb, 0, 255)
    return Image.fromarray(a.astype(np.uint8))


def rain_shadow_mask(rain: Image.Image, box: tuple[int, int, int, int]) -> np.ndarray:
    """The rain layer's alpha under the character's box, blurred, 0..1."""
    m = rain.split()[3].filter(ImageFilter.GaussianBlur(RAIN_BLUR_PX)).crop(box)
    return np.asarray(m).astype(np.float32) / 255.0


def vignette(frame: Image.Image, strength: float) -> Image.Image:
    if strength <= 0:
        return frame
    W, H = frame.size
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    dist = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H * VIGNETTE_CENTRE_Y) / (H / 2)) ** 2)
    m = 1.0 - strength * np.clip(dist - VIGNETTE_START, 0, 1) ** 1.5
    a = np.asarray(frame.convert("RGB")).astype(np.float32) * m[..., None]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def describe(look: dict | None) -> dict | None:
    if not look:
        return None
    return {"name": look.get("name") or Path(look["_path"]).stem, "path": common.rel(Path(look["_path"])),
            "sha256": common.file_hash(Path(look["_path"])),
            "dials": {k: v for k, v in look.items() if k in DIALS}}
