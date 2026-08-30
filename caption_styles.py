#!/usr/bin/env python3
"""Caption look and placement. The whole point of this file is that you can
restyle every video by editing it, without touching render.py.

Coordinates are in pixels on the 1080x1920 output canvas.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

CANVAS_W = 1080
CANVAS_H = 1920

# --- safe area --------------------------------------------------------------
# Roughly where TikTok's own chrome sits. Captions are laid out inside this box
# so nothing important ends up under the UI.
#   top     status bar + the For You / Following tabs
#   bottom  username, description, music ticker
SAFE_TOP = 220
SAFE_BOTTOM = 480
SAFE_Y0 = SAFE_TOP
SAFE_Y1 = CANVAS_H - SAFE_BOTTOM          # 1440
SAFE_H = SAFE_Y1 - SAFE_Y0                 # 1220

# Distance between the baselines of wrapped lines, as a multiple of the font
# size, before line_spacing is added. Measured against drawtext's own layout so
# that render.py - which draws each line separately, to centre them - matches
# what drawtext would have done with a multi-line string.
LINE_HEIGHT_RATIO = 1.17

# Horizontal inset per position. The like/comment/share rail runs down the
# right edge over roughly the lower half, so only lower-third has to clear it;
# upper-third and center can use most of the width.
POSITIONS: dict[str, dict[str, float]] = {
    #                anchor: where the middle of the text block sits, as a
    #                fraction down the safe box
    "upper-third": {"anchor": 0.12, "margin_x": 72},
    "center":      {"anchor": 0.50, "margin_x": 72},
    "lower-third": {"anchor": 0.86, "margin_x": 200},
}

# --- styles -----------------------------------------------------------------
# Every key here maps onto drawtext options in render.py.
#
#   size              font size in px
#   color             fill colour
#   border_w/color    outline; this is what keeps text legible over any footage
#   shadow_*          drop shadow offset and colour
#   box               draw a background box behind the text
#   uppercase         transform the text before drawing
#   line_spacing      extra px between wrapped lines
#   width_frac        fraction of the available width the text may occupy
#   char_width_ratio  average glyph advance as a fraction of the font size,
#                     used to guess where to wrap (drawtext will not wrap)

BASE = {
    "size": 64,
    "color": "white",
    "border_w": 8,
    "border_color": "black@0.92",
    "shadow_x": 0,
    "shadow_y": 5,
    "shadow_color": "black@0.55",
    "box": False,
    "box_color": "black@0.55",
    "box_padding": 22,
    "uppercase": False,
    "line_spacing": 12,
    "width_frac": 1.0,
    "char_width_ratio": 0.58,
}

STYLES: dict[str, dict] = {
    # The first two seconds decide whether the video is watched at all.
    "hook": {
        **BASE,
        "size": 86,
        "border_w": 11,
        "shadow_y": 7,
        "uppercase": True,
        "char_width_ratio": 0.66,   # caps are wider
        "line_spacing": 16,
        "width_frac": 0.95,
    },
    # Ordinary narration.
    "body": {
        **BASE,
        "size": 60,
        "border_w": 7,
        "width_frac": 0.92,
    },
    # A single line you want to land, e.g. over a drop.
    "emphasis": {
        **BASE,
        "size": 96,
        "color": "#FFE44D",
        "border_w": 12,
        "border_color": "black",
        "shadow_y": 8,
        "uppercase": True,
        "char_width_ratio": 0.66,
        "line_spacing": 14,
        "width_frac": 0.9,
    },
}

# --- font -------------------------------------------------------------------
# A bold sans, wherever it lives on this machine. Override with LD_CAPTION_FONT.

FONT_CANDIDATES = [
    # macOS
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    # Windows
    "C:/Windows/Fonts/arialbd.ttf",
]


class FontNotFound(RuntimeError):
    pass


def resolve_font() -> str:
    override = os.environ.get("LD_CAPTION_FONT")
    if override:
        if not Path(override).exists():
            raise FontNotFound(f"LD_CAPTION_FONT points at a file that does not exist: {override}")
        return override

    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate

    # Last resort: ask fontconfig for any bold sans it knows about.
    if shutil.which("fc-match"):
        proc = subprocess.run(
            ["fc-match", "-f", "%{file}", "sans:bold"],
            capture_output=True, text=True,
        )
        path = proc.stdout.strip()
        if path and Path(path).exists():
            return path

    raise FontNotFound(
        "No bold sans font found. Set LD_CAPTION_FONT to a .ttf/.otf, or add one to "
        "FONT_CANDIDATES in caption_styles.py."
    )


def get_style(name: str) -> dict:
    try:
        return STYLES[name]
    except KeyError:
        raise KeyError(
            f"unknown caption style {name!r}; caption_styles.STYLES defines "
            f"{sorted(STYLES)}"
        ) from None


def layout(position: str) -> dict[str, float]:
    try:
        return POSITIONS[position]
    except KeyError:
        raise KeyError(
            f"unknown caption position {position!r}; caption_styles.POSITIONS defines "
            f"{sorted(POSITIONS)}"
        ) from None


if __name__ == "__main__":
    print(f"canvas     {CANVAS_W}x{CANVAS_H}")
    print(f"safe box   y {SAFE_Y0}..{SAFE_Y1}  (height {SAFE_H})")
    print(f"font       {resolve_font()}")
    print("styles:")
    for name, st in STYLES.items():
        print(f"  {name:<9} {st['size']}px  outline {st['border_w']}px  "
              f"{'UPPERCASE' if st['uppercase'] else 'as typed'}  {st['color']}")
    print("positions:")
    for name, pos in POSITIONS.items():
        y = SAFE_Y0 + pos["anchor"] * SAFE_H
        print(f"  {name:<12} centre y={y:.0f}px  side margin {pos['margin_x']:.0f}px")
