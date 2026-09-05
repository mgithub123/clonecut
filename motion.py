"""Stage 8b - A small motion library.

Animation principles as curves: easing, anticipation, overshoot, follow-through,
settle, squash and stretch, and the lag a child carries behind its parent.
Everything here is pure numpy on frame-indexed arrays, so it can be checked
without a renderer, and every amplitude and duration is a named constant with
its unit. Nothing in a frame loop is a bare number.

Units: frames, and pixels on the rig's render canvas (the 3840-wide plates,
which is what rig.json's geometry is in). The caller scales to whatever canvas
it draws on. Angles are degrees, counter-clockwise on screen.
"""
from __future__ import annotations

import numpy as np

FPS = 24

# --- beat hits ---------------------------------------------------------------
# A hit is a small dip before it and a bounce after, never a step. The body
# dips on the anticipation, drops through on the hit, overshoots up, and settles.
# Sized against the Stage 7 render that read right: its head travelled 12px per
# beat in the finished frame. Body and head here add, and the squash below adds
# again at the head, so each is smaller than it looks.
BEAT = dict(
    anticipation_frames=3,   # frames before the hit the dip starts
    anticipation_px=0.8,     # how far the body lifts before dropping (px, up)
    hit_px=2.2,              # how far it drops on the hit (px, down)
    overshoot_px=0.8,        # how far it bounces past rest afterwards (px, up)
    settle_frames=9,         # frames from the hit to rest
    head_delay_frames=1,     # the head follows the body this much later
    head_scale=1.8,          # and moves this much more, being on a neck
)

# --- squash and stretch ------------------------------------------------------
# Tiny: the character must never visibly change proportion at rest. Squash on the
# hit, stretch on the release; volume is kept by moving width the other way.
# Scale is about the body's pivot at the feet, so a percent of height is the
# whole figure's height: on a 3700px doctor 1.8% moved the head 33px, a pogo.
SQUASH = dict(
    hit_pct=0.2,             # % shorter on the hit
    release_pct=0.15,        # % taller on the release
    release_offset_frames=3, # release follows the hit by this
    frames=5,                # duration of each half
)

# --- breathing and drift -----------------------------------------------------
BREATH = dict(period_s=4.2, body_px=1.6, head_px=1.6)
DRIFT = dict(
    head_dy_period_s=3.1, head_dy_px=1.1, head_dy_phase=0.7,
    head_dx_period_s=5.3, head_dx_px=1.8,
    head_rot_period_s=6.1, head_rot_deg=1.25, head_rot_phase=1.2,
)

# --- phrase ends -------------------------------------------------------------
# At the end of a sung line the whole body relaxes down and the head tilts,
# then holds until the next onset.
SETTLE = dict(
    frames=8,                # frames to relax into the settled pose
    body_px=2.4,             # body sinks by this (px, down)
    head_px=3.2,             # head sinks by this on top of the body
    head_tilt_deg=1.7,       # and tilts
    release_frames=5,        # frames to come back up at the next onset
)

# --- follow-through ----------------------------------------------------------
# Children lag their parents: a per-bone number in rig.json, defaults by name.
LAG_DEFAULT_FRAMES = {"ear": 2, "hair": 2, "bangs": 1, "strand": 3, "hand": 1,
                      "glove": 1, "tail": 3, "tag": 2, "collar": 1}
LAG_DAMP = 0.85              # how much of the parent's motion a lagging child keeps

# --- blink -------------------------------------------------------------------
BLINK = dict(every_s=2.5, curve=(0.62, 0.18, 0.0, 0.38, 0.82))   # eye height fractions

# --- head turns and gaze -----------------------------------------------------
TURN = dict(
    step_frames=2,           # frames each intermediate pose is held while turning
    eye_lead_frames=3,       # the eyes dart this many frames before the head turns
    dart_px=6.0,             # how far a pupil moves for a dart (px, toward the turn)
    dart_hold_frames=6,      # the dart holds this long, then eases back
)

# --- camera ------------------------------------------------------------------
CAMERA = dict(push_pct=4.5)  # slow push in over the whole shot, in %


# ------------------------------------------------------------------ easing

def ease_in_out(t: np.ndarray) -> np.ndarray:
    """Smoothstep: slow in, slow out."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def ease_out(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return 1 - (1 - t) ** 3


def ease_out_back(t: np.ndarray, overshoot: float = 1.70158) -> np.ndarray:
    """Overshoots the target, then comes back - the bounce after a hit."""
    t = np.clip(t, 0.0, 1.0)
    c1 = overshoot
    c3 = c1 + 1
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def decay_bounce(t: np.ndarray, cycles: float = 1.5) -> np.ndarray:
    """A damped sine from 1 to 0: what a thing does after it is let go."""
    t = np.clip(t, 0.0, 1.0)
    return np.cos(np.pi * cycles * t) * (1 - t) ** 2


# ------------------------------------------------------------------ curves

def impulse(n: int, hits: list[int], *, pre: int, pre_amp: float, amp: float,
            post_amp: float, settle: int) -> np.ndarray:
    """A beat hit per frame: anticipation before, drop on the hit, overshoot
    and settle after. Positive is down; the anticipation lifts.

    Overlapping hits add, so a fast passage builds rather than clicks.
    """
    out = np.zeros(n)
    for h in hits:
        # anticipation: ease up to -pre_amp, arriving on the hit
        for k in range(pre):
            f = h - pre + k
            if 0 <= f < n:
                out[f] += -pre_amp * ease_in_out(np.array((k + 1) / pre))
        # the hit and after: from amp down through rest to -post_amp and back
        for k in range(settle + 1):
            f = h + k
            if 0 <= f < n:
                t = k / settle
                out[f] += amp * decay_bounce(np.array(t), cycles=1.0) \
                    - post_amp * np.sin(np.pi * np.clip(t, 0, 1)) * (1 - t)
    return out


def squash_stretch(n: int, hits: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (scale_x, scale_y) multipliers, 1.0 at rest.

    Squash on the hit, stretch shortly after on the release; width goes the
    other way to keep volume. Never more than a couple of percent.
    """
    sy = np.ones(n)
    fr = SQUASH["frames"]
    for h in hits:
        for k in range(fr):
            f = h + k
            if 0 <= f < n:
                sy[f] -= SQUASH["hit_pct"] / 100 * np.sin(np.pi * (k + 0.5) / fr)
            f2 = h + SQUASH["release_offset_frames"] + k
            if 0 <= f2 < n:
                sy[f2] += SQUASH["release_pct"] / 100 * np.sin(np.pi * (k + 0.5) / fr)
    sx = 1.0 / np.sqrt(sy)          # keep the area
    return sx, sy


def breath(n: int, amp: float, period_s: float = BREATH["period_s"], phase: float = 0.0) -> np.ndarray:
    t = np.arange(n) / FPS
    return amp * np.sin(2 * np.pi * t / period_s + phase)


def settle(n: int, ends: list[int], onsets: list[int], *, amp: float) -> np.ndarray:
    """Relax by `amp` at each phrase end, hold, and release at the next onset."""
    out = np.zeros(n)
    onsets = sorted(onsets)
    for e in ends:
        nxt = next((o for o in onsets if o > e + 1), n)
        for f in range(e, min(n, nxt + SETTLE["release_frames"])):
            k = f - e
            if f < nxt:
                out[f] += amp * ease_in_out(np.array(min(1.0, k / SETTLE["frames"])))
            else:
                out[f] += amp * (1 - ease_out(np.array((f - nxt + 1) / SETTLE["release_frames"])))
    return out


def lag(track: np.ndarray, frames: int, damp: float = LAG_DAMP) -> np.ndarray:
    """A child's copy of its parent's motion, `frames` late and a little smaller.

    Only the *change* is delayed - the child starts where the parent starts -
    so a lagging ear does not jump on frame 1.
    """
    if frames <= 0:
        return track.copy()
    out = np.empty_like(track)
    out[:frames] = track[0]
    out[frames:] = track[:-frames]
    return track[0] + damp * (out - track[0])


def blink_track(n: int, every_s: float = BLINK["every_s"]) -> np.ndarray:
    """Eye height fraction per frame, 1.0 open."""
    b = np.ones(n)
    step = max(1, int(every_s * FPS))
    for s in range(step, n, step):
        for k, v in enumerate(BLINK["curve"]):
            if s + k < n:
                b[s + k] = v
    return b


def turn_track(n: int, schedule: list[tuple[int, int]], usable: list[int], start: int) -> list[int]:
    """Pose per frame. `schedule` is (frame, target pose); each turn steps through
    the usable poses between the current and the target, holding each for
    TURN['step_frames'], so a head turn eases rather than snaps."""
    poses = [start] * n
    cur = start
    order = sorted(usable)
    for at, target in sorted(schedule):
        if target not in order or cur not in order or at >= n:
            continue
        i0, i1 = order.index(cur), order.index(target)
        path = order[i0:i1 + 1] if i1 >= i0 else order[i1:i0 + 1][::-1]
        f = at
        for p in path[1:]:
            for _ in range(TURN["step_frames"]):
                if f < n:
                    poses[f] = p
                f += 1
        for g in range(f, n):
            poses[g] = target
        cur = target
    return poses


def gaze_track(n: int, poses: list[int], usable: list[int]) -> np.ndarray:
    """Pupil x offset per frame: a dart toward the coming turn a few frames
    before the head moves, held, then eased back to centre."""
    out = np.zeros(n)
    order = sorted(usable)
    for f in range(1, n):
        if poses[f] != poses[f - 1]:
            direction = 1.0 if order.index(poses[f]) > order.index(poses[f - 1]) else -1.0
            s = f - TURN["eye_lead_frames"]
            for k in range(TURN["dart_hold_frames"] + TURN["eye_lead_frames"]):
                g = s + k
                if 0 <= g < n:
                    out[g] = direction * TURN["dart_px"]
            for k in range(6):
                g = s + TURN["dart_hold_frames"] + TURN["eye_lead_frames"] + k
                if 0 <= g < n:
                    out[g] = direction * TURN["dart_px"] * (1 - ease_out(np.array((k + 1) / 6)))
    return out


def camera_push(n: int, pct: float = CAMERA["push_pct"]) -> np.ndarray:
    """Zoom factor per frame, 1.0 at the start."""
    return 1.0 + pct / 100 * np.arange(n) / max(1, n - 1)
