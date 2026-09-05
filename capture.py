#!/usr/bin/env python3
"""Stage 8c - Performance capture from a phone video.

    uv run capture.py raw/me-singing.mp4 --start 0 --duration 9
    uv run capture.py raw/me-singing.mp4 --out cache/capture/me.json

Given a video of a person singing the line, writes a JSON track of what their
face and body did, per frame at 24 fps: head yaw, pitch and roll (degrees),
eyebrow raise, eye openness and mouth openness (0-1), and, when the body is in
frame, shoulder lean and each hand's position relative to the shoulders. The
track is in the shape voice.py's analysis uses - start, duration, per-frame
arrays with timestamps - so perform.py --capture can map it onto the puppet.

MediaPipe does the seeing: its face landmarker gives 52 blendshape weights and a
head transform per frame, its pose landmarker gives shoulders and wrists. Model
files download from Google on first use into cache/models/, the way whisper's
weights do.

Raw landmark jitter on a flat drawing reads as a tremor, so every track goes
through a One Euro filter (Casiez, Cochet, Roussel 2012): a low-pass whose
cutoff rises with speed, so it kills noise when the face is still and lets a
fast turn through without lag. Parameters are named constants below.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

import common
import config
from common import ToolError

FPS = 24
WIDTH = 640                      # analysis width; landmarks do not need more
MODEL_DIR = config.CACHE_DIR / "models"
MODELS = {
    "face": ("face_landmarker.task",
             "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"),
    "pose": ("pose_landmarker_lite.task",
             "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"),
}

# One Euro filter: min_cutoff is the smoothing at rest (Hz, lower = smoother),
# beta is how much the cutoff opens with speed (higher = less lag on fast moves).
ONE_EURO = dict(min_cutoff=0.8, beta=0.03, d_cutoff=1.0)

# blendshape names MediaPipe reports, and how they combine into our tracks
BROW = ("browInnerUp", "browOuterUpLeft", "browOuterUpRight")
EYE_BLINK = ("eyeBlinkLeft", "eyeBlinkRight")
MOUTH_OPEN = ("jawOpen",)
POSE_MIN_VISIBILITY = 0.5


# ------------------------------------------------------------------ smoothing

def one_euro(x: np.ndarray, fps: float = FPS, *, min_cutoff: float = ONE_EURO["min_cutoff"],
             beta: float = ONE_EURO["beta"], d_cutoff: float = ONE_EURO["d_cutoff"]) -> np.ndarray:
    """One Euro filter over a 1-D track. NaN samples (no face) are held."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    dt = 1.0 / fps

    def alpha(cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    prev, dprev = None, 0.0
    for i, v in enumerate(x):
        if np.isnan(v):
            out[i] = prev if prev is not None else np.nan
            continue
        if prev is None:
            prev, out[i] = v, v
            continue
        dx = (v - prev) / dt
        dhat = dprev + alpha(d_cutoff) * (dx - dprev)
        cutoff = min_cutoff + beta * abs(dhat)
        s = prev + alpha(cutoff) * (v - prev)
        out[i], prev, dprev = s, s, dhat
    return out


# ------------------------------------------------------------------ frames

def _model(kind: str) -> Path:
    name, url = MODELS[kind]
    p = MODEL_DIR / name
    if not p.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {name}")
        try:
            urllib.request.urlretrieve(url, p)
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"could not download {name} from {url}: {exc}") from exc
    return p


def frames_of(video: Path, start: float, dur: float | None) -> tuple[np.ndarray, float]:
    """RGB frames at FPS, WIDTH wide, via ffmpeg. Returns (frames, aspect)."""
    common.require_binary(config.FFMPEG)
    info = common.ffprobe_json(video)
    vs = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if not vs:
        raise ToolError(f"{common.rel(video)} has no video stream")
    w, h = int(vs["width"]), int(vs["height"])
    rot = 0
    for sd in vs.get("side_data_list", []) or []:
        rot = int(float(sd.get("rotation", 0))) or rot
    if abs(rot) in (90, 270):
        w, h = h, w
    height = int(round(WIDTH * h / w / 2)) * 2
    cmd = [config.FFMPEG, "-v", "error", "-ss", str(start)]
    if dur:
        cmd += ["-t", str(dur)]
    cmd += ["-i", str(video), "-vf", f"fps={FPS},scale={WIDTH}:{height}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    n = len(raw) // (WIDTH * height * 3)
    if n == 0:
        raise ToolError(f"no frames decoded from {common.rel(video)}")
    return np.frombuffer(raw, np.uint8)[:n * WIDTH * height * 3].reshape(n, height, WIDTH, 3), w / h


def _head_angles(matrix: np.ndarray) -> tuple[float, float, float]:
    """Yaw, pitch, roll in degrees from MediaPipe's facial transformation matrix."""
    R = np.asarray(matrix)[:3, :3]
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(-R[2, 0], sy))          # nod
        yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))          # turn
        roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))         # tilt
    else:
        pitch = math.degrees(math.atan2(-R[2, 0], sy))
        yaw = 0.0
        roll = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
    # MediaPipe's matrix is camera-facing: its rotation about Y is the head turn
    # and about X the nod; the atan2 assignment above is by matrix row, so name
    # them by what they do to a face looking at the camera
    return yaw, pitch, roll


def analyse(video: Path, start: float = 0.0, dur: float | None = None, *, body: bool = True) -> dict:
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise ToolError("mediapipe is not installed; uv add mediapipe") from exc

    frames, aspect = frames_of(video, start, dur)
    n = len(frames)
    print(f"  {n} frames at {FPS} fps, {frames.shape[2]}x{frames.shape[1]}")

    face_opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_model("face"))),
        running_mode=vision.RunningMode.VIDEO, num_faces=1,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True)
    raw = {k: np.full(n, np.nan) for k in ("yaw", "pitch", "roll", "brow", "eye", "mouth")}
    shoulders = np.full((n, 2), np.nan)     # lean dx (as a fraction of shoulder width), dy
    hands = np.full((n, 2, 2), np.nan)      # per hand: x, y relative to shoulder centre, in shoulder widths
    seen_face = seen_body = 0
    with vision.FaceLandmarker.create_from_options(face_opts) as fl:
        pose_ctx = None
        if body:
            pose_opts = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(_model("pose"))),
                running_mode=vision.RunningMode.VIDEO, num_poses=1)
            pose_ctx = vision.PoseLandmarker.create_from_options(pose_opts)
        for i in range(n):
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frames[i]))
            ts = int(i * 1000 / FPS)
            res = fl.detect_for_video(img, ts)
            if res.face_blendshapes:
                seen_face += 1
                bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
                raw["brow"][i] = float(np.mean([bs.get(k, 0.0) for k in BROW]))
                raw["eye"][i] = 1.0 - float(np.mean([bs.get(k, 0.0) for k in EYE_BLINK]))
                raw["mouth"][i] = float(np.mean([bs.get(k, 0.0) for k in MOUTH_OPEN]))
                if res.facial_transformation_matrixes:
                    y, p_, r_ = _head_angles(res.facial_transformation_matrixes[0])
                    raw["yaw"][i], raw["pitch"][i], raw["roll"][i] = y, p_, r_
            if pose_ctx is not None:
                pr = pose_ctx.detect_for_video(img, ts)
                if pr.pose_landmarks:
                    lm = pr.pose_landmarks[0]
                    ls, rs, lw, rw = lm[11], lm[12], lm[15], lm[16]
                    if min(ls.visibility, rs.visibility) >= POSE_MIN_VISIBILITY:
                        seen_body += 1
                        cx, cy = (ls.x + rs.x) / 2, (ls.y + rs.y) / 2
                        sw = max(1e-3, abs(ls.x - rs.x))
                        shoulders[i] = [(cx - 0.5) / sw, (ls.y - rs.y) / sw]
                        for k, wr in enumerate((lw, rw)):
                            if wr.visibility >= POSE_MIN_VISIBILITY:
                                hands[i, k] = [(wr.x - cx) / sw, (wr.y - cy) / sw]
        if pose_ctx is not None:
            pose_ctx.close()

    tracks = {k: one_euro(v) for k, v in raw.items()}
    if seen_body:
        tracks["lean"] = one_euro(shoulders[:, 0])
        tracks["shoulder_tilt"] = one_euro(shoulders[:, 1])
        tracks["hand_l_x"], tracks["hand_l_y"] = one_euro(hands[:, 0, 0]), one_euro(hands[:, 0, 1])
        tracks["hand_r_x"], tracks["hand_r_y"] = one_euro(hands[:, 1, 0]), one_euro(hands[:, 1, 1])
    print(f"  face seen on {seen_face}/{n} frames, body on {seen_body}/{n}")
    if not seen_face:
        raise ToolError(f"no face found in {common.rel(video)}")

    def clean(a):
        return [None if np.isnan(v) else round(float(v), 4) for v in a]

    return {"video": common.rel(video), "video_sha256": common.file_hash(video),
            "start": common.r3(start), "duration": common.r3(n / FPS), "fps": FPS, "frames": n,
            "t": [round(i / FPS, 4) for i in range(n)],
            "face_frames": seen_face, "body_frames": seen_body,
            "tracks": {k: clean(v) for k, v in tracks.items()},
            "units": {"yaw": "degrees, + turns to the subject's left", "pitch": "degrees, + nods down",
                      "roll": "degrees, + tilts to the subject's left", "brow": "0-1 raise",
                      "eye": "0-1 open", "mouth": "0-1 open",
                      "lean": "shoulder centre offset from frame centre, in shoulder widths",
                      "hand_*": "wrist offset from the shoulder centre, in shoulder widths, y down"},
            "smoothing": {"filter": "one euro", **ONE_EURO},
            "mediapipe": __import__("mediapipe").__version__}


def load(path: Path) -> dict:
    d = common.read_json(path)
    for k in ("tracks", "frames", "fps"):
        if k not in d:
            raise ToolError(f"{common.rel(path)} is not a capture track (no {k})")
    return d


def resample(track: list, n: int) -> np.ndarray:
    """A capture track stretched or cut to the shot's frame count, NaN held."""
    a = np.array([np.nan if v is None else v for v in track], dtype=float)
    if len(a) == 0:
        return np.full(n, np.nan)
    idx = np.clip(np.round(np.arange(n) * (len(a) - 1) / max(1, n - 1)).astype(int), 0, len(a) - 1)
    out = a[idx]
    # hold the last seen value across gaps
    last = np.nan
    for i in range(n):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("video")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--no-body", action="store_true", help="skip the pose landmarker")
    p.add_argument("--out", default=None, help=f"default {common.rel(config.CACHE_DIR / 'capture')}/<stem>.json")
    a = p.parse_args(argv)
    video = Path(a.video).expanduser()
    if not video.exists():
        raise ToolError(f"no such video: {video}")
    print(f"capturing {common.rel(video)} {a.start}s" + (f" +{a.duration}s" if a.duration else ""))
    data = analyse(video, a.start, a.duration, body=not a.no_body)
    out = Path(a.out) if a.out else config.CACHE_DIR / "capture" / f"{video.stem}.json"
    common.write_json(out, data)
    tr = data["tracks"]
    yaw = [v for v in tr["yaw"] if v is not None]
    print(f"  yaw {min(yaw):+.0f}..{max(yaw):+.0f} deg, mouth open {np.nanmean([v or 0 for v in tr['mouth']]):.2f} mean, "
          f"eye {np.nanmean([v or 0 for v in tr['eye']]):.2f} mean")
    print("wrote", common.rel(out))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
