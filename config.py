"""Central configuration for the Lucky Dog tool.

Everything tunable lives here so the pipeline stages stay boring.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

RAW_DIR = ROOT / "raw"
MUSIC_DIR = ROOT / "music"
CACHE_DIR = ROOT / "cache"
KEYFRAME_DIR = CACHE_DIR / "keyframes"
PROFILE_DIR = ROOT / "profiles"
PLANS_DIR = ROOT / "plans"
OUT_DIR = ROOT / "out"
DB_PATH = ROOT / "luckydog.db"

FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE_BIN", "ffprobe")

# Bump when the shape of a cached analysis changes; invalidates every cache entry.
INGEST_VERSION = 2

# --- video analysis ---------------------------------------------------------
SCENE_THRESHOLD = 27.0          # PySceneDetect ContentDetector
MOTION_FPS = 4                  # frames/sec decoded for frame differencing
MOTION_W, MOTION_H = 64, 36     # differencing resolution (tiny on purpose)
KEYFRAME_INTERVAL = 2.0         # seconds between sampled keyframes
KEYFRAME_WIDTH = 512            # keyframes get base64'd into the plan prompt

# --- audio analysis ---------------------------------------------------------
SILENCE_NOISE_DB = -30          # ffmpeg silencedetect noise floor
SILENCE_MIN_DUR = 0.35          # ffmpeg silencedetect minimum silence length
ANALYSIS_SR = 22050             # librosa load rate
SECTION_TARGET_LEN = 15.0       # rough seconds per section in the section map

# --- transcription ----------------------------------------------------------
WHISPER_MODEL = os.environ.get("LD_WHISPER_MODEL", "base")
WHISPER_COMPUTE_TYPE = os.environ.get("LD_WHISPER_COMPUTE", "int8")
WHISPER_DEVICE = os.environ.get("LD_WHISPER_DEVICE", "cpu")

# --- planning ---------------------------------------------------------------
# The planner runs in manual mode: it writes a prompt for you to paste into the
# Claude app you already have, and reads the reply back. Nothing here calls an
# API, so the whole pipeline is offline.
PLAN_VARIANTS = 3               # how many genuinely different edits to ask for
PLAN_KEYFRAME_BUDGET = 12       # how many keyframes to put in front of the model
MIN_HISTORY_FOR_RETRIEVAL = 8   # below this, history is reported as not meaningful


def ensure_dirs() -> None:
    for d in (RAW_DIR, MUSIC_DIR, CACHE_DIR, KEYFRAME_DIR, PROFILE_DIR, PLANS_DIR, OUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
