#!/usr/bin/env python3
"""Generate synthetic footage + a music track so the pipeline can be exercised
without real media. Not part of the pipeline - a fixture generator.

    uv run tools/make_test_media.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

SR = 44100


def write_music(path: Path, bpm: float = 120.0, bars: int = 16) -> None:
    """A 4/4 click-and-bass track with three energy sections, so beat tracking
    and the section map have something real to find."""
    beat = 60.0 / bpm
    duration = bars * 4 * beat
    t = np.arange(int(duration * SR)) / SR
    audio = np.zeros_like(t)

    for i in range(int(duration / beat)):
        start = i * beat
        idx = int(start * SR)
        env_t = np.arange(int(0.25 * SR)) / SR
        # section gain: quiet intro, loud middle, medium outro
        frac = start / duration
        gain = 0.35 if frac < 0.25 else (1.0 if frac < 0.7 else 0.6)
        kick = np.sin(2 * np.pi * 55 * env_t) * np.exp(-18 * env_t)
        hat = np.random.default_rng(i).normal(0, 1, env_t.size) * np.exp(-90 * env_t) * 0.15
        hit = (kick + (hat if i % 2 else 0)) * gain
        end = min(idx + hit.size, audio.size)
        audio[idx:end] += hit[: end - idx]

    pad = 0.25 * np.sin(2 * np.pi * 110 * t) * (0.4 + 0.6 * np.sin(2 * np.pi * t / duration) ** 2)
    audio = np.clip(audio + pad, -1.0, 1.0) * 0.8
    sf.write(path, audio.astype(np.float32), SR)
    print(f"  wrote {path}  ({duration:.1f}s, {bpm:g} BPM)")


def write_voice_track(path: Path, duration: float) -> None:
    """Tone bursts separated by real silence, so silencedetect has regions."""
    t = np.arange(int(duration * SR)) / SR
    audio = np.zeros_like(t)
    for start, end, freq in [(0.5, 2.4, 220), (3.6, 5.9, 330), (8.0, 11.2, 180), (13.0, 16.5, 260)]:
        if start >= duration:
            break
        mask = (t >= start) & (t < min(end, duration))
        wobble = 1 + 0.05 * np.sin(2 * np.pi * 5 * t[mask])
        audio[mask] = 0.4 * np.sin(2 * np.pi * freq * t[mask] * wobble)
    sf.write(path, audio.astype(np.float32), SR)


def write_video(path: Path, tmp_audio: Path, seconds_per_scene: int = 6) -> None:
    """Three visually distinct scenes with different motion levels, concatenated."""
    # Three visually distinct sources with different motion levels. Deliberately
    # cheap to render - mandelbrot and friends are ~20x slower and add nothing.
    sources = [
        "testsrc2=size=1280x720:rate=30",       # busy, high motion
        "smptebars=size=1280x720:rate=30",      # static, ~zero motion
        "gradients=size=1280x720:rate=30:speed=0.05",  # slow drift, low motion
    ]
    inputs: list[str] = []
    for src in sources:
        inputs += ["-f", "lavfi", "-t", str(seconds_per_scene), "-i", src]
    inputs += ["-i", str(tmp_audio)]

    n = len(sources)
    concat = "".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]"
    cmd = [
        config.FFMPEG, "-v", "error", "-nostdin", "-y", *inputs,
        "-filter_complex", concat,
        "-map", "[v]", "-map", f"{n}:a",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest",
        str(path),
    ]
    subprocess.run(cmd, check=True)
    print(f"  wrote {path}  ({n * seconds_per_scene}s, {n} scenes)")


def main() -> int:
    config.ensure_dirs()
    total = 18.0
    voice = config.CACHE_DIR / "_testvoice.wav"
    write_voice_track(voice, total)
    write_video(config.RAW_DIR / "test-a.mp4", voice)
    write_voice_track(voice, total)
    write_video(config.RAW_DIR / "test-b.mp4", voice, seconds_per_scene=5)
    write_music(config.MUSIC_DIR / "goodbye-party.wav")
    voice.unlink(missing_ok=True)
    print("\nNow run:\n  uv run ingest.py --video raw/test-a.mp4 raw/test-b.mp4 "
          "--audio music/goodbye-party.wav")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
