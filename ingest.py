#!/usr/bin/env python3
"""Stage 1 - Ingest.

Turns raw video files plus a music track into a JSON "footage profile" that the
planner reads. Everything here is offline and deterministic, and every per-file
analysis is cached by content hash, so re-ingesting unchanged files is instant.

    uv run ingest.py --video raw/a.mp4 raw/b.mp4 --audio music/track.wav
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config
import common
from common import ToolError, r3, rel, run


# ---------------------------------------------------------------------------
# video: container probe
# ---------------------------------------------------------------------------

def probe_clip(path: Path) -> dict[str, Any]:
    info = common.ffprobe_json(path)
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise ToolError(f"{path} has no video stream.")

    fmt = info.get("format", {})
    duration = float(fmt.get("duration") or video.get("duration") or 0.0)
    if duration <= 0:
        raise ToolError(f"Could not determine a duration for {path}.")

    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)

    # Phone footage is often stored landscape with a rotation flag.
    rotation = 0
    for side in video.get("side_data_list", []) or []:
        if "rotation" in side:
            rotation = int(round(float(side["rotation"]))) % 360
    if not rotation:
        tag_rot = (video.get("tags") or {}).get("rotate")
        if tag_rot:
            rotation = int(round(float(tag_rot))) % 360
    if rotation in (90, 270):
        width, height = height, width

    fps = common.parse_fraction(video.get("avg_frame_rate")) or common.parse_fraction(
        video.get("r_frame_rate"), 30.0
    )

    return {
        "duration": r3(duration),
        "width": width,
        "height": height,
        "fps": round(fps, 4),
        "rotation": rotation,
        "codec": video.get("codec_name"),
        "has_audio": audio is not None,
        "audio_codec": (audio or {}).get("codec_name"),
    }


# ---------------------------------------------------------------------------
# video: scene cuts
# ---------------------------------------------------------------------------

def detect_scenes(path: Path, duration: float) -> tuple[list[float], list[dict[str, float]]]:
    from scenedetect import ContentDetector, detect

    try:
        raw = detect(str(path), ContentDetector(threshold=config.SCENE_THRESHOLD))
    except Exception as exc:  # pragma: no cover - decoder-specific
        print(f"  ! scene detection failed ({exc}); continuing with one scene", file=sys.stderr)
        raw = []

    scenes = [{"start": r3(a.seconds), "end": r3(b.seconds)} for a, b in raw]
    if not scenes:
        scenes = [{"start": 0.0, "end": r3(duration)}]
    cuts = [s["start"] for s in scenes[1:]]
    return cuts, scenes


# ---------------------------------------------------------------------------
# video: per-second motion energy (frame differencing)
# ---------------------------------------------------------------------------

def motion_energy(path: Path, duration: float) -> list[dict[str, float]]:
    import numpy as np

    common.require_binary(config.FFMPEG)
    w, h = config.MOTION_W, config.MOTION_H
    frame_bytes = w * h
    cmd = [
        config.FFMPEG, "-v", "error", "-nostdin",
        "-i", str(path),
        "-an", "-sn",
        "-vf", f"fps={config.MOTION_FPS},scale={w}:{h},format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    buckets: dict[int, list[float]] = {}
    prev = None
    index = 0
    assert proc.stdout is not None
    while True:
        buf = proc.stdout.read(frame_bytes)
        if not buf or len(buf) < frame_bytes:
            break
        frame = np.frombuffer(buf, dtype=np.uint8).astype(np.int16)
        t = index / config.MOTION_FPS
        if prev is not None:
            diff = float(np.abs(frame - prev).mean()) / 255.0
            buckets.setdefault(int(t), []).append(diff)
        prev = frame
        index += 1

    proc.stdout.close()
    stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
    proc.wait()
    if proc.returncode not in (0, None) and not buckets:
        raise ToolError(f"ffmpeg could not decode {path} for motion analysis:\n{stderr[-800:]}")

    total_seconds = max(1, int(duration))
    series = []
    for sec in range(total_seconds):
        vals = buckets.get(sec, [])
        series.append({"t": float(sec), "energy": r3(sum(vals) / len(vals)) if vals else 0.0})

    peak = max((p["energy"] for p in series), default=0.0)
    for p in series:
        p["energy_norm"] = r3(p["energy"] / peak) if peak > 0 else 0.0
    return series


# ---------------------------------------------------------------------------
# audio-in-video: silence / speech regions
# ---------------------------------------------------------------------------

_SILENCE_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


def silence_regions(path: Path, duration: float) -> list[dict[str, float]]:
    common.require_binary(config.FFMPEG)
    proc = run(
        [
            config.FFMPEG, "-v", "info", "-nostats", "-nostdin",
            "-i", str(path),
            "-af", f"silencedetect=noise={config.SILENCE_NOISE_DB}dB:d={config.SILENCE_MIN_DUR}",
            "-f", "null", "-",
        ],
        check=False,
    )
    text = proc.stderr or ""
    regions: list[dict[str, float]] = []
    pending: float | None = None
    for line in text.splitlines():
        m = _SILENCE_START.search(line)
        if m:
            pending = max(0.0, float(m.group(1)))
        m = _SILENCE_END.search(line)
        if m and pending is not None:
            end = min(duration, float(m.group(1)))
            if end > pending:
                regions.append({"start": r3(pending), "end": r3(end)})
            pending = None
    if pending is not None and duration > pending:
        regions.append({"start": r3(pending), "end": r3(duration)})
    return regions


def invert_regions(regions: list[dict[str, float]], duration: float, min_len: float = 0.15) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    cursor = 0.0
    for reg in sorted(regions, key=lambda r: r["start"]):
        if reg["start"] - cursor >= min_len:
            out.append({"start": r3(cursor), "end": r3(reg["start"])})
        cursor = max(cursor, reg["end"])
    if duration - cursor >= min_len:
        out.append({"start": r3(cursor), "end": r3(duration)})
    return out


# ---------------------------------------------------------------------------
# transcription
# ---------------------------------------------------------------------------

def extract_wav(src: Path, dest: Path, sample_rate: int = 16000) -> None:
    common.require_binary(config.FFMPEG)
    dest.parent.mkdir(parents=True, exist_ok=True)
    run([
        config.FFMPEG, "-v", "error", "-nostdin", "-y",
        "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-f", "wav", str(dest),
    ])


_WHISPER_CACHE: dict[tuple[str, str, str], Any] = {}


def _whisper_model(name: str):
    from faster_whisper import WhisperModel

    key = (name, config.WHISPER_DEVICE, config.WHISPER_COMPUTE_TYPE)
    if key not in _WHISPER_CACHE:
        try:
            _WHISPER_CACHE[key] = WhisperModel(
                name, device=config.WHISPER_DEVICE, compute_type=config.WHISPER_COMPUTE_TYPE
            )
        except Exception as exc:
            raise ToolError(
                f"could not load the faster-whisper model '{name}': {exc}\n"
                f"    Weights are fetched from huggingface.co on first use and cached in "
                f"~/.cache/huggingface. If that host is unreachable, either pre-download the "
                f"model once on a connected machine or pass --no-transcript."
            ) from exc
    return _WHISPER_CACHE[key]


def transcribe(path: Path, model_name: str) -> dict[str, Any]:
    """Word-level transcript via faster-whisper. Returns an empty transcript on failure."""
    wav = config.CACHE_DIR / "wav" / f"{path.stem}-{common.file_hash(path)[:12]}.wav"
    if not wav.exists():
        extract_wav(path, wav)

    model = _whisper_model(model_name)
    segments, info = model.transcribe(str(wav), word_timestamps=True, vad_filter=True)

    seg_out: list[dict[str, Any]] = []
    words_out: list[dict[str, Any]] = []
    for seg in segments:
        seg_out.append({"start": r3(seg.start), "end": r3(seg.end), "text": seg.text.strip()})
        for word in (seg.words or []):
            words_out.append({
                "word": word.word.strip(),
                "start": r3(word.start),
                "end": r3(word.end),
                "prob": round(float(word.probability), 3),
            })

    return {
        "language": info.language,
        "language_probability": round(float(info.language_probability or 0.0), 3),
        "model": model_name,
        "text": " ".join(s["text"] for s in seg_out).strip(),
        "segments": seg_out,
        "words": words_out,
    }


EMPTY_TRANSCRIPT: dict[str, Any] = {
    "language": None, "language_probability": 0.0, "model": None,
    "text": "", "segments": [], "words": [],
}


# ---------------------------------------------------------------------------
# keyframes
# ---------------------------------------------------------------------------

def sample_keyframes(path: Path, file_hash_hex: str, duration: float) -> list[dict[str, Any]]:
    common.require_binary(config.FFMPEG)
    out_dir = config.KEYFRAME_DIR / file_hash_hex
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob("kf_*.jpg"))
    if not existing:
        run([
            config.FFMPEG, "-v", "error", "-nostdin", "-y",
            "-i", str(path),
            "-vf", f"fps=1/{config.KEYFRAME_INTERVAL},scale={config.KEYFRAME_WIDTH}:-2",
            "-q:v", "3",
            str(out_dir / "kf_%04d.jpg"),
        ])
        existing = sorted(out_dir.glob("kf_*.jpg"))

    frames = []
    for i, frame_path in enumerate(existing):
        t = i * config.KEYFRAME_INTERVAL
        if t > duration:
            break
        frames.append({"t": r3(t), "path": rel(frame_path)})
    return frames


# ---------------------------------------------------------------------------
# music analysis
# ---------------------------------------------------------------------------

def analyse_music(path: Path) -> dict[str, Any]:
    import librosa
    import numpy as np

    y, sr = librosa.load(str(path), sr=config.ANALYSIS_SR, mono=True)
    duration = float(len(y) / sr)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_times = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    onsets_arr = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units="time")
    beats, bpm, grid_notes = _beat_grid(beat_times, onsets_arr, duration, tempo)
    onset_times = librosa.times_like(onset_env, sr=sr)
    peak = float(onset_env.max()) if onset_env.size else 0.0
    onset_curve = {
        "times": [r3(t) for t in onset_times.tolist()],
        "values": [r3(v / peak) if peak else 0.0 for v in onset_env.tolist()],
    }

    # Per-second RMS energy, used both directly and to label sections.
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_times = librosa.times_like(rms, sr=sr, hop_length=hop)
    rms_peak = float(rms.max()) if rms.size else 0.0
    energy_curve = {
        "times": [r3(t) for t in rms_times.tolist()],
        "values": [r3(v / rms_peak) if rms_peak else 0.0 for v in rms.tolist()],
    }

    sections = _section_map(y, sr, rms, rms_times, duration)

    return {
        "duration": r3(duration),
        "sample_rate": sr,
        "bpm": round(bpm, 2),
        "bpm_raw": round(tempo, 2),
        "beat_interval": r3(60.0 / bpm) if bpm > 0 else None,
        "beat_grid": grid_notes,
        "beats": beats,
        "downbeats": beats[::4],  # naive 4/4 assumption, useful as a coarse grid
        "onsets": [r3(t) for t in onsets_arr.tolist()],
        "onset_strength": onset_curve,
        "rms_energy": energy_curve,
        "sections": sections,
    }


def _offbeat_occupancy(onsets, phase: float, period: float, tol_frac: float = 0.18) -> float:
    """Fraction of the *offbeat* lines of a period/2 grid that have an onset on them.

    A half-period grid trivially fits onsets better than the full-period one - its
    lines are a superset. So instead of comparing fit, ask whether the lines the
    halved grid *adds* are actually played. If most of them are, the tracker
    locked to half time.
    """
    import numpy as np

    if period <= 0 or len(onsets) < 6:
        return 0.0
    onsets = np.asarray(onsets, dtype=float)
    tol = period * tol_frac
    lines = np.arange(phase + period / 2, onsets[-1], period)
    lines = lines[lines >= onsets[0]]
    if lines.size == 0:
        return 0.0
    hits = sum(1 for line in lines if np.min(np.abs(onsets - line)) <= tol)
    return hits / lines.size


def _beat_grid(beat_times, onsets, duration: float, tempo: float):
    """Turn librosa's tracked beats into a full-coverage, phase-locked grid.

    Raw tracker output is not good enough to snap cuts to for two reasons:
      * it frequently locks to half tempo on sparse or quiet material
      * it only spans the stretch it was confident about, so intros and outros
        have no beat to snap to at all
    We recover the period and phase from the tracked beats, correct an obvious
    half-time lock against the onsets, then lay a uniform grid over the track.
    """
    import numpy as np

    beat_times = np.asarray(beat_times, dtype=float)
    if beat_times.size >= 4:
        period = float(np.median(np.diff(beat_times)))
        phase = float(beat_times[0])
    elif tempo > 0:
        period, phase = 60.0 / tempo, 0.0
    else:
        return [r3(t) for t in beat_times.tolist()], tempo, {"method": "none"}

    notes = {"method": "phase-locked grid", "tracked_beats": int(beat_times.size),
             "tracked_span": [r3(beat_times[0]), r3(beat_times[-1])] if beat_times.size else None,
             "halved": False}

    occupancy = _offbeat_occupancy(onsets, phase, period)
    notes["offbeat_occupancy"] = r3(occupancy)
    if occupancy >= 0.6:
        period /= 2.0
        notes["halved"] = True

    # Extend the grid backwards to the start of the track and forwards to the end.
    first = phase - period * np.floor(phase / period + 1e-9)
    grid = np.arange(first, duration, period)
    bpm = 60.0 / period if period > 0 else tempo
    notes["extended_to"] = [r3(float(grid[0])), r3(float(grid[-1]))] if grid.size else None
    return [r3(t) for t in grid.tolist()], bpm, notes


def _section_map(y, sr, rms, rms_times, duration: float) -> list[dict[str, Any]]:
    """Rough energy-over-time map: timbre-based boundaries, energy-based labels."""
    import librosa
    import numpy as np

    n_sections = int(max(3, min(12, round(duration / config.SECTION_TARGET_LEN))))
    bounds_t: list[float]
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        feat = np.vstack([librosa.util.normalize(mfcc, axis=1),
                          librosa.util.normalize(chroma, axis=1)])
        idx = librosa.segment.agglomerative(feat, n_sections)
        bounds_t = [float(t) for t in librosa.frames_to_time(idx, sr=sr).tolist()]
    except Exception:
        bounds_t = list(np.linspace(0.0, duration, n_sections + 1)[:-1])

    bounds_t = sorted({0.0, *[b for b in bounds_t if 0.0 < b < duration]})
    edges = bounds_t + [duration]

    rms_peak = float(rms.max()) if rms.size else 0.0
    raw = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end - start < 0.5:
            continue
        mask = (rms_times >= start) & (rms_times < end)
        energy = float(rms[mask].mean()) if mask.any() else 0.0
        raw.append({"start": r3(start), "end": r3(end),
                    "energy": r3(energy / rms_peak) if rms_peak else 0.0})

    if raw:
        # Label relative to this track's own range rather than by tercile: with only
        # a handful of sections a tercile split can leave nothing labelled "high".
        levels = [s["energy"] for s in raw]
        lo, hi = min(levels), max(levels)
        span = hi - lo
        for i, s in enumerate(raw):
            frac = (s["energy"] - lo) / span if span > 1e-6 else 0.5
            s["index"] = i
            s["energy_rel"] = r3(frac)
            s["label"] = "high" if frac >= 0.66 else ("low" if frac <= 0.33 else "mid")
    return raw


# ---------------------------------------------------------------------------
# per-file orchestration (with cache)
# ---------------------------------------------------------------------------

def ingest_clip(path: Path, *, whisper_model: str, do_transcript: bool, force: bool) -> dict[str, Any]:
    path = path.resolve()
    if not path.exists():
        raise ToolError(f"Video not found: {path}")
    digest = common.file_hash(path)

    cached = None if force else common.cache_load("clips", digest)
    if cached is not None:
        wanted = whisper_model if do_transcript else None
        have = (cached.get("transcript") or {}).get("model")
        if wanted == have:
            cached["path"] = rel(path)
            print(f"  cache hit  {path.name}  ({digest[:12]})")
            return cached

    print(f"  analysing  {path.name}  ({digest[:12]})")
    meta = probe_clip(path)
    duration = meta["duration"]

    cuts, scenes = detect_scenes(path, duration)
    motion = motion_energy(path, duration)
    silence = silence_regions(path, duration) if meta["has_audio"] else []
    speech = invert_regions(silence, duration) if meta["has_audio"] else []
    keyframes = sample_keyframes(path, digest, duration)

    if do_transcript and meta["has_audio"]:
        try:
            transcript = transcribe(path, whisper_model)
        except Exception as exc:
            print(f"  ! transcription failed ({exc}); continuing without it", file=sys.stderr)
            transcript = dict(EMPTY_TRANSCRIPT)
    else:
        transcript = dict(EMPTY_TRANSCRIPT)

    record = {
        "path": rel(path),
        "hash": digest,
        **meta,
        "scene_cuts": cuts,
        "scenes": scenes,
        "motion": motion,
        "silence": silence,
        "speech": speech,
        "transcript": transcript,
        "keyframes": keyframes,
    }
    common.cache_store("clips", digest, record)
    return record


def ingest_audio(path: Path, *, force: bool) -> dict[str, Any]:
    path = path.resolve()
    if not path.exists():
        raise ToolError(f"Audio not found: {path}")
    digest = common.file_hash(path)

    cached = None if force else common.cache_load("audio", digest)
    if cached is not None:
        cached["path"] = rel(path)
        print(f"  cache hit  {path.name}  ({digest[:12]})")
        return cached

    print(f"  analysing  {path.name}  ({digest[:12]})")
    record = {"path": rel(path), "hash": digest, **analyse_music(path)}
    common.cache_store("audio", digest, record)
    return record


def build_profile(videos: list[Path], audio: Path, *, whisper_model: str,
                  do_transcript: bool, force: bool) -> dict[str, Any]:
    config.ensure_dirs()
    print("clips:")
    clips = [ingest_clip(v, whisper_model=whisper_model, do_transcript=do_transcript, force=force)
             for v in videos]
    print("music:")
    track = ingest_audio(audio, force=force)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ingest_version": config.INGEST_VERSION,
        "clips": clips,
        "audio": track,
    }


# ---------------------------------------------------------------------------
# summary printing
# ---------------------------------------------------------------------------

def print_summary(profile: dict[str, Any]) -> None:
    print("\n" + "=" * 62)
    print("FOOTAGE PROFILE")
    print("=" * 62)
    for clip in profile["clips"]:
        print(f"\n{clip['path']}")
        print(f"  {clip['width']}x{clip['height']} @ {clip['fps']:g}fps   "
              f"{clip['duration']:.2f}s   audio={'yes' if clip['has_audio'] else 'no'}")
        cuts = clip["scene_cuts"]
        print(f"  scenes: {len(clip['scenes'])}"
              + (f"  cuts at {', '.join(f'{c:.2f}' for c in cuts[:8])}"
                 + (" ..." if len(cuts) > 8 else "") if cuts else ""))
        motion = clip["motion"]
        if motion:
            top = sorted(motion, key=lambda m: -m["energy"])[:5]
            print("  motion peaks: " + ", ".join(f"{m['t']:.0f}s={m['energy']:.3f}" for m in top))
            print("  motion       " + _sparkline([m["energy_norm"] for m in motion]))
        print(f"  silence regions: {len(clip['silence'])}   speech regions: {len(clip['speech'])}")
        print(f"  keyframes: {len(clip['keyframes'])}"
              + (f"  -> {Path(clip['keyframes'][0]['path']).parent}" if clip["keyframes"] else ""))
        tr = clip["transcript"]
        if tr["text"]:
            preview = tr["text"][:160] + ("..." if len(tr["text"]) > 160 else "")
            print(f"  transcript ({tr['language']}, {len(tr['words'])} words): {preview}")
        else:
            print("  transcript: (none)")

    a = profile["audio"]
    print(f"\n{a['path']}")
    grid = a.get("beat_grid") or {}
    corrected = "  (corrected from %.1f, half-time lock)" % a["bpm_raw"] if grid.get("halved") else ""
    print(f"  {a['duration']:.2f}s   {a['bpm']:.1f} BPM{corrected}")
    print(f"  beat grid: {len(a['beats'])} beats spanning "
          f"{a['beats'][0]:.2f}-{a['beats'][-1]:.2f}s   {len(a['onsets'])} onsets")
    print("  energy      " + _sparkline(a["rms_energy"]["values"][::max(1, len(a['rms_energy']['values']) // 60)]))
    print("  sections:")
    for s in a["sections"]:
        print(f"    [{s['index']:>2}] {s['start']:7.2f} - {s['end']:7.2f}  "
              f"{s['label']:<4} energy={s['energy']:.3f}")
    print()


def _sparkline(values: list[float]) -> str:
    if not values:
        return ""
    blocks = " ▁▂▃▄▅▆▇█"
    hi = max(values) or 1.0
    return "".join(blocks[min(8, int(round(v / hi * 8)))] for v in values)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 1: analyse footage + music into a JSON profile.")
    parser.add_argument("--video", "-v", nargs="+", required=True, type=Path, help="one or more video files")
    parser.add_argument("--audio", "-a", required=True, type=Path, help="the music track")
    parser.add_argument("--out", "-o", type=Path, default=None,
                        help="profile output path (default: profiles/<audio-stem>.json)")
    parser.add_argument("--whisper-model", default=config.WHISPER_MODEL,
                        help=f"faster-whisper model size (default: {config.WHISPER_MODEL})")
    parser.add_argument("--no-transcript", action="store_true", help="skip transcription")
    parser.add_argument("--force", action="store_true", help="ignore the cache and re-analyse")
    parser.add_argument("--quiet", action="store_true", help="skip the human-readable summary")
    args = parser.parse_args(argv)

    try:
        profile = build_profile(
            args.video, args.audio,
            whisper_model=args.whisper_model,
            do_transcript=not args.no_transcript,
            force=args.force,
        )
    except ToolError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1

    out = args.out or (config.PROFILE_DIR / f"{args.audio.stem}.json")
    common.write_json(out, profile)

    if not args.quiet:
        print_summary(profile)
    print(f"profile written to {rel(out)} "
          f"({out.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
