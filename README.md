# clonecut — Lucky Dog TikTok tool

Turns raw footage plus a music track into short-form vertical videos, in two
stages that stay strictly separated:

1. **Planning** (AI) produces a JSON Edit Decision List. It never touches ffmpeg.
2. **Rendering** (deterministic) consumes that EDL and produces an mp4. No API calls.

So any plan can be inspected and hand-edited before a render is paid for, and a
render can be repeated exactly without new API calls.

## Setup

```bash
uv sync
```

Requires `ffmpeg` and `ffprobe` on PATH (`brew install ffmpeg` /
`apt install ffmpeg`), and Python 3.11+.

## Build status

| Stage | File | Status |
|---|---|---|
| 1. Ingest | `ingest.py` | done |
| 2. Plan | `plan.py` | not started |
| 3. Render | `render.py` | not started |
| 4. Learn | `log.py` | not started |

---

## Stage 1 — Ingest

```bash
uv run ingest.py --video raw/a.mp4 raw/b.mp4 --audio music/track.wav
```

Writes a footage profile to `profiles/<audio-stem>.json` and prints a summary.

Per clip: duration, resolution, fps, rotation; scene cuts (PySceneDetect);
per-second motion energy (frame differencing); silence and speech regions
(ffmpeg `silencedetect`); a word-level transcript (faster-whisper); and
keyframes sampled every ~2s, written to `cache/keyframes/<hash>/` with paths
recorded in the profile.

For the music: BPM, a beat grid, an onset-strength curve, an RMS energy curve,
and a rough section map labelled low/mid/high.

### Caching

Every per-file analysis is cached under `cache/` keyed by the file's SHA-256, so
re-ingesting unchanged files is instant (~65ms vs ~44s cold on the test
fixtures). Hashes are memoised on size+mtime, but a file whose stat changed is
always re-hashed, so the cache key follows content, never timestamps. `--force`
ignores the cache; bumping `INGEST_VERSION` in `config.py` invalidates it.

### Two things worth knowing

**The beat grid is not raw librosa output.** `beat_track` frequently locks to
half tempo on sparse material, and only spans the stretch it was confident
about — on the test track it reported 60 BPM over 8.5s–32s when the truth was
120 BPM from 0s. Since Stage 2 snaps cuts to this grid, `_beat_grid()` recovers
the period and phase from the tracked beats, checks whether the offbeat lines
of a half-period grid are actually played (if they are, the tracker was in half
time), and then lays a uniform grid across the whole track. The uncorrected
value is kept as `bpm_raw`, and `beat_grid` records what was done.

**Section labels are relative to the track's own range**, not terciles — with
only three or four sections a tercile split can leave nothing labelled `high`.

### Tuning

Everything is in `config.py`: scene threshold, motion sampling resolution,
keyframe interval, silence floor, whisper model size.

Transcription downloads model weights from `huggingface.co` on first use
(cached in `~/.cache/huggingface`). Use `--no-transcript` to skip it.

## Test fixtures

No footage needed to try the pipeline:

```bash
uv run tools/make_test_media.py   # synthetic clips + a 120 BPM track
uv run tools/selfcheck.py         # pure-logic assertions, no media or network
```
