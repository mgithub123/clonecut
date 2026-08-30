#!/usr/bin/env python3
"""Fast assertions over the pure logic in the pipeline. No media, no network.

    uv run tools/selfcheck.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import common  # noqa: E402
import ingest  # noqa: E402

CHECKS: list[tuple[str, callable]] = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@check("parse_fraction handles rationals, plain floats and junk")
def _():
    assert abs(common.parse_fraction("30000/1001") - 29.97) < 0.01
    assert common.parse_fraction("25") == 25.0
    assert common.parse_fraction("0/0", 30.0) == 30.0
    assert common.parse_fraction(None, 12.0) == 12.0


@check("invert_regions turns silence into speech and covers the tail")
def _():
    silence = [{"start": 0.0, "end": 1.0}, {"start": 3.0, "end": 4.0}]
    speech = ingest.invert_regions(silence, duration=6.0)
    assert speech == [{"start": 1.0, "end": 3.0}, {"start": 4.0, "end": 6.0}], speech
    # fully silent clip yields no speech
    assert ingest.invert_regions([{"start": 0.0, "end": 5.0}], 5.0) == []
    # overlapping/unsorted input does not produce negative-length regions
    out = ingest.invert_regions([{"start": 2.0, "end": 4.0}, {"start": 0.0, "end": 3.0}], 5.0)
    assert all(r["end"] > r["start"] for r in out), out


@check("_beat_grid corrects a half-time lock and covers the whole track")
def _():
    period, duration = 0.5, 32.0                      # 120 BPM
    onsets = np.arange(0.02, duration, period)        # every real beat is played
    tracked = np.arange(8.52, duration, period * 2)   # tracker locked to half time, late start
    beats, bpm, notes = ingest._beat_grid(tracked, onsets, duration, tempo=60.0)
    assert notes["halved"] is True, notes
    assert abs(bpm - 120.0) < 1.0, bpm
    assert beats[0] < period, f"grid must reach the start of the track, got {beats[0]}"
    assert beats[-1] > duration - period, f"grid must reach the end, got {beats[-1]}"
    gaps = np.diff(beats)
    assert np.allclose(gaps, period, atol=0.01), gaps[:5]


@check("_beat_grid leaves a correctly-tracked grid at its own tempo")
def _():
    period, duration = 0.5, 20.0
    tracked = np.arange(0.0, duration, period)
    onsets = np.arange(0.0, duration, period)   # onsets only on the beat: no offbeats
    beats, bpm, notes = ingest._beat_grid(tracked, onsets, duration, tempo=120.0)
    assert notes["halved"] is False, notes
    assert abs(bpm - 120.0) < 1.0, bpm


@check("_section_map labels the loudest section high and the quietest low")
def _():
    sr = 22050
    t = np.arange(int(30 * sr)) / sr
    y = np.sin(2 * np.pi * 220 * t) * np.where(t < 10, 0.1, np.where(t < 20, 1.0, 0.4))
    hop = 512
    import librosa
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop)
    sections = ingest._section_map(y, sr, rms, times, 30.0)
    assert sections, "expected at least one section"
    labels = {s["label"] for s in sections}
    loudest = max(sections, key=lambda s: s["energy"])
    quietest = min(sections, key=lambda s: s["energy"])
    assert loudest["label"] == "high", sections
    assert quietest["label"] == "low", sections
    assert "high" in labels and "low" in labels


@check("transcribe() flattens faster-whisper segments into word timings")
def _(tmp=Path("cache/wav")):
    # The weights are a network dependency; the parsing around them is not.
    words = [SimpleNamespace(word=" hello", start=0.1, end=0.4, probability=0.9),
             SimpleNamespace(word=" world", start=0.4, end=0.8, probability=0.8)]
    seg = SimpleNamespace(start=0.1, end=0.8, text=" hello world", words=words)
    info = SimpleNamespace(language="en", language_probability=0.99)

    class FakeModel:
        def transcribe(self, path, **kwargs):
            assert kwargs.get("word_timestamps") is True
            return iter([seg]), info

    key = ("__fake__", ingest.config.WHISPER_DEVICE, ingest.config.WHISPER_COMPUTE_TYPE)
    ingest._WHISPER_CACHE[key] = FakeModel()
    src = Path("raw/test-a.mp4")
    if not src.exists():
        print("    (skipped: raw/test-a.mp4 missing - run tools/make_test_media.py)")
        return
    out = ingest.transcribe(src, "__fake__")
    assert out["language"] == "en"
    assert out["text"] == "hello world", out["text"]
    assert [w["word"] for w in out["words"]] == ["hello", "world"]
    assert out["words"][0]["start"] == 0.1 and out["words"][1]["end"] == 0.8
    assert out["segments"] == [{"start": 0.1, "end": 0.8, "text": "hello world"}]


def main() -> int:
    failures = 0
    for name, fn in CHECKS:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}\n        {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {name}\n        {type(exc).__name__}: {exc}")
        else:
            print(f"  ok    {name}")
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
