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
import schema  # noqa: E402

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


# --- schema -----------------------------------------------------------------

def _edl(**over):
    base = {
        "variant_name": "test-variant",
        "strategy_notes": "a note",
        "target_duration": 2.0,
        "audio": {"source": "music/goodbye-party.wav", "start": 0.0},
        "segments": [{"clip": "raw/test-a.mp4", "in": 1.0, "out": 3.0}],
        "captions": [],
    }
    base.update(over)
    return base


@check("EDL accepts the hand-written reference and derives its features")
def _():
    edl = schema.load_edl("examples/hand-written-reference.json")
    assert edl.variant_name == "hand-written-reference"
    assert len(edl.segments) == 8
    f = edl.derived_features()
    assert f["cut_count"] == 8
    assert f["beat_synced"] is True
    assert f["uses_speed_ramp"] is True, "segment 5 runs at 2x"
    assert f["hook_type"] == "we recorded this in one take"
    assert abs(f["avg_segment_length"] - edl.duration / 8) < 0.01
    # a 2x segment occupies half its source length in the output
    fast = next(s for s in edl.segments if s.speed == 2.0)
    assert abs(fast.output_duration - fast.source_duration / 2) < 1e-9


@check("EDL rejects unknown fields rather than silently ignoring them")
def _():
    from pydantic import ValidationError
    for bad in ({"duration": 5.0}, {"segments": [{"clip": "a.mp4", "in": 0, "out": 1, "fx": "glow"}]}):
        try:
            schema.EDL.model_validate(_edl(**bad))
        except ValidationError:
            continue
        raise AssertionError(f"expected {bad} to be rejected")


@check("EDL rejects reversed times, bad speeds and unsafe variant names")
def _():
    from pydantic import ValidationError
    cases = {
        "out before in": _edl(segments=[{"clip": "a.mp4", "in": 5.0, "out": 2.0}]),
        "zero-length segment": _edl(segments=[{"clip": "a.mp4", "in": 2.0, "out": 2.0}]),
        "negative speed": _edl(segments=[{"clip": "a.mp4", "in": 0.0, "out": 1.0, "speed": -1.0}]),
        "caption end before start": _edl(captions=[{"text": "x", "start": 3.0, "end": 1.0}]),
        "unknown caption style": _edl(captions=[{"text": "x", "start": 0.0, "end": 1.0, "style": "glow"}]),
        "path traversal in variant_name": _edl(variant_name="../../etc/passwd"),
        "spaces in variant_name": _edl(variant_name="my variant"),
        "no segments": _edl(segments=[]),
        "negative audio start": _edl(audio={"source": "a.wav", "start": -1.0}),
    }
    for label, payload in cases.items():
        try:
            schema.EDL.model_validate(payload)
        except ValidationError:
            continue
        raise AssertionError(f"expected {label!r} to be rejected")


@check("validate_media catches out-of-range timestamps and a track that runs out")
def _():
    if not Path("raw/test-a.mp4").exists():
        print("    (skipped: run tools/make_test_media.py)")
        return
    edl = schema.load_edl("examples/broken-out-of-range.json")
    errors, _ = schema.validate_media(edl)
    assert any("runs past the end of" in e for e in errors), errors
    assert any("only 32.000s long" in e for e in errors), errors
    try:
        schema.require_valid_media(edl)
    except schema.EdlError as exc:
        assert "runs past the end" in str(exc)
    else:
        raise AssertionError("require_valid_media should have raised")

    good = schema.load_edl("examples/hand-written-reference.json")
    errors, warnings = schema.validate_media(good)
    assert not errors, errors
    assert not warnings, warnings


@check("resolve_media finds a bare filename in the conventional directories")
def _():
    if not Path("raw/test-a.mp4").exists():
        print("    (skipped: run tools/make_test_media.py)")
        return
    assert schema.resolve_media("test-a.mp4").exists()
    assert schema.resolve_media("raw/test-a.mp4").exists()
    assert not schema.resolve_media("nope-does-not-exist.mp4").exists()


@check("an EDL survives a write/read round trip with 'in' and 'out' preserved")
def _():
    import json, tempfile
    edl = schema.load_edl("examples/hand-written-reference.json")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "rt.json"
        edl.write(p)
        raw = json.loads(p.read_text())
        assert "in" in raw["segments"][0] and "out" in raw["segments"][0], raw["segments"][0]
        assert "in_" not in raw["segments"][0]
        again = schema.load_edl(p)
    assert again.model_dump() == edl.model_dump()


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
