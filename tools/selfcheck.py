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
import config  # noqa: E402
import ingest  # noqa: E402
import schema  # noqa: E402
import render  # noqa: E402
import caption_styles as cs  # noqa: E402

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


# --- render -----------------------------------------------------------------

@check("q() escapes quotes so caption text cannot break the filtergraph")
def _():
    assert render.q("plain") == "'plain'"
    # a quote must close, escape and reopen - it cannot appear inside the quotes
    assert render.q("it's") == r"'it'\''s'"
    # colons and commas are literal inside quotes, so they need no escaping
    assert render.q("a:b,c") == "'a:b,c'"


@check("captions wrap to the available width and honour uppercase")
def _():
    style = dict(cs.STYLES["hook"])
    long = "we recorded this in one take and it took all afternoon"
    wrapped = render.wrap_caption(long, style, box_width=900)
    assert wrapped == wrapped.upper(), "hook style is uppercase"
    per_char = style["size"] * style["char_width_ratio"]
    for line in wrapped.split("\n"):
        assert len(line) * per_char <= 900 * 1.05, f"line too wide: {line!r}"
    assert len(wrapped.split("\n")) > 1, "expected this to wrap"
    # a short caption is left on one line
    assert "\n" not in render.wrap_caption("short", dict(cs.STYLES["body"]), 900)


@check("each wrapped line becomes its own centred drawtext")
def _():
    import tempfile
    cap = schema.Caption(text="we recorded this in one take and it took all afternoon",
                         start=0.0, end=2.0, style="hook", position="upper-third")
    with tempfile.TemporaryDirectory() as d:
        filters = render.caption_filters(cap, "/font.ttf", Path(d), 0)
        assert len(filters) > 1, "expected a multi-line caption to produce several filters"
        for f in filters:
            # centring each line is the whole point of drawing them separately
            assert "x='(w-text_w)/2'" in f, f
            assert "enable='between(t,0.000,2.000)'" in f
            # drawtext drops a whole line on a bare % and expands %{...} into
            # ffmpeg metadata unless expansion is switched off
            assert "expansion=none" in f, f
        # rows are stacked around the anchor, so the y values must all differ
        ys = [f.split("y='")[1].split("'")[0] for f in filters]
        assert len(set(ys)) == len(ys), ys
        # and the text files actually got written
        assert len(list(Path(d).glob("cap_000_*.txt"))) == len(filters)


@check("captions stay inside the TikTok safe area, however many lines they wrap to")
def _():
    import re, tempfile
    # A one-word-per-line caption is the worst case for block height.
    texts = {1: "short", 2: "one two three four five six seven",
             3: "one two three four five six seven eight nine ten eleven twelve",
             4: "a b c d e f g h i j k l m n o p q r s t u v w x y z aa bb cc dd ee ff"}
    with tempfile.TemporaryDirectory() as d:
        for pos_name in cs.POSITIONS:
            for style_name, style in cs.STYLES.items():
                for text in texts.values():
                    cap = schema.Caption(text=text, start=0.0, end=1.0,
                                         style=style_name, position=pos_name)
                    filters = render.caption_filters(cap, "/f.ttf", Path(d), 0)
                    half = style["size"] * cs.LINE_HEIGHT_RATIO / 2
                    ys = [float(re.search(r"y='([-\d.]+)-text_h/2'", f).group(1)) for f in filters]
                    top, bottom = min(ys) - half, max(ys) + half
                    label = f"{style_name}/{pos_name} on {len(filters)} lines"
                    assert top >= cs.SAFE_Y0 - 1, f"{label} rides above the safe area ({top:.0f})"
                    assert bottom <= cs.SAFE_Y1 + 1, f"{label} rides below the safe area ({bottom:.0f})"
    # Text is centred in the full frame, so a block of the widest permitted width
    # must still clear each position's side margins.
    for pos_name, pos in cs.POSITIONS.items():
        widest = (cs.CANVAS_W - 2 * pos["margin_x"]) * max(st["width_frac"] for st in cs.STYLES.values())
        assert (cs.CANVAS_W - widest) / 2 >= pos["margin_x"] - 1, pos_name


@check("build_command produces one input per segment and replaces the audio")
def _():
    import tempfile
    edl = schema.load_edl("examples/hand-written-reference.json")
    with tempfile.TemporaryDirectory() as d:
        cmd = render.build_command(edl, Path("out/x.mp4"), Path(d), "/font.ttf")
    assert cmd.count("-i") == len(edl.segments) + 1, "one input per segment, plus the music"
    fg = cmd[cmd.index("-filter_complex") + 1]
    assert f"concat=n={len(edl.segments)}:v=1:a=0" in fg
    # only the music input is mapped to audio: no source audio survives
    assert f"[{len(edl.segments)}:a]" in fg
    assert not any(f"[{i}:a]" in fg for i in range(len(edl.segments))), "source audio must not be used"
    assert cmd[cmd.index("-map") + 1] == "[vout]"
    assert "[aout]" in cmd
    # every segment is cropped to fill the vertical canvas
    assert fg.count(f"crop={cs.CANVAS_W}:{cs.CANVAS_H}") == len(edl.segments)
    assert fg.count("force_original_aspect_ratio=increase") == len(edl.segments)
    # the speed ramp reaches setpts
    fast = next(sg for sg in edl.segments if sg.speed != 1.0)
    assert f"setpts=(PTS-STARTPTS)/{fast.speed}" in fg


@check("segment in/out reach ffmpeg as input seeks, not full decodes")
def _():
    import tempfile
    edl = schema.load_edl("examples/hand-written-reference.json")
    with tempfile.TemporaryDirectory() as d:
        cmd = render.build_command(edl, Path("out/x.mp4"), Path(d), "/font.ttf")
    for seg in edl.segments:
        i = cmd.index(f"{seg.in_:.6f}")
        assert cmd[i - 1] == "-ss", "seek must come before -i to avoid decoding the whole clip"
        assert cmd[i + 1] == "-t" and cmd[i + 2] == f"{seg.source_duration:.6f}"
    # the music is seeked to audio.start and cut to the edit length
    ai = cmd.index(f"{edl.audio.start:.6f}")
    assert cmd[ai - 1] == "-ss"
    assert cmd[ai + 2] == f"{edl.duration:.6f}"


@check("render refuses an EDL whose timestamps exceed the media")
def _():
    if not Path("raw/test-a.mp4").exists():
        print("    (skipped: run tools/make_test_media.py)")
        return
    try:
        render.render(Path("examples/broken-out-of-range.json"), dry_run=True)
    except schema.EdlError as exc:
        assert "runs past the end" in str(exc), exc
    else:
        raise AssertionError("render should have refused the broken EDL")


@check("a caption containing % and %{} survives drawtext intact")
def _():
    import subprocess, tempfile
    font = cs.resolve_font()
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        cap = schema.Caption(text="100% live %{n}", start=0.0, end=1.0,
                             style="body", position="center")
        filters = render.caption_filters(cap, font, d, 0)
        assert len(filters) == 1, filters
        out = d / "probe.png"
        proc = subprocess.run(
            [config.FFMPEG, "-v", "error", "-nostdin", "-y",
             "-f", "lavfi", "-i", "color=black:s=1080x1920:d=1",
             "-vf", filters[0], "-frames:v", "1", str(out)],
            capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        assert "Stray %" not in proc.stderr, proc.stderr
        # something must actually have been drawn on the black frame
        import numpy as np
        raw = subprocess.run(
            [config.FFMPEG, "-v", "error", "-i", str(out), "-f", "rawvideo",
             "-pix_fmt", "gray", "-"], capture_output=True).stdout
        assert np.frombuffer(raw, dtype=np.uint8).max() > 200, "no text was drawn"


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
