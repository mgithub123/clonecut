#!/usr/bin/env python3
"""Fast assertions over the pure logic in the pipeline. No media, no network.

    uv run tools/selfcheck.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json  # noqa: E402
import numpy as np  # noqa: E402

import common  # noqa: E402
import config  # noqa: E402
import ingest  # noqa: E402
import schema  # noqa: E402
import render  # noqa: E402
import caption_styles as cs  # noqa: E402
import plan  # noqa: E402
import log  # noqa: E402

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


# --- plan -------------------------------------------------------------------

_FIXTURE: dict | None = None


def _fixture() -> dict:
    """A synthetic ingest profile, built once in a temp dir.

    These checks used to read ``profiles/goodbye-party.json``, which only exists
    after ``make_test_media.py`` and ``ingest.py`` have been run - so seven of
    them failed on any checkout that had not, despite the file's promise of "no
    media". The shapes here mirror what ``ingest.py`` writes, and the durations
    mirror ``tools/make_test_media.py`` (test-a 18s, test-b 15s, a 32s track at
    120 BPM) so the same EDLs stay valid against the real fixtures when those do
    exist. The media files are empty: enough for ``resolve_media`` to find them,
    and a failed probe is a warning, not an error.
    """
    global _FIXTURE
    if _FIXTURE is not None:
        return _FIXTURE
    import atexit
    import shutil
    import tempfile
    from PIL import Image

    root = Path(tempfile.mkdtemp(prefix="clonecut-selfcheck-"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    for d in ("raw", "music", "keyframes", "profiles"):
        (root / d).mkdir()
    for name in ("raw/test-a.mp4", "raw/test-b.mp4", "music/goodbye-party.wav"):
        (root / name).write_bytes(b"")

    def clip(stem: str, duration: float, scene_len: int) -> dict:
        scenes = [{"start": float(s), "end": float(min(s + scene_len, duration))}
                  for s in range(0, int(duration), scene_len)]
        motion = [{"t": float(t), "energy": round(((t * 7) % 10) / 10, 3),
                   "energy_norm": round(((t * 7) % 10) / 10, 3)}
                  for t in range(int(duration))]
        keyframes = []
        for t in range(0, int(duration), 2):
            p = root / "keyframes" / f"{stem}-{t:04d}.png"
            Image.new("RGB", (8, 8), (t * 10 % 255, 40, 40)).save(p)
            keyframes.append({"t": float(t), "path": str(p)})
        silence = [{"start": 2.4, "end": 3.6}, {"start": 5.9, "end": 8.0}]
        return {
            "path": f"raw/{stem}.mp4", "hash": "0" * 64, "duration": duration,
            "width": 1280, "height": 720, "fps": 30.0, "rotation": 0, "has_audio": True,
            "scene_cuts": [s["start"] for s in scenes[1:]], "scenes": scenes,
            "motion": motion, "silence": silence,
            "speech": ingest.invert_regions(silence, duration),
            "transcript": {"text": "", "segments": [], "words": []},
            "keyframes": keyframes,
        }

    duration, beat = 32.0, 0.5
    beats = [round(0.035 + beat * i, 3) for i in range(int(duration / beat))]
    times = [round(i * 0.1, 3) for i in range(int(duration * 10))]
    sections = [
        {"start": 0.0, "end": 8.0, "energy": 0.35, "index": 0, "energy_rel": 0.0, "label": "low"},
        {"start": 8.0, "end": 22.4, "energy": 1.0, "index": 1, "energy_rel": 1.0, "label": "high"},
        {"start": 22.4, "end": duration, "energy": 0.6, "index": 2, "energy_rel": 0.385, "label": "mid"},
    ]
    profile = {
        "created_at": "2026-07-01T00:00:00+00:00",
        "ingest_version": config.INGEST_VERSION,
        "clips": [clip("test-a", 18.0, 6), clip("test-b", 15.0, 5)],
        "audio": {
            "path": "music/goodbye-party.wav", "hash": "1" * 64,
            "duration": duration, "sample_rate": 22050, "bpm": 120.0, "bpm_raw": 60.0,
            "beat_interval": beat, "beat_grid": {"halved": True},
            "beats": beats, "downbeats": beats[::4], "onsets": beats,
            "onset_strength": {"times": times, "values": [float(i % 5 == 0) for i in range(len(times))]},
            "rms_energy": {"times": times,
                           "values": [next(s["energy"] for s in sections if s["start"] <= t < s["end"])
                                      for t in times]},
            "sections": sections,
        },
    }
    common.write_json(root / "profiles" / "goodbye-party.json", profile)
    _FIXTURE = {"root": root, "profile": profile}
    return _FIXTURE


def _profile():
    return _fixture()["profile"]


class _in_fixture_dir:
    """Run with the fixture root as cwd, so ``schema.resolve_media`` finds the
    placeholder media through its ``Path.cwd()`` fallback."""

    def __enter__(self):
        import os
        self.prev = os.getcwd()
        os.chdir(_fixture()["root"])

    def __exit__(self, *exc):
        import os
        os.chdir(self.prev)


def _plan_edl(segments, audio_start=0.035, **over):
    base = {
        "variant_name": "snap-test", "strategy_notes": "n", "target_duration": 1.0,
        "audio": {"source": "music/goodbye-party.wav", "start": audio_start},
        "segments": segments, "captions": [],
    }
    base.update(over)
    return schema.EDL.model_validate(base)


@check("downsample_curve thins a dense curve but keeps its shape")
def _():
    times = [i * 0.01 for i in range(1000)]
    values = [float(i < 500) for i in range(1000)]
    out = plan.downsample_curve({"times": times, "values": values}, points=20)
    assert len(out) == 20, len(out)
    assert out[0][1] == 1.0 and out[-1][1] == 0.0, out
    assert out[0][0] == 0.0
    # a short curve is passed through untouched
    short = {"times": [0.0, 1.0], "values": [0.5, 0.6]}
    assert plan.downsample_curve(short, points=20) == [[0.0, 0.5], [1.0, 0.6]]
    assert plan.downsample_curve({"times": [], "values": []}) == []


@check("extract_json_objects copes with how a chat reply is actually shaped")
def _():
    obj = '{"a": 1}'
    cases = {
        "bare object": obj,
        "bare array": f"[{obj}]",
        "one fenced block": f"```json\n[{obj}]\n```",
        "prose around a fence": f"Here you go:\n```json\n[{obj}]\n```\nHope that helps!",
        "several fenced blocks": f"one\n```json\n{obj}\n```\ntwo\n```json\n{obj}\n```",
        "unlabelled fence": f"```\n[{obj}]\n```",
        "prose, no fence": f"Sure. {obj} That is variant one.",
    }
    for label, text in cases.items():
        found = plan.extract_json_objects(text)
        assert found and all(f == {"a": 1} for f in found), f"{label}: {found}"
    # braces inside strings must not confuse the scanner
    tricky = '{"text": "a } b { c", "n": 2}'
    assert plan.extract_json_objects(tricky) == [{"text": "a } b { c", "n": 2}]
    assert plan.extract_json_objects("no json at all") == []


@check("snapping puts every snapped cut on a beat and leaves the others alone")
def _():
    profile = _profile()
    beats = profile["audio"]["beats"]
    edl = _plan_edl([
        {"clip": "raw/test-a.mp4", "in": 12.0, "out": 12.83, "snap_to_beat": True},
        {"clip": "raw/test-b.mp4", "in": 5.2, "out": 6.1, "snap_to_beat": True},
        {"clip": "raw/test-a.mp4", "in": 2.0, "out": 5.9, "speed": 2.0, "snap_to_beat": True},
        {"clip": "raw/test-b.mp4", "in": 0.2, "out": 3.4, "snap_to_beat": False},
    ], audio_start=8.17)
    snapped, report = plan.snap_edl_to_beats(edl, profile)
    assert snapped.audio.start in beats, "the grid itself must be aligned first"

    t = 0.0
    for i, seg in enumerate(snapped.segments):
        t += seg.output_duration
        music_t = snapped.audio.start + t
        off = min(abs(music_t - b) for b in beats)
        if seg.snap_to_beat:
            assert off < 0.002, f"segment {i} lands {off*1000:.0f}ms off the beat"
        else:
            assert off > 0.05, f"segment {i} was not asked to snap but did"
    assert abs(snapped.target_duration - t) < 0.002, "target_duration must follow the snap"
    assert report, "snapping should say what it moved"


@check("snapping never pushes a segment past the end of its clip")
def _():
    profile = _profile()
    dur = next(c["duration"] for c in profile["clips"] if c["path"] == "raw/test-b.mp4")
    # out is close enough to the clip end that the nearest beat forward overruns it
    edl = _plan_edl([{"clip": "raw/test-b.mp4", "in": 14.2, "out": 14.95, "snap_to_beat": True}])
    snapped, report = plan.snap_edl_to_beats(edl, profile)
    seg = snapped.segments[0]
    assert seg.out_ <= dur, f"snapped out {seg.out_} exceeds clip duration {dur}"
    assert seg.out_ > seg.in_
    # it should have fallen back to the earlier beat rather than giving up
    assert seg.snap_to_beat is True, report
    with _in_fixture_dir():
        errors, _ = schema.validate_media(snapped)
    assert not errors, errors


@check("snapping refuses to collapse a segment to nothing")
def _():
    profile = _profile()
    # a very short segment whose nearest beat is behind its start
    edl = _plan_edl([{"clip": "raw/test-a.mp4", "in": 1.0, "out": 1.06, "snap_to_beat": True}],
                    audio_start=0.035)
    snapped, report = plan.snap_edl_to_beats(edl, profile)
    seg = snapped.segments[0]
    assert seg.output_duration >= plan.MIN_SEGMENT or seg.snap_to_beat is False, \
        f"left a {seg.output_duration:.3f}s segment"
    if not seg.snap_to_beat:
        assert any("unsnapped" in line for line in report), report


@check("snapping is a no-op when nothing asks for it")
def _():
    profile = _profile()
    segs = [{"clip": "raw/test-a.mp4", "in": 1.0, "out": 3.7, "snap_to_beat": False}]
    edl = _plan_edl(segs, audio_start=1.234)
    snapped, _ = plan.snap_edl_to_beats(edl, profile)
    assert snapped.audio.start == 1.234, "audio must not move if no segment snaps"
    assert snapped.segments[0].out_ == 3.7


@check("keyframe selection respects the budget and covers every clip")
def _():
    profile = _profile()
    for budget in (2, 6, 12, 500):
        picked = plan.select_keyframes(profile, budget)
        assert len(picked) <= budget, f"budget {budget}: got {len(picked)}"
        assert len({(k["clip"], k["t"]) for k in picked}) == len(picked), "duplicates"
        for k in picked:
            assert Path(k["image"]).exists(), k
        if budget >= 2 * len(profile["clips"]):
            covered = {k["clip"] for k in picked}
            assert covered == {c["path"] for c in profile["clips"]}, \
                f"budget {budget} left a clip unseen: {covered}"


@check("the prompt names the real clips, the beat grid and the missing history")
def _():
    import tempfile
    profile = _profile()
    keyframes = plan.select_keyframes(profile, 6)
    with tempfile.TemporaryDirectory() as d:
        # an empty database, so this does not depend on the real one
        prompt, history, logged = plan.build_prompt(
            profile, variants=3, notes="try a slow burn", keyframes=keyframes,
            db_path=Path(d) / "empty.db")
    assert logged == 0
    low = prompt.lower()
    assert "below the" in low and "no history" in low, "must not imply history it does not have"
    assert "strongest" not in low, "must not rank anything with nothing logged"
    for clip in profile["clips"]:
        assert clip["path"] in prompt, clip["path"]
    assert "music/goodbye-party.wav" in prompt
    assert str(profile["audio"]["bpm"]) in prompt
    assert "try a slow burn" in prompt, "trend notes must reach the prompt"
    assert "3 EDLs" in prompt
    # the schema in the prompt must be the one that is actually enforced
    assert "^[a-z0-9]+(?:-[a-z0-9]+)*$" in prompt
    assert "snap_to_beat" in prompt
    # and it must tell the model not to do the arithmetic itself
    assert "Do not\ntry to compute beat-aligned timestamps yourself" in prompt


@check("compact_numbers collapses numeric arrays without changing the data")
def _():
    import json as _json
    data = {"beats": [0.035, 0.534, 1.033], "curve": [[0.0, 0.2], [1.0, 0.3]], "bpm": 120.19}
    text = plan.compact_numbers(_json.dumps(data, indent=2))
    assert _json.loads(text) == data, "collapsing must not alter the JSON"
    assert "[0.035, 0.534, 1.033]" in text
    assert text.count("\n") < _json.dumps(data, indent=2).count("\n")


# --- log --------------------------------------------------------------------

def _tmp_db(fn):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        return fn(log.connect(Path(d) / "t.db"))


def _seed_video(conn, tmp: Path, name="v", hook="we recorded this in one take",
                avg_len=1.0, cuts=8, synced=True, audio_start=9.0, posted="2026-07-01"):
    import json as _json
    segs = [{"clip": "raw/test-a.mp4", "in": 0.0, "out": avg_len, "speed": 1.0,
             "snap_to_beat": synced, "reason": ""} for _ in range(cuts)]
    caps = ([{"text": hook, "start": 0.1, "end": 2.0, "style": "hook",
              "position": "upper-third"}] if hook else [])
    duration = avg_len * cuts
    video = tmp / f"{name}.mp4"
    video.write_bytes(b"")
    video.with_suffix(".json").write_text(_json.dumps({
        "video": video.name, "rendered_at": "2026-07-01T00:00:00+00:00",
        "edl_path": f"plans/{name}.json",
        "edl": {"variant_name": name, "strategy_notes": "s", "target_duration": duration,
                "audio": {"source": "music/goodbye-party.wav", "start": audio_start,
                          "fade_in": 0.0, "fade_out": 0.0},
                "segments": segs, "captions": caps},
        "derived_features": {
            "duration": duration, "cut_count": cuts, "avg_segment_length": avg_len,
            "shortest_segment": avg_len, "longest_segment": avg_len,
            "cuts_per_second": cuts / duration, "caption_count": len(caps),
            "caption_density": len(caps) / duration, "beat_synced": synced,
            "uses_speed_ramp": False, "hook_type": hook,
            "hook_position": "upper-third" if hook else None,
            "clips_used": ["raw/test-a.mp4"], "audio_start": audio_start},
    }))
    return log.record_video(conn, video, posted_at=posted, caption=None,
                            hashtags=None, profile=None)


@check("hook captions are classified into groupable types")
def _():
    cases = {
        "why does this sound like that?": "question",
        "How did we record this": "question",
        "pov you found the band early": "pov",
        "3 takes, one survived": "number",
        "no click track": "negation",
        "listen to the bass": "command",
        "we recorded this in one take": "statement",
        None: "none",
        "": "none",
        "   ": "none",
    }
    for text, expected in cases.items():
        assert log.classify_hook(text) == expected, f"{text!r} -> {log.classify_hook(text)}"


@check("metrics input accepts the shapes a person actually types")
def _():
    assert log.parse_count("12,400") == 12400
    assert log.parse_count("31k") == 31000
    assert log.parse_count("1.2M") == 1200000
    assert log.parse_count("  980 ") == 980
    assert log.parse_count("") is None
    assert log.parse_rate("42%") == 0.42
    assert log.parse_rate("0.42") == 0.42
    assert log.parse_rate("42") == 0.42, "a bare number above 1 is a percentage"
    assert log.parse_rate("0.9") == 0.9
    assert log.parse_rate("") is None
    for bad in ("twelve", "abc"):
        try:
            log.parse_count(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} should be rejected")
    for bad in ("150%", "-5%"):
        try:
            log.parse_rate(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} should be rejected as a rate")


@check("posting a render stores its EDL and derived features")
def _():
    import tempfile
    # The song section is looked up from a profile on disk, matched on the
    # track's filename. Point that lookup at the fixture's profiles/ so the
    # check exercises the real lookup without needing a real ingest.
    prev_dir = config.PROFILE_DIR
    config.PROFILE_DIR = _fixture()["root"] / "profiles"
    try:
        with tempfile.TemporaryDirectory() as d:
            _check_post(Path(d))
    finally:
        config.PROFILE_DIR = prev_dir


def _check_post(d: Path) -> None:
    conn = log.connect(d / "t.db")
    vid = _seed_video(conn, d, name="alpha", hook="why does this sound like that?",
                      avg_len=0.8, cuts=10)
    row = conn.execute("SELECT * FROM videos WHERE id = ?", (vid,)).fetchone()
    assert row["variant_name"] == "alpha"
    assert row["hook_type"] == "question"
    assert row["hook_text"] == "why does this sound like that?"
    assert row["cut_count"] == 10
    assert abs(row["avg_segment_length"] - 0.8) < 1e-6
    assert row["beat_synced"] == 1
    assert row["posted_at"] == "2026-07-01"
    assert json.loads(row["edl_json"])["variant_name"] == "alpha"
    # audio_start 9.0 sits in the fixture's "high" section (8.0 to 22.4)
    assert row["song_section"] == "high", row["song_section"]
    # re-posting the same file updates rather than duplicating
    again = _seed_video(conn, d, name="alpha", posted="2026-07-05")
    assert again == vid
    assert conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 1
    assert conn.execute("SELECT posted_at FROM videos").fetchone()[0] == "2026-07-05"


@check("posting a video with no render sidecar fails readably")
def _():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        conn = log.connect(d / "t.db")
        orphan = d / "orphan.mp4"
        orphan.write_bytes(b"")
        try:
            log.record_video(conn, orphan, posted_at="2026-07-01",
                             caption=None, hashtags=None)
        except common.ToolError as exc:
            assert "sidecar" in str(exc).lower(), exc
        else:
            raise AssertionError("expected a ToolError about the missing sidecar")


@check("the latest pull is used even when pulls are entered out of order")
def _():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        conn = log.connect(d / "t.db")
        vid = _seed_video(conn, d, name="beta")
        # entered newest-first, i.e. a backfill of an earlier date afterwards
        log.record_metrics(conn, vid, {"views": 900, "watch_through_rate": 0.5}, "2026-07-10")
        log.record_metrics(conn, vid, {"views": 100, "watch_through_rate": 0.2}, "2026-07-03")
        latest = log.latest_metrics(conn)
        assert vid in latest, "the video must not drop out when pulls are out of order"
        assert latest[vid]["views"] == 900, latest[vid]["views"]
        assert len(log.build_rows(conn)) == 1


@check("metrics are rejected for a video that does not exist")
def _():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        conn = log.connect(Path(d) / "t.db")
        try:
            log.record_metrics(conn, 999, {"views": 1}, "2026-07-01")
        except common.ToolError as exc:
            assert "999" in str(exc)
        else:
            raise AssertionError("expected a ToolError for an unknown video id")


@check("a video with no metrics is excluded from the report")
def _():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        conn = log.connect(d / "t.db")
        _seed_video(conn, d, name="unmeasured")
        assert log.build_rows(conn) == []


@check("grouping averages per bucket and counts the members")
def _():
    rows = [
        {"k": "a", "views": 100, "watch_through_rate": 0.4, "share_rate": 0.01, "like_rate": 0.05},
        {"k": "a", "views": 300, "watch_through_rate": 0.2, "share_rate": 0.03, "like_rate": 0.07},
        {"k": "b", "views": 200, "watch_through_rate": 0.5, "share_rate": None, "like_rate": None},
    ]
    groups = {g["group"]: g for g in log.summarise(rows, "k")}
    assert groups["a"]["n"] == 2 and groups["b"]["n"] == 1
    assert abs(groups["a"]["watch_through"] - 0.3) < 1e-9
    assert abs(groups["a"]["share_rate"] - 0.02) < 1e-9
    assert groups["a"]["median_views"] == 200
    assert groups["b"]["share_rate"] is None, "missing values must not become zero"
    # sorted best-first by watch-through
    assert log.summarise(rows, "k")[0]["group"] == "b"


@check("pace buckets split on the cut lengths the report claims")
def _():
    assert log.pace_bucket(0.5).startswith("fast")
    assert log.pace_bucket(0.99).startswith("fast")
    assert log.pace_bucket(1.0).startswith("medium")
    assert log.pace_bucket(1.99).startswith("medium")
    assert log.pace_bucket(2.0).startswith("slow")
    assert log.pace_bucket(None) == "unknown"


@check("history stays honest below the threshold and useful above it")
def _():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        db = d / "t.db"
        conn = log.connect(db)
        for i in range(config.MIN_HISTORY_FOR_RETRIEVAL - 1):
            vid = _seed_video(conn, d, name=f"v{i}")
            log.record_metrics(conn, vid, {"views": 100, "watch_through_rate": 0.3}, "2026-07-05")
        conn.close()
        text, n = log.history_for_prompt(5, db)
        assert n == config.MIN_HISTORY_FOR_RETRIEVAL - 1
        assert "below" in text.lower() and "no history" in text.lower(), text
        assert "strongest" not in text.lower(), "must not rank anything below the threshold"

        conn = log.connect(db)
        vid = _seed_video(conn, d, name="winner", hook="no click track")
        log.record_metrics(conn, vid, {"views": 5000, "watch_through_rate": 0.9,
                                       "shares": 100}, "2026-07-05")
        conn.close()
        text, n = log.history_for_prompt(5, db)
        assert n == config.MIN_HISTORY_FOR_RETRIEVAL
        assert "winner" in text, text
        low = text.lower()
        assert "hint" in low and "not a rule" in low, \
            "must still frame a small sample as a hint"


# --- retrieval ---------------------------------------------------------------

@check("similarity prefers the same track over a better-performing different one")
def _():
    here = {"audio_source": "music/a.wav", "bpm": 120.0,
            "clips": ["raw/x.mp4"], "notes": None}
    same = {"audio_source": "music/a.wav", "bpm": 120.0, "clips": ["raw/x.mp4"]}
    other = {"audio_source": "music/b.wav", "bpm": 88.0, "clips": ["raw/y.mp4"]}
    s_same, why = similarity_of(here, same)
    s_other, _ = similarity_of(here, other)
    assert s_same > s_other, (s_same, s_other)
    assert s_same > 0.9, s_same
    assert "same track" in why


def similarity_of(a, b):
    return log.similarity(a, b)


@check("similarity drops a missing component instead of scoring it zero")
def _():
    here = {"audio_source": "music/a.wav", "bpm": 120.0, "clips": ["raw/x.mp4"], "notes": None}
    full = {"audio_source": "music/a.wav", "bpm": 120.0, "clips": ["raw/x.mp4"]}
    # an older row with no BPM recorded must not be punished for it
    no_bpm = {"audio_source": "music/a.wav", "bpm": None, "clips": ["raw/x.mp4"]}
    assert abs(log.similarity(here, full)[0] - log.similarity(here, no_bpm)[0]) < 1e-9
    # nothing comparable at all scores zero and says so
    score, why = log.similarity({"notes": None}, {})
    assert score == 0.0 and "nothing comparable" in why[0]


@check("typed notes pull matching past edits up the ranking")
def _():
    here = {"audio_source": "music/a.wav", "bpm": 120.0, "clips": ["raw/x.mp4"],
            "notes": "fast cutting, no click track feel"}
    plain = {"audio_source": "music/a.wav", "bpm": 120.0, "clips": ["raw/x.mp4"],
             "strategy_notes": "long slow holds", "variant_name": "slow-one"}
    matching = {"audio_source": "music/a.wav", "bpm": 120.0, "clips": ["raw/x.mp4"],
                "strategy_notes": "fast cutting throughout", "hook_text": "no click track",
                "variant_name": "fast-one"}
    s_match, why = log.similarity(here, matching)
    s_plain, _ = log.similarity(here, plain)
    assert s_match > s_plain, (s_match, s_plain)
    assert any("notes" in r for r in why), why


@check("tempo similarity falls off with distance and ignores octave-far tracks")
def _():
    base = {"audio_source": "music/a.wav", "clips": [], "notes": None}
    def score(bpm_here, bpm_there):
        return log.similarity({**base, "bpm": bpm_here},
                              {"audio_source": "music/b.wav", "bpm": bpm_there, "clips": []})[0]
    assert score(120, 120) > score(120, 130) > score(120, 155)
    assert score(120, 200) == 0.0, "far apart tempos contribute nothing"


@check("retrieval degrades honestly when nothing is comparable")
def _():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        db = d / "t.db"
        conn = log.connect(db)
        for i in range(config.MIN_HISTORY_FOR_RETRIEVAL):
            vid = _seed_video(conn, d, name=f"v{i}")
            log.record_metrics(conn, vid, {"views": 100 * (i + 1),
                                           "watch_through_rate": 0.2 + i * 0.01}, "2026-07-05")
        conn.close()
        # a track and footage the log has never seen
        alien = {"audio": {"path": "music/never-seen.wav", "bpm": 150.0},
                 "clips": [{"path": "raw/brand-new.mp4"}]}
        text, n = log.history_for_prompt(3, db, profile=alien)
        assert n == config.MIN_HISTORY_FOR_RETRIEVAL
        assert "Nothing logged was made from material like this" in text, text
        assert "closest past edits" not in text, "must not claim similarity it does not have"
        # and with the matching material it does claim it
        matching = {"audio": {"path": "music/goodbye-party.wav", "bpm": 120.19},
                    "clips": [{"path": "raw/test-a.mp4"}]}
        text, _ = log.history_for_prompt(3, db, profile=matching)
        assert "closest past edits" in text, text
        # a missing profile must not crash the planner
        text, _ = log.history_for_prompt(3, db, profile=None)
        assert text.strip()


@check("a v1 database migrates in place and keeps its rows")
def _():
    import sqlite3, tempfile
    V1 = """
    CREATE TABLE videos (
        id INTEGER PRIMARY KEY, video_path TEXT NOT NULL UNIQUE, variant_name TEXT NOT NULL,
        edl_path TEXT, edl_json TEXT NOT NULL, features_json TEXT NOT NULL,
        rendered_at TEXT, posted_at TEXT, caption TEXT, hashtags TEXT,
        hook_type TEXT, hook_text TEXT, hook_position TEXT, cut_count INTEGER,
        avg_segment_length REAL, cuts_per_second REAL, caption_count INTEGER,
        caption_density REAL, duration REAL, audio_start REAL, song_section TEXT,
        song_section_index INTEGER, beat_synced INTEGER, uses_speed_ramp INTEGER,
        created_at TEXT NOT NULL);
    CREATE TABLE metrics (
        id INTEGER PRIMARY KEY, video_id INTEGER NOT NULL REFERENCES videos(id),
        pulled_at TEXT NOT NULL, views INTEGER, watch_through_rate REAL, likes INTEGER,
        shares INTEGER, comments INTEGER, follows INTEGER, saves INTEGER,
        created_at TEXT NOT NULL);
    """
    edl = {"variant_name": "legacy", "strategy_notes": "old",
           "audio": {"source": "music/goodbye-party.wav", "start": 9.0},
           "segments": [{"clip": "raw/test-a.mp4", "in": 0, "out": 1, "speed": 1.0}],
           "captions": []}
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "old.db"
        raw = sqlite3.connect(db)
        raw.executescript(V1)
        raw.execute("PRAGMA user_version = 1")
        raw.execute("INSERT INTO videos (video_path, variant_name, edl_json, features_json,"
                    " posted_at, created_at) VALUES (?,?,?,?,?,?)",
                    ("out/legacy.mp4", "legacy", json.dumps(edl), "{}", "2026-07-01", "t"))
        raw.execute("INSERT INTO metrics (video_id, pulled_at, views, watch_through_rate,"
                    " created_at) VALUES (1,'2026-07-03',500,0.4,'t')")
        raw.commit()
        raw.close()

        conn = log.connect(db)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == log.SCHEMA_VERSION
        row = conn.execute("SELECT * FROM videos").fetchone()
        assert row["variant_name"] == "legacy", "the existing row must survive"
        assert row["audio_source"] == "music/goodbye-party.wav", "backfilled from the stored EDL"
        assert json.loads(row["clips_json"]) == ["raw/test-a.mp4"]
        assert len(log.build_rows(conn)) == 1, "the report must still see it"
        conn.close()
        # re-opening an already-migrated database is a no-op
        for _ in range(2):
            log.connect(db).close()



# ---- Harmony is on a trial licence expiring 2026-10-03 -------------------------
# It is the only thing that can turn these rigs into images, so the pipeline has to
# be able to run without it, and the baked plates have to be right. Both are easy
# to break silently and neither shows up in a render that happens to still work.

@check("every rig is renderable with Harmony absent")
def _():
    import json as _json
    root = config.ROOT / "assets" / "rigs"
    rigs = sorted(p.name for p in root.iterdir() if (p / "rig.json").exists())
    assert rigs, "no configured rigs"
    for name in rigs:
        cfg = _json.loads((root / name / "rig.json").read_text())
        w = cfg["render"]["resolution"][0]
        for part in ("head", "body"):
            p = root / name / "plates" / f"plate-{part}-{w}.png"
            assert p.exists(), (
                f"{name} has no tracked {part} plate at {w}px. Harmony's licence "
                f"expires 2026-10-03; bake it before then or the rig is unusable.")
        for shape in cfg["mouth_ladder"]:
            m = root / name / "mouths" / f"{shape}.png"
            assert m.exists(), f"{name} is missing mouth shape {shape}"


@check("no rig lists a mouth element among its head layers")
def _():
    import json as _json
    root = config.ROOT / "assets" / "rigs"
    for p in sorted(root.glob("*/rig.json")):
        cfg = _json.loads(p.read_text())
        layers = cfg["layers"]
        head = set(layers["head"])
        mouthy = {layers["mouth"]} | set(layers.get("mouth_family", []))
        clash = head & mouthy
        assert not clash, (
            f"{p.parent.name}: head layers {sorted(clash)} are mouth elements. "
            f"The head plate would carry a mouth that the compositor then draws "
            f"another mouth over.")
        assert not (head & set(layers["body"])), (
            f"{p.parent.name}: an element is in both head and body")


@check("a plate that already has alpha is not un-premultiplied again")
def _():
    import perform as _perform
    from PIL import Image as _Image
    tmp = config.CACHE_DIR / "selfcheck-plate.png"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    # a dark shape on transparency, as TGA4 plates arrive
    a = np.zeros((8, 8, 4), np.uint8)
    a[2:6, 2:6] = (20, 20, 20, 255)
    _Image.fromarray(a).save(tmp)
    got = np.asarray(_perform._matted(tmp))
    assert got[0, 0, 3] == 0, "transparent corner became opaque"
    assert got[3, 3, 3] == 255 and got[3, 3, 0] == 20, "the art was altered"
    # and a white-background plate still gets its matte recovered
    b = np.full((8, 8, 4), 255, np.uint8)
    b[2:6, 2:6] = (0, 0, 0, 255)
    _Image.fromarray(b).save(tmp)
    got = np.asarray(_perform._matted(tmp))
    assert got[0, 0, 3] < 8, "white background stayed opaque"
    assert got[3, 3, 3] > 200, "the art was matted away"


@check("every rig has usable baked view angles")
def _():
    import json as _json
    root = config.ROOT / "assets" / "rigs"
    for p in sorted(root.glob("*/rig.json")):
        man = p.parent / "poses" / "manifest.json"
        assert man.exists(), (
            f"{p.parent.name} has no baked poses. Harmony's licence expires "
            f"2026-10-03 and it is the only thing that can bake them.")
        poses = _json.loads(man.read_text())["poses"]
        usable = [v for v in poses.values() if v["usable"]]
        assert usable, f"{p.parent.name}: no usable pose"
        for k, v in poses.items():
            if not v["usable"]:
                continue
            for part in ("head", "body"):
                assert (p.parent / "poses" / v[part]).exists(), \
                    f"{p.parent.name} pose {k}: missing {part} plate"


@check("no pose puts the mouth outside the head")
def _():
    import json as _json
    root = config.ROOT / "assets" / "rigs"
    for p in sorted(root.glob("*/poses/manifest.json")):
        for k, v in _json.loads(p.read_text())["poses"].items():
            if not v["usable"] or not v.get("mouth_bbox"):
                continue
            mb, hb = v["mouth_bbox"], v["head_bbox"]
            assert (mb[0] >= hb[0] and mb[2] <= hb[2]
                    and mb[1] >= hb[1] and mb[3] <= hb[3]), (
                f"{p.parent.parent.name} pose {k}: mouth {mb} falls outside the "
                f"head {hb} - this is how the dog's mouth ended up on its nose. "
                f"Measuring the mouth family instead of the mouth element does it: "
                f"a hanging tongue drags the box down onto the neck.")


@check("an unusable or absent pose is refused, not rendered")
def _():
    import json as _json
    import perform as _perform
    root = config.ROOT / "assets" / "rigs"
    name = sorted(d.name for d in root.iterdir() if (d / "rig.json").exists())[0]
    r = _perform.load_rig(name)
    poses = _json.loads((r["_dir"] / "poses" / "manifest.json").read_text())["poses"]
    try:
        _perform.pose_geometry(r, 10 ** 6)
        raise AssertionError("a pose that does not exist was accepted")
    except common.ToolError:
        pass
    bad = [k for k, v in poses.items() if not v["usable"]]
    if bad:
        try:
            _perform.pose_geometry(r, int(bad[0]))
            raise AssertionError(f"unusable pose {bad[0]} was accepted")
        except common.ToolError:
            pass


@check("the Blender spike measures a rest frame against perform's composite honestly")
def _():
    import importlib.util
    from PIL import Image
    spec = importlib.util.spec_from_file_location(
        "blender_spike", config.ROOT / "tools" / "blender_spike.py")
    spike = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(spike)
    assert isinstance(config.BLENDER, str) and config.BLENDER, "config.BLENDER is unset"
    head = Image.new("RGBA", (60, 60), (0, 0, 0, 0))
    body = Image.new("RGBA", (60, 60), (0, 0, 0, 0))
    head.paste((200, 40, 40, 255), (20, 10, 40, 30))
    body.paste((40, 40, 200, 255), (15, 25, 45, 55))
    ref = spike._rest_reference(head, body, (10, 5, 50, 58))
    assert ref.size == (40, 53), ref.size
    assert ref.getpixel((15, 10))[:3] == (200, 40, 40), "head is not drawn over the body"
    assert ref.getpixel((15, 40))[:3] == (40, 40, 200), "body is missing"
    same = spike._compare(ref, ref)
    assert same["coverage_mismatch_px"] == 0 and same["rgb_mean_abs_diff_solid"] == 0.0, same
    shifted = Image.new("RGBA", ref.size, (0, 0, 0, 0))
    shifted.alpha_composite(ref, (3, 0))
    moved = spike._compare(shifted, ref)
    assert moved["coverage_mismatch_px"] > 0, "a 3px shift went unnoticed"
    assert spike._alpha_bbox(head) == [20, 10, 40, 30], spike._alpha_bbox(head)


@check("the puppet's draw order is peeled from the truth, top layer first")
def _():
    import puppet
    from PIL import Image
    canvas = (60, 60)
    def plate(colour, box):
        im = Image.new("RGBA", (box[2] - box[0] + 1, box[3] - box[1] + 1), colour + (255,))
        p = puppet.Plate(Path("unused.png"), box)
        p._arr = np.asarray(im).astype(np.int16)
        return p
    # three overlapping squares; the truth was drawn 3 over 1 over 2
    plates = {"1": plate((200, 0, 0), [5, 5, 35, 35]),
              "2": plate((0, 200, 0), [20, 20, 50, 50]),
              "3": plate((0, 0, 200), [15, 15, 25, 45])}
    truth = puppet.composite([{"id": i} for i in ("2", "1", "3")], plates, canvas, mattes=False)
    order, stat = puppet.measure_order(plates, np.asarray(truth).astype(np.int16))
    assert order == ["2", "1", "3"], order
    assert not stat["hidden"], stat
    again = puppet.composite([{"id": i} for i in order], plates, canvas, mattes=False)
    assert puppet.compare(again, truth)["differing_px"] == 0
    # a plate nothing shows is reported hidden and sent to the bottom
    plates["4"] = plate((9, 9, 9), [21, 21, 24, 24])
    order, stat = puppet.measure_order(plates, np.asarray(truth).astype(np.int16))
    assert stat["hidden"] == ["4"] and order[0] == "4", (order, stat)


@check("a pivot is the joined end of a part, not its free end")
def _():
    import puppet
    child = np.zeros((80, 80), bool); child[30:70, 20:30] = True       # a shin
    thigh = np.zeros((80, 80), bool); thigh[10:34, 18:32] = True       # overlaps at the knee
    boot = np.zeros((80, 80), bool);  boot[66:78, 16:34] = True        # overlaps at the ankle
    # against the whole leg the shin's free end is the ankle, so the knee wins
    pv, how = puppet.measure_pivot(child, thigh | boot)
    assert 29 <= pv[1] <= 34, (pv, how)
    # against the boot alone the joint is the ankle
    pv, how = puppet.measure_pivot(child, boot)
    assert 65 <= pv[1] <= 70, (pv, how)
    # an arm that runs down the side of a chest still pivots at its top
    arm = np.zeros((80, 80), bool); arm[10:70, 40:50] = True
    chest = np.zeros((80, 80), bool); chest[5:60, 0:46] = True
    pv, how = puppet.measure_pivot(arm, chest)
    assert pv[1] <= 20, (pv, how)
    pv, how = puppet.measure_pivot(child, np.zeros((80, 80), bool))
    assert pv == [24, 69] and "root" in how, (pv, how)


@check("a plate drawn twice splits into one layer per blob, left to right")
def _():
    import puppet
    from PIL import Image
    im = Image.new("RGBA", (100, 40), (0, 0, 0, 0))
    im.paste((10, 10, 10, 255), (2, 5, 30, 35))
    im.paste((10, 10, 10, 255), (60, 8, 98, 38))
    p = puppet.Plate(Path("unused.png"), [500, 300, 599, 339])
    p._arr = np.asarray(im).astype(np.int16)
    parts = puppet.split_instances(p, 2)
    assert [b for b, _ in parts] == [[502, 305, 529, 334], [560, 308, 597, 337]], parts
    assert parts[0][1].arr[..., 3].max() == 255 and parts[0][1].arr.shape[1] == 28
    assert puppet.split_instances(p, 3) == [], "asked for three blobs of a two-blob plate"
    assert puppet._idkey("36#2") == (36, 2) and puppet._idkey("7") == (7, 0)


@check("every rig's puppet recomposites to within 3% of its cutter-free ground truth")
def _():
    import puppet
    from PIL import Image
    root = config.ROOT / "assets" / "rigs"
    for p in sorted(root.glob("*/rig.json")):
        r = json.loads(p.read_text())
        pp = r.get("puppet")
        assert pp, f"{p.parent.name} has no puppet block - run: uv run puppet.py build {p.parent.name}"
        m = pp["measured"]["recomposite_vs_nocut"]
        assert m["differing_pct"] < 3.0, f"{p.parent.name}: {m['differing_pct']}% recorded at build"
        assert all(l["bone"] in pp["bones"] for l in pp["layers"]), f"{p.parent.name}: a layer hangs from no bone"
        for k, b in pp["bones"].items():
            assert b["parent"] is None or b["parent"] in pp["bones"], f"{p.parent.name}: bone {k} orphaned"
        truth = p.parent / pp["ground_truth"]["nocut"]
        assert truth.exists(), f"{p.parent.name}: {truth.name} missing"


@check("a gen template lays out exactly the cells the job asks for, where face.py will find them")
def _():
    import gen, face
    r = gen.load_rig("robot")
    for job, spec in gen.JOBS.items():
        im, boxes = gen.template(r, job)
        tiles = face.sheet_tiles(np.asarray(im.convert("RGB")).astype(int))
        assert len(tiles) == len(spec["cells"]) == len(boxes), (job, len(tiles), len(spec["cells"]))
        assert [tuple(t) for t in tiles] == boxes, f"{job}: tiles found out of reading order"


@check("a gen prompt names every cell, every rule and every attachment")
def _():
    import gen
    r = gen.load_rig("robot")
    text = gen.prompt_text(r, "mouths", ["ref-head.png", "template.png"])
    for name in gen.JOBS["mouths"]["cells"]:
        assert f"**{name}**" in text, name
    for rule in gen.RULES:
        assert rule in text
    assert "ref-head.png, template.png" in text
    assert "ref-current" not in text, "claimed a current-shapes image it did not attach"


@check("gen measures line weight from the outline, not from a filled mouth cavity")
def _():
    import gen
    from PIL import Image, ImageDraw
    im = Image.new("RGBA", (300, 200), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([20, 20, 280, 180], radius=30, fill=(200, 120, 130, 255),
                        outline=(0, 0, 0, 255), width=6)
    thin = gen.stroke_width(im)
    assert 4.5 <= thin <= 7.5, thin
    d.ellipse([90, 70, 210, 130], fill=(0, 0, 0, 255))        # a black cavity
    with_cavity = gen.stroke_width(im)
    assert abs(with_cavity - thin) <= 1.5, (thin, with_cavity)
    assert gen.palette_fraction(im, [(0, 0, 0), (200, 120, 130)]) > 0.95
    assert gen.palette_fraction(im, [(0, 0, 255)]) < 0.5


@check("gen ingest refuses a sheet with the wrong number of cells")
def _():
    import gen, json, tempfile
    from PIL import Image, ImageDraw
    r = gen.load_rig("robot")
    with tempfile.TemporaryDirectory() as td:
        folder = Path(td) / "bundle"
        folder.mkdir()
        tmpl, boxes = gen.template(r, "brows")
        json.dump({"rig": "robot", "job": "brows", "cells": list(gen.JOBS["brows"]["cells"]),
                   "template_boxes": boxes, "palette": [[0, 0, 0]], "prompt_sha256": "x"},
                  open(folder / "meta.json", "w"))
        sheet = tmpl.copy()
        d = ImageDraw.Draw(sheet)
        x0, y0, x1, y1 = boxes[0]
        d.rectangle([x0 - 4, y0 - 4, x1 + 4, y1 + 4], fill=(255, 255, 255))   # kill one cell outright
        sheet.save(folder / "bad.png")
        try:
            gen.ingest(folder, folder / "bad.png")
            raise AssertionError("a three-cell sheet was accepted for a four-cell job")
        except common.ToolError as exc:
            assert "found 3 tiles" in str(exc), str(exc)


@check("a beat hit dips before it and bounces after, and overlapping hits add")
def _():
    import motion
    n = 48
    one = motion.impulse(n, [12], pre=3, pre_amp=1.6, amp=4.4, post_amp=1.4, settle=9)
    assert one[9] < 0 and one[11] < one[9], "no anticipation lift before the hit"
    assert one[12] == max(one) and one[12] > 4.0, "the hit is not the peak"
    assert min(one[13:22]) < 0, "no overshoot past rest after the hit"
    assert abs(one[30]) < 1e-9, "did not settle"
    two = motion.impulse(n, [12, 14], pre=3, pre_amp=1.6, amp=4.4, post_amp=1.4, settle=9)
    assert two[14] > one[14], "a second hit did not add"


@check("squash and stretch keep volume and stay under three percent")
def _():
    import motion
    sx, sy = motion.squash_stretch(40, [10])
    assert sy.min() > 0.97 and sy.max() < 1.03, (sy.min(), sy.max())
    assert np.allclose(sx * sx * sy, 1.0), "volume not kept"
    assert np.argmin(sy) < np.argmax(sy), "stretch came before squash"
    assert sy[0] == 1.0 and sy[-1] == 1.0


@check("a phrase end settles down, holds, and releases on the next onset")
def _():
    import motion
    st = motion.settle(60, [10], [30], amp=2.4)
    assert st[9] == 0 and abs(st[18] - 2.4) < 1e-6 and abs(st[29] - 2.4) < 1e-6, st[8:31]
    assert 0 < st[31] < 2.4 and st[40] == 0, "no release after the onset"
    # a phrase end with no later onset holds to the end
    assert abs(motion.settle(30, [10], [], amp=1.0)[-1] - 1.0) < 1e-6


@check("a lagging child starts where its parent starts and follows late and smaller")
def _():
    import motion
    parent = np.zeros(20); parent[5:] = 10.0
    child = motion.lag(parent, 2)
    assert child[0] == 0 and child[5] == 0 and child[6] == 0, "the child moved on time"
    assert abs(child[7] - 10 * motion.LAG_DAMP) < 1e-9, child[7]
    assert np.array_equal(motion.lag(parent, 0), parent)
    off = np.full(20, 3.0); off[5:] = 13.0
    assert motion.lag(off, 2)[0] == 3.0, "a child with an offset start jumped on frame 1"


@check("a head turn steps through the poses between, and the eyes dart first")
def _():
    import motion
    poses = motion.turn_track(30, [(6, 4)], [1, 2, 3, 4, 5], 1)
    assert poses[:6] == [1] * 6 and poses[-1] == 4
    steps = [p for i, p in enumerate(poses) if i == 0 or p != poses[i - 1]]
    assert steps == [1, 2, 3, 4], steps
    for k in (2, 3):
        assert poses.count(k) == motion.TURN["step_frames"], poses
    gaze = motion.gaze_track(30, poses, [1, 2, 3, 4, 5])
    first = int(np.argmax(np.abs(gaze) > 0))
    assert first == 6 - motion.TURN["eye_lead_frames"], first
    assert gaze[first] > 0, "a turn to a higher pose should dart the same way"
    assert gaze[-1] == 0, "the dart never eased back"
    assert motion.turn_track(10, [(2, 9)], [1, 2, 3], 1) == [1] * 10, "an unusable pose was turned to"


@check("the acting cast finds root, body, head, eyes and hands from the rig's own layer lists")
def _():
    import act, puppet
    for name in ("doctor", "dog", "robot"):
        r, pp, _ = puppet.load_puppet(name)
        parts = act.cast(r, pp)
        bones = pp["bones"]
        assert bones[parts["root"]]["parent"] in (None, "") or bones[parts["root"]]["parent"] not in bones
        assert parts["head"] != parts["body"], f"{name}: head and body are the same bone"
        assert parts["head"] in act._subtree(bones, parts["body"]), f"{name}: the head is not under the body"
        assert parts["eyes"], f"{name}: no eye layers"
        for lid in parts["eyes"]:
            assert any(l["id"] == lid for l in pp["layers"])


@check("an automatic turn schedule leaves at every second long breath and comes back")
def _():
    import act
    track = ["AH"] * 30 + ["CLOSED"] * 15 + ["EE"] * 60 + ["CLOSED"] * 20 + ["OH"] * 40 + ["CLOSED"] * 14 + ["AH"] * 37
    sched = act.auto_turns(track, list(range(1, 12)), 1)
    assert sched == [(30, 2), (105, 1), (165, 2)], sched
    assert act.auto_turns(track, [1], 1) == [], "turned with a single usable pose"
    assert act.auto_turns(["AH"] * 50, [1, 2, 3], 1) == [], "turned with no breath"

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
