#!/usr/bin/env python3
"""Stage 7a - Voice analysis.

Reads a vocal stem and works out three things a character rig needs in order to
perform to it: when a syllable starts, which vowel is being sung, and where the
beat is.

    uv run voice.py "music/goodbyparty 1.4 vox.wav" --start 41.2 --duration 9
    uv run voice.py vox.wav --start 41.2 --duration 9 --track "goodbyparty 1.4"

The vowel part is the reason this file exists. Driving a mouth off loudness alone
gives you *how far* it opens but never *what shape* it makes, so a loud "eee" comes
out as a wide gape and a quiet "ahh" as a small one. Vowels are distinguished by
their first two formants - F1 tracks jaw openness, F2 tracks tongue position - so
we estimate those with LPC and classify against the clip's own medians. Absolute
formant frequencies vary hugely between singers; relative position within one
performance is stable, which is why the thresholds are per-clip medians and not
constants from a textbook.

Runs on the vocal stem, never the full mix. On a mix the envelope reads drums and
guitar as singing and the mouth barely closes - see HOW-TO.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

import common
import config
from common import ToolError

SR = 16000                 # analysis rate; formant work only needs up to ~4 kHz
ONSET_HOP = 32             # 2 ms at 16 kHz
ONSET_THRESH = 0.25        # fraction of the 97th-percentile envelope
ONSET_MIN_GAP = 0.18       # seconds; below this two peaks are one syllable
LPC_ORDER = 16
FORMANT_WIN = 640          # 40 ms
FORMANT_STEP = 320         # 20 ms
VOWELS = ("EE", "AH", "OH", "OO")


def _decode(audio: str, start: float, dur: float, *, band: bool) -> np.ndarray:
    """Mono 16 kHz float samples. `band` applies the 250-3500 Hz speech band.

    Onset detection wants the band (it is looking for syllable energy); formant
    estimation must NOT have it, because the filter would reshape the spectrum we
    are about to measure.
    """
    config.ensure_dirs()
    tag = "band" if band else "flat"
    tmp = config.CACHE_DIR / f"voice-{tag}-{abs(hash((audio, start, dur))):x}.wav"
    cmd = [config.FFMPEG, "-y", "-loglevel", "error",
           "-ss", str(start), "-t", str(dur), "-i", audio]
    if band:
        cmd += ["-af", "highpass=f=250,lowpass=f=3500"]
    cmd += ["-ac", "1", "-ar", str(SR), str(tmp)]
    common.run(cmd)
    with wave.open(str(tmp)) as w:
        raw = w.readframes(w.getnframes())
    tmp.unlink(missing_ok=True)
    if not raw:
        raise ToolError(f"no audio decoded from {audio} at {start}s for {dur}s")
    return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0


def onsets(audio: str, start: float, dur: float) -> list[float]:
    """Syllable start times, in seconds from the beginning of the window.

    A 2 ms-hop RMS envelope, normalised to its own 97th percentile so it does not
    matter how hot the stem was bounced, then rising-edge crossings of a fixed
    fraction of that. Peaks closer together than ONSET_MIN_GAP are treated as one
    syllable - without it, a single sung note with vibrato fires several times.
    """
    x = _decode(audio, start, dur, band=True)
    n = len(x) // ONSET_HOP
    if n < 2:
        return []
    env = np.array([np.sqrt(np.mean(x[i * ONSET_HOP:(i + 1) * ONSET_HOP] ** 2) + 1e-12)
                    for i in range(n)])
    env = env / (np.percentile(env, 97) + 1e-12)
    out: list[float] = []
    for i in range(1, len(env)):
        if env[i] > ONSET_THRESH >= env[i - 1]:
            t = i * ONSET_HOP / SR
            if not out or t - out[-1] > ONSET_MIN_GAP:
                out.append(round(float(t), 3))
    return out


def _lpc(sig: np.ndarray, order: int) -> np.ndarray | None:
    """Levinson-Durbin. Returns prediction coefficients, or None if ill-conditioned."""
    sig = sig * np.hamming(len(sig))
    r = np.correlate(sig, sig, "full")[len(sig) - 1:len(sig) + order]
    if len(r) <= order or r[0] <= 0:
        return None
    a = np.zeros(order + 1)
    a[0] = 1.0
    err = r[0]
    for i in range(1, order + 1):
        acc = r[i] + sum(a[j] * r[i - j] for j in range(1, i))
        k = -acc / err
        nxt = a.copy()
        for j in range(1, i):
            nxt[j] = a[j] + k * a[i - j]
        nxt[i] = k
        a = nxt
        err *= (1 - k * k)
        if err <= 0:
            return None
    return a


def _formants(sig: np.ndarray) -> tuple[float, float] | None:
    """F1 and F2 from the LPC roots of one window.

    Pre-emphasis lifts the high end so the upper formants survive the fit. Roots
    become resonances; we keep the ones in vocal range with a narrow enough
    bandwidth to be a real formant rather than fitting noise.
    """
    pre = np.append(sig[0], sig[1:] - 0.97 * sig[:-1])
    a = _lpc(pre, LPC_ORDER)
    if a is None:
        return None
    found = []
    for root in np.roots(a):
        if np.imag(root) <= 0.01:
            continue
        f = float(np.arctan2(np.imag(root), np.real(root)) * SR / (2 * np.pi))
        bw = float(-0.5 * (SR / (2 * np.pi)) * np.log(abs(root)))
        if 200 < f < 4000 and bw < 500:
            found.append(f)
    found.sort()
    return (found[0], found[1]) if len(found) >= 2 else None


def vowels(audio: str, start: float, dur: float,
           at: list[float]) -> tuple[list[str], list[tuple[float, float]]]:
    """One vowel label per onset, plus the (F1, F2) each was decided from.

    Measured over the first 300 ms of each syllable at most - past that a sung
    note has usually moved on. Classification is a quadrant split about the
    clip's own median F1/F2, so it adapts to the singer:

        high F2, low  F1 -> EE      high F2, high F1 -> AH
        low  F2, high F1 -> OH      low  F2, low  F1 -> OO
    """
    x = _decode(audio, start, dur, band=False)
    pairs: list[tuple[float, float] | None] = []
    for i, t in enumerate(at):
        end = at[i + 1] if i + 1 < len(at) else t + 0.25
        s0 = int(t * SR)
        s1 = min(int(min(end, t + 0.30) * SR), len(x))
        f1s, f2s = [], []
        for p in range(s0, max(s0 + 1, s1 - FORMANT_WIN), FORMANT_STEP):
            got = _formants(x[p:p + FORMANT_WIN])
            if got:
                f1s.append(got[0])
                f2s.append(got[1])
        pairs.append((float(np.median(f1s)), float(np.median(f2s))) if f1s else None)

    ok = [p for p in pairs if p]
    if not ok:
        raise ToolError("no formants could be measured - is this really a vocal stem?")
    f1_mid = float(np.median([p[0] for p in ok]))
    f2_mid = float(np.median([p[1] for p in ok]))

    labels = []
    for p in pairs:
        if p is None:
            labels.append("OH")            # neutral fallback for an unmeasurable syllable
            continue
        hi1, hi2 = p[0] > f1_mid, p[1] > f2_mid
        labels.append("AH" if (hi1 and hi2) else
                      "EE" if hi2 else
                      "OH" if hi1 else "OO")
    return labels, [p or (0.0, 0.0) for p in pairs]


def beats(track: str, start: float, dur: float) -> dict:
    """Beat and downbeat times (relative to the window) from an ingest profile."""
    prof = config.PROFILE_DIR / f"{track}.json"
    if not prof.exists():
        raise ToolError(f"no profile for {track!r} - run ingest.py first ({common.rel(prof)})")
    audio = json.loads(prof.read_text()).get("audio", {})
    win = lambda ts: [round(t - start, 3) for t in (ts or []) if start <= t <= start + dur]
    return {"bpm": audio.get("bpm"),
            "beats": win(audio.get("beats")),
            "downbeats": win(audio.get("downbeats"))}


def analyse(audio: str, start: float, dur: float, track: str | None = None) -> dict:
    """Everything a performance needs from the audio, in one dict."""
    at = onsets(audio, start, dur)
    labels, formants = vowels(audio, start, dur, at)
    out = {"audio": common.rel(Path(audio)), "start": common.r3(start),
           "duration": common.r3(dur), "onsets": at, "vowels": labels,
           "formants": [[common.r3(a), common.r3(b)] for a, b in formants]}
    if track:
        out["beat"] = beats(track, start, dur)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("audio")
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--track", help="profile name for the beat grid, e.g. 'goodbyparty 1.4'")
    p.add_argument("--json", action="store_true", help="emit the analysis as JSON")
    a = p.parse_args(argv)

    if not Path(a.audio).exists():
        raise ToolError(f"no such audio: {a.audio}")
    data = analyse(a.audio, a.start, a.duration, a.track)

    if a.json:
        print(json.dumps(data, indent=2))
        return 0
    print(f"{len(data['onsets'])} syllable onsets over {a.duration}s")
    from collections import Counter
    print("vowel usage:", dict(Counter(data["vowels"]).most_common()))
    for t, v, (f1, f2) in zip(data["onsets"], data["vowels"], data["formants"]):
        print(f"  {t:6.2f}s  F1 {f1:6.0f}  F2 {f2:6.0f}   {v}")
    if "beat" in data:
        b = data["beat"]
        print(f"\nbpm {b['bpm']}  beats in window {len(b['beats'])}  downbeats {len(b['downbeats'])}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
