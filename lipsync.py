#!/usr/bin/env python3
"""lipsync.py - write a Harmony mouth track from an audio file.

Harmony's own Auto Lip-Sync Detection reads whatever you hand it, so on a full mix it
hears drums and guitar as speech and leaves the mouth open almost constantly. It also
maps phonemes to whatever drawings it likes - including the rig's PROFILE mouths, which
look sideways on a front-facing head.

This does the job directly instead: takes the vocal-band envelope, quantises it into a
ladder of hand-picked FRONT-facing drawings, and rewrites the mouth element's exposure
in the .xstage. Harmony must be closed (or the scene closed without saving) when this
runs, or Harmony will write its in-memory copy back over the top.

    python3 lipsync.py --scene ~/Desktop/LuckyDog-Harmony/Luckydog_DOG/Luckydog_DOG.xstage \\
        --element 34 --audio "music/days like this 1.3.mp3" --start 63.5 --duration 9

The ladder is closed -> wide. Defaults are the Lucky Dog rig's front mouths, read off a
rendered contact sheet of all 20 drawings (see README).
"""
from __future__ import annotations
import argparse, re, shutil, subprocess, sys, wave
from pathlib import Path
import numpy as np

import common
import config
from common import ToolError

LADDER = ["16", "17", "10", "1", "11"]   # closed -> wide, front-facing only


def envelope(audio: str, start: float, dur: float, fps: float) -> np.ndarray:
    config.ensure_dirs()
    tmp = str(config.CACHE_DIR / "lipsync-band.wav")
    subprocess.run([config.FFMPEG, "-y", "-loglevel", "error", "-ss", str(start), "-t", str(dur),
                    "-i", audio, "-af", "highpass=f=250,lowpass=f=3500",
                    "-ac", "1", "-ar", "16000", tmp], check=True)
    w = wave.open(tmp)
    sr = w.getframerate()
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    n = int(len(x) / sr * fps)
    e = np.array([np.sqrt(np.mean(x[int(i * sr / fps):int((i + 1) * sr / fps)] ** 2) + 1e-9) for i in range(n)])
    return np.clip(e / (np.percentile(e, 96) + 1e-9), 0, 1)


def levels(env: np.ndarray, ladder, gate: float, hold: int) -> list[str]:
    """Envelope -> drawing per frame, with a minimum hold so it doesn't strobe."""
    k = len(ladder)
    idx = np.zeros(len(env), dtype=int)
    for i, v in enumerate(env):
        idx[i] = 0 if v < gate else int(np.clip(1 + (v - gate) / (1 - gate) * (k - 1), 1, k - 1))
    out, i = [], 0
    while i < len(idx):
        j = i
        while j < len(idx) and idx[j] == idx[i]:
            j += 1
        run = j - i
        if run < hold and out:                      # too short: absorb into the previous shape
            idx[i:j] = out[-1]
        out.extend(idx[i:j].tolist())
        i = j
    return [ladder[v] for v in idx]


def runs(seq):
    i = 0
    while i < len(seq):
        j = i
        while j < len(seq) and seq[j] == seq[i]:
            j += 1
        yield i + 1, j, seq[i]
        i = j


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scene", required=True)
    p.add_argument("--element", default="34", help="element id of the mouth layer")
    p.add_argument("--audio", required=True)
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--gate", type=float, default=0.14, help="below this the mouth is shut")
    p.add_argument("--hold", type=int, default=2, help="minimum frames per shape")
    p.add_argument("--ladder", default=",".join(LADDER))
    a = p.parse_args(argv)

    scene = Path(a.scene)
    s = scene.read_text()
    ladder = a.ladder.split(",")
    env = envelope(a.audio, a.start, a.duration, a.fps)
    seq = levels(env, ladder, a.gate, a.hold)
    n = len(seq)

    cols = re.findall(r'<column type="0"[^>]*>.*?</column>', s, re.S)
    col = next((c for c in cols if f'id="{a.element}"' in c), None)
    if col is None:
        raise ToolError(f"no type-0 column with element id {a.element}")
    shutil.copy(scene, scene.with_suffix(".xstage.bak-lipsync"))

    body = "\n".join(f'     <elementSeq exposures="{x0}-{x1}" val="{v}" id="{a.element}"/>'
                     if x1 > x0 else f'     <elementSeq exposures="{x0}" val="{v}" id="{a.element}"/>'
                     for x0, x1, v in runs(seq))
    new_col = re.sub(r'(\s*<elementSeq[^>]*/>)+', "\n" + body, col, count=1)
    s = s.replace(col, new_col, 1)
    s = re.sub(r'(<scene name="Top"[^>]*nbframes=")\d+(")', rf'\g<1>{n}\g<2>', s)
    s = re.sub(r'(<scene name="Top"[^>]*stopFrame=")\d+(")', rf'\g<1>{n}\g<2>', s)
    scene.write_text(s)

    import collections
    c = collections.Counter(seq)
    shut = sum(v for k, v in c.items() if k == ladder[0])
    print(f"{n} frames @ {a.fps}fps, {len(list(runs(seq)))} mouth changes")
    print("drawing usage:", dict(c.most_common()))
    print(f"mouth shut on {shut}/{n} frames ({100*shut/n:.0f}%), mean level when shut "
          f"{env[np.array(seq) == ladder[0]].mean():.2f}, when open {env[np.array(seq) != ladder[0]].mean():.2f}")
    print("backup:", scene.with_suffix('.xstage.bak-lipsync').name)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
