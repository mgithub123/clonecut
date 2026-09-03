# Stage 7 — Harmony character performance

Turns a Toon Boom Harmony character rig and a vocal stem into a finished vertical
music-video shot, and cuts several shots together.

```bash
# one shot
uv run perform.py --rig doctor \
    --audio "~/Desktop/LuckyDog-Harmony/goodbyparty 1.4 vox.wav" \
    --start 41.2 --duration 9 --track "goodbyparty 1.4" \
    --mix "music/goodbyparty 1.4.wav" \
    --background rainy-window --rain --tears 6 \
    --char-width 705 --char-top 430 \
    --text "this is our song. it's called goodbye party" \
    --name doc

# several shots, cutting between characters
uv run montage.py examples/shots-two-rigs.json --dry-run   # check the cut plan
uv run montage.py examples/shots-two-rigs.json
```

## The pieces

| | |
|---|---|
| `voice.py` | 7a. Syllable onsets, vowels from formants, beat grid |
| `face.py` | 7b. Split a generated mouth sheet; un-premultiply art drawn on white |
| `rig.py` | 7c. Harmony scene surgery and headless render |
| `perform.py` | 7. One shot, end to end |
| `montage.py` | 7d. Several shots, cut to the beat |
| `assets/rigs/<name>/rig.json` | Per-rig geometry and capabilities |
| `prototype/` | The original working code, kept as the reference |

## What makes it work

**Harmony renders headless.** `Harmony Premium -batch scene.xstage` — 216 frames at
4K in under twenty seconds, no GUI. This is what makes the whole thing scriptable.

**The scene file is XML.** Everything the GUI did unreliably is done by rewriting
it: freezing the turnaround pegs, filling exposure gaps, setting resolution,
rendering a chosen subset of elements as a plate.

**Vowels come from formants, not loudness.** A mouth driven by the envelope alone
gets *how far* it opens but never *what shape* it makes, so a loud "eee" becomes a
wide gape. `voice.py` estimates the first two formants with 16th-order LPC and
classifies against the clip's own medians — F1 tracks jaw openness, F2 tongue
position. Absolute formants vary hugely between singers; relative position within
one performance is stable.

**The mouth runs two frames early.** Measured lag from the RMS window is about
32 ms, and viewers perceive audio slightly late, so animators place mouths ahead.

## Adding a rig

1. `uv run rig.py info <rig-dir>` — see what it has and what is wrong with it.
2. Get a mouth library:
   - the rig has a real mouth chart (the dog):
     `uv run rig.py mouths <rig-dir> --element 34 --out assets/rigs/<name>/mouths --map CLOSED=16,SMALL=17,...`
   - it does not (the doctor, the robot): generate eight shapes and
     `uv run face.py sheet <sheet.png> --out assets/rigs/<name>/mouths`.
     The prompt that produces correctly-styled shapes, and the reference stills to
     go with it, are in `assets/rigs/doctor/reference/`.
3. Write `assets/rigs/<name>/rig.json`. Measure geometry from a rendered plate
   rather than by hand — `rig.py measure` derives some of it.
4. Fill in `capabilities` honestly. It is what stops a shot rendering wrong.

## The rigs, and what they can actually do

| | Dog | Doctor | Robot |
|---|---|---|---|
| Mouth drawings | **20** (only 5 usable) | 3 (= its view angles) | **1** |
| Mouth library | from its own art | generated | **none yet** |
| Blink | yes | yes | untested |
| Tears | not measured | yes | untested |
| Gaze | no mechanism | impossible — no pupils | untested |
| Hands | 5 poses | 5 poses | **19 poses** |
| Palettes | complete | 4 of 8 missing | **16 of 20 missing** |

**None of these rigs was built to lip-sync.** All three are turnarounds — model
sheets where the character rotates through view angles and then holds. The dog's
twenty mouths cluster at two extremes (openness 8.5k–14k and 24k–28k) with nothing
between, so it can be shut or gaping but has no mid-range. The doctor and robot
have one usable mouth drawing each.

That is why generated mouth libraries are a first-class part of this pipeline
rather than a workaround.

## Things that will bite

**Use the vocal stem, never the full mix.** On a mix the envelope reads drums and
guitar as singing: the mouth is shut 6% of the time instead of 34%.

**The robot is missing 16 of its 20 palettes** and the doctor 4 of 8 — every
`PALETTE_LIST` points at `C:/Users/sophb/…` from the machine the rigs were built
on. Render a test frame before planning shots around either.

**Art drawn on white needs un-premultiplying, not thresholding.** Composited onto
a dark scene, a threshold leaves a pale halo tracing the whole silhouette and
erosion eats the black outline. `face.unmul` inverts the blend instead.

**The caption must stay inside `caption_styles`' safe box.** Below `SAFE_Y1` is
where the platform draws its own UI. `perform.py` warns and names a `--char-width`
that fits rather than quietly overlapping.

**`LuckyDog-Practice/frames/` holds 329 `final-*.tga`, not 216** — duplicates from
earlier renders, which silently produce a 13.7s video instead of 9s.

## Rig masters

Working copies live in `~/Desktop/LuckyDog-Harmony/`. The dog's master is
`~/Desktop/LuckyDog-Harmony/_masters/LD_DOG_MA` (read-only), extracted from
`LD_DOG_MA.7z` in iCloud — there is no `7z` binary on this Mac, `bsdtar` reads it.
The doctor and robot masters are in iCloud under `Lucky Dog content /Rig/`.
Nothing in this pipeline writes to a rig: `rig.py` copies the scene first.
