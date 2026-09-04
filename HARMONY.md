# Stage 7 — Harmony character performance

Turns a Toon Boom Harmony character rig and a vocal stem into a finished vertical
music-video shot, and cuts several shots together.

```bash
# one shot, front-on
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

# a three-quarter view, singing, with the hands changing between lines
uv run perform.py --rig dog --pose 13 --gesture \
    --audio "…vox.wav" --start 41.2 --duration 4 --track "goodbyparty 1.4"
```

## View angles

A turnaround is a rotation, so every rig already contains the angles a music video
wants. `rig.py poses` bakes each one as **the head/body plate pair the front-on
path already uses**, which is the whole trick: a turned character then goes through
the same compositing, mouth anchoring, blink and camera code. Treating a pose as a
flat full-figure image would have meant a second path and hand-measured geometry
per angle.

Pegs animate over time, so pose *p* exists only on frame *p* and a subset cannot be
moved onto frame 1 — but rendering the turnaround **unfrozen** with a chosen
element subset yields that subset at every pose in one pass. Five passes cover a
rig.

Geometry is measured, never typed in. `centre_x`, `lip_line`, mouth width and the
eye boxes come from differencing head-plus-mouth against head, at every angle.
Measured against the hand-tuned front values, the dog agrees to 4px — which is the
check that the whole chain is right, and it is what stops a mouth landing on a nose.

| | Dog | Doctor | Robot |
|---|---|---|---|
| Usable poses | **22** | 11 | 8 |
| Of those, can lip-sync | **16** | 9 | 6 |
| Hand poses | 5 | 5 | **10 + 9** |
| Independent hands | no | no | **yes** |

Two limits are recorded per pose rather than hidden:

**A rig has no profile mouth.** As the head turns, the visible mouth narrows — the
robot's from 112px head-on to 10px — and scaling a front-drawn shape into that
reads as a smudge. Each pose carries its foreshortening and a `lipsync` flag at a
35% threshold; below it the mouth is held shut and `perform.py` says so.

**The tail of a turnaround is not a pose.** The doctor's last frames render a
collar, a skirt and boots with no head — the rig part-disassembled after the
rotation ends. Those are marked unusable; three of its fourteen.

## The pieces

| | |
|---|---|
| `voice.py` | 7a. Syllable onsets, vowels from formants, beat grid |
| `face.py` | 7b. Split a generated mouth sheet; un-premultiply art drawn on white |
| `rig.py` | 7c. Harmony scene surgery and headless render |
| `perform.py` | 7. One shot, end to end |
| `montage.py` | 7d. Several shots, cut to the beat |
| `assets/rigs/<name>/rig.json` | Per-rig geometry and capabilities |
| `assets/rigs/<name>/plates/` | **The baked rigs.** Element plates, drawing libraries, turnaround poses |
| `assets/rigs/<name>/poses/` | **The view angles.** Head/body plates and measured geometry per pose |
| `prototype/` | The original working code, kept as the reference |

## What makes it work

**Harmony renders headless.** `Harmony Premium -batch scene.xstage` — 216 frames at
4K in under twenty seconds, no GUI. This is what makes the whole thing scriptable.

**And nothing downstream needs it.** Harmony's only job is turning rigs into PNGs.
Voice analysis, mouth tracks, compositing, motion, rain, montage and the mux are all
our own Python. Everything has been baked, so a shot renders with the binary gone:

```bash
CLONECUT_NO_HARMONY=1 uv run perform.py --rig robot ...
```

That variable makes `rig.render()` raise instead of rendering, so anything that still
reaches for Harmony fails loudly rather than silently working on this machine only.

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
   - it does not (the doctor, the robot): either generate eight shapes and
     `uv run face.py sheet <sheet.png> --out assets/rigs/<name>/mouths` — the prompt
     and reference stills are in `assets/rigs/doctor/reference/` — or draw them:
     `uv run face.py draw --style grille --out assets/rigs/robot/mouths`.
     Match the style to the face. The robot's mouth is not lips but an outlined slot
     holding pale-green bars; `grille()` opens the aperture and shrinks the teeth
     band, the way a rigid jaw would, and the organic `DRAWN_SPEC` lips would read as
     a different character.
3. Write `assets/rigs/<name>/rig.json`. Measure geometry from a rendered plate
   rather than by hand — `rig.py measure` derives some of it.
4. Fill in `capabilities` honestly. It is what stops a shot rendering wrong.

## The trial, and why everything is baked

The licence is a **trial expiring 2026-10-03**. Harmony is the only thing that can
produce these images, so all three rigs are baked to `assets/rigs/<name>/plates/`
and committed — 364 images at 7680×4320:

| | per rig |
|---|---|
| Element plates | every element, alone, cropped to its own art with the offset recorded |
| `drawings/` | every drawing of every multi-drawing element — the pose libraries |
| `turnaround/` | the view angles, which `freeze_pegs()` otherwise throws away |
| `_ALL.png` | the rig as authored: what a recomposite should look like |
| `_ALL_NOCUT.png` | the same with cutters bypassed: what it can be checked against |
| `plate-head-<w>.png`, `plate-body-<w>.png` | what `perform.py` actually composites |

```bash
uv run rig.py bake ~/Desktop/LuckyDog-Harmony/LD_ROBOT_RIG_MA \
    --out assets/rigs/robot/plates --resolution 7680x4320
```

`.gitignore` calls generated media "regenerable, never committed". Plates are the
exception and are called out there by name — after expiry they *are* the rigs.

## The rigs, and what they can actually do

| | Dog | Doctor | Robot |
|---|---|---|---|
| Mouth drawings | **20** (only 3 head-on) | 3 (= its view angles) | **1** |
| Mouth library | derived from its own art | generated | drawn (`--style grille`) |
| Blink | yes | yes | yes |
| Tears | not measured | yes | not measured |
| Gaze | no mechanism | impossible — no pupils | impossible — no pupils |
| Hands | 5 poses | 5 poses | **19 poses** |
| Palettes | complete | 4 of 8 missing | 16 of 20 missing — **but it renders in full colour** |

**None of these rigs was built to lip-sync.** All three are turnarounds — model
sheets where the character rotates through view angles and then holds. The dog's
twenty mouths cluster at two extremes (openness 8.5k–14k and 24k–28k) with nothing
between, so it can be shut or gaping but has no mid-range. The doctor and robot
have one usable mouth drawing each.

That is why generated mouth libraries are a first-class part of this pipeline
rather than a workaround.

## Things that will bite

**Measure the mouth from the mouth element, not the mouth family.** The dog's
family includes `Tongue_OUT`, a tongue that hangs to its chest, so measuring the
family put the box 137px too low and pasted the mouth on the dog's neck.

**A colour card fills the alpha channel.** The dog's and doctor's scenes each hold
a white `COLOR_CARD`; the robot's does not, which is why only the robot first baked
with a usable matte. `Scene.drop_colour_cards()` unwires them. Checking that no
element baked *empty* does not catch this — on a white canvas nothing ever is.

**Use the vocal stem, never the full mix.** On a mix the envelope reads drums and
guitar as singing: the mouth is shut 6% of the time instead of 34%.

**Cutters make element-wise baking lie.** A Cutter takes an image on port 0 and a
matte on port 1. Baking one element at a time blanks every other element, which
empties those mattes, and a Cutter with no matte outputs *nothing* — ten of the
robot's elements baked blank before `Scene.bypass_cutters()` rewired all 31 out of
the graph. Bypassing is right rather than a workaround: the mattes are baked too,
so the masking is reconstructible.

**The Write node defaults to 24-bit TGA.** Transparent renders as the scene's
background colour. The doctor's scene composites on white, which `face.unmul()` can
invert; the robot's composites on **black**, where the same inversion makes every
pixel opaque and the character vanishes. Plates are written `TGA4` now, and
`perform._matted()` decides by looking at the alpha rather than by filename.

**Element drawings live in `elementFolder`, not `elementName`.** The robot has two
elements both called `Hand` — ids 35 and 44, folders `Hand` and `Hand.44`, with 10
and 9 poses.

**The robot is missing 16 of its 20 palettes** and the doctor 4 of 8 — every
`PALETTE_LIST` points at `C:/Users/sophb/…` from the machine the rigs were built
on. In practice all three render in full colour; the missing palettes are unused.

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
