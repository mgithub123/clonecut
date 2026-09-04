# Stage 8 — Puppet: make the character performances better without new art

Paste everything below this line into a Claude Code session running locally in
`~/Desktop/clonecut` on the Mac that has the baked rigs.

---

You are working in `clonecut`, a Lucky Dog TikTok tool. Read `README.md` and
`HARMONY.md` first, then `perform.py`, `montage.py`, `face.py`, `rig.py`,
`assets/rigs/*/rig.json` and `assets/rigs/*/plates/manifest.json`. Do not start
writing code until you have read those.

## Where things stand

Stage 7 renders one shot of a Toon Boom Harmony character singing a vocal stem.
The character is a two-layer cutout puppet: a head plate and a body plate, with a
mouth image swapped in from an eight-shape library every frame. `build_frames()`
in `perform.py` is the whole animation system today: mouth shape from vowels,
eye-box squash for a blink, drawn tear streaks, body bob on the beat, a breathing
sine, a slow head drift, and a camera push. Nothing else moves. The result reads
as a talking sticker.

Harmony is out of the picture. `CLONECUT_NO_HARMONY=1` must stay honoured, and
nothing you write may need the Harmony binary. All the art the rigs have is
already baked under `assets/rigs/<name>/plates/`: every element as its own
cropped plate with its offset recorded in `manifest.json`, every drawing of every
multi-drawing element under `drawings/`, and every turnaround pose under
`turnaround/`. The robot has 19 baked hand poses. Nobody is drawing anything new
by hand and no image model is being paid for; the only new art comes from the
Gemini app by copy-paste, in 8g.

**Before anything else:** confirm the plates actually exist on this machine
(`ls assets/rigs/*/plates | head`) and that they are backed up somewhere outside
this repo. GitHub has only the manifests. If they are not backed up, zip
`assets/rigs/` to iCloud or Drive now and say that you did.

## The goal

A nine-second `perform.py` shot should look like a character giving a
performance rather than a sticker bobbing to a metronome. Four sub-stages, in
this order: 8a, 8g, 8b, 8c. Commit each one separately, and stop after each to render a shot and
look at it before going on.

### 8a — A layered puppet built from the baked plates

Replace the head-plus-body model with a layer tree read from the rig config.
Each layer is one baked element plate (or a set of drawings for a multi-drawing
element), placed at its manifest offset, with a pivot and a parent. Ears, brows,
lids, arms, hands, collar, hair, tail: whatever the rig actually has. The dog and
doctor rigs are already partly described in `rig.json` (`layers.head`,
`layers.body`); extend that into a real tree rather than replacing it, so the old
two-plate path keeps working until the new one is verified.

- Treat the format as the product. After 8a a "rig" is nothing more than a
  folder of PNG plates plus `rig.json`. Harmony produced these three, but the
  format must not assume Harmony: a later stage will compile characters from
  other sources (generated images, cut-up drawings) into the same folder layout.
  Write the format down in `RIGS.md`: every key in `rig.json`, what it means,
  what unit it is in, and how it was measured. If a key only makes sense for a
  Harmony rig, say so.
- Composite the tree in draw order and confirm the result is pixel-identical (or
  near enough, and say how near) to the rig's `_ALL_NOCUT.png` ground truth at
  rest. Add a selfcheck for this. That is the proof the layer offsets are right.
- Measure pivots from the plates, not by hand, the way Stage 7 did: an arm's
  pivot is near the top of its bbox where it meets the body, an ear's where it
  meets the head. Record every pivot in `rig.json` with a note saying how it was
  measured.
- Expose the turnaround poses as a discrete head-angle set and the hand drawings
  as a pose set, both selectable per frame.
- Cutter mattes were bypassed at bake time so every element is raw art. Where a
  layer visibly needs its matte back (a lid over an eye, a sleeve over a hand),
  reconstruct it from the baked matte plate rather than hand-drawing a mask.

### 8g — `gen.py`: image generation in manual mode

The only image model available is the Gemini app (a Gemini Pro membership: no
API credits, and the free API tier has no image quota, which is what the Stage 7
detour ran into). So do exactly what `plan.py` does with the Claude app: write a
bundle, the user pastes it in, and read the result back.

- `gen.py prompt --rig robot --job mouths` writes `gen/<stamp>-<rig>-<job>/`
  with `prompt.md`, the reference images to attach (the baked head plate, the
  current mouth shapes if any, and a blank eight-cell grid template at the
  target size so the model lays the sheet out where `face.py sheet` expects it),
  and `meta.json` recording what was sent and the SHA-256 of every reference.
  Jobs to support first: `mouths` (the eight-shape ladder in `mouth_ladder`),
  `brows` (raised, neutral, furrowed, asymmetric), `eyes` (open, half, shut,
  plus pupils where the rig has none), `expressions` (four full faces).
- `gen.py ingest <folder> <downloaded.png>` splits the sheet with the existing
  `face.py sheet` code, un-premultiplies white backgrounds with `face.unmul`,
  and rejects a sheet whose cells are empty, whose line weight or palette is
  far from the source plate's (measure both, print the numbers), or whose
  cell count is wrong. Accepted shapes are written under
  `assets/rigs/<rig>/gen/<job>/` with a sidecar per shape naming the prompt
  bundle. Generated art is not regenerable: it goes in `assets/`, in git.
- The prompt text is generated from the rig config, not typed: the character's
  name, the reference plate, the requested cells, and hard rules (flat colour,
  same outline weight, front view, white background, no shading, one cell per
  shape, nothing outside the grid). Keep it in one place so it can be tuned.
- First job: a robot mouth sheet. Render a shot with it and one with the drawn
  grille, put both contact sheets side by side, and let the user pick. Do not
  decide for them which has the character's hand.

The user does the paste. When you reach the point where a sheet is needed,
write the bundle, print the folder path, and stop so they can run the round.

### 8b — Animation principles, procedurally

Replace the sines in `build_frames()` with a small motion library: easing curves,
anticipation, overshoot, follow-through and settle. Then use it:

- A beat hit is a small dip before it and a bounce after, not a step.
- Squash on the hit, stretch on the release, tiny amounts. Measure against the
  ground truth so the character never visibly changes proportion at rest.
- Children lag their parents: ears and hair trail the head by one or two frames,
  hands trail arms. Make the lag a per-layer number in `rig.json`.
- Eyes dart to the new direction a few frames before the head turns. Where a rig
  has no pupils (doctor, robot), use the `eyes` sheet from 8g if one was accepted;
  otherwise draw one in the eye's own outline weight and colour, the way
  `face.py draw` builds the robot's grille mouth. Either way record the geometry
  in `rig.json`.
- Phrase ends (the existing `phrase_ends()`) get a settle: the whole body relaxes
  down and the head tilts slightly, then holds until the next onset.
- Head turns step through the turnaround poses with an ease, and the mouth and
  eyes are re-placed per pose. If per-pose face geometry is needed, measure it
  from the pose plates and store it.

Keep every amplitude and duration as a named constant, grouped, with units.
Nothing hard-coded inside the frame loop.

### 8c — Performance capture from a phone video

Add `capture.py`: given a video of a person singing the line, extract per-frame
head yaw, pitch and roll, eyebrow raise, eye openness, mouth openness and, if the
body is in frame, shoulder and hand positions. Use MediaPipe (add it with
`uv add mediapipe`; check it installs cleanly on this Python and this Mac before
committing to it, and say what version). Write the result as a JSON track,
timestamped, in the same shape `voice.py` uses for its analysis, and add a
`--capture` option to `perform.py` that maps the track onto the puppet: yaw to
turnaround pose, pitch and roll to head rotation, brows to brow layers, eye
openness to blink, shoulders to body lean, hands to the nearest baked hand pose.

The mouth keeps coming from `voice.py`. Vowels from the vocal stem are more
reliable than a phone camera's lip reading, and the two are already aligned to
the same audio. Capture only decides open-versus-shut if the vowel track is
unsure.

Smooth the capture with a small filter and say which one. Raw landmark jitter on
a flat drawing looks like a tremor.

## Looking ahead, not in scope now

The reason 8a must produce a documented, Harmony-free rig format: the next stage
after this one is compiling a character from generated images. A prompt makes a
variant of the doctor, a segmentation model cuts it into parts, inpainting fills
what sits behind each part, pivots are measured the way 8a measures them, and
the result is a rig folder this pipeline animates like the other three. Do not
build any of that now. Do keep it possible: no step in 8a or 8b may depend on a
rig having come from Harmony.

## Conventions, which the earlier stages follow and you must too

- `config.py` for paths and binaries, `common.py` for helpers and `ToolError`,
  `common.write_json` for sidecars. Never a literal `/tmp`, `ffmpeg`, or
  `1080x1920` in a module.
- `tools/selfcheck.py` is the test suite: pure logic, no media, no network. Add
  checks for anything that can be checked without media. It currently reports
  41/48 with 7 pre-existing failures. Do not make that number worse; if you fix
  any of the 7, say which.
- Every render writes a JSON sidecar with the inputs, settings and source hashes.
- Generated media stays out of git. `assets/` is the one exception and only for
  what cannot be regenerated.
- One commit per sub-stage, titled `Stage 8a: ...`, `Stage 8g: ...`, `Stage 8b:
  ...`, `Stage 8c: ...`. The commit body says what was built, what was measured, what went wrong
  and was fixed, and what still does not work. The existing log is the model for
  tone. No model names in commits.
- Do not touch anything under `~/Desktop/LuckyDog-Harmony/` or the rig masters.
- `README.md` has uncommitted edits in progress. Do not revert them. Add a
  Stage 8 row to the build-status table and a short Stage 8 section; fold the
  `HARMONY.md` pointer in while you are there.

## How to check your work

```bash
uv run tools/selfcheck.py
CLONECUT_NO_HARMONY=1 uv run perform.py --rig doctor \
    --audio "~/Desktop/LuckyDog-Harmony/goodbyparty 1.4 vox.wav" \
    --start 41.2 --duration 9 --track "goodbyparty 1.4" \
    --mix "music/goodbyparty 1.4.wav" --background rainy-window --rain \
    --text "this is our song. it's called goodbye party" --name doc-stage8
uv run tools/contact_sheet.py out/doc-stage8-*.mp4 --at 0.5 2.0 4.5 7.0
```

Render all three rigs after each sub-stage, look at the contact sheets, and
describe what you see honestly. If something looks wrong, say so and fix it or
record it in the commit body as a known limit. Do not report a sub-stage done on
the strength of the selfcheck alone.

Start with 8a. When it is committed and a shot has been rendered and inspected,
summarise what changed and what you saw, then continue to 8g, then 8b, then 8c.
