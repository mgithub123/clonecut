# clonecut — Lucky Dog TikTok tool

Turns raw footage plus a music track into short-form vertical videos, in two
stages that stay strictly separated:

1. **Planning** (AI) produces a JSON Edit Decision List. It never touches ffmpeg.
2. **Rendering** (deterministic) consumes that EDL and produces an mp4.

So any plan can be inspected and hand-edited before rendering, and a render can
be repeated exactly without going back to the model.

Because the AI's only output is JSON, the handoff is plain text — which means it
can cross that gap by copy-paste just as well as by HTTP. The planner runs in
**manual mode**: it writes a prompt you paste into the Claude app you already
have, and reads the reply back. Nothing in this project calls an API, so
**the whole pipeline runs offline** and there is no per-run cost.

## Setup

```bash
uv sync
```

Requires `ffmpeg` and `ffprobe` on PATH (`brew install ffmpeg` /
`apt install ffmpeg`), and Python 3.11+.

If you would rather not use a terminal at all, skip to
[Stage 6](#stage-6--the-app): `uv run app.py` opens the whole pipeline in your
browser, and there is a double-clickable launcher for Mac and Windows.

## Build status

| Stage | File | Status |
|---|---|---|
| 1. Ingest | `ingest.py` | done |
| 2. EDL schema | `schema.py` | done |
| 2b. Plan (manual) | `plan.py` | done |
| 3. Render | `render.py` | done |
| 4. Learn | `log.py` + SQLite | done |
| 5. Retrieval | `log.py` -> `plan.py` | done |
| 6. App | `app.py` + `webui/` | done |
| 7. Perform | `perform.py` + `montage.py` + `rig.py` — see [HARMONY.md](HARMONY.md) | done |
| 8. Puppet | `puppet.py` + `act.py` + `motion.py` + `gen.py` — see [RIGS.md](RIGS.md) | 8z, 8a, 8g, 8b done |

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


---

## The EDL

`schema.py` defines the Edit Decision List — the only thing the AI produces and
the only thing the renderer consumes. Validation is in two deliberately
separate layers:

* The **Pydantic models** check structure: field names, types, ranges,
  ordering. They touch no media, so a model response can be validated before
  any file is opened. Unknown fields are rejected — a model writing `duration`
  where the schema says `out` should fail the retry, not silently render a
  different edit. `variant_name` is constrained to kebab-case because it
  becomes part of an output filename.
* **`validate_media()`** checks the EDL against the actual files: that clips
  exist, that no segment reaches past the end of one, and that the music is
  long enough for the edit. It returns errors (the render would be wrong)
  separately from warnings (it will render, but probably not as intended —
  a caption starting after the video ends, say).

`snap_to_beat` is a record, not an instruction. Snapping happens in code inside
`plan.py` after the model responds; by the time an EDL is on disk its
timestamps are already snapped.

Validate any EDL from the command line:

```bash
uv run schema.py examples/*.json      # exit 1 if any file is invalid
uv run schema.py --print-schema       # the JSON schema handed to the model
```

### Example EDLs

`examples/hand-written-reference.json` was written by hand against the test
fixtures, to drive `render.py` before any AI is involved. Every cut length is a
whole number of beats at 120 BPM and the music starts on a beat inside the
track's loudest section, so a mistimed render shows up as an obvious drift
rather than something subtle. One segment runs at 2x to exercise the speed
path.

`examples/broken-out-of-range.json` is deliberately invalid — a segment reaching
past the end of a clip, and a music start that leaves the track too short. It
exists so the loud-failure path has something to fail on.


---

## Stage 3 — Render

```bash
uv run render.py examples/hand-written-reference.json
uv run render.py plans/*.json --verbose
uv run render.py some.json --dry-run     # print the ffmpeg command and stop
```

Writes `out/<variant_name>-<timestamp>.mp4` plus a sidecar `.json` recording the
EDL that produced it, the derived features, the render settings, the SHA-256 of
every source file, and the exact ffmpeg command.

One ffmpeg invocation does the whole job: one input per segment (seeking with
`-ss` *before* `-i`, so a two-second cut out of a ten-minute clip does not decode
ten minutes), each scaled with `force_original_aspect_ratio=increase` and
cropped back to 1080x1920 — crop-to-fill, no bars — then concatenated. The
source audio is never mapped; only the music input reaches the output.

Before encoding, `schema.validate_media()` runs and any error aborts the render
with every problem listed at once. Afterwards the output is probed to confirm it
is really 1080x1920 with both streams and the expected duration.

### Restyling captions

Everything about how captions look and where they sit is in `caption_styles.py`:
the three styles (`hook`, `body`, `emphasis`), the three positions, and the safe
area. Edit that file and re-render; `render.py` does not need touching.

The safe area defaults to 220px off the top and 480px off the bottom, which is
roughly what TikTok's own chrome covers. `lower-third` also insets 200px each
side to clear the like/comment/share rail.

### Three things that had to be handled

**Captions are drawn one line at a time.** Handing `drawtext` a multi-line
string centres the block but left-aligns the lines inside it, which reads as a
mistake. ffmpeg only grew a `text_align` option in 7.0, so each line gets its own
centred `drawtext` — that works on whatever ffmpeg is already installed.

**`expansion=none` is not optional.** By default `drawtext` treats `%` as
special: a caption reading "100% live" logs `Stray % near ''` and renders as
*nothing at all*, silently. `%{n}` is worse — it expands to the frame number.
Both are covered by a self-check that renders the text and asserts pixels
changed.

**Caption blocks are clamped to the safe area.** The anchor positions the middle
of the block, so a hook that wraps onto three lines reaches 120px further up than
a one-liner and would ride under the UI. The block is pulled back inside instead.

## Checking a render

```bash
uv run tools/contact_sheet.py out/some-video.mp4                    # evenly spaced frames
uv run tools/contact_sheet.py out/v.mp4 --from-edl plans/v.json     # one frame per caption
uv run tools/contact_sheet.py out/v.mp4 --at 0.5 2.0 7.0
```

Red bands mark where TikTok's UI sits, so a caption that would be covered is
obvious at a glance.


---

## Stage 2 — Plan (manual mode)

```bash
uv run plan.py prompt --profile profiles/goodbye-party.json \
    --notes "trying a fast-cut hook where the first shot is under half a second"
```

Writes three things to `plans/`:

- `<stamp>-prompt.md` — paste this whole file into Claude
- `<stamp>-keyframes/` — attach every image in here to the same message
- `<stamp>-meta.json` — what was sent, for traceability

Then save Claude's reply to a file and read it back:

```bash
uv run plan.py ingest reply.txt --profile profiles/goodbye-party.json
uv run plan.py ingest - --profile profiles/...   # or pipe it in
```

Each variant is validated against the schema, snapped to the beat grid, and
written to `plans/<stamp>-<variant>.json`. Invalid variants are rejected
individually with the exact field errors — one bad variant does not lose the
others. Then render whichever you like:

```bash
uv run render.py plans/20260830-135954-fast-cut-hook.json
```

### What goes into the prompt

The footage profile with the dense curves downsampled (raw curves run to
thousands of points and crowd out everything useful), a selected spread of
keyframes, the transcript, the beat grid and section map, your free-text trend
notes, and the retrieval history. The schema section is **generated from the
Pydantic models**, so the prompt can never drift from what validation actually
enforces.

Until at least 8 videos are logged, the prompt says so plainly rather than
implying a history it does not have.

### Snapping happens in code, not in the prompt

The model is told to place cuts musically and *approximately*, and explicitly
told not to compute beat-aligned timestamps itself — arithmetic across a beat
grid is where models drift, and a cut 40ms off the beat is visible.

`snap_edl_to_beats()` then makes them exact. A segment's position in the
finished video is the running sum of the segments before it, so the moment that
matters musically is at `audio.start + timeline_position`. The whole grid is
aligned first by snapping `audio.start` itself — otherwise every downstream cut
inherits that offset. Two guards: a snap that would push a segment past the end
of its clip falls back to the previous beat, and one that would collapse a
segment below 0.15s is skipped and reported. Segments with
`snap_to_beat: false` are left exactly as written.

### Reading the reply back

`extract_json_objects()` handles what a chat reply actually looks like: a fenced
block, several fenced blocks, a bare array, or JSON with prose around it. Brace
scanning ignores brackets inside strings, so a caption reading `a } b { c` does
not break parsing.


---

## Stage 4 — Learn

A local SQLite database (`luckydog.db`) with two tables. `videos` holds the
rendered file, its EDL, the derived features and what you posted with it;
`metrics` holds one row per pull, so a video's numbers can be recorded again
after a few days and you can see how it matured.

```bash
uv run log.py post out/fast-cut-hook-20260830-140027.mp4 \
    --posted-at 2026-08-30 --caption "we recorded this in one take" --hashtags "#luckydog"
uv run log.py list                  # ids for everything logged
uv run log.py metrics 1             # type in the TikTok numbers
uv run log.py report
```

`post` reads the sidecar `render.py` wrote next to the mp4, so the EDL and the
derived features come across without you retyping anything. The song section a
variant used is resolved automatically by matching the EDL's audio path against
the profiles in `profiles/`.

`metrics` prompts field by field. It accepts the shapes you would actually type
— `12,400`, `31k`, `1.2M`, `42%`, `0.42`, and a bare `42` (read as a percentage,
since TikTok shows it that way). Blank skips a field, and a value it cannot
parse re-asks instead of storing a wrong number.

### The report

Groups by hook type, cut pace, song section and beat sync, printing median
views, mean watch-through, share rate and like rate per group, **with counts**.
Groups below 3 are marked *(too few to trust)*, and below 8 logged videos the
report says outright that none of it is worth acting on.

Hooks are bucketed into types — `question`, `pov`, `number`, `negation`,
`command`, `statement`, `none` — because grouping by the literal caption text
would put one video in each group and tell you nothing. What you want to know is
whether *questions* beat *claims*.

### One bug worth recording

`latest_metrics` originally picked each video's most recent pull with
`MAX(pulled_at)` and `MAX(id)` in the same subquery. Those can come from
different rows: enter a pull, then backfill one with an *earlier* date, and no
row satisfies both conditions — so the video vanishes from the report entirely,
with no error. Backfilling an earlier date is a completely normal thing to do.
It now picks the row by `ORDER BY pulled_at DESC, id DESC LIMIT 1`, and there is
a regression test that enters pulls out of order.

### Seeing the report before you have data

```bash
uv run tools/seed_demo_db.py --db demo.db     # 12 invented videos, 2 pulls each
uv run log.py --db demo.db report
uv run plan.py prompt --profile profiles/x.json --db demo.db   # retrieval in the prompt
```

The numbers are made up. The tool refuses to seed `luckydog.db`.


---

## Stage 5 — Retrieval

The planner asks *"what did I do last time the material looked like this"*,
which is a different question from *"what did best"*. A video shot to another
song at another tempo is weak evidence about this one however well it did.

So `plan.py prompt` passes the profile and your trend notes into
`log.history_for_prompt`, which ranks past videos by similarity to what you are
planning now:

| Component | Weight | |
|---|---|---|
| Same track | 0.40 | the strongest signal — same song, same structure |
| Tempo | 0.25 | linear falloff, zero at 40 BPM apart |
| Clip overlap | 0.20 | Jaccard over the source clips |
| Your notes | 0.15 | word overlap with past strategy notes and hooks |

Every component is a named number in 0–1 with a fixed weight, and the prompt
prints **why** each video was retrieved (`same track, similar tempo (120 vs 120
BPM), shares 2 clip(s), matches your notes (click, fast, track)`). Nothing about
the ranking is hidden.

A component that cannot be computed — no BPM recorded, no notes typed — is
dropped and the remaining weights renormalise, rather than scoring zero. Scoring
zero would silently punish older rows for missing data they never had a chance
to record.

### It refuses to overclaim

If the best match scores below 0.25 the header changes from *"the closest past
edits"* to *"nothing logged was made from material like this, so these are simply
the strongest so far — weak evidence for this edit"*. A new song and new footage
should not produce a confident-sounding list.

Verified against a seeded database where the **other** track's videos have
higher watch-through: planning against `goodbye-party` retrieves the
goodbye-party edits, not the better-performing back-room ones.

### Schema migration

Retrieval needs context the v1 schema did not store, so `videos` gained
`audio_source`, `bpm` and `clips_json`. `connect()` migrates in place, guarded by
`PRAGMA user_version`, and backfills the new columns from each row's stored EDL —
BPM comes from the ingest profile if it is still on disk, and stays NULL if not.
An existing database upgrades on next open with no action from you, and there is
a test that builds a real v1 database and opens it.

---

## Stage 6 — The app

```bash
uv run app.py            # opens your browser at 127.0.0.1:8765
uv run app.py --port 9000 --no-browser
```

Or double-click **Open Clonecut.command** (Mac) / **open-clonecut.bat**
(Windows), which check for `ffmpeg` and `uv` and say what is missing before
starting anything.

Four tabs, matching the stages: drop in footage and a track and analyse it;
write the prompt and paste Claude's reply back; tick the plans you want and
render them; log what you posted and type in the numbers. Rendered videos play
in the page, and the console at the bottom streams whatever the stage is
printing, so a long ingest is visibly working rather than apparently hung.

### It drives the CLI rather than reimplementing it

Every button shells out to `ingest.py`, `plan.py`, `render.py` or `log.py` as a
subprocess and streams stdout back. Nothing about the pipeline is reimplemented
in the app, so the app cannot drift from what the commands actually do, and
anything you can do in the UI you can still do in a terminal against the same
files. `log.py metrics` prompts field by field, so the app feeds its seven
answers in on stdin — the same parser, including the re-ask on a value it
cannot read.

**Stdlib only.** `http.server` and `sqlite3`, no framework and no build step:
adding an app must not add an install step, and `uv sync` is unchanged.

### Three things that had to be handled

**A browser hanging up mid-video is not an error.** `<video>` fetches a byte
range, gets what it needs and closes the socket, which surfaces as
`ConnectionResetError` inside the write. The generic handler caught it and
printed a traceback into the window the user was told to leave open, which
reads as a crash. Client disconnects are now caught before the generic handler
and dropped silently.

**Media is streamed, never buffered.** Both directions originally read whole
files into memory. Phone footage runs to hundreds of megabytes and every scrub
of a preview re-read the file, so uploads now stream to a `.part` file that is
renamed once the expected byte count arrives, and serving seeks to the
requested range and writes it in chunks.

**Paths from the browser are not trusted.** Every client-supplied path resolves
under `raw/`, `music/`, `profiles/`, `plans/` or `out/` and is rejected if it
escapes; uploads are re-based to a bare filename and checked against a suffix
whitelist. The server binds `127.0.0.1` only.


---

## The whole loop

```bash
uv run ingest.py --video raw/*.mp4 --audio music/track.wav
uv run plan.py prompt --profile profiles/track.json --notes "..."
#   paste into Claude, attach the keyframes, save the reply
uv run plan.py ingest reply.txt --profile profiles/track.json
uv run render.py plans/<stamp>-<variant>.json
uv run log.py post out/<variant>-<stamp>.mp4 --posted-at YYYY-MM-DD --caption "..."
uv run log.py metrics <id>          # again after a few days
uv run log.py report
```

Then the next `plan.py prompt` draws on everything logged so far. Or run
`uv run app.py` and do all of it in a browser.

```bash
uv run tools/selfcheck.py           # 45 checks, no media or network needed
```


---

## Stage 7 — Character performance

A Toon Boom Harmony rig and a vocal stem become a vertical music-video shot, and
several shots cut together on the beat. Everything about it, including why the
three rigs are baked to PNG plates and committed, is in [HARMONY.md](HARMONY.md).

---

## Stage 8 — Puppet

Stage 7's character is a head plate over a body plate with a mouth swapped in each
frame: a talking sticker. Stage 8 makes it a performance without any new art.

The first question was what should move the pieces: code in Pillow, which is what
`perform.build_frames()` is today, or Blender, which is free with no licence to
expire, has a real bone hierarchy and easing curves, and produces a file that can
be opened and nudged by hand. It was settled with a one-hour test rather than an
argument:

```bash
uv run tools/blender_spike.py          # doctor, 24 frames, Eevee, headless
```

The spike loads the doctor's baked head and body plates as image planes at their
recorded offsets, builds a two-bone armature (head parented to body, pivots at the
rig's measured head pivot and the bottom-centre of the body), keyframes a
one-second beat bob and counter-tilt with Blender's own easing, renders headless
with alpha through an orthographic camera framed on the rig's crop box, and pushes
the frames through the real finishing path (`perform.composite()`, `perform.mux()`).
It also measures its own rest frame against the two-plate composite `perform.py`
draws from the same plates, and reports the number.

**It passed.** Blender 5.2.1 rendered 24 frames in about 8 seconds with no window,
and the rest frame differs from the Pillow composite by about one 8-bit level on
average, with the only visible difference a one-pixel softening along outlines
from the render's pixel filter. So the puppet engine will be built in Blender:
8a builds and documents the rig format with a Blender scene builder as its
compositor, 8b builds the armature from `rig.json` and animates it with Blender's
curves, 8c writes performance capture as bone keyframes. `perform.py` keeps the
same command line and sidecar either way.

Blender is the one binary this project needs beyond ffmpeg (`brew install --cask
blender`). `config.BLENDER` finds it on PATH or under `/Applications`, and
`BLENDER_BIN` overrides that. The `bpy` module from PyPI was tried so that
`uv sync` could stay the only setup step: it installs and imports on this
Python, but its Eevee renders on this Mac come out with magenta textures on an
opaque white ground, and it pins numpy back to 1.26 for the whole project, so
the spike uses the binary and the dependency was not kept.

### 8a — A layered puppet built from the baked plates

```bash
uv run puppet.py build doctor      # measure the puppet, write rig.json["puppet"]
uv run puppet.py tree doctor       # bones, pivots, layers
uv run puppet.py verify doctor     # recomposite against both ground truths
uv run puppet.py render doctor     # the puppet in Blender, at rest, checked
```

A rig is now a folder of PNG plates plus `rig.json`, and the format is written
down in [RIGS.md](RIGS.md). `puppet.py build` turns the baked plates into a
puppet: a tree of bones, each with a parent and a pivot, and a stack of layers
in draw order, each hanging from a bone. Three things are measured rather than
typed in, because measuring is what makes the same code work for a rig that did
not come from Harmony:

- **Draw order** is peeled from the rig's own cutter-free render, top layer
  first. Element-id order is 12–28% wrong against that truth across the three
  rigs; the peeled order is 0.2–1.9%.
- **Pivots** are the joined end of each part's overlap with its parent's art:
  hips, knees, ankles, shoulders, ear roots. Each one records in words which rule
  found it, so a wrong one can be found.
- **Mattes** are read back from the difference between the two ground truths -
  the plate whose shape best explains what a layer loses when the cutters are on
  is its matte - and a layer the rig draws see-through gets its opacity fitted
  the same way, and a drawing the rig hides at rest (a closed lid) is marked
  hidden rather than matted. With those applied the recomposite is within 2.3%
  (doctor) and 6.3% (dog) of the authored render; the rest is Harmony's
  deformers, which a plate cannot carry. The robot stays at 16% because its glass
  dome's shine is a node the bake did not keep.

Parenting comes from the peg hierarchy in the tracked scene file, which is the
one Harmony-only input and is labelled as such in the output. An element the
scene draws twice - both arms, both legs, both ears - splits into one layer per
blob on its own peg, so limbs can move apart. Every turnaround angle and every
hand drawing is exposed as a set a frame can pick from.

`blenderpuppet.py` builds the same puppet as a Blender scene: one plane per
layer, one bone per puppet bone with its pivot and parent, planes bound to bones
by vertex group, mattes and opacity in the material. The doctor's 63 bones and
58 layers render in about two seconds a frame, and the `.blend` can be opened and
nudged by hand. Stage 7's two-plate `perform.py` path is untouched and still
renders the same shot.

### 8g — `gen.py`: image generation in manual mode

```bash
uv run gen.py prompt --rig robot --job mouths      # writes gen/<stamp>-robot-mouths/
uv run gen.py ingest gen/<stamp>-robot-mouths downloaded.png
```

The only image model available is the Gemini app, so this does what `plan.py`
does with the Claude app: writes a bundle you paste in, and reads the result
back. `prompt` writes `prompt.md`, the images to attach (the rig's head from its
tracked plate, the shapes it already has for the job, and a blank grid of green
cells sized from the rig's own face width so the model lays the sheet out where
`face.py sheet` expects it), and `meta.json` with the SHA-256 of everything sent.
Jobs: `mouths` (the eight-shape ladder), `brows`, `eyes` (open, half, shut,
and pupils for the rigs that have none), `expressions`. The prompt text is
built from the rig config in one place, so it can be tuned.

`ingest` splits the sheet with the existing `face.py` code, snapping colours to
the rig's own measured palette rather than the doctor's, un-premultiplies a cell
drawn on white, and rejects a sheet whose cell count is wrong, whose cells are
empty, or whose line weight or palette is far from the rig's head - both
measured against the reference and printed either way. Line weight is the lower
quartile of the outline's spine thickness, so a filled mouth cavity does not
read as a thick line. Accepted shapes go to `assets/rigs/<rig>/gen/<job>/` with
a sidecar naming the bundle; generated art is not regenerable, so it is tracked.

The first job, a robot mouth sheet, is written and waiting for the paste. Once
it is back, a shot with it and a shot with the drawn grille go side by side and
the user picks - the tool does not decide which has the character's hand.

### 8b — Animation principles, procedurally

```bash
uv run perform.py --rig doctor ... --turns auto      # the Blender puppet, by default
uv run perform.py --rig doctor ... --engine pillow   # Stage 7's two-plate path
```

`perform.py` keeps its command line and sidecar; with a puppet and Blender
present it now renders through them (`--engine auto`), and the Pillow path
stays as `--engine pillow`. `montage.py` follows the same rule per shot.

`motion.py` is the motion library: easing, anticipation, overshoot,
follow-through, settle, squash and stretch, blink, head turns, gaze darts and
the camera push, all as pure numpy curves on frame arrays, every amplitude and
duration a named constant with its unit, nothing bare inside a frame loop.
`act.py` decides which bone gets which curve and writes every frame as a key on
the armature, so what Blender renders is exactly what the curves say and the
curves are still there to nudge in the graph editor:

- **A beat hit** is a small lift before it, a drop on it, a bounce past rest
  and a settle - never a step. The head gets the same a frame later and larger.
- **Squash on the hit, stretch on the release**, about the body's pivot at the
  feet. Half a percent: at 1.8% the doctor's head moved 33px per hit, a pogo.
- **Children lag their parents** by a per-bone number of frames recorded in
  `rig.json` (ears and hair 2, strands and tail 3, hands 1); the lag is the
  parent's motion late and damped, applied as a delta so nothing jumps.
- **Eyes dart** toward a coming turn three frames before the head moves. The dog
  moves its pupil layers; the doctor and robot, which have none, get a disc drawn
  in the eye's own darkest colour at a size measured from the eye box, recorded
  in `rig.json`.
- **A phrase end** sinks the body, tilts the head, holds, and releases on the
  next onset.
- **A head turn** steps through the baked view angles two frames each; a turned
  frame shows that angle's head/body plates in place of the layer stack, with
  the mouth re-placed from that angle's measured geometry. `--turns auto` turns
  at every second breath and comes back on the next.
- **Blinks** scale each eye layer about its own centre on a bone made for it;
  **mouths** are one sprite per shape at the face anchor, shown by the mouth
  track; the rig's own mouth drawings stay out, as the head plate kept them out
  in Stage 7.

Tears and hand gestures are not on the Blender path yet and fall back to the
Pillow engine when asked for. The doctor's nine-second shot renders in 24s.
