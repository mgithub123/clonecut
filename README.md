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

## Build status

| Stage | File | Status |
|---|---|---|
| 1. Ingest | `ingest.py` | done |
| 2. EDL schema | `schema.py` | done |
| 2b. Plan (manual) | `plan.py` | done |
| 3. Render | `render.py` | done |
| 4. Learn | `log.py` + SQLite | done |

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
