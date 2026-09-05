# The rig format

A rig is a folder under `assets/rigs/<name>/` holding PNG plates and one
`rig.json`. Nothing in the pipeline needs anything else: Harmony produced the
three rigs that exist, but the format does not assume Harmony, and a later stage
will compile characters from other sources - generated images, cut-up drawings -
into the same layout. Where a key only makes sense for a Harmony rig, this file
says so.

```
assets/rigs/<name>/
  rig.json                 geometry, capabilities, and the measured puppet
  plates/                  the baked art
    manifest.json          what was baked, from where, with every offset
    <id>-<Name>.png        one element, alone, cropped to its art
    drawings/<id>-<d>.png  every drawing of every multi-drawing element
    turnaround/pose-NNNN.png  the view angles, whole figure
    _ALL.png               the rig as authored, full canvas (the authored truth)
    _ALL_NOCUT.png         the same with cutters bypassed (the cutter-free truth)
    plate-head-<w>.png, plate-body-<w>.png   what Stage 7's perform.py composites
  poses/                   view angles as head/body plate pairs (Stage 7e)
    manifest.json
  mouths/<SHAPE>.png       the eight-shape mouth library
```

Units are pixels on the plate canvas unless a key says otherwise. Every bbox is
`[x0, y0, x1, y1]`, inclusive at both ends, in canvas pixels; a cropped plate is
put back on the canvas by pasting at `(x0, y0)`.

## `plates/manifest.json` - what was baked

Written by `rig.py bake`. Harmony-only in origin, but everything downstream reads
only the fields below, which any baker can write.

| Key | Meaning |
|---|---|
| `resolution` | `[w, h]` of the plate canvas. The three rigs bake at 7680x4320. |
| `cropped` | true: element plates are cropped to their art; `bbox` restores them. |
| `elements.<id>.name` | the element's name as authored |
| `elements.<id>.file` | its plate, relative to `plates/` |
| `elements.<id>.bbox` | where the plate sits on the canvas; `null` for an element that drew nothing (guides, reference layers) |
| `elements.<id>.drawings` | how many drawings the element has |
| `elements.<id>.library.<drawing>` | `{file, bbox}` per drawing of a multi-drawing element, under `plates/drawings/`; `null` for a blank drawing |
| `elements.<id>.folder` | Harmony: the folder the .tvg files lived in. Unused downstream. |
| `ground_truth` | `_ALL.png`: the rig as authored |
| `ground_truth_nocut` | `_ALL_NOCUT.png`: the same with cutters bypassed, which is what element plates can be checked against |
| `turnaround.poses[]` | `{frame, file, bbox}` per turnaround frame under `plates/turnaround/` |
| `scene`, `scene_sha256`, `cutters_bypassed`, `colour_cards_dropped` | Harmony: provenance of the bake |

## `rig.json` - the hand-measured geometry (Stage 7)

These keys were measured from rendered plates when each rig was set up, mostly
by differencing renders (see HARMONY.md), and a few were tuned by eye. Stage 7's
`perform.py` reads them; Stage 8's puppet keeps them working.

| Key | Meaning, unit, how measured |
|---|---|
| `name` | the rig's folder name |
| `scene` | Harmony: path of the working scene copy. Only needed to bake more. |
| `render.resolution` | `[w, h]` the head/body plates and poses are rendered at (3840x2160); half the plate canvas |
| `layers.head`, `layers.body` | element ids that make up the head plate and the body plate. Chosen by hand from `rig.py info`. |
| `layers.mouth`, `layers.mouth_family` | the mouth element, and the supporting elements (teeth, tongue) that are excluded from the head plate with it |
| `layers.eyes` | the eye element, used to measure eye boxes per pose |
| `layers.hands` | groups of element ids that move together as one hand; the first id carries the pose drawings |
| `face.width` | mouth region width in render-resolution pixels, measured at the lip line |
| `face.lip_line`, `face.centre_x` | where the mouth's top edge sits and its centre, render-resolution pixels, by differencing head-with-mouth against head |
| `face.anchor`, `face.anchor_y` | which edge stays put as the mouth opens (`top` or `bottom`), and the fixed edge's y |
| `face.max_mouth_height`, `face.chin` | how tall a mouth may be before it crosses the chin or collar |
| `face.head_pivot` | `[x, y]` the head rotates about, render-resolution pixels; hand-tuned to the neck |
| `eyes` | two eye boxes, render-resolution pixels, from `rig.py measure` |
| `eye_colour` | RGB of the eye fill, so a blink can be drawn |
| `mouth_ladder` | the eight shape names, shut to wide |
| `mouth_width_pct.<SHAPE>` | each shape's width as a fraction of `face.width`; tuned by eye |
| `mouth_source` | how the mouth library was made: derived from drawings, drawn, or generated |
| `vowel_map`, `vowel_hold` | how voice.py's vowels become shapes, and which shapes hold while a note decays |
| `tears` | tear paths and y-range, render-resolution pixels; only where `capabilities.tears` |
| `crop` | the render-resolution box `perform.py` cuts a shot to; generous enough that head motion never reaches its edge |
| `capabilities` | what this rig can honestly do: `lipsync`, `blink`, `tears`, `gaze`, `hair_lag`, `hands`, `poses`, `gesture`. `montage.py` refuses a shot that asks for more. |
| `poses` | summary of `poses/manifest.json`: which view angles exist and which can lip-sync |

## `rig.json["puppet"]` - the measured puppet (Stage 8a)

Written by `uv run puppet.py build <name>`; do not edit by hand, rebuild. It is
measured from the plates and the two ground truths, so any source that can
produce those can produce a puppet. Everything is in plate-canvas pixels.

| Key | Meaning, how measured |
|---|---|
| `canvas` | `[w, h]` of the plate canvas |
| `ground_truth.nocut`, `.authored` | the two images the puppet was measured against |
| `parenting_source` | where the bone tree came from. `peg hierarchy of <scene> (Harmony)` when the tracked .xstage was read; `overlap geometry` otherwise. |
| `bones.<key>` | one bone per peg the scene needed (pegs with no layer and one child are collapsed). `name` is the peg's own name; `parent` is another bone key or `null` at the root. Keys are the peg's scope path in the scene, so two pegs called `Peg` in different groups stay distinct. |
| `bones.<key>.pivot` | `[x, y]`. A part has a free end - the point of its art farthest from its parent's centre: the hand of an arm, the foot of a leg - and a joined end. The pivot is the centre of the quarter of the overlap between this bone's art (its whole subtree) and its parent's art that lies farthest from the free end: a leg's hip, a shin's knee, a boot's ankle, an arm's shoulder even though the arm also runs down the chest. The parent's art is tried in order: the parent bone's own layers, each ancestor's own layers, then everything else in the rig. |
| `bones.<key>.pivot_method` | which of those rules produced the pivot, in words, so a wrong one can be found |
| `bones.<key>.layers` | the layer ids attached directly to this bone |
| `layers[]` | the plates in draw order, bottom first. `z` is the position in that order. |
| `layers[].id` | the element id, or `<id>#<n>` for one instance of an element the scene draws more than once |
| `layers[].file`, `.bbox` | the plate and where it sits on the canvas |
| `layers[].bone` | the bone it hangs from |
| `layers[].library` | for a multi-drawing element, `{drawing: {file, bbox}}` - the hand poses, the mouth chart, the bangs; a frame picks one by name |
| `layers[].element`, `.instance`, `.crop` | instance layers: the shared element, which blob (left to right), and that blob's window within the shared plate. Blobs are matched to pegs in the order the scene lists them, left to right, which is an assumption and is said so. |
| `layers[].instances` | an element drawn more than once whose plate did not fall into that many blobs (legs drawn touching); the copies move together |
| `layers[].matte` | `{layer, inverted, cut_fraction, fit}`. The plate whose alpha, or its complement, best explains what this layer loses between the cutter-free and the authored truth: a highlight kept inside the hair, a decal kept on its panel. Fitted top layer first, on observable pixels only (where the layer differs from what lies under it - a black lid over black hair says nothing), and only when at least a tenth of those are cut and the best plate explains three quarters of it. Applied as an alpha multiply at composite time. |
| `layers[].hidden_at_rest` | a layer whose observable pixels are all gone in the authored truth: a drawing the rig keeps for later - the doctor's closed lower lid, the robot's head-angle guide - and hides at rest. Not drawn unless a frame picks it. |
| `layers[].opacity` | a layer the rig draws see-through, as the least-squares blend of the plate over what lies below it that fits the authored truth. Only recorded between 0.1 and 0.9 and only where the blend explains the truth better than opaque does. |
| `angles` | the view angles from `poses/manifest.json`: `usable` pose numbers, the `default`, and their `resolution`. A frame at a non-default angle swaps the whole layer stack for that pose's head/body pair. |
| `hands.<lead>` | per hand group, the drawing names a frame can pick, and the ids drawn with it |
| `measured.recomposite_vs_nocut` | how far the layers in draw order, with no mattes, are from `_ALL_NOCUT.png` - the proof the offsets and order are right |
| `measured.matted_vs_authored` | the same with mattes and opacity applied, against `_ALL.png` |

### Draw order

Peeled from the cutter-free truth from the top. The layer on top is the one
whose opaque pixels the truth still shows; take it, claim its pixels, ask again.
Each next layer is judged only where nothing above it has been drawn. A layer
nothing shows at rest (a hidden guide, a closed-eye drawing) is reported and
sent to the bottom.

### What the numbers were on the three rigs

| | Doctor | Dog | Robot |
|---|---|---|---|
| Plates with art | 31 | 29 | 33 |
| Bones (pegs kept) | 63 | 58 | 59 |
| Instance layers split out | 27 | 28 | 18 |
| Mattes reconstructed | 5 | 7 | 1 |
| Layers hidden at rest | 1 | 2 | 1 |
| See-through layers | 0 | 0 | 3 (dome 0.19, eye lines 0.72) |
| Recomposite vs `_ALL_NOCUT`, pixels differing | 1.39% | 1.86% | 0.19% |
| With mattes and opacity vs `_ALL` | 2.31% | 6.32% | 16.14% |
| Same, element-id order, no measurement | 23.2% | 12.3% | 27.9% |

The doctor's remaining authored difference is its bangs, which Harmony draws
through a cutter whose matte is another hair element of the same colour, so the
fit cannot tell them apart; the dog's is deformers: Harmony bends its collar and
stitches with envelope deformers and auto-patches at render time, which a plate
cannot carry. The robot's is its glass dome: the plate
bakes it opaque, the rig draws it see-through with a shine, and the fitted
opacity of 0.19 gets the face back but not the shine's gradient. One robot
shoulder pivot also lands nearer the chest's centre than the armpit, because the
arm plate carries hidden art behind the chest; the joint rule cannot tell hidden
art from visible.

## The Blender scene

`uv run puppet.py render <name>` writes a job from the puppet block and runs
`blenderpuppet.py` inside Blender: one image plane per layer at its offset, a
small Z step per draw-order position, one armature bone per puppet bone with its
pivot and parent, each plane bound to its bone by vertex group, an orthographic
camera framed on the character, and Eevee rendering PNG with alpha. Mattes are
the matte plate's alpha sampled in canvas space and multiplied in; opacity is a
multiply on alpha. The `.blend` is saved beside the frames and can be opened and
nudged by hand.
