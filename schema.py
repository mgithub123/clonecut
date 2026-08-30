#!/usr/bin/env python3
"""The Edit Decision List: the only thing the AI produces, and the only thing
the renderer consumes.

Two layers of validation live here, deliberately separate:

* The Pydantic models check *structure* - field names, types, ranges, ordering.
  They know nothing about the media, so they can validate a model response
  before any file is touched.
* ``validate_media()`` checks the EDL *against the actual files* - that clips
  exist and that no segment reaches past the end of one. This is the check that
  has to fail loudly at render time.

A note on ``snap_to_beat``: snapping happens in code inside plan.py, after the
model responds. By the time an EDL is on disk its timestamps are already
snapped, and the flag is a record of that, not an instruction to the renderer.
"""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

import common
import config

SCHEMA_VERSION = 1

# How far past a clip's probed duration a segment may reach before it is an
# error. One frame of slack at 30fps; ffprobe durations are not exact.
DURATION_EPSILON = 0.034


class EdlError(RuntimeError):
    """An EDL is structurally valid but does not match the media it references."""


class CaptionStyle(str, Enum):
    """Names must match the style table in caption_styles.py."""
    HOOK = "hook"
    BODY = "body"
    EMPHASIS = "emphasis"


class CaptionPosition(str, Enum):
    """Vertical placement, resolved to safe-area-aware pixels by the renderer."""
    UPPER_THIRD = "upper-third"
    CENTER = "center"
    LOWER_THIRD = "lower-third"


class Strict(BaseModel):
    """Reject unknown fields everywhere.

    A model that writes "duration" where the schema says "out" should fail the
    validation retry, not silently render the wrong edit.
    """
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class AudioSpec(Strict):
    source: str = Field(description="path to the music file, relative to the project root")
    start: float = Field(ge=0, description="offset into the track where the video's audio begins")
    fade_in: float = Field(default=0.0, ge=0, le=10)
    fade_out: float = Field(default=0.0, ge=0, le=10)


class Segment(Strict):
    clip: str = Field(description="path to the source video, relative to the project root")
    in_: float = Field(alias="in", ge=0, description="in point in the source clip, seconds")
    out_: float = Field(alias="out", gt=0, description="out point in the source clip, seconds")
    speed: float = Field(default=1.0, gt=0.1, le=8.0)
    snap_to_beat: bool = Field(default=False, description="set by plan.py once boundaries are snapped")
    reason: str = Field(default="", max_length=300, description="why this shot, in the model's words")

    @model_validator(mode="after")
    def _ordered(self) -> "Segment":
        if self.out_ <= self.in_:
            raise ValueError(f"segment out ({self.out_}) must be greater than in ({self.in_})")
        return self

    @property
    def source_duration(self) -> float:
        """Length consumed from the source clip."""
        return self.out_ - self.in_

    @property
    def output_duration(self) -> float:
        """Length occupied in the finished video, after the speed change."""
        return self.source_duration / self.speed


class Caption(Strict):
    text: str = Field(min_length=1, max_length=120)
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    style: CaptionStyle = CaptionStyle.BODY
    position: CaptionPosition = CaptionPosition.CENTER

    @model_validator(mode="after")
    def _ordered(self) -> "Caption":
        if self.end <= self.start:
            raise ValueError(f"caption end ({self.end}) must be after start ({self.start}): {self.text!r}")
        return self

    @property
    def duration(self) -> float:
        return self.end - self.start


class EDL(Strict):
    variant_name: str = Field(
        min_length=1, max_length=60, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
        description="kebab-case; becomes part of the output filename, so no slashes or spaces",
    )
    strategy_notes: str = Field(min_length=1, max_length=800,
                                description="why this edit was chosen, 2-3 sentences")
    target_duration: float = Field(gt=0, le=180)
    audio: AudioSpec
    segments: list[Segment] = Field(min_length=1)
    captions: list[Caption] = Field(default_factory=list)

    @field_validator("segments")
    @classmethod
    def _segments_present(cls, v: list[Segment]) -> list[Segment]:
        if not v:
            raise ValueError("an EDL needs at least one segment")
        return v

    # --- derived ------------------------------------------------------------

    @property
    def duration(self) -> float:
        """Actual duration implied by the segments. This, not target_duration,
        is what the renderer produces."""
        return sum(s.output_duration for s in self.segments)

    @property
    def beat_synced(self) -> bool:
        return any(s.snap_to_beat for s in self.segments)

    def derived_features(self) -> dict[str, Any]:
        """Features Stage 4 correlates against performance.

        Only what the EDL alone can answer; the song section a variant used
        needs the audio profile and is filled in by log.py.
        """
        dur = self.duration
        lengths = [s.output_duration for s in self.segments]
        hook = next((c for c in sorted(self.captions, key=lambda c: c.start)
                     if c.style is CaptionStyle.HOOK), None)
        return {
            "duration": common.r3(dur),
            "cut_count": len(self.segments),
            "avg_segment_length": common.r3(sum(lengths) / len(lengths)),
            "shortest_segment": common.r3(min(lengths)),
            "longest_segment": common.r3(max(lengths)),
            "cuts_per_second": common.r3(len(self.segments) / dur) if dur else 0.0,
            "caption_count": len(self.captions),
            "caption_density": common.r3(len(self.captions) / dur) if dur else 0.0,
            "beat_synced": self.beat_synced,
            "uses_speed_ramp": any(abs(s.speed - 1.0) > 1e-6 for s in self.segments),
            "hook_type": hook.text[:80] if hook else None,
            "hook_position": hook.position.value if hook else None,
            "clips_used": sorted({s.clip for s in self.segments}),
            "audio_start": self.audio.start,
        }

    # --- io -----------------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(self.model_dump(by_alias=True, mode="json"), indent=2, ensure_ascii=False)

    def write(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_json())


def load_edl(path: str | Path) -> EDL:
    """Read an EDL from disk, reporting a bad one in terms of the file."""
    path = Path(path)
    if not path.exists():
        raise EdlError(f"EDL not found: {path}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise EdlError(f"{path} is not valid JSON: {exc}") from exc
    try:
        return EDL.model_validate(raw)
    except ValidationError as exc:
        raise EdlError(f"{path} is not a valid EDL:\n{format_validation_error(exc)}") from exc


def format_validation_error(exc: ValidationError) -> str:
    """Pydantic errors, rendered for a human and for feeding back to the model."""
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# media-aware validation
# ---------------------------------------------------------------------------

def resolve_media(ref: str) -> Path:
    """Resolve a path from an EDL.

    The model is told to use project-relative paths, but plans get hand-edited
    and a bare filename is a natural thing to type, so fall back to the
    conventional media directories before giving up.
    """
    candidate = Path(ref)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for base in (config.ROOT, config.RAW_DIR, config.MUSIC_DIR, Path.cwd()):
        p = base / ref
        if p.exists():
            return p.resolve()
        # also try the bare filename inside the conventional directories
        p = base / candidate.name
        if p.exists():
            return p.resolve()
    return (config.ROOT / ref)  # non-existent; the caller reports it


def _probe_duration(path: Path) -> float | None:
    try:
        info = common.ffprobe_json(path)
    except common.ToolError:
        return None
    fmt = info.get("format", {})
    if fmt.get("duration"):
        return float(fmt["duration"])
    for stream in info.get("streams", []):
        if stream.get("duration"):
            return float(stream["duration"])
    return None


def validate_media(edl: EDL) -> tuple[list[str], list[str]]:
    """Check an EDL against the files it references.

    Returns (errors, warnings). Errors mean the render would be wrong or would
    fail partway; warnings mean it will render but probably not as intended.
    """
    errors: list[str] = []
    warnings: list[str] = []
    durations: dict[str, float | None] = {}

    def duration_of(ref: str) -> float | None:
        if ref not in durations:
            path = resolve_media(ref)
            durations[ref] = _probe_duration(path) if path.exists() else None
        return durations[ref]

    # --- segments -----------------------------------------------------------
    for i, seg in enumerate(edl.segments):
        path = resolve_media(seg.clip)
        if not path.exists():
            errors.append(f"segment {i}: clip not found: {seg.clip}")
            continue
        dur = duration_of(seg.clip)
        if dur is None:
            warnings.append(f"segment {i}: could not probe a duration for {seg.clip}")
            continue
        if seg.in_ >= dur:
            errors.append(
                f"segment {i}: in point {seg.in_:.3f}s is at or past the end of "
                f"{seg.clip} ({dur:.3f}s)"
            )
        elif seg.out_ > dur + DURATION_EPSILON:
            errors.append(
                f"segment {i}: out point {seg.out_:.3f}s runs past the end of "
                f"{seg.clip} ({dur:.3f}s) by {seg.out_ - dur:.3f}s"
            )

    # --- audio --------------------------------------------------------------
    audio_path = resolve_media(edl.audio.source)
    if not audio_path.exists():
        errors.append(f"audio: track not found: {edl.audio.source}")
    else:
        adur = duration_of(edl.audio.source)
        needed = edl.duration
        if adur is None:
            warnings.append(f"audio: could not probe a duration for {edl.audio.source}")
        elif edl.audio.start >= adur:
            errors.append(
                f"audio: start {edl.audio.start:.3f}s is at or past the end of "
                f"{edl.audio.source} ({adur:.3f}s)"
            )
        elif edl.audio.start + needed > adur + DURATION_EPSILON:
            errors.append(
                f"audio: the edit needs {needed:.3f}s from {edl.audio.start:.3f}s, but "
                f"{edl.audio.source} is only {adur:.3f}s long "
                f"(short by {edl.audio.start + needed - adur:.3f}s)"
            )

    # --- soft checks --------------------------------------------------------
    total = edl.duration
    if abs(total - edl.target_duration) > 0.5:
        warnings.append(
            f"segments total {total:.2f}s but target_duration says {edl.target_duration:.2f}s; "
            f"the render will be {total:.2f}s"
        )
    if edl.audio.fade_out > total:
        warnings.append(f"fade_out ({edl.audio.fade_out}s) is longer than the video ({total:.2f}s)")
    for i, cap in enumerate(edl.captions):
        if cap.start >= total:
            warnings.append(
                f"caption {i} ({cap.text!r}) starts at {cap.start:.2f}s, after the video ends "
                f"at {total:.2f}s; it will never appear"
            )
        elif cap.end > total + 0.05:
            warnings.append(
                f"caption {i} ({cap.text!r}) runs to {cap.end:.2f}s, past the end of the "
                f"video at {total:.2f}s; it will be cut short"
            )
    return errors, warnings


def require_valid_media(edl: EDL, *, source: str = "EDL") -> list[str]:
    """Raise EdlError listing every problem, or return the warnings."""
    errors, warnings = validate_media(edl)
    if errors:
        raise EdlError(
            f"{source} does not match the media it references:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
    return warnings


# ---------------------------------------------------------------------------
# introspection, used to build the prompt in plan.py
# ---------------------------------------------------------------------------

def json_schema() -> dict[str, Any]:
    return EDL.model_json_schema(by_alias=True)


def describe_schema() -> str:
    """A compact, readable field spec for the planning prompt.

    Generated from the models rather than written out by hand, so the prompt can
    never drift from what validation actually enforces.
    """
    schema = EDL.model_json_schema(by_alias=True)
    defs = schema.get("$defs", {})
    lines: list[str] = []

    def render(title: str, node: dict[str, Any], indent: str = "  ") -> None:
        lines.append(f"{title}:")
        required = set(node.get("required", []))
        for name, field in node.get("properties", {}).items():
            lines.append(f"{indent}{name}{' (required)' if name in required else ''}"
                         f"  {_type_of(field, defs)}")
            desc = field.get("description")
            if desc:
                lines.append(f"{indent}    {desc}")

    render("EDL (top level)", schema)
    for name in ("AudioSpec", "Segment", "Caption"):
        if name in defs:
            lines.append("")
            render(name, defs[name])
    return "\n".join(lines)


def _type_of(field: dict[str, Any], defs: dict[str, Any]) -> str:
    """Render one field's type and constraints the way a human would read them."""
    ref = field.get("$ref") or (field.get("allOf") or [{}])[0].get("$ref")
    if ref:
        target = defs.get(ref.rsplit("/", 1)[-1], {})
        if "enum" in target:
            return "one of: " + ", ".join(repr(v) for v in target["enum"])
        return ref.rsplit("/", 1)[-1]

    kind = field.get("type", "any")
    if kind == "array":
        items = field.get("items", {})
        inner = _type_of(items, defs) if items else "any"
        kind = f"array of {inner}"

    bits = [kind]
    for key, label in (("minimum", ">="), ("exclusiveMinimum", ">"),
                       ("maximum", "<="), ("exclusiveMaximum", "<"),
                       ("minLength", "min length"), ("maxLength", "max length"),
                       ("minItems", "min items")):
        if key in field:
            bits.append(f"{label} {field[key]}")
    if "pattern" in field:
        bits.append(f"matching {field['pattern']}")
    if "default" in field:
        bits.append(f"default {field['default']!r}")
    return ", ".join(bits)


def main(argv: list[str] | None = None) -> int:
    """Validate EDL files from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate one or more EDL files.")
    parser.add_argument("edl", nargs="*", type=Path, help="EDL files to check")
    parser.add_argument("--print-schema", action="store_true", help="dump the JSON schema")
    args = parser.parse_args(argv)

    if args.print_schema:
        print(json.dumps(json_schema(), indent=2))
        return 0
    if not args.edl:
        parser.error("give at least one EDL path, or --print-schema")

    failed = 0
    for path in args.edl:
        try:
            edl = load_edl(path)
        except EdlError as exc:
            failed += 1
            print(f"INVALID  {path}\n{exc}\n")
            continue
        errors, warnings = validate_media(edl)
        status = "INVALID" if errors else ("OK     " if not warnings else "OK*    ")
        print(f"{status}  {path}")
        print(f"         {edl.variant_name}: {len(edl.segments)} segments, "
              f"{len(edl.captions)} captions, {edl.duration:.2f}s"
              f"{', beat-synced' if edl.beat_synced else ''}")
        for e in errors:
            print(f"         error:   {e}")
        for w in warnings:
            print(f"         warning: {w}")
        failed += bool(errors)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
