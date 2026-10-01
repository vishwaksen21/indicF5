"""Phase 4B-1: offline technical draft render (placeholder visuals + burned subtitles + original audio).

    .venv/bin/python scripts/assemble_video.py --job job_20261001_103028_eb8ca2

Reads (never modifies): speech.wav, metadata.json, alignment/words.json, scene_plan/scene_plan.json
(re-validated in memory with the Phase 3.1 planner; timeline must match exactly), assets/asset_manifest.json,
subtitles/{captions_highlight.ass, captions.srt, subtitle_report.json}.
Writes (refuses to overwrite): renders/phase4b1_draft.mp4, renders/phase4b1_draft_report.json,
renders/phase4b1_draft_review.png.

How it renders (deterministic, offline, no models):
  - Pillow draws a neutral grey placeholder per scene, marked DRAFT / TEST — NOT FINAL (no logo, no brand colours).
  - The installed FFmpeg has no libass, so captions_highlight.ass is parsed by a strict parser here and every
    Dialogue event is rasterised by Pillow into an RGBA overlay using the event's exact start/end times. Any ASS
    feature outside the subset written by subtitles.py (colour overrides, \\N) stops the render.
  - Every video frame n (t = n / 25) is composited in Pillow: scene placeholder, transitions from the plan
    (fade in, centred cross-dissolve, fade out), then the subtitle event with start <= t < end.
  - FFmpeg encodes the PNG sequence (libx264, yuv420p, 25 fps) and muxes speech.wav as the only audio (AAC,
    same 24 kHz mono: no resampling, filters or loudness processing). AAC is lossy; the report says so.
  - The result is validated (ffprobe, decoded frames vs expected frames, blackdetect, decoded audio vs WAV) on a
    temporary file and only then atomically renamed into place. A failed render leaves no output.
Exit codes: 0 success; 1 failed (nothing written).
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

W, H, FPS = 1280, 720, 25
FRAME_S = 1.0 / FPS
FADE_IN_S = 0.25
FADE_OUT_MAX_S = 0.5
DRAFT_TEXT = "DRAFT / TEST — NOT FINAL"
FONTS = {"Helvetica Neue": ("/System/Library/Fonts/HelveticaNeue.ttc", 0)}
UI_FONT = FONTS["Helvetica Neue"]
SCENE_BG = [(48, 48, 48), (84, 84, 84), (64, 64, 64), (100, 100, 100)]   # neutral greys, cycled per scene
PROBE_BOX = (1170, 600, 1270, 700)          # x0,y0,x1,y1: plain background, outside subtitle + label areas
SUB_BOX = (128, 480, 1152, 690)             # where subtitles may appear at 720p with the ASS margins
PSNR_MIN_DB = 30.0
DEFAULT_OUT = "renders/phase4b1_draft.mp4"
ASS_EVENT_FORMAT = "Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
ASS_STYLE_FIELDS = ("Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
                    "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                    "Alignment, MarginL, MarginR, MarginV, Encoding")


class RenderError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


# ================================================================ ASS (strict subset parser)


@dataclass
class AssEvent:
    index: int
    start_s: float
    end_s: float
    style: str
    lines: list            # [[(text, rgba), ...], ...] one list of runs per visual line
    raw: str = ""


@dataclass
class AssDoc:
    play_res: tuple
    style: dict
    events: list = field(default_factory=list)


def ass_colour(code: str) -> tuple:
    m = re.fullmatch(r"&H([0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})&?", code)
    if not m:
        raise RenderError("ASS_MALFORMED", f"bad ASS colour {code!r}")
    hx = m.group(1).rjust(8, "0")
    a, b, g, r = (int(hx[i:i + 2], 16) for i in range(0, 8, 2))
    return (r, g, b, 255 - a)


def ass_time(s: str) -> float:
    m = re.fullmatch(r"(\d+):(\d\d):(\d\d)\.(\d\d)", s.strip())
    if not m:
        raise RenderError("ASS_MALFORMED", f"bad ASS time {s!r}")
    h, mi, se, cs = map(int, m.groups())
    return round(h * 3600 + mi * 60 + se + cs / 100, 2)


def parse_text(text: str, primary: tuple) -> list:
    """Supported: {\\c&H..&} / {\\1c&H..&} colour overrides and \\N line breaks. Anything else is refused."""
    lines, runs, colour, buf, i = [], [], primary, "", 0
    while i < len(text):
        ch = text[i]
        if ch == "{":
            j = text.find("}", i)
            if j < 0:
                raise RenderError("ASS_MALFORMED", f"unbalanced '{{' in {text!r}")
            inner = text[i + 1:j]
            tags = [t for t in inner.split("\\")]
            if tags[0] != "" or len(tags) < 2:
                raise RenderError("ASS_UNSUPPORTED", f"override block {{{inner}}} is a comment or not a tag")
            for tag in tags[1:]:
                m = re.fullmatch(r"1?c(&H[0-9A-Fa-f]{6,8}&)", tag)
                if not m:
                    raise RenderError("ASS_UNSUPPORTED", f"unsupported ASS override tag \\{tag} (only colour \\c is "
                                                         "supported; karaoke/positioning/font tags are not)")
                if buf:
                    runs.append((buf, colour))
                    buf = ""
                colour = ass_colour(m.group(1))
            i = j + 1
        elif ch == "}":
            raise RenderError("ASS_MALFORMED", f"unbalanced '}}' in {text!r}")
        elif ch == "\\":
            nxt = text[i + 1:i + 2]
            if nxt != "N":
                raise RenderError("ASS_UNSUPPORTED", f"unsupported ASS escape \\{nxt} (only \\N is supported)")
            if buf:
                runs.append((buf, colour))
                buf = ""
            lines.append(runs)
            runs = []
            i += 2
        else:
            buf += ch
            i += 1
    if buf:
        runs.append((buf, colour))
    lines.append(runs)
    if not any(r for ln in lines for r in ln):
        raise RenderError("ASS_MALFORMED", "event has no visible text")
    return lines


def parse_ass(text: str) -> AssDoc:
    sections, cur = {}, None
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith(";"):
            continue
        if s.startswith("[") and s.endswith("]"):
            cur = s[1:-1]
            sections[cur] = []
        elif cur is None:
            raise RenderError("ASS_MALFORMED", f"content before first section: {s[:40]!r}")
        else:
            sections[cur].append(s)
    for need in ("Script Info", "V4+ Styles", "Events"):
        if need not in sections:
            raise RenderError("ASS_MALFORMED", f"missing [{need}] section")
    info = dict(l.split(":", 1) for l in sections["Script Info"] if ":" in l)
    info = {k.strip(): v.strip() for k, v in info.items()}
    try:
        play_res = (int(info["PlayResX"]), int(info["PlayResY"]))
    except (KeyError, ValueError):
        raise RenderError("ASS_MALFORMED", "PlayResX/PlayResY missing or invalid")
    if info.get("ScriptType", "").lower() != "v4.00+":
        raise RenderError("ASS_UNSUPPORTED", f"ScriptType {info.get('ScriptType')!r} (need v4.00+)")
    if info.get("WrapStyle") != "2":
        raise RenderError("ASS_UNSUPPORTED", "only WrapStyle 2 (no automatic wrapping) is supported")
    if info.get("ScaledBorderAndShadow", "").lower() != "yes":
        raise RenderError("ASS_UNSUPPORTED", "only ScaledBorderAndShadow: yes is supported")

    st = sections["V4+ Styles"]
    if not st or st[0] != "Format: " + ASS_STYLE_FIELDS:
        raise RenderError("ASS_MALFORMED", "unexpected [V4+ Styles] Format line")
    keys = [k.strip() for k in ASS_STYLE_FIELDS.split(",")]
    styles = {}
    for l in st[1:]:
        if not l.startswith("Style:"):
            raise RenderError("ASS_MALFORMED", f"unexpected style line {l[:40]!r}")
        vals = [v.strip() for v in l[len("Style:"):].split(",")]
        if len(vals) != len(keys):
            raise RenderError("ASS_MALFORMED", f"style has {len(vals)} fields, expected {len(keys)}")
        styles[vals[0]] = dict(zip(keys, vals))
    if len(styles) != 1:
        raise RenderError("ASS_UNSUPPORTED", f"exactly one style supported, found {len(styles)}")
    style = next(iter(styles.values()))
    unsupported = {k: style[k] for k, want in (("Bold", "0"), ("Italic", "0"), ("Underline", "0"), ("StrikeOut", "0"),
                                               ("ScaleX", "100"), ("ScaleY", "100"), ("Spacing", "0"), ("Angle", "0"),
                                               ("BorderStyle", "1"), ("Alignment", "2")) if style[k] != want}
    if unsupported:
        raise RenderError("ASS_UNSUPPORTED", f"unsupported style settings {unsupported}")
    if style["Fontname"] not in FONTS:
        raise RenderError("FONT_UNAVAILABLE", f"font {style['Fontname']!r} has no known local file")
    parsed_style = {"name": style["Name"], "font": style["Fontname"], "size": float(style["Fontsize"]),
                    "primary": ass_colour(style["PrimaryColour"]), "outline_colour": ass_colour(style["OutlineColour"]),
                    "back_colour": ass_colour(style["BackColour"]), "outline": float(style["Outline"]),
                    "shadow": float(style["Shadow"]), "margin_l": int(style["MarginL"]),
                    "margin_r": int(style["MarginR"]), "margin_v": int(style["MarginV"])}

    ev = sections["Events"]
    if not ev or ev[0] != "Format: " + ASS_EVENT_FORMAT:
        raise RenderError("ASS_MALFORMED", "unexpected [Events] Format line")
    doc = AssDoc(play_res, parsed_style)
    for l in ev[1:]:
        if not l.startswith("Dialogue:"):
            raise RenderError("ASS_UNSUPPORTED", f"unsupported event type {l.split(':')[0]!r}")
        parts = l[len("Dialogue:"):].split(",", 9)
        if len(parts) != 10:
            raise RenderError("ASS_MALFORMED", f"dialogue has {len(parts)} fields: {l[:60]!r}")
        layer, start, end, sname, name, ml, mr, mv, effect, txt = [p.strip() if k < 9 else p for k, p in enumerate(parts)]
        if layer != "0" or name or (ml, mr, mv) != ("0", "0", "0") or effect:
            raise RenderError("ASS_UNSUPPORTED", f"event uses unsupported Layer/Name/Margin/Effect: {l[:60]!r}")
        if sname not in styles:
            raise RenderError("ASS_MALFORMED", f"event references unknown style {sname!r}")
        s, e = ass_time(start), ass_time(end)
        if not s < e:
            raise RenderError("ASS_MALFORMED", f"event {len(doc.events)} has start {s} >= end {e}")
        doc.events.append(AssEvent(len(doc.events), s, e, sname, parse_text(txt, parsed_style["primary"]), raw=l))
    if not doc.events:
        raise RenderError("ASS_MALFORMED", "no Dialogue events")
    for a, b in zip(doc.events, doc.events[1:]):
        if b.start_s < a.end_s - 1e-9:
            raise RenderError("ASS_UNSUPPORTED", f"events {a.index} and {b.index} overlap in time "
                                                 "(simultaneous captions are not supported)")
    return doc


# ================================================================ drawing


def font(size: float):
    path, idx = UI_FONT
    return ImageFont.truetype(path, max(1, round(size)), index=idx)


def render_overlay(ev: AssEvent, doc: AssDoc) -> Image.Image:
    """Rasterise one ASS event at W x H. ASS PlayRes coordinates are scaled to the video size."""
    sx, sy = W / doc.play_res[0], H / doc.play_res[1]
    if abs(sx - sy) > 1e-6:
        raise RenderError("ASS_UNSUPPORTED", f"PlayRes {doc.play_res} aspect differs from {W}x{H}")
    st = doc.style
    f = font(st["size"] * sx)
    outline = max(1, round(st["outline"] * sx))
    shadow = round(st["shadow"] * sx)
    ascent, descent = f.getmetrics()
    line_h = ascent + descent
    ml, mr, mv = st["margin_l"] * sx, st["margin_r"] * sx, st["margin_v"] * sy
    avail = W - ml - mr
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    top = H - mv - line_h * len(ev.lines)
    placed = []
    for li, runs in enumerate(ev.lines):
        width = sum(f.getlength(t) for t, _ in runs)
        if width > avail + 0.5:
            raise RenderError("SUBTITLE_OVERFLOW", f"event {ev.index} line {li + 1} is {width:.0f}px wide, "
                                                   f"exceeds safe width {avail:.0f}px")
        x, y = ml + (avail - width) / 2, top + li * line_h
        for t, c in runs:
            placed.append((x, y, t, c))
            x += f.getlength(t)
    # pass 1 shadow, pass 2 outline, pass 3 fill: no run's outline can cover a neighbour's fill
    if shadow:
        for x, y, t, _ in placed:
            d.text((x + shadow, y + shadow), t, font=f, fill=st["back_colour"], stroke_width=outline,
                   stroke_fill=st["back_colour"])
    for x, y, t, _ in placed:
        d.text((x, y), t, font=f, fill=st["outline_colour"], stroke_width=outline, stroke_fill=st["outline_colour"])
    for x, y, t, c in placed:
        d.text((x, y), t, font=f, fill=c)
    return img


def placeholder(scene: dict, k: int, n_scenes: int) -> Image.Image:
    bg = SCENE_BG[k % len(SCENE_BG)]
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, 56), fill=(205, 205, 205))
    d.text((W / 2, 28), DRAFT_TEXT, font=font(30), fill=(20, 20, 20), anchor="mm")
    d.text((32, 76), f"{scene['scene_id']} of {n_scenes}  ·  {scene['visual_mode']}  ·  "
                     f"{scene['start_s']:.3f}–{scene['end_s']:.3f} s", font=font(24), fill=(225, 225, 225))
    d.text((32, 108), "technical placeholder · not footage · not the final advertisement",
           font=font(18), fill=(170, 170, 170))
    line = (150, 150, 150)
    cx, cy = W / 2, 290
    if scene["visual_mode"] == "talking_head":
        d.ellipse((cx - 55, cy - 120, cx + 55, cy - 10), outline=line, width=4)
        d.rounded_rectangle((cx - 120, cy + 5, cx + 120, cy + 110), radius=40, outline=line, width=4)
        label = "PRESENTER PLACEHOLDER — no footage, no lip-sync"
    elif scene["visual_mode"] == "text_graphic":
        d.rectangle((cx - 260, cy - 110, cx + 260, cy + 110), outline=line, width=4)
        d.line((cx - 260, cy - 110, cx + 260, cy + 110), fill=line, width=2)
        d.line((cx - 260, cy + 110, cx + 260, cy - 110), fill=line, width=2)
        label = "BRAND CARD PLACEHOLDER — no logo or brand assets supplied"
    else:
        d.rectangle((cx - 260, cy - 110, cx + 260, cy + 110), outline=line, width=4)
        label = f"{scene['visual_mode'].upper()} PLACEHOLDER"
    d.text((cx, cy + 150), label, font=font(26), fill=(230, 230, 230), anchor="mm")
    return img


# ================================================================ timeline


@dataclass
class Timeline:
    duration_s: float
    n_frames: int
    scenes: list
    dissolves: list        # (boundary_s, d) per scene boundary, d == 0 for hard cuts
    fade_in_end: float
    fade_out_start: float


def parse_transitions(plan: dict, duration: float) -> tuple:
    sc = plan["scenes"]
    dissolves = []
    for i, s in enumerate(sc[:-1]):
        tr = s.get("transition") or ""
        m = re.match(r"cross-dissolve ([\d.]+) s", tr)
        if m:
            d = float(m.group(1))
            b = s["end_s"]
            if not (s["narration_end_s"] <= b - d / 2 and b + d / 2 <= sc[i + 1]["narration_start_s"] + 1e-6):
                raise RenderError("INVALID_TRANSITION", f"{s['scene_id']}: dissolve {d}s at {b}s leaves the pause")
            dissolves.append((b, d))
        elif tr.startswith("hard cut"):
            dissolves.append((s["end_s"], 0.0))
        else:
            raise RenderError("UNSUPPORTED_TRANSITION", f"{s['scene_id']}: transition {tr!r}")
    if not (sc[0].get("transition_in") or "").startswith("fade in from black"):
        raise RenderError("UNSUPPORTED_TRANSITION", f"first scene transition_in {sc[0].get('transition_in')!r}")
    if not (sc[-1].get("transition") or "").startswith("fade out to black"):
        raise RenderError("UNSUPPORTED_TRANSITION", f"last scene transition {sc[-1].get('transition')!r}")
    fade_out = max(sc[-1]["narration_end_s"], duration - FADE_OUT_MAX_S)
    return dissolves, FADE_IN_S, fade_out


def check_timeline_structure(plan: dict, duration: float):
    sc = plan.get("scenes") or []
    if not sc:
        raise RenderError("PLAN_INVALID", "scene plan has no scenes")
    for s in sc:
        vals = [s.get(k) for k in ("start_s", "end_s", "duration_s", "narration_start_s", "narration_end_s")]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
            raise RenderError("INVALID_SCENE_DURATION", f"{s.get('scene_id')}: non-numeric timeline values")
        if not s["end_s"] - s["start_s"] >= FRAME_S - 1e-9:
            raise RenderError("INVALID_SCENE_DURATION", f"{s['scene_id']}: duration {s['end_s'] - s['start_s']:.3f}s "
                                                        f"is shorter than one frame ({FRAME_S:.2f}s)")
        if abs(s["duration_s"] - (s["end_s"] - s["start_s"])) > 1e-3:
            raise RenderError("INVALID_SCENE_DURATION", f"{s['scene_id']}: duration_s disagrees with start/end")
    if abs(sc[0]["start_s"]) > 1e-6 or abs(sc[-1]["end_s"] - duration) > 1e-3:
        raise RenderError("INVALID_SCENE_DURATION", f"scenes cover {sc[0]['start_s']}–{sc[-1]['end_s']}s, audio is "
                                                    f"0–{duration:.3f}s")
    for a, b in zip(sc, sc[1:]):
        if abs(a["end_s"] - b["start_s"]) > 1e-6:
            raise RenderError("INVALID_SCENE_DURATION", f"gap/overlap between {a['scene_id']} and {b['scene_id']}")


def scene_at(tl: Timeline, t: float) -> int:
    for i, s in enumerate(tl.scenes):
        if s["start_s"] <= t < s["end_s"]:
            return i
    return len(tl.scenes) - 1


class FrameRenderer:
    """Deterministic expected frame for any index n; used for rendering and for validation."""

    def __init__(self, tl: Timeline, doc: AssDoc):
        self.tl, self.doc = tl, doc
        self.bases = [placeholder(s, k, len(tl.scenes)) for k, s in enumerate(tl.scenes)]
        self.overlays = [render_overlay(e, doc) for e in doc.events]
        self.black = Image.new("RGB", (W, H), (0, 0, 0))

    def event_at(self, t: float):
        for e in self.doc.events:
            if e.start_s <= t + 1e-9 < e.end_s:
                return e.index
        return None

    def frame(self, n: int, force_event="auto") -> Image.Image:
        tl, t = self.tl, n / FPS
        k = scene_at(tl, t)
        img = self.bases[k]
        for i, (b, d) in enumerate(tl.dissolves):
            if d > 0 and b - d / 2 <= t < b + d / 2:
                img = Image.blend(self.bases[i], self.bases[i + 1], (t - (b - d / 2)) / d)
        if t < tl.fade_in_end:
            img = Image.blend(self.black, img, t / tl.fade_in_end)
        if t >= tl.fade_out_start:
            img = Image.blend(img, self.black, min(1.0, (t - tl.fade_out_start) / (tl.duration_s - tl.fade_out_start)))
        ev = self.event_at(t) if force_event == "auto" else force_event
        if ev is not None:
            img = Image.alpha_composite(img.convert("RGBA"), self.overlays[ev]).convert("RGB")
        return img


# ================================================================ inputs


def load_inputs(job_dir: Path) -> dict:
    if not job_dir.is_dir():
        raise RenderError("JOB_NOT_FOUND", f"{job_dir}")
    wav = job_dir / "speech.wav"
    if not wav.is_file():
        raise RenderError("AUDIO_MISSING", f"{wav} not found")
    try:
        info = sf.info(str(wav))
    except Exception as e:  # noqa: BLE001
        raise RenderError("AUDIO_INVALID", f"{wav}: {e}")
    duration = info.frames / info.samplerate
    meta_p, plan_p = job_dir / "metadata.json", job_dir / "scene_plan" / "scene_plan.json"
    if not plan_p.is_file():
        raise RenderError("PLAN_MISSING", f"{plan_p} not found")
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        plan = json.loads(plan_p.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise RenderError("METADATA_MISSING", str(e))
    except json.JSONDecodeError as e:
        raise RenderError("INVALID_JSON", str(e))
    if abs(duration - meta["output"]["duration_s"]) > 1e-3 or abs(duration - plan.get("audio_duration_s", -1)) > 1e-3:
        raise RenderError("DURATION_MISMATCH", f"speech.wav {duration:.3f}s vs metadata "
                                               f"{meta['output']['duration_s']} vs plan {plan.get('audio_duration_s')}")
    check_timeline_structure(plan, duration)

    # stale-plan protection: re-validate in memory with the current planner; timeline must match exactly
    from asset_manifest import ManifestError, check_plan, resolve

    brief = resolve(plan.get("inputs", {}).get("brief"), common.PROJECT_ROOT)
    if brief is None or not brief.is_file():
        raise RenderError("BRIEF_MISSING", f"brief recorded in plan not found: {brief}")
    try:
        verdict = check_plan(job_dir, plan, brief)
    except ManifestError as e:
        raise RenderError(e.code, str(e))
    if verdict["status"] == "failed":
        raise RenderError("PLAN_INVALID", "scene plan validation status is 'failed'")

    man_p = job_dir / "assets" / "asset_manifest.json"
    if not man_p.is_file():
        raise RenderError("MANIFEST_MISSING", f"{man_p} not found (run asset_manifest.py)")
    man = json.loads(man_p.read_text(encoding="utf-8"))
    if not man.get("validation", {}).get("ok"):
        raise RenderError("MANIFEST_INVALID", "asset manifest validation is not ok")
    if [(e["scene_id"], e["timeline"]["start_s"], e["timeline"]["end_s"]) for e in man["scenes"]] != \
            [(s["scene_id"], s["start_s"], s["end_s"]) for s in plan["scenes"]]:
        raise RenderError("MANIFEST_STALE", "asset manifest scenes/timeline differ from the scene plan")
    provided = [e["scene_id"] for e in man["scenes"] if e["status"] != "unresolved"]
    if provided:
        raise RenderError("UNSUPPORTED_IN_4B1", f"scenes {provided} have real assets; Phase 4B-1 renders placeholders only")

    sub = job_dir / "subtitles"
    paths = {k: sub / k for k in ("captions_highlight.ass", "captions.srt", "subtitle_report.json")}
    for k, p in paths.items():
        if not p.is_file():
            raise RenderError("SUBTITLES_MISSING", f"{p} not found (run subtitles.py)")
    sub_report = json.loads(paths["subtitle_report.json"].read_text(encoding="utf-8"))
    if not sub_report.get("validation", {}).get("ok"):
        raise RenderError("SUBTITLES_INVALID", "subtitle_report validation is not ok")
    ass_text = paths["captions_highlight.ass"].read_text(encoding="utf-8")
    doc = parse_ass(ass_text)

    # staleness check: the ASS events must be exactly what subtitles.py produces from the current words.json
    import subtitles as subs

    words = json.loads((job_dir / "alignment" / "words.json").read_text(encoding="utf-8"))["words"]
    expected_ass, _ = subs.render_ass(subs.build_cues(words), tuple(sub_report["play_resolution"]), subs.PLACEHOLDER_STYLE)
    got = [l for l in ass_text.splitlines() if l.startswith("Dialogue:")]
    want = [l for l in expected_ass.splitlines() if l.startswith("Dialogue:")]
    if got != want:
        raise RenderError("SUBTITLES_STALE", "captions_highlight.ass does not match the current alignment/words.json")
    if doc.events[-1].end_s > duration + 0.01:
        raise RenderError("SUBTITLES_INVALID", "an ASS event ends after the audio")

    dissolves, fade_in, fade_out = parse_transitions(plan, duration)
    tl = Timeline(duration, int(round(duration * FPS)), plan["scenes"], dissolves, fade_in, fade_out)
    return {"job_dir": job_dir, "wav": wav, "wav_info": info, "duration": duration, "plan": plan, "verdict": verdict,
            "manifest": man, "doc": doc, "sub_report": sub_report, "timeline": tl, "words": words}


# ================================================================ ffmpeg helpers


def run(cmd: list, what: str) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RenderError("RENDER_FAILED", f"{what} failed (exit {r.returncode}): {r.stderr.strip()[-400:]}")
    return r


def encode(ctx: dict, frames_dir: Path, out: Path, ffmpeg: str):
    run([ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
         "-framerate", str(FPS), "-i", str(frames_dir / "%05d.png"), "-i", str(ctx["wav"]),
         "-map", "0:v:0", "-map", "1:a:0",
         "-vf", "scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p",
         "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-r", str(FPS),
         "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
         "-c:a", "aac", "-b:a", "128k",
         "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact",
         "-movflags", "+faststart", str(out)], "ffmpeg encode")


def probe(path: Path, ffprobe: str) -> dict:
    r = run([ffprobe, "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(path)],
            "ffprobe")
    return json.loads(r.stdout)


def decode_frames(path: Path, indices: list, ffmpeg: str) -> dict:
    idx = sorted(set(indices))
    sel = "+".join(f"eq(n\\,{i})" for i in idx)
    r = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path), "-vf",
                        f"select='{sel}',scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb24",
                        "-fps_mode", "passthrough", "-f", "rawvideo", "-"], capture_output=True)
    if r.returncode != 0:
        raise RenderError("VALIDATION_FAILED", f"frame decode failed: {r.stderr.decode()[-300:]}")
    data = np.frombuffer(r.stdout, np.uint8)
    if data.size != len(idx) * W * H * 3:
        raise RenderError("VALIDATION_FAILED", f"decoded {data.size} bytes for {len(idx)} frames")
    return {i: data[k * W * H * 3:(k + 1) * W * H * 3].reshape(H, W, 3) for k, i in enumerate(idx)}


def decode_audio(path: Path, sr: int, ffmpeg: str) -> np.ndarray:
    r = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path), "-map", "0:a:0",
                        "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"], capture_output=True)
    if r.returncode != 0:
        raise RenderError("VALIDATION_FAILED", f"audio decode failed: {r.stderr.decode()[-300:]}")
    return np.frombuffer(r.stdout, np.float32)


def blackdetect(path: Path, ffmpeg: str) -> list:
    r = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path), "-vf", "blackdetect=d=0.04:pix_th=0.10", "-an",
                        "-f", "null", "-"], capture_output=True, text=True)
    return [(float(a), float(b)) for a, b in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", r.stderr)]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse == 0 else 10 * math.log10(255 ** 2 / mse)


# ================================================================ validation


def check_durations(video_s: float, audio_out_s: float, source_s: float) -> dict:
    return {"video_vs_source_s": round(video_s - source_s, 6), "audio_vs_source_s": round(audio_out_s - source_s, 6),
            "within_one_frame": abs(video_s - source_s) <= FRAME_S + 1e-9 and abs(audio_out_s - source_s) <= FRAME_S + 1e-9}


def audio_compare(src: np.ndarray, out: np.ndarray, sr: int) -> dict:
    max_lag = int(0.1 * sr)
    n = min(len(src), len(out))
    a, b = src[:n].astype(np.float64), out[:n].astype(np.float64)
    size = 1 << int(np.ceil(np.log2(2 * n)))
    xc = np.fft.irfft(np.fft.rfft(b, size) * np.conj(np.fft.rfft(a, size)), size)
    lags = np.concatenate([xc[:max_lag + 1], xc[-max_lag:]])
    lag_vals = np.concatenate([np.arange(0, max_lag + 1), np.arange(-max_lag, 0)])
    lag = int(lag_vals[int(np.argmax(lags))])
    if lag >= 0:
        sa, sb = a[:n - lag], b[lag:n]
    else:
        sa, sb = a[-lag:n], b[:n + lag]
    err = sb - sa
    corr = float(np.corrcoef(sa, sb)[0, 1])
    snr = 10 * math.log10(float(np.sum(sa ** 2)) / max(float(np.sum(err ** 2)), 1e-20))
    thr = 10 ** (-40 / 20)
    first = lambda x: int(np.argmax(np.abs(x) > thr)) if np.any(np.abs(x) > thr) else None  # noqa: E731
    return {"source_samples": int(len(src)), "decoded_samples": int(len(out)), "sample_rate": sr,
            "source_duration_s": round(len(src) / sr, 6), "decoded_duration_s": round(len(out) / sr, 6),
            "best_lag_samples": lag, "best_lag_ms": round(1000 * lag / sr, 3), "correlation": round(corr, 6),
            "snr_db": round(snr, 2), "max_abs_error": round(float(np.max(np.abs(err))), 6),
            "rms_error_dbfs": round(20 * math.log10(max(float(np.sqrt(np.mean(err ** 2))), 1e-12)), 2),
            "first_sample_above_-40dBFS": {"source": first(a), "decoded": first(b)},
            "bit_identical": bool(len(src) == len(out) and np.array_equal(src, out))}


def validate(ctx: dict, mp4: Path, fr: FrameRenderer, ffmpeg: str, ffprobe: str) -> dict:
    tl, dur = ctx["timeline"], ctx["duration"]
    checks, details = {}, {}
    checks["exists_nonempty"] = mp4.is_file() and mp4.stat().st_size > 0
    pr = probe(mp4, ffprobe)
    vs = [s for s in pr["streams"] if s["codec_type"] == "video"]
    as_ = [s for s in pr["streams"] if s["codec_type"] == "audio"]
    checks["one_video_one_audio_stream"] = len(vs) == 1 and len(as_) == 1 and len(pr["streams"]) == 2
    v, a = vs[0], as_[0]
    checks["resolution_1280x720"] = (v["width"], v["height"]) == (W, H)
    checks["fps_25_constant"] = v["r_frame_rate"] == f"{FPS}/1" and v["avg_frame_rate"] == f"{FPS}/1"
    checks["video_h264_yuv420p"] = v["codec_name"] == "h264" and v["pix_fmt"] == "yuv420p"
    checks["audio_aac_24k_mono"] = a["codec_name"] == "aac" and int(a["sample_rate"]) == 24000 and a["channels"] == 1
    n_frames = int(v.get("nb_read_frames") or v.get("nb_frames"))
    checks["frame_count"] = n_frames == tl.n_frames
    vdur, adur = float(v["duration"]), float(a["duration"])
    details["durations"] = {"source_wav_s": round(dur, 6), "video_stream_s": vdur, "audio_stream_s": adur,
                            "container_s": float(pr["format"]["duration"]), "frames": n_frames,
                            **check_durations(vdur, adur, dur)}
    checks["duration_within_one_frame"] = details["durations"]["within_one_frame"]

    # frames: scene midpoints, around cuts/dissolves, every subtitle event (first + middle frame), gaps, fades
    sample = {0, 3, 6, tl.n_frames - 1, tl.n_frames - 4}
    for s in tl.scenes:
        sample.add(int(((s["narration_start_s"] + s["narration_end_s"]) / 2) * FPS))
    for b, d in tl.dissolves:
        c = int(round(b * FPS))
        sample.update(range(max(0, c - 6), min(tl.n_frames, c + 7)))
    first_frames = []
    for e in fr.doc.events:
        f0 = math.ceil(e.start_s * FPS - 1e-9)
        f1 = math.ceil(e.end_s * FPS - 1e-9) - 1
        if f1 >= f0:
            first_frames.append((e.index, f0))
            sample.update({f0, (f0 + f1) // 2, f1})
    dec = decode_frames(mp4, sorted(i for i in sample if 0 <= i < tl.n_frames), ffmpeg)
    worst, per = 99.0, {}
    for n, got in dec.items():
        p = psnr(got, np.asarray(fr.frame(n)))
        per[n] = round(p, 2)
        worst = min(worst, p)
    details["frame_psnr_db"] = {"frames_checked": len(dec), "min": round(worst, 2), "threshold": PSNR_MIN_DB}
    checks["decoded_frames_match_expected"] = worst >= PSNR_MIN_DB

    # scene identity + order via plain-background probe region (frames outside dissolves/fades)
    x0, y0, x1, y1 = PROBE_BOX
    order, misclassified = [], []
    for n in sorted(dec):
        t = n / FPS
        if t < tl.fade_in_end or t >= tl.fade_out_start or any(d > 0 and b - d / 2 <= t < b + d / 2 for b, d in tl.dissolves):
            continue
        lum = float(dec[n][y0:y1, x0:x1].mean())
        want = scene_at(tl, t)
        got_k = int(np.argmin([abs(lum - np.mean(SCENE_BG[k % len(SCENE_BG)])) for k in range(len(tl.scenes))]))
        if got_k != want:
            misclassified.append(n)
        if not order or order[-1] != got_k:
            order.append(got_k)
    details["scene_order_observed"] = [tl.scenes[k]["scene_id"] for k in order]
    checks["scenes_in_order"] = order == list(range(len(tl.scenes))) and not misclassified
    cut = [int(round(b * FPS)) for b, _ in tl.dissolves]
    details["cut_frames"] = cut

    # subtitles: present (text pixels) during events, absent outside; each event's first frame shows that event
    bx0, by0, bx1, by1 = SUB_BOX
    vis_ok, timing_ok, timing = True, True, []
    for n, img in dec.items():
        ev = fr.event_at(n / FPS)
        region = img[by0:by1, bx0:bx1].astype(int)
        white = int(np.sum(np.all(region > 225, axis=2)))
        yellow = int(np.sum((region[..., 0] > 200) & (region[..., 1] > 200) & (region[..., 2] < 80)))
        if ev is None and white + yellow > 50:
            vis_ok = False
        if ev is not None:
            hl = any(c[:3] == (255, 255, 0) for ln in fr.doc.events[ev].lines for _, c in ln)
            if white < 200 or (hl and yellow < 50):
                vis_ok = False
    for ei, f0 in first_frames:
        if f0 not in dec:
            continue
        own = psnr(dec[f0], np.asarray(fr.frame(f0, force_event=ei)))
        alt_ev = ei - 1 if ei > 0 and fr.doc.events[ei - 1].end_s >= fr.doc.events[ei].start_s - 1e-9 else None
        alt = psnr(dec[f0], np.asarray(fr.frame(f0, force_event=alt_ev)))
        timing.append({"event": ei, "first_frame": f0, "t_s": round(f0 / FPS, 3),
                       "event_start_s": fr.doc.events[ei].start_s, "psnr_own": round(own, 2), "psnr_previous": round(alt, 2)})
        if not own > alt:
            timing_ok = False
    details["subtitle_event_onsets"] = timing
    checks["subtitles_visible_only_during_events"] = vis_ok
    checks["subtitle_timing_matches_ass"] = timing_ok and len(timing) == len(fr.doc.events)

    # black frames only inside intentional fades
    black = blackdetect(mp4, ffmpeg)
    details["black_intervals"] = black
    checks["no_unexpected_black"] = all(s >= -1e-6 and (e <= tl.fade_in_end + FRAME_S or s >= tl.fade_out_start - FRAME_S)
                                        for s, e in black)

    # audio
    src, sr = sf.read(str(ctx["wav"]), dtype="float32")
    out = decode_audio(mp4, sr, ffmpeg)
    ac = audio_compare(src, out, sr)
    ac["stream_start_time_s"] = float(a.get("start_time", 0))
    ac["codec_note"] = ("AAC-LC at 128 kb/s is lossy: decoded samples differ from speech.wav. speech.wav itself was "
                        "not modified; it is the only audio source and was not resampled, filtered or normalised.")
    details["audio"] = ac
    checks["audio_aligned_at_start"] = abs(ac["best_lag_samples"]) <= 1 and abs(ac["stream_start_time_s"]) < 1e-3
    checks["audio_correlation_ge_0.99"] = ac["correlation"] >= 0.99
    ac["sample_count_difference"] = ac["decoded_samples"] - ac["source_samples"]
    checks["decoded_audio_duration_within_one_frame"] = abs(ac["sample_count_difference"]) <= round(sr * FRAME_S)
    return {"ok": all(checks.values()), "checks": checks, "details": details, "frame_psnr_by_index": per}


def review_sheet(mp4: Path, idx: list, ffmpeg: str, out: Path):
    dec = decode_frames(mp4, idx, ffmpeg)
    tw, th = W // 4, H // 4
    cols = 4
    rows = math.ceil(len(idx) / cols)
    sheet = Image.new("RGB", (cols * tw, rows * (th + 22)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for k, n in enumerate(sorted(dec)):
        x, y = (k % cols) * tw, (k // cols) * (th + 22)
        sheet.paste(Image.fromarray(dec[n]).resize((tw, th)), (x, y + 22))
        d.text((x + 4, y + 3), f"frame {n}  t={n / FPS:.2f}s", font=font(14), fill=(0, 0, 0))
    sheet.save(out)


# ================================================================ main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True, help="job id under --outputs-dir, or a job folder path")
    ap.add_argument("--outputs-dir", type=Path, default=common.OUTPUTS_DIR)
    ap.add_argument("--output", default=DEFAULT_OUT, help="path relative to the job folder")
    ap.add_argument("--ffmpeg", default="ffmpeg", help=argparse.SUPPRESS)     # test hook
    ap.add_argument("--ffprobe", default="ffprobe", help=argparse.SUPPRESS)   # test hook
    args = ap.parse_args(argv)
    job_dir = Path(args.job) if Path(args.job).is_dir() else args.outputs_dir / args.job
    out = job_dir / args.output
    report_p = out.with_name(out.stem + "_report.json")
    sheet_p = out.with_name(out.stem + "_review.png")
    tmp = out.with_name("." + out.stem + ".tmp.mp4")
    for p in (out, report_p, sheet_p):
        if p.exists():
            sys.exit(f"[ERROR] OUTPUT_EXISTS: {p} already exists; refusing to overwrite")
    t0 = time.perf_counter()
    try:
        ctx = load_inputs(job_dir)
        tl = ctx["timeline"]
        fr = FrameRenderer(tl, ctx["doc"])
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="phase4b1_frames_") as fd:
            for n in range(tl.n_frames):
                fr.frame(n).save(Path(fd) / f"{n:05d}.png", compress_level=1)
            encode(ctx, Path(fd), tmp, args.ffmpeg)
        val = validate(ctx, tmp, fr, args.ffmpeg, args.ffprobe)
        if not val["ok"]:
            raise RenderError("VALIDATION_FAILED", f"failed checks: {[k for k, v in val['checks'].items() if not v]}")
        os.replace(tmp, out)
    except RenderError as e:
        tmp.unlink(missing_ok=True)
        sys.exit(f"[ERROR] {e.code}: {e}")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    ver = subprocess.run([args.ffmpeg, "-hide_banner", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    key = sorted({0, 4, 50, 95, 99, 100, 101, 104, 120, 150, tl.n_frames - 1} & set(range(tl.n_frames)))
    review_sheet(out, key, args.ffmpeg, sheet_p)
    report = {
        "job_id": ctx["plan"].get("job_id"), "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "renderer": "assemble_video.py (Phase 4B-1, draft, placeholders only)", "mode": "draft",
        "output": str(out), "review_sheet": str(sheet_p), "ffmpeg": ver,
        "inputs": {"audio": str(ctx["wav"]), "scene_plan_validation": ctx["verdict"]["status"],
                   "scene_plan_source": ctx["verdict"]["source"], "asset_manifest": "assets/asset_manifest.json",
                   "subtitles": "subtitles/captions_highlight.ass (parsed; rasterised with Pillow, no libass)"},
        "format": {"width": W, "height": H, "fps": FPS, "video_codec": "h264 (libx264, crf 16, yuv420p, bt709)",
                   "audio_codec": "aac (128 kb/s, 24 kHz mono)"},
        "timeline": {"frames": tl.n_frames, "cut_frames": [int(round(b * FPS)) for b, _ in tl.dissolves],
                     "dissolves": [{"boundary_s": b, "duration_s": d} for b, d in tl.dissolves],
                     "fade_in_s": [0.0, tl.fade_in_end], "fade_out_s": [tl.fade_out_start, tl.duration_s]},
        "subtitle_events": [{"index": e.index, "start_s": e.start_s, "end_s": e.end_s,
                             "text": " / ".join("".join(t for t, _ in ln) for ln in e.lines),
                             "highlight": ["".join(t for t, c in ln if c[:3] == (255, 255, 0)) for ln in e.lines]}
                            for e in ctx["doc"].events],
        "subtitle_rendering_note": ("FFmpeg here lacks libass; events were rasterised by Pillow with Helvetica Neue at "
                                    "the ASS style's size/outline/shadow scaled to 720p. Timing is exact; glyph "
                                    "rendering approximates libass."),
        "validation": val, "render_seconds": round(time.perf_counter() - t0, 1),
        "not_used": "no AI models, no cloud APIs, no stock/brand assets, no lip-sync, no audio processing",
    }
    report_p.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    d = val["details"]
    print(f"OK  {out}")
    print(f"  {W}x{H} @ {FPS} fps, h264 + aac; frames {d['durations']['frames']}; video {d['durations']['video_stream_s']}s, "
          f"audio {d['durations']['audio_stream_s']}s, source {d['durations']['source_wav_s']}s")
    print(f"  scenes observed: {d['scene_order_observed']}; cut frames {report['timeline']['cut_frames']}; "
          f"min frame PSNR {d['frame_psnr_db']['min']} dB over {d['frame_psnr_db']['frames_checked']} frames")
    a = d["audio"]
    print(f"  audio: lag {a['best_lag_samples']} samples, corr {a['correlation']}, SNR {a['snr_db']} dB, "
          f"samples {a['decoded_samples']}/{a['source_samples']}, bit-identical {a['bit_identical']}")
    print(f"  report {report_p}\n  review {sheet_p}")


if __name__ == "__main__":
    main()
