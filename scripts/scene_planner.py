"""Phase 3: timestamp-aware scene planning from a voicegen job + its Phase 2 word alignment.

    .venv/bin/python scripts/scene_planner.py --job <JOB_ID> --brief texts/briefs/sample_brief.json

Reads (never modifies): <job>/metadata.json, <job>/speech.wav (header only),
<job>/alignment/words.json, <job>/alignment/alignment_report.json, the brief and brand profile.
Writes: <job>/scene_plan/scene_plan.json and scene_plan.md.

Deterministic and local: no models, no network. Every time comes from Phase 2 word
timestamps; nothing is estimated from text length.

Pipeline:
  quality gate (alignment report + re-validated words)
  -> narration units = Phase 2 sentences
  -> split units longer than max_scene_s at the longest internal pause (between words only)
  -> merge units shorter than min_scene_s into a neighbour within the same paragraph gap rules
  -> timeline: each scene spans from the midpoint of the pause before its first word to the
     midpoint of the pause after its last word (first scene from 0, last to audio end)
  -> visual direction (brief overrides > brand card > greeting/presenter > brief keywords > template)
  -> assets + brand constraints (missing brand data => REQUIRES_USER_INPUT, never invented)
  -> validation -> JSON + Markdown (Markdown re-parsed and checked against JSON)
"""

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

REQUIRES_INPUT = "REQUIRES_USER_INPUT"
MODES = ("talking_head", "voiceover_broll", "product_visual", "text_graphic", "environmental_visual")
BRAND_FIELDS = ("brand_name", "logo_asset_path", "brand_colors", "typography", "tone", "visual_restrictions",
                "required_disclaimers", "prohibited_visual_elements")
GREETING = re.compile(r"^(hello|hi|welcome|namaste|good (morning|afternoon|evening))\b", re.I)
SHOT = {  # mode -> (shot type, camera movements to rotate through)
    "talking_head": ("medium close-up, eye level", ["static", "slow push-in"]),
    "voiceover_broll": ("medium / wide b-roll", ["slow dolly right", "slow dolly left", "gentle handheld drift"]),
    "product_visual": ("close-up product detail", ["slow orbit", "slow push-in"]),
    "text_graphic": ("full-frame graphic", ["static"]),
    "environmental_visual": ("wide establishing", ["slow pan", "slow aerial glide"]),
}
TIME_TOL_S = 0.001          # 1 ms tolerance for float rounding in Phase 2 output
MIN_SAFE_GAP_S = 0.04       # 2 Whisper frames: below this, a "gap" is not a reliable silence
DEFAULT_PLANNING = {"min_scene_s": 1.5, "max_scene_s": 8.0, "dissolve_min_pause_s": 0.6,
                    "min_coverage_percent": 95.0, "max_uncertain_fraction": 0.2}


class PlanError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def r3(x):
    return round(float(x), 3)


def ms(x):
    return int(round(float(x) * 1000))


# ---------------------------------------------------------------- inputs + quality gate


def load_inputs(job_dir: Path, brief_path: Path, cfg_override: dict | None = None):
    for p, code in ((job_dir / "metadata.json", "JOB_NOT_FOUND"),
                    (job_dir / "alignment" / "words.json", "ALIGNMENT_MISSING"),
                    (job_dir / "alignment" / "alignment_report.json", "ALIGNMENT_MISSING")):
        if not p.is_file():
            raise PlanError(code, f"{p} not found (run Phase 1/2 first)")
    if not brief_path.is_file():
        raise PlanError("BRIEF_MISSING", f"Brief not found: {brief_path}")
    try:
        meta = json.loads((job_dir / "metadata.json").read_text(encoding="utf-8"))
        words_doc = json.loads((job_dir / "alignment" / "words.json").read_text(encoding="utf-8"))
        report = json.loads((job_dir / "alignment" / "alignment_report.json").read_text(encoding="utf-8"))
        brief = json.loads(brief_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise PlanError("INVALID_JSON", f"Could not parse input JSON: {e}")

    brand = brief.get("brand_profile") or {}
    brand_src = "inline"
    if isinstance(brand, str):
        bp = (brief_path.parent / brand) if not Path(brand).is_absolute() else Path(brand)
        if not bp.is_file():
            raise PlanError("BRAND_PROFILE_MISSING", f"Brand profile not found: {bp}")
        brand, brand_src = json.loads(bp.read_text(encoding="utf-8")), str(bp)
    cfg = {**DEFAULT_PLANNING, **(brief.get("planning") or {}), **(cfg_override or {})}

    wav = job_dir / "speech.wav"
    if not wav.is_file():
        raise PlanError("AUDIO_MISSING", f"{wav} not found")
    import soundfile as sf

    duration = sf.info(str(wav)).duration
    return meta, words_doc, report, brief, brand, brand_src, cfg, duration


def _is_time(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def gate(words: list, report: dict, duration: float, cfg: dict) -> dict:
    """Re-validate Phase 2 output ourselves; stop rather than plan on bad timings.
    Gaps (pauses) between words are fine; overlaps, bad values and untimed 'aligned' words are not."""
    problems = []
    if not report.get("validation_ok"):
        problems.append("Phase 2 alignment_report.validation_ok is false")
    if [w.get("index") for w in words] != list(range(len(words))):
        problems.append("word indices are not contiguous 0..N-1")
    timed = []
    for w in words:
        i, s, e, st = w.get("index"), w.get("start_s"), w.get("end_s"), w.get("status")
        tag = f"word {i} {w.get('word')!r}"
        if s is None and e is None:
            if st != "unaligned":
                problems.append(f"{tag}: missing timestamps but status is {st!r} (only 'unaligned' words may lack times)")
            continue
        if s is None or e is None:
            problems.append(f"{tag}: incomplete timestamps (start={s}, end={e})")
            continue
        if not (_is_time(s) and _is_time(e)):
            problems.append(f"{tag}: non-numeric or non-finite timestamps (start={s!r}, end={e!r})")
            continue
        if st == "unaligned":
            problems.append(f"{tag}: status 'unaligned' but carries times")
        if s < 0:
            problems.append(f"{tag}: negative start {s}")
        if not s < e:
            problems.append(f"{tag}: start {s} is not before end {e}")
        if e > duration + TIME_TOL_S:
            problems.append(f"{tag}: end {e} exceeds audio duration {duration:.3f}")
        timed.append(w)
    for a, b in zip(timed, timed[1:]):
        if b["start_s"] < a["start_s"] - TIME_TOL_S:
            problems.append(f"words {a['index']}->{b['index']}: start {b['start_s']} precedes previous start {a['start_s']}")
        elif a["end_s"] > b["start_s"] + TIME_TOL_S:
            problems.append(f"words {a['index']}->{b['index']}: overlap of {a['end_s'] - b['start_s']:.3f} s "
                            f"({a['word']!r} ends {a['end_s']} after {b['word']!r} starts {b['start_s']})")
    cov = 100.0 * len(timed) / len(words) if words else 0.0
    unc = [w for w in words if w.get("status") != "aligned"]
    if cov < cfg["min_coverage_percent"]:
        problems.append(f"timestamp coverage {cov:.1f}% < {cfg['min_coverage_percent']}%")
    if words and len(unc) / len(words) > cfg["max_uncertain_fraction"]:
        problems.append(f"{len(unc)}/{len(words)} words uncertain/unaligned (> {cfg['max_uncertain_fraction']:.0%})")
    if words and timed and (timed[0]["index"] != 0 or timed[-1]["index"] != len(words) - 1):
        problems.append("first or last word has no timestamp, so scene edges cannot be anchored")
    if problems:
        raise PlanError("ALIGNMENT_INSUFFICIENT", "Alignment quality insufficient for planning:\n  - " +
                        "\n  - ".join(problems[:20]))
    return {"coverage_percent": round(cov, 1), "uncertain_or_unaligned": [
        {"index": w["index"], "word": w["word"], "status": w["status"], "flags": w.get("flags", [])} for w in unc]}


# ---------------------------------------------------------------- segmentation


def span(ws):
    timed = [w for w in ws if w.get("start_s") is not None]
    return timed[0]["start_s"], timed[-1]["end_s"]


def split_long(unit: list, max_s: float) -> list[list]:
    """Recursively split a unit longer than max_s at a word boundary: prefer splits leaving >= 2 words
    per side, then the longest real pause, then clause punctuation, then the middle. Units that still
    exceed max_s (e.g. one very long word) are kept whole and reported by plan validation."""
    s, e = span(unit)
    if e - s <= max_s or len(unit) < 2:
        return [unit]
    best, best_key = None, None
    for k in range(1, len(unit)):
        a, b = unit[k - 1], unit[k]
        if a.get("end_s") is None or b.get("start_s") is None:
            continue                         # never cut next to a word whose extent is unknown
        gap = b["start_s"] - a["end_s"]
        both_sides = 1 if min(k, len(unit) - k) >= 2 else 0
        clause = 1 if re.search(r"[,;:]$", a["word"]) else 0
        mid = -abs(k - len(unit) / 2)
        key = (both_sides, round(gap, 2), clause, mid)
        if best_key is None or key > best_key:
            best, best_key = k, key
    if best is None:
        return [unit]
    return split_long(unit[:best], max_s) + split_long(unit[best:], max_s)


def segment(words: list, cfg: dict) -> list[list]:
    units = []
    for si in sorted({w["sentence_index"] for w in words}):
        sent = [w for w in words if w["sentence_index"] == si]
        if not any(w.get("start_s") is not None for w in sent):
            # a sentence with no timed word cannot anchor a scene: attach it to its neighbour
            (units[-1].extend(sent) if units else units.append(sent))
            continue
        if units and not any(w.get("start_s") is not None for w in units[-1]):
            sent = units.pop() + sent
        units += split_long(sent, cfg["max_scene_s"])
    # merge too-short units into the neighbour with the smaller pause, if the result stays <= max
    changed = True
    while changed and len(units) > 1:
        changed = False
        for i, u in enumerate(units):
            s, e = span(u)
            if e - s >= cfg["min_scene_s"]:
                continue
            cands = []
            if i > 0:
                ps, pe = span(units[i - 1])
                cands.append((s - pe, i - 1, i, e - ps))
            if i < len(units) - 1:
                ns, ne = span(units[i + 1])
                cands.append((ns - e, i, i + 1, ne - s))
            cands = [c for c in cands if c[3] <= cfg["max_scene_s"]]
            if cands:
                _, a, b, _ = min(cands)
                units[a:b + 1] = [units[a] + units[b]]
                changed = True
                break
    return units


# ---------------------------------------------------------------- visual direction


def brand_words(brand: dict) -> set:
    return {t.lower() for t in re.findall(r"[A-Za-z]+", brand.get("brand_name") or "")}


def direct(scenes: list, brief: dict, brand: dict, duration: float):
    bw = brand_words(brand)
    presenter = brief.get("presenter") or {}
    on_cam_budget = float(presenter.get("max_on_camera_ratio", 0.0) or 0.0) * duration if presenter.get("available") else 0.0
    overrides = brief.get("scene_overrides") or []
    rotation = {m: 0 for m in MODES}
    for sc in scenes:
        text = sc["narration_text"]
        toks = [t.lower() for t in re.findall(r"[A-Za-z']+", text)]
        mode, desc, source, notes = None, None, None, []
        for ov in overrides:   # explicit user direction wins
            if ov.get("scene_id") == sc["scene_id"] or (ov.get("match") and ov["match"].lower() in text.lower()):
                mode, desc, source = ov.get("visual_mode"), ov.get("visual_description"), "brief.scene_overrides"
                break
        if mode is None and bw and toks and sum(t in bw for t in toks) / len(toks) >= 0.5 and sc["duration_s"] <= 5:
            mode, source = "text_graphic", "rule: brand-name card"
            desc = f"Brand card: {brand.get('brand_name')} name and logo on a clean background"
        if mode is None and GREETING.match(text.strip()) and presenter.get("available"):
            mode, source = "talking_head", "rule: greeting with available presenter"
            desc = f"Presenter ({presenter.get('description', 'narrator')}) speaking directly to camera"
        if mode is None:
            for kv in brief.get("keyword_visuals") or []:
                if any(k.lower() in toks for k in kv.get("keywords", [])):
                    mode, desc, source = kv["visual_mode"], kv["visual_description"], "brief.keyword_visuals"
                    break
        if mode is None:
            mode, source = "voiceover_broll", "template (no brief match; review)"
            desc = f"B-roll illustrating the narration: \"{text}\""
            notes.append("visual_description is a template; supply a scene override for real direction")
        if mode not in MODES:
            raise PlanError("INVALID_BRIEF", f"Unsupported visual_mode {mode!r} (allowed: {MODES})")
        if mode == "talking_head":
            if sc["duration_s"] <= on_cam_budget + 1e-6:
                on_cam_budget -= sc["duration_s"]
            else:
                notes.append("talking_head demoted: presenter on-camera budget exhausted")
                mode, desc, source = "voiceover_broll", f"B-roll illustrating the narration: \"{text}\"", "template"
        shot, moves = SHOT[mode]
        sc.update({"visual_mode": mode, "visual_description": desc, "visual_description_source": source,
                   "shot_type": shot, "camera_movement": moves[rotation[mode] % len(moves)],
                   "speaker_required": mode == "talking_head", "direction_notes": notes})
        rotation[mode] += 1


def transitions(scenes: list, cfg: dict):
    for i, sc in enumerate(scenes):
        if i == 0:
            sc["transition_in"] = "fade in from black"
        if i == len(scenes) - 1:
            sc["transition"] = "fade out to black (after narration ends)"
            continue
        pause = scenes[i + 1]["narration_start_s"] - sc["narration_end_s"]
        if not sc.get("boundary_after", {}).get("safe", True):
            sc["transition"] = "hard cut on a word edge (narration continuous across the cut; review)"
        elif pause >= cfg["dissolve_min_pause_s"]:
            d = min(0.4, pause / 2)
            sc["transition"] = f"cross-dissolve {d:.2f} s centred in the {pause:.2f} s pause"
        else:
            sc["transition"] = f"hard cut in the {pause:.2f} s pause"


def field(brand, key):
    v = brand.get(key)
    return v if v not in (None, "", [], {}) else REQUIRES_INPUT


def assets_and_brand(sc: dict, brand: dict, brief: dict, project_root: Path, is_last: bool):
    presenter = brief.get("presenter") or {}
    req = []
    if sc["visual_mode"] == "talking_head":
        consent = presenter.get("consent_confirmed")
        req.append({"type": "presenter_video", "description": f"{presenter.get('description', 'narrator')} framed "
                    f"{sc['shot_type']}, >= {sc['duration_s']:.2f} s, neutral mouth for lip-sync",
                    "status": "to_generate" if consent else REQUIRES_INPUT,
                    "note": None if consent else "confirm consent/likeness rights for the presenter"})
        req.append({"type": "lip_sync_audio", "description": f"speech.wav {sc['start_s']:.3f}-{sc['end_s']:.3f} s",
                    "status": "available"})
    elif sc["visual_mode"] == "text_graphic":
        logo = brand.get("logo_asset_path")
        if logo:
            lp = Path(logo) if Path(logo).is_absolute() else project_root / logo
            req.append({"type": "logo", "path": str(lp), "status": "available" if lp.is_file() else "missing"})
        else:
            req.append({"type": "logo", "path": None, "status": REQUIRES_INPUT})
        req.append({"type": "typography", "value": field(brand, "typography"),
                    "status": "available" if field(brand, "typography") != REQUIRES_INPUT else REQUIRES_INPUT})
    else:
        req.append({"type": "video_clip", "description": f"{sc['visual_mode']} clip, {sc['shot_type']}, "
                    f"{sc['camera_movement']}, >= {sc['duration_s']:.2f} s, no on-screen text or logos",
                    "status": "to_generate"})
    req.append({"type": "brand_colors", "value": field(brand, "brand_colors"),
                "status": "available" if field(brand, "brand_colors") != REQUIRES_INPUT else REQUIRES_INPUT})
    disc = field(brand, "required_disclaimers")
    sc["asset_requirements"] = req
    sc["brand_constraints"] = {
        "brand_name": field(brand, "brand_name"), "colors": field(brand, "brand_colors"),
        "typography": field(brand, "typography"), "tone": field(brand, "tone"),
        "visual_restrictions": field(brand, "visual_restrictions"),
        "prohibited_visual_elements": field(brand, "prohibited_visual_elements"),
        "required_disclaimers": disc if is_last else ("see final scene" if disc != REQUIRES_INPUT else REQUIRES_INPUT),
    }


def prompt(sc: dict, brief: dict) -> str:
    bc = sc["brand_constraints"]
    parts = [sc["visual_description"] + ".", f"Shot: {sc['shot_type']}; camera: {sc['camera_movement']}.",
             f"Style: {brief.get('visual_style', REQUIRES_INPUT)}.", f"Aspect ratio {brief.get('aspect_ratio', '16:9')}, "
             f"duration {sc['duration_s']:.2f} s."]
    if bc["colors"] != REQUIRES_INPUT:
        parts.append(f"Palette: {bc['colors']}.")
    if sc["visual_mode"] != "text_graphic":
        parts.append("No on-screen text, captions or logos.")
    if sc["visual_mode"] == "talking_head":
        parts.append("Mouth clearly visible, minimal head motion (lip-sync applied later).")
    neg = bc["prohibited_visual_elements"]
    if neg != REQUIRES_INPUT:
        parts.append(f"Avoid: {neg}.")
    return " ".join(parts)


# ---------------------------------------------------------------- build + validate


def build_plan(job_dir: Path, brief_path: Path, cfg_override=None) -> dict:
    meta, words_doc, report, brief, brand, brand_src, cfg, duration = load_inputs(job_dir, brief_path, cfg_override)
    words = words_doc["words"]
    if abs(duration - words_doc["duration_s"]) > 1e-3:
        raise PlanError("DURATION_MISMATCH", f"speech.wav is {duration:.3f} s but words.json says {words_doc['duration_s']}")
    quality = gate(words, report, duration, cfg)
    units = segment(words, cfg)

    scenes = []
    for i, u in enumerate(units):
        ns, ne = span(u)
        scenes.append({"scene_id": f"S{i + 1:02d}", "narration_text": " ".join(w["word"] for w in u),
                       "first_word_index": u[0]["index"], "last_word_index": u[-1]["index"],
                       "word_count": len(u), "narration_start_s": r3(ns), "narration_end_s": r3(ne),
                       "uncertain_words": [{"index": w["index"], "word": w["word"], "status": w["status"],
                                            "flags": w.get("flags", [])} for w in u if w["status"] != "aligned"]})
    for i, sc in enumerate(scenes):   # contiguous timeline: cut at pause midpoints
        sc["start_s"] = 0.0 if i == 0 else r3((scenes[i - 1]["narration_end_s"] + sc["narration_start_s"]) / 2)
        sc["end_s"] = r3(duration) if i == len(scenes) - 1 else r3((sc["narration_end_s"] + scenes[i + 1]["narration_start_s"]) / 2)
        sc["duration_s"] = r3(sc["end_s"] - sc["start_s"])
        sc["start_ms"], sc["end_ms"] = ms(sc["start_s"]), ms(sc["end_s"])
        sc["pause_before_s"] = r3(sc["narration_start_s"] - (scenes[i - 1]["narration_end_s"] if i else 0.0))
        sc["pause_after_s"] = r3((scenes[i + 1]["narration_start_s"] if i < len(scenes) - 1 else duration) - sc["narration_end_s"])
    timed = [w for w in words if w.get("start_s") is not None]
    for i in range(len(scenes) - 1):
        a, b = scenes[i], scenes[i + 1]
        t, gap = a["end_s"], b["narration_start_s"] - a["narration_end_s"]
        untimed_edge = [words[k]["index"] for k in (a["last_word_index"], b["first_word_index"])
                        if words[k].get("start_s") is None]
        inside = [w["index"] for w in timed if w["start_s"] + TIME_TOL_S < t < w["end_s"] - TIME_TOL_S]
        if inside:
            reason = f"boundary {t:.3f} s falls inside word(s) {inside}"
        elif untimed_edge:
            reason = f"adjacent word(s) {untimed_edge} have no timestamps, so their real extent is unknown"
        elif gap < MIN_SAFE_GAP_S:
            reason = (f"no reliable silence between {words[a['last_word_index']]['word']!r} and "
                      f"{words[b['first_word_index']]['word']!r} (gap {gap * 1000:.0f} ms); narration is continuous "
                      "across this cut, which lands on a word edge")
        else:
            reason = None
        a["boundary_after"] = {"time_s": t, "gap_s": r3(gap), "safe": reason is None, "words_inside": inside,
                               "reason": reason or f"cut in a {gap:.2f} s silence gap"}
    direct(scenes, brief, brand, duration)
    transitions(scenes, cfg)
    for i, sc in enumerate(scenes):
        assets_and_brand(sc, brand, brief, common.PROJECT_ROOT, i == len(scenes) - 1)
        sc["generation_prompt"] = prompt(sc, brief)
        issues = []
        if sc["uncertain_words"]:
            issues.append(("review", "uncertain_words", f"uncertain words: {[w['word'] for w in sc['uncertain_words']]}"))
        for edge in ([scenes[i - 1]["boundary_after"]] if i else []) + ([sc["boundary_after"]] if "boundary_after" in sc else []):
            if edge["words_inside"]:
                issues.append(("error", "boundary_inside_word", edge["reason"]))
            elif not edge["safe"]:
                issues.append(("review", "unsafe_boundary", edge["reason"]))
        span_s = sc["narration_end_s"] - sc["narration_start_s"]
        if span_s > cfg["max_scene_s"] + TIME_TOL_S:
            issues.append(("review", "exceeds_max_scene_s",
                           f"narration spans {span_s:.2f} s > max_scene_s {cfg['max_scene_s']} s and cannot be split "
                           "at a word boundary with known timestamps"))
        elif sc["duration_s"] > cfg["max_scene_s"] + TIME_TOL_S:
            issues.append(("review", "timeline_exceeds_max_scene_s",
                           f"scene timeline {sc['duration_s']:.2f} s > max_scene_s {cfg['max_scene_s']} s because of "
                           "the surrounding pauses"))
        issues += [("review", "direction", n) for n in sc["direction_notes"]]
        issues += [("missing_input", "missing_input", f"missing input: {a['type']}")
                   for a in sc["asset_requirements"] if a["status"] in (REQUIRES_INPUT, "missing")]
        sc["validation_issues"] = [{"severity": sev, "code": code, "message": msg} for sev, code, msg in issues]
        set_scene_status(sc)

    missing_brand = [k for k in BRAND_FIELDS if field(brand, k) == REQUIRES_INPUT]
    if (brief.get("presenter") or {}).get("available") and not (brief.get("presenter") or {}).get("consent_confirmed"):
        missing_brand_note = ["presenter.consent_confirmed"]
    else:
        missing_brand_note = []
    target = brief.get("target_duration_s")
    plan = {
        "schema_version": 1, "job_id": meta["job_id"],
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "planner": "scene_planner.py (deterministic rules; no models, no network)",
        "inputs": {"audio": "speech.wav", "words": "alignment/words.json", "alignment_report": "alignment/alignment_report.json",
                   "brief": str(brief_path), "brand_profile": brand_src,
                   "alignment_engine": words_doc.get("alignment_engine"),
                   "timing_precision_note": words_doc.get("precision_note")},
        "audio_duration_s": r3(duration),
        "target_duration_s": target,
        "duration_note": (None if target is None else
                          f"audio is {duration:.2f} s vs target {target} s ({duration - target:+.2f} s); "
                          "audio is not stretched. Any intro/outro padding is an editorial decision."),
        "planning_config": cfg,
        "alignment_quality": quality,
        "brand": {k: field(brand, k) for k in BRAND_FIELDS},
        "missing_inputs": missing_brand + missing_brand_note,
        "timeline_note": ("start_s/end_s partition the audio contiguously, cutting at the midpoint of the pause between "
                          "scenes; narration_start_s/narration_end_s are the exact first/last word times."),
        "scenes": scenes,
    }
    plan["validation"] = validate(plan, words, duration)
    return plan


SCENE_STATUS = (("error", "failed"), ("review", "needs_review"), ("missing_input", "missing_inputs"))


def set_scene_status(sc: dict):
    """failed > needs_review > missing_inputs > ok. validation_notes kept as plain strings (v1 field)."""
    sev = {i["severity"] for i in sc["validation_issues"]}
    sc["validation_status"] = next((st for s_, st in SCENE_STATUS if s_ in sev), "ok")
    sc["validation_notes"] = [i["message"] for i in sc["validation_issues"]]


def finalize_status(plan: dict):
    """Plan status: failed (hard timeline error) > needs_review > valid_with_missing_inputs > valid.
    Missing brand/creative inputs never make a timeline invalid."""
    v, sc = plan["validation"], plan["scenes"]
    hard = not all(v["checks"].values()) or any(s["validation_status"] == "failed" for s in sc)
    if hard:
        v["status"] = "failed"
    elif any(s["validation_status"] == "needs_review" for s in sc):
        v["status"] = "needs_review"
    elif plan["missing_inputs"] or any(s["validation_status"] == "missing_inputs" for s in sc):
        v["status"] = "valid_with_missing_inputs"
    else:
        v["status"] = "valid"
    v["ok"] = v["status"] != "failed"          # ok = timeline data valid (unchanged meaning)
    v["failed_checks"] = [k for k, ok in v["checks"].items() if not ok]
    v["scenes_failed"] = [s["scene_id"] for s in sc if s["validation_status"] == "failed"]
    v["scenes_needing_review"] = [s["scene_id"] for s in sc if s["validation_status"] == "needs_review"]
    v["scenes_missing_inputs"] = [s["scene_id"] for s in sc if s["validation_status"] == "missing_inputs"]


def validate(plan: dict, words: list, duration: float) -> dict:
    sc = plan["scenes"]
    idx = [i for s in sc for i in range(s["first_word_index"], s["last_word_index"] + 1)]
    texts_ok = all(s["narration_text"] == " ".join(w["word"] for w in words[s["first_word_index"]:s["last_word_index"] + 1])
                   for s in sc)
    checks = {
        "timestamps_valid": all(0 <= s["start_s"] <= s["narration_start_s"] < s["narration_end_s"] <= s["end_s"]
                                for s in sc),
        "within_audio_duration": all(s["end_s"] <= duration + 1e-3 for s in sc),
        "covers_full_audio": bool(sc) and sc[0]["start_s"] == 0.0 and abs(sc[-1]["end_s"] - duration) < 1e-3,
        "contiguous_no_overlap": all(abs(a["end_s"] - b["start_s"]) < 1e-6 for a, b in zip(sc, sc[1:])),
        "chronological": all(a["narration_end_s"] <= b["narration_start_s"] + 1e-6 for a, b in zip(sc, sc[1:])),
        "durations_consistent": all(abs(s["duration_s"] - (s["end_s"] - s["start_s"])) < 1e-3 and
                                    s["start_ms"] == ms(s["start_s"]) and s["end_ms"] == ms(s["end_s"]) for s in sc),
        "all_words_assigned": sorted(idx) == list(range(len(words))),
        "no_word_in_multiple_scenes": len(idx) == len(set(idx)),
        "no_missing_word_indices": set(idx) == set(range(len(words))),
        "narration_text_matches_words": texts_ok,
        "valid_visual_modes": all(s["visual_mode"] in MODES for s in sc),
        "timestamps_finite": all(_is_time(s[k]) for s in sc for k in ("start_s", "end_s", "narration_start_s",
                                                                       "narration_end_s", "duration_s")),
        "no_boundary_inside_word": not any(s.get("boundary_after", {}).get("words_inside") for s in sc),
    }
    for s in sc:   # scene-local hard failures
        bad = []
        if not (_is_time(s["start_s"]) and _is_time(s["end_s"]) and
                0 <= s["start_s"] <= s["narration_start_s"] < s["narration_end_s"] <= s["end_s"] <= duration + 1e-3):
            bad.append("scene times invalid or outside the audio")
        if abs(s["duration_s"] - (s["end_s"] - s["start_s"])) >= 1e-3:
            bad.append("duration_s disagrees with start/end")
        for m in bad:
            s["validation_issues"].append({"severity": "error", "code": "invalid_scene_timeline", "message": m})
        if bad:
            set_scene_status(s)
    v = {"ok": all(checks.values()), "checks": checks,
         "uncertain_words_surfaced": sum(len(s["uncertain_words"]) for s in sc)}
    plan["validation"] = v
    finalize_status(plan)
    return v


def to_markdown(plan: dict) -> str:
    L = [f"# Scene plan: {plan['job_id']}", "",
         f"- Audio: {plan['audio_duration_s']:.3f} s; target: {plan['target_duration_s']} s. {plan['duration_note'] or ''}",
         f"- Alignment: {plan['inputs']['alignment_engine']} — coverage {plan['alignment_quality']['coverage_percent']}%, "
         f"{len(plan['alignment_quality']['uncertain_or_unaligned'])} uncertain/unaligned word(s)",
         f"- Validation status: **{plan['validation'].get('status', 'unknown')}** (timeline "
         f"{'valid' if plan['validation']['ok'] else 'INVALID'}); needing review: "
         f"{', '.join(plan['validation']['scenes_needing_review']) or 'none'}; missing inputs only: "
         f"{', '.join(plan['validation'].get('scenes_missing_inputs', [])) or 'none'}",
         f"- Missing inputs: {', '.join(plan['missing_inputs']) or 'none'}", "",
         "| Scene | Start (s) | End (s) | Duration (s) | Words | Mode | Speaker | Narration |",
         "|---|---|---|---|---|---|---|---|"]
    for s in plan["scenes"]:
        L.append(f"| {s['scene_id']} | {s['start_s']:.3f} | {s['end_s']:.3f} | {s['duration_s']:.3f} | "
                 f"{s['first_word_index']}-{s['last_word_index']} | {s['visual_mode']} | "
                 f"{'yes' if s['speaker_required'] else 'no'} | {s['narration_text']} |")
    for s in plan["scenes"]:
        L += ["", f"## {s['scene_id']} — {s['visual_mode']} ({s['start_s']:.3f}–{s['end_s']:.3f} s)", "",
              f"- **Narration** ({s['narration_start_s']:.3f}–{s['narration_end_s']:.3f} s): {s['narration_text']}",
              f"- **Visual** ({s['visual_description_source']}): {s['visual_description']}",
              f"- **Shot / camera:** {s['shot_type']} / {s['camera_movement']}",
              f"- **Transition out:** {s['transition']}" + (f" (in: {s['transition_in']})" if s.get("transition_in") else ""),
              f"- **Assets:** " + "; ".join(f"{a['type']} [{a['status']}]" for a in s["asset_requirements"]),
              f"- **Prompt:** {s['generation_prompt']}",
              f"- **Status:** {s['validation_status']}" + (f" — {'; '.join(s['validation_notes'])}" if s["validation_notes"] else "")]
    L += ["", "Timing precision: " + (plan["inputs"]["timing_precision_note"] or "n/a")]
    return "\n".join(L) + "\n"


ROW = re.compile(r"^\| (S\d+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| (\d+)-(\d+) \| (\w+) \|")


def check_markdown(md: str, plan: dict) -> bool:
    rows = [m.groups() for m in map(ROW.match, md.splitlines()) if m]
    want = [(s["scene_id"], f"{s['start_s']:.3f}", f"{s['end_s']:.3f}", f"{s['duration_s']:.3f}",
             str(s["first_word_index"]), str(s["last_word_index"]), s["visual_mode"]) for s in plan["scenes"]]
    return rows == want


EXIT = {"valid": 0, "valid_with_missing_inputs": 0, "needs_review": 2, "failed": 1}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="exit codes: 0 valid / valid_with_missing_inputs, 2 needs_review, "
                                        "1 failed (hard error; no plan files written)")
    ap.add_argument("--job", required=True, help="job id under --outputs-dir, or a job folder path")
    ap.add_argument("--brief", type=Path, required=True)
    ap.add_argument("--outputs-dir", type=Path, default=common.OUTPUTS_DIR)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build and validate only; write nothing")
    args = ap.parse_args()
    job_dir = Path(args.job) if Path(args.job).is_dir() else args.outputs_dir / args.job
    out = job_dir / "scene_plan"
    if not args.dry_run and out.exists() and any(out.iterdir()) and not args.overwrite:
        sys.exit(f"[ERROR] PLAN_EXISTS: {out} already has a plan; pass --overwrite (or --dry-run to only validate)")
    try:
        plan = build_plan(job_dir, args.brief)
    except PlanError as e:
        sys.exit(f"[ERROR] {e.code}: {e}")
    md = to_markdown(plan)
    plan["validation"]["checks"]["markdown_matches_json"] = check_markdown(md, plan)
    finalize_status(plan)
    md = to_markdown(plan)  # re-render with the final verdict
    text = json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False)  # strict JSON: no NaN/Infinity
    json.loads(text)
    v = plan["validation"]
    if v["status"] == "failed":
        print(f"[ERROR] PLAN_INVALID: hard validation failure; no plan written. failed checks: {v['failed_checks']}; "
              f"failed scenes: {v['scenes_failed']}")
        for sc in plan["scenes"]:
            for i in sc["validation_issues"]:
                if i["severity"] == "error":
                    print(f"  {sc['scene_id']}: {i['message']}")
        sys.exit(EXIT["failed"])
    if not args.dry_run:
        out.mkdir(exist_ok=True)
        (out / "scene_plan.json").write_text(text, encoding="utf-8")
        (out / "scene_plan.md").write_text(md, encoding="utf-8")
    print(f"{v['status'].upper()}  {'(dry run, nothing written)' if args.dry_run else out}  ({len(plan['scenes'])} scenes)")
    for sc in plan["scenes"]:
        print(f"  {sc['scene_id']} {sc['start_s']:6.3f}-{sc['end_s']:6.3f}s  {sc['visual_mode']:20} "
              f"{sc['validation_status']:15} {sc['narration_text']}")
        for i in sc["validation_issues"]:
            if i["severity"] != "missing_input":
                print(f"      [{i['severity']}] {i['message']}")
    if plan["missing_inputs"]:
        print(f"  missing inputs: {', '.join(plan['missing_inputs'])}")
    sys.exit(EXIT[v["status"]])


if __name__ == "__main__":
    main()
