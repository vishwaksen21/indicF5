"""Phase 4B-0 (Stage D): SRT + word-highlight ASS subtitles from Phase 2 word timestamps.

    .venv/bin/python scripts/subtitles.py --job <JOB_ID>

Reads (never modifies): <job>/alignment/words.json, <job>/speech.wav (header only, for duration),
<job>/scene_plan/scene_plan.json (optional: aspect ratio via its brief).
Writes: <job>/subtitles/{captions.srt, captions_highlight.ass, subtitle_report.json} (refuses to overwrite).

Timing comes only from words.json; nothing is estimated. Cue rules mirror Phase 2's
align_words.build_cues (sentence cues, <= 42 chars/line, 2 lines, <= 7 s) and SRT formatting and
checking reuse align_words.srt_time / check_srt. Unlike build_cues, words without timestamps are
kept in the cue text (never dropped); cue times come from the cue's timestamped words only.

Highlighting: a cue is word-highlighted only if every word in it is 'aligned'. A cue containing an
'uncertain' or 'unaligned' word is shown whole, without highlighting (its word boundaries are not
trustworthy). The style is a neutral placeholder: brand typography/colours are not invented.
Exit codes: 0 ok; 2 written but needs review (uncertain/unaligned words); 1 failed (nothing written).
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
from align_words import SRT_LINE_CHARS, SRT_MAX_CUE_S, check_srt, srt_time  # noqa: E402

TOL = 0.001
RESOLUTIONS = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080), "4:5": (1080, 1350)}
# Neutral placeholder style (ASS colours are &HAABBGGRR): white text, black outline, active word yellow.
PLACEHOLDER_STYLE = {"font": "Helvetica Neue", "primary": "&H00FFFFFF", "highlight": "&H0000FFFF",
                     "outline_colour": "&H00000000", "back_colour": "&H80000000", "outline": 3, "shadow": 1,
                     "note": "neutral placeholder; brand typography/colours are REQUIRES_USER_INPUT"}


class SubtitleError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def is_time(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def check_words(words: list, duration: float) -> list[str]:
    """All timestamp problems, with word indices. Pauses between words are fine."""
    problems, timed = [], []
    if [w.get("index") for w in words] != list(range(len(words))):
        problems.append("word indices are not contiguous 0..N-1")
    for w in words:
        i, s, e, st = w.get("index"), w.get("start_s"), w.get("end_s"), w.get("status")
        tag = f"word {i} {w.get('word')!r}"
        if st not in ("aligned", "uncertain", "unaligned"):
            problems.append(f"{tag}: unknown status {st!r}")
        if s is None and e is None:
            if st != "unaligned":
                problems.append(f"{tag}: missing timestamps but status is {st!r}")
            continue
        if s is None or e is None or not (is_time(s) and is_time(e)):
            problems.append(f"{tag}: invalid timestamps (start={s!r}, end={e!r})")
            continue
        if st == "unaligned":
            problems.append(f"{tag}: 'unaligned' but carries timestamps")
        if s < 0 or not s < e:
            problems.append(f"{tag}: start {s} / end {e} invalid")
        if e > duration + TOL:
            problems.append(f"{tag}: end {e} exceeds audio duration {duration:.3f}")
        timed.append(w)
    for a, b in zip(timed, timed[1:]):
        if b["start_s"] < a["start_s"] - TOL:
            problems.append(f"words {a['index']}->{b['index']}: not chronological")
        elif a["end_s"] > b["start_s"] + TOL:
            problems.append(f"words {a['index']}->{b['index']}: overlap {a['end_s'] - b['start_s']:.3f} s")
    return problems


def build_cues(words: list) -> list[dict]:
    """Same grouping rules as align_words.build_cues, but untimed words stay in their cue."""
    cues = []
    for si in sorted({w["sentence_index"] for w in words}):
        group = []
        for w in (x for x in words if x["sentence_index"] == si):
            timed = [x for x in group if x.get("start_s") is not None]
            text = " ".join(x["word"] for x in group + [w])
            too_long = len(text) > 2 * SRT_LINE_CHARS or (
                timed and w.get("end_s") is not None and w["end_s"] - timed[0]["start_s"] > SRT_MAX_CUE_S)
            if group and too_long:
                cues.append(group)
                group = []
            group.append(w)
        if group:
            cues.append(group)
    out = []
    for g in cues:
        timed = [w for w in g if w.get("start_s") is not None]
        if not timed:
            raise SubtitleError("CUE_UNTIMED", f"cue with words {[w['index'] for w in g]} has no timestamped word; "
                                               "it cannot be placed without inventing times")
        text = " ".join(w["word"] for w in g)
        lines = [g]
        if len(text) > SRT_LINE_CHARS and len(g) > 1:
            cut = min(range(1, len(g)), key=lambda k: abs(len(" ".join(w["word"] for w in g[:k])) - len(text) / 2))
            lines = [g[:cut], g[cut:]]
        out.append({"words": g, "lines": lines, "start_s": timed[0]["start_s"], "end_s": timed[-1]["end_s"],
                    "highlight": all(w["status"] == "aligned" for w in g)})
    return out


def ass_time(t: float) -> str:
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def ass_escape(text: str) -> str:
    # '{' '}' start override blocks and '\' starts tags in ASS; neutralise them in caption text
    return text.replace("\\", "⧵").replace("{", "(").replace("}", ")")


def cue_text_ass(cue: dict, active: int | None, style: dict) -> str:
    parts = []
    for line in cue["lines"]:
        ws = []
        for w in line:
            t = ass_escape(w["word"])
            if active is not None and w["index"] == active:
                t = f"{{\\c{style['highlight']}&}}{t}{{\\c{style['primary']}&}}"
            ws.append(t)
        parts.append(" ".join(ws))
    return "\\N".join(parts)


def render_ass(cues: list, res: tuple, style: dict) -> tuple[str, list[dict]]:
    w, h = res
    margin_v, margin_lr = round(0.08 * h), round(0.10 * w)
    font_size = round(h * 0.055) if w >= h else round(w * 0.06)
    head = ["[Script Info]", "; Generated by subtitles.py (Phase 4B-0). Timing from alignment/words.json.",
            "ScriptType: v4.00+", f"PlayResX: {w}", f"PlayResY: {h}", "WrapStyle: 2", "ScaledBorderAndShadow: yes", "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
            "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
            "MarginR, MarginV, Encoding",
            f"Style: Caption,{style['font']},{font_size},{style['primary']},{style['highlight']},{style['outline_colour']},"
            f"{style['back_colour']},0,0,0,0,100,100,0,0,1,{style['outline']},{style['shadow']},2,{margin_lr},{margin_lr},"
            f"{margin_v},1", "",
            "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    events = []
    for ci, cue in enumerate(cues):
        if cue["highlight"]:
            ws = cue["words"]
            for k, wd in enumerate(ws):   # word k active from its start to the next word's start
                s = wd["start_s"]
                e = ws[k + 1]["start_s"] if k + 1 < len(ws) else cue["end_s"]
                events.append({"cue": ci, "word_index": wd["index"], "start_s": s, "end_s": e,
                               "text": cue_text_ass(cue, wd["index"], style)})
        else:
            events.append({"cue": ci, "word_index": None, "start_s": cue["start_s"], "end_s": cue["end_s"],
                           "text": cue_text_ass(cue, None, style)})
    lines = head + [f"Dialogue: 0,{ass_time(ev['start_s'])},{ass_time(ev['end_s'])},Caption,,0,0,0,,{ev['text']}"
                    for ev in events]
    return "\n".join(lines) + "\n", events


ASS_DIALOGUE = re.compile(r"^Dialogue: 0,(\d+):(\d\d):(\d\d)\.(\d\d),(\d+):(\d\d):(\d\d)\.(\d\d),Caption,,0,0,0,,(.*)$")


def check_ass(text: str, duration: float, expected_events: int) -> dict:
    rows, bad = [], []
    for ln in text.splitlines():
        if ln.startswith("Dialogue:"):
            m = ASS_DIALOGUE.match(ln)
            if not m:
                bad.append(ln[:60])
                continue
            g = list(map(int, m.groups()[:8]))
            rows.append((g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 100, g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 100))
    order_ok = all(b[0] >= a[0] - 1e-9 for a, b in zip(rows, rows[1:]))
    within = all(0 <= s < e <= duration + 0.01 for s, e in rows)   # ASS has 10 ms (centisecond) resolution
    return {"events": len(rows), "expected_events": expected_events, "malformed": bad,
            "chronological": order_ok, "within_duration_and_positive": within,
            "ok": not bad and order_ok and within and len(rows) == expected_events}


def generate(job_dir: Path) -> tuple[dict, str, str]:
    wpath = job_dir / "alignment" / "words.json"
    wav = job_dir / "speech.wav"
    if not wpath.is_file():
        raise SubtitleError("WORDS_MISSING", f"{wpath} not found (run align_words.py first)")
    if not wav.is_file():
        raise SubtitleError("AUDIO_MISSING", f"{wav} not found")
    try:
        doc = json.loads(wpath.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SubtitleError("INVALID_JSON", f"{wpath}: {e}")
    import soundfile as sf

    duration = sf.info(str(wav)).duration
    if abs(duration - float(doc.get("duration_s", -1))) > TOL:
        raise SubtitleError("DURATION_MISMATCH", f"speech.wav {duration:.3f} s vs words.json {doc.get('duration_s')}")
    words = doc.get("words") or []
    if not words:
        raise SubtitleError("NO_WORDS", "words.json has no words")
    problems = check_words(words, duration)
    if problems:
        raise SubtitleError("INVALID_TIMESTAMPS", "word timestamps invalid:\n  - " + "\n  - ".join(problems[:20]))

    aspect, aspect_src = "16:9", "default"
    plan_path = job_dir / "scene_plan" / "scene_plan.json"
    if plan_path.is_file():
        brief_ref = json.loads(plan_path.read_text(encoding="utf-8")).get("inputs", {}).get("brief")
        bp = (common.PROJECT_ROOT / brief_ref) if brief_ref and not Path(brief_ref).is_absolute() else Path(brief_ref or "")
        if brief_ref and bp.is_file():
            aspect, aspect_src = json.loads(bp.read_text(encoding="utf-8")).get("aspect_ratio") or "16:9", str(bp)
    if aspect not in RESOLUTIONS:
        raise SubtitleError("INVALID_ASPECT", f"unsupported aspect ratio {aspect!r}")
    res = RESOLUTIONS[aspect]

    cues = build_cues(words)
    srt = "".join(f"{k}\n{srt_time(c['start_s'])} --> {srt_time(c['end_s'])}\n"
                  f"{chr(10).join(' '.join(w['word'] for w in ln) for ln in c['lines'])}\n\n"
                  for k, c in enumerate(cues, 1))
    ass, events = render_ass(cues, res, PLACEHOLDER_STYLE)

    covered = [w["index"] for c in cues for w in c["words"]]
    flagged = [{"index": w["index"], "word": w["word"], "status": w["status"], "flags": w.get("flags", [])}
               for w in words if w["status"] != "aligned"]
    expected_hl = sum(len(c["words"]) for c in cues if c["highlight"])
    report = {
        "job_id": doc.get("job_id"), "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "generator": "subtitles.py (Phase 4B-0)", "timing_source": str(wpath),
        "alignment_engine": doc.get("alignment_engine"), "audio_duration_s": round(duration, 3),
        "aspect_ratio": aspect, "aspect_source": aspect_src, "play_resolution": list(res),
        "style": PLACEHOLDER_STYLE,
        "placement": {"alignment": "bottom-centre", "margin_v_px": round(0.08 * res[1]),
                      "margin_lr_px": round(0.10 * res[0]), "max_chars_per_line": SRT_LINE_CHARS, "max_lines": 2},
        "cues": [{"cue": k, "start_s": c["start_s"], "end_s": c["end_s"],
                  "first_word_index": c["words"][0]["index"], "last_word_index": c["words"][-1]["index"],
                  "text": " ".join(w["word"] for w in c["words"]), "highlighted": c["highlight"],
                  "not_highlighted_because": None if c["highlight"] else
                  [f"{w['word']} ({w['status']})" for w in c["words"] if w["status"] != "aligned"]}
                 for k, c in enumerate(cues, 1)],
        "word_coverage": {"words": len(words), "in_cues": len(covered),
                          "every_word_exactly_once": sorted(covered) == list(range(len(words))) and
                          len(covered) == len(set(covered)),
                          "text_matches_transcript": " ".join(w["word"] for c in cues for w in c["words"]) ==
                          " ".join(w["word"] for w in words),
                          "highlight_events": sum(1 for e in events if e["word_index"] is not None),
                          "expected_highlight_events": expected_hl},
        "uncertain_or_unaligned_words": flagged,
        "precision_note": (doc.get("precision_note") or "") + " SRT is written in ms, ASS in centiseconds "
                          "(10 ms); neither is a precision claim.",
    }
    return report, srt, ass


def validate(report: dict, srt_path: Path, ass_text: str) -> dict:
    d = report["audio_duration_s"]
    cues = report["cues"]
    wc = report["word_coverage"]
    checks = {
        "cues_chronological": all(b["start_s"] >= a["end_s"] - TOL for a, b in zip(cues, cues[1:])),
        "cues_within_audio": all(0 <= c["start_s"] < c["end_s"] <= d + TOL for c in cues),
        "every_word_in_exactly_one_cue": wc["every_word_exactly_once"],
        "cue_text_matches_transcript": wc["text_matches_transcript"],
        "highlight_events_match_aligned_words": wc["highlight_events"] == wc["expected_highlight_events"],
    }
    srt_check = check_srt(srt_path, d)
    ass_check = check_ass(ass_text, d, wc["highlight_events"] + sum(1 for c in cues if not c["highlighted"]))
    checks["srt_valid"], checks["ass_valid"] = srt_check["ok"], ass_check["ok"]
    return {"ok": all(checks.values()), "checks": checks, "srt": srt_check, "ass": ass_check}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True, help="job id under --outputs-dir, or a job folder path")
    ap.add_argument("--outputs-dir", type=Path, default=common.OUTPUTS_DIR)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    job_dir = Path(args.job) if Path(args.job).is_dir() else args.outputs_dir / args.job
    if not job_dir.is_dir():
        sys.exit(f"[ERROR] JOB_NOT_FOUND: {job_dir}")
    out = job_dir / "subtitles"
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        sys.exit(f"[ERROR] SUBTITLES_EXIST: {out} already has subtitles; pass --overwrite")
    try:
        report, srt, ass = generate(job_dir)
    except SubtitleError as e:
        sys.exit(f"[ERROR] {e.code}: {e}")
    out.mkdir(exist_ok=True)
    srt_path = out / "captions.srt"
    srt_path.write_text(srt, encoding="utf-8")
    v = validate(report, srt_path, ass)
    if not v["ok"]:
        srt_path.unlink()
        if not any(out.iterdir()):
            out.rmdir()
        sys.exit(f"[ERROR] SUBTITLES_INVALID: failed checks {[k for k, ok in v['checks'].items() if not ok]}")
    report["validation"] = v
    report["status"] = "needs_review" if report["uncertain_or_unaligned_words"] else "ok"
    (out / "captions_highlight.ass").write_text(ass, encoding="utf-8")
    (out / "subtitle_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                                              encoding="utf-8")
    wc = report["word_coverage"]
    print(f"{report['status'].upper()}  {out}  ({len(report['cues'])} cues, {wc['in_cues']}/{wc['words']} words, "
          f"{wc['highlight_events']} highlight events)")
    for c in report["cues"]:
        print(f"  cue {c['cue']} {srt_time(c['start_s'])} --> {srt_time(c['end_s'])}  "
              f"{'highlight' if c['highlighted'] else 'NO highlight: ' + ', '.join(c['not_highlighted_because'])}"
              f"  {c['text']}")
    sys.exit(2 if report["status"] == "needs_review" else 0)


if __name__ == "__main__":
    main()
