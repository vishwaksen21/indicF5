"""Phase 4B-0 (Stage A): per-scene asset specification from a validated scene plan.

    .venv/bin/python scripts/asset_manifest.py --job <JOB_ID> [--brief texts/briefs/sample_brief.json]

Reads (never modifies): <job>/scene_plan/scene_plan.json, the brief and brand profile recorded in
the plan (or --brief), and - for plans written before Phase 3.1 - the inputs needed to
re-validate the plan in memory with the current planner (nothing is written for that).
Writes: <job>/assets/asset_manifest.json and asset_manifest.md (refuses to overwrite).

Brand data is copied verbatim from the brand profile; anything missing is REQUIRES_USER_INPUT.
Nothing is generated, downloaded or fetched.
Exit codes: 0 manifest written; 1 failed (nothing written).
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

REQUIRES_INPUT = "REQUIRES_USER_INPUT"
BRAND_FIELDS = ("brand_name", "logo_asset_path", "brand_colors", "typography", "tone", "visual_restrictions",
                "required_disclaimers", "prohibited_visual_elements")
RESOLUTIONS = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080), "4:5": (1080, 1350)}
DEFAULT_FPS = 25
ASSET_TYPE = {"talking_head": "presenter_video", "text_graphic": "brand_card", "voiceover_broll": "video_clip",
              "environmental_visual": "video_clip", "product_visual": "product_footage"}
DISSOLVE = re.compile(r"cross-dissolve ([\d.]+) s")
TIMELINE_KEYS = ("scene_id", "start_s", "end_s", "first_word_index", "last_word_index", "visual_mode", "narration_text")


class ManifestError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def verbatim(v):
    return REQUIRES_INPUT if v in (None, "", [], {}) else v


def resolve(path_str: str | None, base: Path) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    return p if p.is_absolute() else base / p


def load_json(path: Path, code: str):
    if not path.is_file():
        raise ManifestError(code, f"{path} not found")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ManifestError("INVALID_JSON", f"{path}: {e}")


def check_plan(job_dir: Path, plan: dict, brief_path: Path) -> dict:
    """Return the plan's validation verdict. Plans from Phase 3.0 lack validation.status: rebuild the
    plan in memory with the current planner and require an identical timeline (nothing is written)."""
    v = plan.get("validation") or {}
    if "status" in v:
        return {"status": v["status"], "source": "scene_plan.json", "boundaries": None}
    import scene_planner

    try:
        rebuilt = scene_planner.build_plan(job_dir, brief_path)
    except scene_planner.PlanError as e:
        raise ManifestError("PLAN_INVALID", f"in-memory re-validation failed: {e.code}: {e}")
    saved = [{k: s.get(k) for k in TIMELINE_KEYS} for s in plan.get("scenes", [])]
    fresh = [{k: s.get(k) for k in TIMELINE_KEYS} for s in rebuilt["scenes"]]
    if saved != fresh:
        raise ManifestError("PLAN_OUTDATED", "saved scene_plan.json predates Phase 3.1 and its timeline differs "
                                             "from a fresh plan; re-run scene_planner.py (needs --overwrite approval)")
    return {"status": rebuilt["validation"]["status"], "source": "re-validated in memory (plan predates Phase 3.1)",
            "boundaries": {s["scene_id"]: s.get("boundary_after") for s in rebuilt["scenes"]},
            "rebuilt_validation": rebuilt["validation"]}


def handles(scenes: list) -> list[dict]:
    """Extra clip time needed around each scene for dissolves (half the dissolve on each side)."""
    out = [{"head": 0.0, "tail": 0.0} for _ in scenes]
    for i, s in enumerate(scenes[:-1]):
        m = DISSOLVE.search(s.get("transition") or "")
        if m:
            half = round(float(m.group(1)) / 2, 3)
            out[i]["tail"] = half
            out[i + 1]["head"] = half
    return out


def build_manifest(job_dir: Path, brief_override: Path | None = None) -> dict:
    plan_path = job_dir / "scene_plan" / "scene_plan.json"
    plan = load_json(plan_path, "PLAN_MISSING")
    if not plan.get("scenes"):
        raise ManifestError("PLAN_INVALID", "scene plan has no scenes")
    root = common.PROJECT_ROOT
    brief_path = brief_override or resolve(plan.get("inputs", {}).get("brief"), root)
    if brief_path is None:
        raise ManifestError("BRIEF_MISSING", "plan records no brief; pass --brief")
    brief = load_json(brief_path, "BRIEF_MISSING")

    verdict = check_plan(job_dir, plan, brief_path)
    if verdict["status"] == "failed":
        raise ManifestError("PLAN_INVALID", "scene plan validation status is 'failed'")

    # brand profile, copied verbatim (inline in the brief, or the file the brief points to)
    bp = brief.get("brand_profile")
    if isinstance(bp, str):
        bpath = resolve(bp, brief_path.parent)
        brand_raw, brand_src = load_json(bpath, "BRAND_PROFILE_MISSING"), str(bpath)
    else:
        brand_raw, brand_src = (bp or {}), "brief (inline)"
    brand = {k: verbatim(brand_raw.get(k)) for k in BRAND_FIELDS}

    aspect = brief.get("aspect_ratio") or "16:9"
    if aspect not in RESOLUTIONS:
        raise ManifestError("INVALID_BRIEF", f"unsupported aspect_ratio {aspect!r} (supported: {list(RESOLUTIONS)})")
    res, fps = RESOLUTIONS[aspect], int(brief.get("fps") or DEFAULT_FPS)
    presenter = brief.get("presenter") or {}
    consent = presenter.get("consent_confirmed")
    incoming = job_dir / "assets" / "incoming"

    entries = []
    for sc, h in zip(plan["scenes"], handles(plan["scenes"])):
        mode = sc["visual_mode"]
        asset_type = ASSET_TYPE.get(mode)
        if asset_type is None:
            raise ManifestError("PLAN_INVALID", f"{sc['scene_id']}: unknown visual_mode {mode!r}")
        found = sorted(incoming.glob(f"{sc['scene_id']}.*")) if incoming.is_dir() else []
        person, lipsync = mode == "talking_head", mode == "talking_head"
        brand_req = {"scene_constraints": sc.get("brand_constraints", {}), "brand_profile": brand}
        acceptance = {
            "min_duration_s": round(sc["duration_s"] + h["head"] + h["tail"], 3),
            "aspect_ratio": aspect, "resolution": list(res), "fps": fps, "frame_rate_mode": "constant",
            "no_audio_track_after_conform": True,
            "no_text_or_logos": mode != "text_graphic",
            "must_not_stretch": True,
        }
        blockers = []
        if person:
            acceptance.update({"lipsync_required": True, "lipsync_audio": "exact speech.wav segment "
                               f"{sc['start_s']:.3f}-{sc['end_s']:.3f} s", "consent_required": True,
                               "consent_confirmed": consent if consent is not None else REQUIRES_INPUT,
                               "identity_source": presenter.get("source_media") or REQUIRES_INPUT})
            if consent is not True:
                blockers.append("presenter consent not confirmed (brief.presenter.consent_confirmed)")
            if not presenter.get("source_media"):
                blockers.append("approved presenter media not specified (brief.presenter.source_media)")
            blockers.append("lip-sync against the exact WAV segment not yet produced or validated")
        if mode == "text_graphic":
            for k in ("logo_asset_path", "typography", "brand_colors"):
                if brand[k] == REQUIRES_INPUT:
                    blockers.append(f"brand {k} missing")
            if brand["logo_asset_path"] != REQUIRES_INPUT:
                lp = resolve(brand["logo_asset_path"], root)
                if not lp.is_file():
                    blockers.append(f"logo file not found: {lp}")
        if found:
            source = {"kind": "user_provided", "provider": "manual", "path": str(found[0]), "approval_ref": None}
            status = "provided_unvalidated"   # Stage B (visual_assets.py) will probe/conform it
        else:
            source = {"kind": "unresolved", "provider": None, "path": None, "approval_ref": None,
                      "options": ["user_provided (drop file at assets/incoming/%s.<ext>)" % sc["scene_id"],
                                  "approved_generation (requires explicit approval)",
                                  "placeholder (draft renders only)"]}
            status = "unresolved"
            blockers.insert(0, "no asset provided yet")
        entries.append({
            "scene_id": sc["scene_id"],
            "timeline": {"start_s": sc["start_s"], "end_s": sc["end_s"], "duration_s": sc["duration_s"],
                         "start_ms": sc["start_ms"], "end_ms": sc["end_ms"],
                         "narration_start_s": sc["narration_start_s"], "narration_end_s": sc["narration_end_s"],
                         "required_clip_s": acceptance["min_duration_s"], "handles_s": h},
            "visual_mode": mode,
            "asset_type": asset_type,
            "description": sc.get("visual_description"),
            "description_source": sc.get("visual_description_source"),
            "shot_type": sc.get("shot_type"), "camera_movement": sc.get("camera_movement"),
            "transition_out": sc.get("transition"), "transition_in": sc.get("transition_in"),
            "aspect_ratio": aspect, "target_resolution": list(res), "fps": fps,
            "person_required": person, "lipsync_required": lipsync,
            "brand_requirements": brand_req,
            "planner_asset_requirements": sc.get("asset_requirements", []),
            "generation_prompt": sc.get("generation_prompt"),
            "source": source, "status": status, "acceptance": acceptance, "blockers": blockers,
            "narration_text": sc.get("narration_text"),
        })

    missing_brand = [k for k, v in brand.items() if v == REQUIRES_INPUT]
    return {
        "schema_version": 1, "job_id": plan.get("job_id"),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "generator": "asset_manifest.py (Phase 4B-0; no generation, no network)",
        "inputs": {"scene_plan": str(plan_path), "brief": str(brief_path), "brand_profile": brand_src},
        "plan_validation": {k: verdict[k] for k in ("status", "source")},
        "plan_boundaries_revalidated": verdict.get("boundaries"),
        "audio_duration_s": plan.get("audio_duration_s"),
        "aspect_ratio": aspect, "target_resolution": list(res), "fps": fps,
        "brand_profile_verbatim": brand,
        "missing_inputs": sorted(set(missing_brand + (["presenter.consent_confirmed"] if consent is not True and
                                                       any(e["person_required"] for e in entries) else []) +
                                     (["presenter.source_media"] if not presenter.get("source_media") and
                                      any(e["person_required"] for e in entries) else []))),
        "summary": {"scenes": len(entries), "unresolved": sum(e["status"] == "unresolved" for e in entries),
                    "needing_person": sum(e["person_required"] for e in entries),
                    "needing_lipsync": sum(e["lipsync_required"] for e in entries)},
        "scenes": entries,
    }


def brand_matches_source(m: dict) -> bool:
    """Re-read the brand profile and confirm every manifest value is the source value or REQUIRES_USER_INPUT."""
    src = m["inputs"]["brand_profile"]
    if src == "brief (inline)":
        raw = (json.loads(Path(m["inputs"]["brief"]).read_text(encoding="utf-8")).get("brand_profile") or {})
    else:
        raw = json.loads(Path(src).read_text(encoding="utf-8"))
    return all(m["brand_profile_verbatim"][k] == verbatim(raw.get(k)) for k in BRAND_FIELDS)


def validate_manifest(m: dict, plan: dict) -> dict:
    sc, ps = m["scenes"], plan["scenes"]
    checks = {
        "one_entry_per_scene": [e["scene_id"] for e in sc] == [s["scene_id"] for s in ps],
        "timeline_copied_exactly": all(e["timeline"][k] == s[k] for e, s in zip(sc, ps)
                                       for k in ("start_s", "end_s", "duration_s")),
        "required_clip_covers_scene": all(e["timeline"]["required_clip_s"] >= e["timeline"]["duration_s"] for e in sc),
        "talking_head_flags_lipsync_and_consent": all(e["lipsync_required"] and e["acceptance"].get("consent_required")
                                                      for e in sc if e["visual_mode"] == "talking_head"),
        "brand_values_verbatim_from_profile": brand_matches_source(m),
        "json_roundtrip": json.loads(json.dumps(m, allow_nan=False)) == m,
    }
    return {"ok": all(checks.values()), "checks": checks}


def to_markdown(m: dict) -> str:
    L = [f"# Asset manifest: {m['job_id']}", "",
         f"- Plan validation: **{m['plan_validation']['status']}** ({m['plan_validation']['source']})",
         f"- Format: {m['aspect_ratio']} {m['target_resolution'][0]}x{m['target_resolution'][1]} @ {m['fps']} fps; "
         f"audio {m['audio_duration_s']} s",
         f"- Missing inputs: {', '.join(m['missing_inputs']) or 'none'}", "",
         "| Scene | Timeline (s) | Required clip (s) | Mode | Asset | Person | Lip-sync | Status |",
         "|---|---|---|---|---|---|---|---|"]
    for e in m["scenes"]:
        t = e["timeline"]
        L.append(f"| {e['scene_id']} | {t['start_s']:.3f}-{t['end_s']:.3f} | {t['required_clip_s']:.3f} | "
                 f"{e['visual_mode']} | {e['asset_type']} | {'yes' if e['person_required'] else 'no'} | "
                 f"{'yes' if e['lipsync_required'] else 'no'} | {e['status']} |")
    for e in m["scenes"]:
        L += ["", f"## {e['scene_id']} — {e['asset_type']}", "",
              f"- **Narration:** {e['narration_text']}",
              f"- **Description** ({e['description_source']}): {e['description']}",
              f"- **Shot / camera:** {e['shot_type']} / {e['camera_movement']}",
              f"- **Acceptance:** " + "; ".join(f"{k}={v}" for k, v in e["acceptance"].items()),
              f"- **Source:** {e['source']['kind']}",
              "- **Blockers:**" + "".join(f"\n  - {b}" for b in e["blockers"]) if e["blockers"] else "- **Blockers:** none"]
    L += ["", "## Brand profile (verbatim)", ""] + [f"- {k}: {v}" for k, v in m["brand_profile_verbatim"].items()]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True, help="job id under --outputs-dir, or a job folder path")
    ap.add_argument("--brief", type=Path, help="override the brief recorded in the scene plan")
    ap.add_argument("--outputs-dir", type=Path, default=common.OUTPUTS_DIR)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    job_dir = Path(args.job) if Path(args.job).is_dir() else args.outputs_dir / args.job
    out = job_dir / "assets"
    if not job_dir.is_dir():
        sys.exit(f"[ERROR] JOB_NOT_FOUND: {job_dir}")
    if (out / "asset_manifest.json").exists() and not args.overwrite:
        sys.exit(f"[ERROR] MANIFEST_EXISTS: {out / 'asset_manifest.json'} exists; pass --overwrite")
    try:
        m = build_manifest(job_dir, args.brief)
        v = validate_manifest(m, json.loads((job_dir / "scene_plan" / "scene_plan.json").read_text(encoding="utf-8")))
    except ManifestError as e:
        sys.exit(f"[ERROR] {e.code}: {e}")
    m["validation"] = v
    if not v["ok"]:
        sys.exit(f"[ERROR] MANIFEST_INVALID: failed checks {[k for k, ok in v['checks'].items() if not ok]}")
    out.mkdir(exist_ok=True)
    (out / "asset_manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (out / "asset_manifest.md").write_text(to_markdown(m), encoding="utf-8")
    print(f"OK  {out / 'asset_manifest.json'}  (plan {m['plan_validation']['status']}, {m['summary']['scenes']} scenes, "
          f"{m['summary']['unresolved']} unresolved)")
    for e in m["scenes"]:
        print(f"  {e['scene_id']} {e['asset_type']:16} clip>={e['timeline']['required_clip_s']:.3f}s  {e['status']:12} "
              f"blockers: {len(e['blockers'])}")
    print(f"  missing inputs: {', '.join(m['missing_inputs']) or 'none'}")


if __name__ == "__main__":
    main()
