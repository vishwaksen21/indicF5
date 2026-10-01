# Phase 4B-2 — Visual Asset & Presenter Readiness (audit + design)

Status: **read-only audit and design.** Nothing here has been implemented, generated, installed or downloaded.
Date: 2026-10-01 · Builds on `docs/phase4_visual_pipeline_design.md` · Reference job: `outputs/job_20261001_103028_eb8ca2`

**Bottom line: the pipeline is not production-ready.** The Phase 4B-1 render proves that timing, audio muxing and subtitle burn-in work end to end with *synthetic* placeholders. No real visual asset can be taken in, validated or rendered yet, and every real asset for the reference job is missing.

---

## 1. Current implementation audit

### 1.1 What each module actually supports (verified in code)

| Module | Supports | Does **not** support |
|---|---|---|
| `scene_planner.py` (3 / 3.1) | 5 visual modes (`talking_head`, `voiceover_broll`, `product_visual`, `text_graphic`, `environmental_visual`). It sets each asset requirement's status (`to_generate`, `REQUIRES_USER_INPUT`, `available`, `missing`), checks that the logo file **exists** if a path is given, reads `presenter.consent_confirmed`, and gives each boundary a `boundary_after` safety record | Any inspection of media content (it never opens an image or video) |
| `asset_manifest.py` (4B-0) | One primary asset per scene. If a file matches `assets/incoming/<scene_id>.*` (glob), the status becomes `provided_unvalidated`; otherwise `unresolved`. It declares acceptance criteria (minimum clip length including dissolve handles, 16:9, 1920×1080, 25 fps, no audio, no stretching) and copies brand data verbatim. It re-validates pre-3.1 plans in memory | **No** probing, hashing, duration or size checks, and no consent or lip-sync tracking per asset. Acceptance criteria are written down but **not enforced**. Only two statuses. Reads `presenter.source_media`, which the brief doesn't define |
| `subtitles.py` (4B-0) | SRT + ASS from `words.json`, a neutral placeholder style, and uncertain/unaligned words handled explicitly | Brand typography and colours; suppressing captions during a brand card |
| `assemble_video.py` (4B-1) | Placeholder-only rendering: Pillow frames, ASS rasterised in Pillow, FFmpeg encoding with `speech.wav` muxed as AAC. Transitions: fade in, centred cross-dissolve, hard cut, fade out. It validates against **expected synthetic frames** (PSNR) | It **refuses** any scene whose manifest status isn't `unresolved` (`UNSUPPORTED_IN_4B1`). It has no footage or image input path, no conform step, and no PCM master. Its expected-frame validation **can't be applied unchanged to real footage**, because there's no synthetic reference to compare with |
| Not yet written | — | `media_utils.py`, `visual_assets.py`, `lipsync.py` and `campaign.py` (all proposed in the Phase 4 design) |

### 1.2 Environment

- **FFmpeg / ffprobe 9.0.2** (Homebrew):
  - Decoders: H.264, HEVC, ProRes, VP9, AV1 (libdav1d), PNG, MJPEG, WebP, GIF.
  - Encoders: libx264, AAC, PCM.
  - Filters: `overlay`, `scale`, `select`, `xfade`, `blackdetect`, `freezedetect`.
  - **No libass and no drawtext/freetype** in this build.
- **Pillow 12.3:** freetype, WebP and AVIF support. The system font Helvetica Neue is available.
- **Cached models:** IndicF5, Vocos and whisper-base.en only. **No face, lip-sync or video models.**

### 1.3 Job state

| Job | Phases done | Notes |
|---|---|---|
| `job_20261001_103028_eb8ca2` (6.76 s, 2 scenes) | 1, 2, 3.0 (plan re-validated in memory), 4B-0, 4B-1 | The saved plan predates 3.1. `renders/phase4b1_draft.mp4` is a placeholder draft (1280×720, 25 fps, 169 frames). Both manifest entries are `unresolved` |
| `job_20261001_115730_0b2ea1` (10.43 s, "Health insurance…") | 1 only | Not yet aligned or planned. Known content issue: "the" in "needs it the most" is probably missing (forced-scoring margin −5.3), so a regeneration decision is open |

### 1.4 What's missing for real assets

1. Any way to take in a file and **verify** it: probe, hash, decode test, duration, size, frame rate, audio presence.
2. A conform step: crop, pad or scale; frame-rate conversion; trim; stripping audio.
3. Per-asset tracking for consent, lip-sync, review and hash.
4. A real-asset mode in `assemble_video.py`, with validation against **conformed source frames** rather than synthetic ones.
5. A lip-sync interface (none exists) and a provider decision.
6. Every brand asset except the brand name.
7. A presenter source and consent.

---

## 2. Asset type specifications

The target format for the next stage is 1280×720 at 25 fps, as in 4B-1. Final delivery per the design is 1920×1080. "Required clip" means scene length plus dissolve handles (`timeline.required_clip_s` in the manifest; e.g. S01 = 4.165 s, S02 = 2.945 s).

| | 1. Static image / photo | 2. B-roll video | 3. Presenter footage (no speech sync) | 4. Presenter footage requiring lip-sync | 5. Brand card / end card | 6. Text / graphic scene | 7. Placeholder |
|---|---|---|---|---|---|---|---|
| **Required inputs** | Image file + licence/source note | Video file + licence/source note | Approved presenter clip + **consent record** | Approved presenter clip or portrait + **consent** + exact WAV segment + lip-sync provider decision | **Official logo** + typography + colours (+ disclaimer if required) | Approved text (verbatim from brief/script) + typography + colours | None (generated deterministically) |
| **Accepted formats** | PNG, JPEG, WebP, AVIF (Pillow); sRGB | MP4/MOV with H.264, HEVC, ProRes, VP9 or AV1 | As B-roll | As B-roll; portrait PNG/JPEG if the provider animates stills | Logo: PNG with alpha (preferred) or high-res JPEG. **SVG not supported** (no rasteriser installed) | Rendered by Pillow | Rendered by Pillow |
| **Duration** | Any; held or animated for the exact scene duration | ≥ required clip | ≥ required clip, continuous (no cuts inside) | Source ≥ segment; **lip-sync output = segment length ±1 frame** | Held for the scene | Held for the scene | Exactly the scene |
| **Aspect handling** | Cover-crop at a declared anchor; pad only if the brief says so | Cover-crop at an anchor, else `ASPECT_DECISION_REQUIRED` | Cover-crop; the face must stay inside a declared safe box | As 3; the crop must not cut the mouth or face | Layout composed at target size; the logo is never cropped | Composed at target size | Native |
| **Resolution** | ≥ target size; upscaling ≤ 1.25× gives NEEDS_REVIEW, > 1.5× gives INVALID | Same rule | Same rule. The current candidate is 1138×640: 1.125× to 720p (NEEDS_REVIEW), 1.69× to 1080p (INVALID for final) | Same, plus whatever the provider outputs | Logo ≥ 2× its displayed size | Vector-quality text at target size | Target |
| **Crop / pad** | Crop yes (anchor recorded); pad if the brief allows | Crop yes; pad if the brief allows | Crop limited to keep the face box; no pad unless the brief allows | Same as 3 | Neither on the logo | n/a | n/a |
| **Audio permitted** | n/a | **No.** Stripped at conform; never mixed | **No.** Stripped; `speech.wav` is the only audio | **No** in output. Provider audio, if returned, is used only to check its offset, then discarded | No | No | No |
| **Lip-sync required** | No | No | No; it may only show the presenter *not speaking* (listening, reacting) unless lip-synced | **Yes**: genuine audio-driven lip-sync to the exact segment | No | No | No |
| **Validation** | Decodes; SHA-256; size; EXIF orientation applied; colour mode; no text or logos unless allowed | ffprobe: codec, size, frame rate (constant or variable), duration, rotation, colour space; decode of first and last frame; `blackdetect`; SHA-256 | As B-roll, plus continuity (`freezedetect`, no cuts), face box declared (no detector installed, so a person marks it), consent recorded | As 3, plus §3.4 checks; **human review mandatory** | Logo hash matches the approved file; colours and fonts exactly as given; disclaimer text verbatim | Text verbatim; within safe area; contrast | Watermarked DRAFT; only allowed in `draft` mode |
| **Missing input** | MISSING_INPUTS; placeholder in draft only | MISSING_INPUTS; placeholder in draft only | MISSING_INPUTS (no consent or media) | MISSING_INPUTS | MISSING_INPUTS; **never invent a logo or colours** | MISSING_INPUTS | n/a |

**General rules:**
- **No silent substitution.** Picture is fitted to the audio timeline; nothing is stretched or slowed.
- **Too short means INVALID.** A clip shorter than its required length is INVALID. A freeze-frame extension only happens with an explicit flag, recorded in the report.
- **No hidden fixes.** Variable-frame-rate sources are converted to constant 25 fps, and the dropped or duplicated frames are reported.
- **Re-checked at render time.** Every file's hash is re-verified when rendering, and a changed file is INVALID.

---

## 3. Presenter and lip-sync readiness

### 3.1 What exists

| Item | Status | Evidence |
|---|---|---|
| Presenter source video | **None in the project.** One candidate *outside* it: `~/Downloads/linkedin-video.mp4` | Stream metadata only (no frames decoded): 1138×640, **30 fps**, H.264, 52.6 s. It's the same edited video the voice reference came from: it contains a background bed and an edit at about 13.5 s, and its suitability (continuous face, framing, light) is **unassessed** |
| Reference still image | None | — |
| Consent | **Not given**: `brief.presenter.consent_confirmed = null` | A video existing is **not** consent, and none is inferred |
| Declared presenter media | None: `brief.presenter.source_media` is absent | `asset_manifest.py` already reports it as missing |
| Audio-to-presenter mapping | Defined: S01 ↔ `speech.wav` 0.000–3.990 s (timeline span; narration 0.150–3.640 s) | `asset_manifest.json` S01 `acceptance.lipsync_audio` |
| Lip-sync implementation | **None.** Only flags (`lipsync_required: true`) | No code or model references in `scripts/` beyond `asset_manifest.py` flags |
| Local lip-sync models | **None cached** | Model cache holds IndicF5, Vocos and whisper-base.en only |
| Licensing notes in the project | The Phase 4 design notes that Wav2Lip's pretrained weights are, as far as I know, research/non-commercial; MuseTalk, LatentSync and portrait-animator licences need checking; hosted services upload face and voice | `docs/phase4_visual_pipeline_design.md` §6.3, §12 |

### 3.2 Three different things

1. **Ordinary presenter footage:** the person on screen, with no claim that their mouth matches the narration. They might be listening, walking or working. It's allowed over the narration only if the mouth isn't visibly speaking other words.
2. **Talking-head footage:** the person filmed speaking to camera. If they say *other* words than our narration, putting our WAV over it looks like bad dubbing, so it is **not acceptable** for a speaking scene without step 3.
3. **Audio-driven lip-sync:** mouth motion is **computed from our exact `speech.wav` segment**, either by editing an existing talking-head clip or by animating a portrait. Only this meets `lipsync_required: true`.

For the same reason, a text-to-video "person talking" clip only ever counts as 1 or 2, never 3.

### 3.3 Future interface: `scripts/lipsync.py` (design only)

```
lipsync.py --job <JOB_ID> --scene S01 --provider manual|<local_name>|<hosted_name> [--approved <approval_ref>]
```

| Step | Behaviour |
|---|---|
| Gate | Refuse unless the manifest has `consent.status = CONFIRMED` with an evidence reference, and the provider is approved for this job (`--approved` matches a recorded approval). Non-`manual` providers additionally need the licence and data-handling review recorded |
| Prepare | Cut the exact samples `[start_s × 24000, end_s × 24000)` of `speech.wav` to `lipsync/S01_audio.wav` (PCM) and record its SHA-256. Conform the presenter clip to 25 fps at the target size, with no audio, as `lipsync/S01_input.mp4` |
| Provider contract | `run(audio_wav, presenter_media, params) → output_video`. **`manual`** imports a file you produced elsewhere. Local and hosted adapters come later and are added only after approval |
| Validate | Output duration = segment ±1 frame; frame count = round(duration × 25); constant 25 fps; size as specified. If the provider returned audio: offset 0 ± 1 ms and correlation ≥ 0.99 against the segment, then **discarded**. Mandatory review (`reviewed_by`, `reviewed_at`, verdict) |
| Output | `lipsync/S01_lipsync.mp4`, registered in the manifest as a `lipsync_output` asset with hash, and `lipsync/lipsync_report.json` |
| Never | Mux provider audio; alter `speech.wav`; send media to a service without approval; mark READY without review |

### 3.4 Lip-sync acceptance checks (for when it exists)

- **Timing:** the mouth is closed or neutral in the pauses (0.000–0.150 s and after 3.640 s in S01) and moving during the narration. That's checked by a person, since no detector is installed.
- **No visual damage:** no visible mouth-region seams, flicker or identity drift. Checked by a person, side by side with the source.
- **Duration and audio:** the duration and offset checks from §3.3.

---

## 4. Brand asset gaps (`texts/briefs/prudential_brand_profile.json`)

| Field | Current value | Needed by | Status |
|---|---|---|---|
| `brand_name` | "Prudential Health India" | Brand card text, prompts | Present. Approval to use the name and mark in an ad is a separate legal question |
| `logo_asset_path` | null | S02 brand card / end card | **MISSING**: an official file from Prudential's brand team (PNG with alpha preferred) |
| `brand_colors` | null | Brand card, subtitle highlight, graphics | **MISSING**: official values (hex/RGB), not sampled from the internet |
| `typography` | null | Brand card, subtitles | **MISSING**: font names plus licensed font files. Only Helvetica Neue (system) is available now, as a placeholder |
| `tone` | null | Visual direction, prompts | **MISSING** |
| `visual_restrictions` | null | All scenes | **MISSING** |
| `required_disclaimers` | null | End card, possibly legally required text | **MISSING**: insurance advertising may need specific disclaimers. Needs your compliance input |
| `prohibited_visual_elements` | null | Negative constraints for every visual | **MISSING** |

Nothing was searched for or downloaded, and no value has been invented.

---

## 5. Proposed manifest schema (v2, backward-compatible)

- **Compatibility:** keep `assets/asset_manifest.json` (v1) exactly as written; the planner and renderer already read it. Add a new per-asset tracking file, `assets/asset_status.json` (the name the Phase 4 design already uses), with `schema_version: 2`. The v1 manifest stays the *specification*; v2 records *what was supplied and how it validated*.
- **Mapping:** v1 status `unresolved` corresponds to v2 `MISSING_INPUTS`, and `provided_unvalidated` to `NEEDS_REVIEW` (**never `READY` without validation**).

### 5.1 Statuses and precedence

| Status | Meaning |
|---|---|
| `INVALID` | An automated check failed: unreadable, too short, wrong length, hash changed, too much upscaling, VFR not converted, audio not stripped, frame decode failed |
| `MISSING_INPUTS` | A required input is absent: file, consent, presenter media, brand field, lip-sync output, crop/pad decision |
| `NEEDS_REVIEW` | Automated checks pass, but a person must approve: likeness, lip-sync, crop near the face, upscale ≤ 1.25×, licence/source note, any `provided_unvalidated` file |
| `READY` | All automated checks pass **and** every required review is recorded |

Precedence when combining: **INVALID > MISSING_INPUTS > NEEDS_REVIEW > READY**, both per asset and per scene. A scene is READY only if all its assets are. A final render requires every scene to be READY; draft renders may include placeholders.

### 5.2 Per-asset record

```json
{
  "asset_id": "S01-presenter_source-3fa2c91e",          // <scene>-<role>-<sha256[:8]>; deterministic
  "scene_id": "S01",
  "role": "presenter_source",                           // primary | presenter_source | lipsync_output | logo | font | overlay
  "asset_type": "presenter_footage_lipsync",            // one of the 7 types in §2
  "source": {"kind": "user_provided", "path": "assets/incoming/S01.mp4",
             "provider": "manual", "licence_note": null, "approval_ref": null},
  "file": {"sha256": "…", "bytes": 0, "container": "mp4", "video_codec": "h264", "width": 1138, "height": 640,
           "fps": "30/1", "fps_mode": "constant", "duration_s": 0.0, "rotation": 0, "color_space": "bt709",
           "has_audio": true, "audio_codec": "aac", "has_alpha": false},
  "conform": {"target": [1280, 720], "fps": 25, "scale": 1.125, "crop": {"x": 0, "y": 0, "w": 1138, "h": 640},
              "anchor": "center", "pad": false, "trim_in_s": 0.0, "trim_out_s": 4.165, "audio_stripped": true,
              "output_path": null, "output_sha256": null},
  "consent": {"required": true, "status": "NOT_CONFIRMED", "evidence_ref": null,
              "confirmed_by": null, "confirmed_at": null, "scope": null},
  "lipsync": {"required": true, "status": "NOT_STARTED",
              "segment": {"start_s": 0.0, "end_s": 3.99, "wav_sha256": null},
              "provider": null, "output_asset_id": null, "report": null},
  "validation": {"status": "MISSING_INPUTS",
                 "checks": {"decodes": null, "min_duration": null, "resolution_ok": null, "audio_stripped": null},
                 "issues": [{"severity": "missing_input", "code": "CONSENT_NOT_CONFIRMED", "message": "…"}]},
  "missing_inputs": ["consent.evidence_ref", "lipsync.provider"],
  "review": {"required": true, "reasons": ["presenter likeness", "lip-sync quality", "upscale 1.125x"],
             "reviewed_by": null, "reviewed_at": null, "verdict": null}
}
```

(Comments are explanatory; the real file is strict JSON. The values shown are illustrative, **not** measurements of any real file.)

Scene-level summary in the same file: `{"scene_id", "readiness", "assets": [asset_id…], "blocking": [...]}`, with readiness computed using the precedence rule.

---

## 6. Production readiness matrix (reference job `job_20261001_103028_eb8ca2`)

| Scene | Mode | Required visual asset(s) | Available now | Blocking inputs | Local processing enough? | External service possibly needed? | Approval needed before execution |
|---|---|---|---|---|---|---|---|
| **S01** 0.000–3.990 s ("Hello, this is my first local voice cloning test.") | `talking_head` | Presenter clip ≥ 4.165 s (continuous, face visible) **+** lip-sync output synced to `speech.wav` 0.000–3.990 s | **No** (only an unassessed candidate outside the project) | Consent; choice of presenter media; lip-sync provider; licence/data-handling review; a crop or face-box decision for the 1138×640, 30 fps source | Conform: **yes**. Lip-sync: **no**, since no local model is installed (a download would be needed) | **Possibly**: a hosted lip-sync API if you don't want a local model | **Yes**: consent; media choice; model download *or* hosted service; licence review |
| **S02** 3.990–6.760 s ("Welcome to Prudential Health India!") | `text_graphic` | Brand card ≥ 2.945 s: official logo, typography, colours (+ disclaimer if required) | **No** | Logo file, colours, typography (fonts plus licence), disclaimer decision | **Yes**: Pillow + FFmpeg are enough once assets exist | **No** | **Yes**: official brand assets from you; compliance sign-off on any disclaimer |

**Alternative if S01 shouldn't use a presenter:** a brief `scene_overrides` entry can turn S01 into `voiceover_broll` or `text_graphic`. That changes the plan, so it needs either a new job or `scene_planner.py --overwrite`, which needs your approval. It would remove the consent and lip-sync blockers but still needs B-roll or graphic assets.

**Health-insurance job `job_20261001_115730_0b2ea1`:** not in the matrix yet. It has no alignment, plan or manifest. Decide first whether to regenerate it for the missing "the", then run Phases 2, 3 and 4B-0 on it before assessing its assets.

---

## 7. Recommended implementation sequence

Each step is small, offline unless stated, and stops on failure.

| Step | Scope | Offline? | Approval? |
|---|---|---|---|
| 4B-3 | `media_utils.py` (ffprobe/decode/hash helpers) + `visual_assets.py intake|probe|check` writing `asset_status.json` v2. Statuses only; **no conform yet**. Tested with synthetic files generated locally by FFmpeg/Pillow (test patterns, not real assets) | ✅ | No |
| 4B-4 | `visual_assets.py conform` for types 1, 2, 5, 6 (images, B-roll, brand card, text) → `conformed/Sxx.mp4`; enforced length, crop/pad, constant 25 fps, audio stripped | ✅ | No (tests use synthetic files) |
| 4B-5 | `assemble_video.py` real-asset mode: reads conformed clips; validates decoded output against **conformed source frames** plus blackdetect and audio checks; adds the lossless PCM `final_master.mov`; `final` mode allowed only when every scene is READY | ✅ | No |
| 4B-6 | First real assets: your official logo, colours and fonts → S02 brand card (READY after review) | ✅ | **Yes**: you supply official assets |
| 4B-7 | Presenter intake for S01 (types 3/4) + `lipsync.py` with the `manual` provider only | ✅ | **Yes**: consent + media choice |
| 4B-8 | Lip-sync provider evaluation on the 3.99 s S01 segment | Local ✅ after download / hosted ❌ | **Yes**: download or hosted service, licence and data review |

---

## 8. Risks and validation requirements

| Risk | Mitigation / required validation |
|---|---|
| Treating a placeholder success as production readiness | `final` mode refused unless every scene is READY. Reports label `mode: draft` |
| Wrong or unofficial brand assets | Only files you supply. Hash recorded; the logo is never cropped; colours and fonts used exactly as given; no internet sourcing |
| Likeness misuse | Consent record with evidence and scope required for types 3/4; never inferred from a file existing |
| Fake lip-sync (talking-head clip with other words, or generated "talking" video) | Type 4 requires a registered `lipsync_output` with its segment hash and a recorded review |
| Lip-sync drift or offset | Duration ±1 frame, frame count, provider-audio offset 0 ± 1 ms, human review of pauses |
| Low-resolution presenter source (1138×640) | Upscale limits: NEEDS_REVIEW ≤ 1.25×, INVALID > 1.5×. Fine for a 720p draft, **not** for 1080p final |
| 30 fps source vs 25 fps timeline | Explicit frame-rate conversion, reporting dropped frames (5 of every 30). Lip-sync is done *after* conversion, at 25 fps |
| Edited source (cuts in the candidate video) | The presenter clip must be continuous for the whole scene; `freezedetect`/scene-cut checks plus review |
| Background audio in source footage | Audio always stripped at conform; ffprobe confirms no audio stream in conformed clips |
| Clip too short / wrong aspect | INVALID / MISSING_INPUTS (crop decision); never stretched; freeze only with an explicit flag |
| File changed after validation | SHA-256 re-checked at render time; mismatch makes it INVALID |
| Subtitle legibility over real footage | Contrast check in the subtitle area during 4B-5; brand typography needs licensed font files |
| Validating real-footage renders | Compare decoded output with conformed source frames (PSNR), plus scene identity and order via per-scene frame fingerprints, plus blackdetect and audio checks |
| Privacy of face and voice data | Local processing by default; hosted services only with approval and a data-retention review; no face data in reports beyond hashes and metadata |

---

## 9. Decisions needed from you before any real asset is processed

1. **S01 presenter:** keep `talking_head` (which needs items 2–4 below) **or** change S01 to B-roll or a graphic (which needs a re-plan: a new job or approved `--overwrite`).
2. **Consent:** explicit confirmation (who, scope, date), recorded in the brief, for using your likeness. It is not inferred from `linkedin-video.mp4` existing.
3. **Presenter media:** which file (the LinkedIn video or a new recording), and if the LinkedIn video, which continuous time range. A new recording made for this purpose (good light, 1080p or better, mouth visible, little head movement) is strongly preferable.
4. **Lip-sync route:** `manual` (you supply a lip-synced clip), a local model (download plus licence review; some weights are non-commercial), or a hosted service (paid; uploads your face and voice).
5. **Brand assets:** official logo file, colour values, typography with licensed font files, tone, restrictions, prohibited elements, all from Prudential's brand guidelines.
6. **Disclaimers:** whether any regulatory or compliance text must appear, and its exact wording.
7. **Delivery format:** 720p draft only for now, or a 1080p final, which rules out the current presenter source unless it's replaced.
8. **Health-insurance job:** accept as is, or regenerate once to fix the missing "the", before aligning and planning it.
9. **Refresh the reference job's Phase 3.0 plan** to the 3.1 schema (`--overwrite`), or keep re-validating in memory.

Until these are answered, every real-asset scene stays at `MISSING_INPUTS`, and only draft renders with placeholders are possible.
