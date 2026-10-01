# Asset manifest: job_20261001_103028_eb8ca2

- Plan validation: **valid_with_missing_inputs** (re-validated in memory (plan predates Phase 3.1))
- Format: 16:9 1920x1080 @ 25 fps; audio 6.76 s
- Missing inputs: brand_colors, logo_asset_path, presenter.consent_confirmed, presenter.source_media, prohibited_visual_elements, required_disclaimers, tone, typography, visual_restrictions

| Scene | Timeline (s) | Required clip (s) | Mode | Asset | Person | Lip-sync | Status |
|---|---|---|---|---|---|---|---|
| S01 | 0.000-3.990 | 4.165 | talking_head | presenter_video | yes | yes | unresolved |
| S02 | 3.990-6.760 | 2.945 | text_graphic | brand_card | no | no | unresolved |

## S01 — presenter_video

- **Narration:** Hello, this is my first local voice cloning test.
- **Description** (rule: greeting with available presenter): Presenter (the narrator (owner of the reference voice), seated, facing camera) speaking directly to camera
- **Shot / camera:** medium close-up, eye level / static
- **Acceptance:** min_duration_s=4.165; aspect_ratio=16:9; resolution=[1920, 1080]; fps=25; frame_rate_mode=constant; no_audio_track_after_conform=True; no_text_or_logos=True; must_not_stretch=True; lipsync_required=True; lipsync_audio=exact speech.wav segment 0.000-3.990 s; consent_required=True; consent_confirmed=REQUIRES_USER_INPUT; identity_source=REQUIRES_USER_INPUT
- **Source:** unresolved
- **Blockers:**
  - no asset provided yet
  - presenter consent not confirmed (brief.presenter.consent_confirmed)
  - approved presenter media not specified (brief.presenter.source_media)
  - lip-sync against the exact WAV segment not yet produced or validated

## S02 — brand_card

- **Narration:** Welcome to Prudential Health India!
- **Description** (rule: brand-name card): Brand card: Prudential Health India name and logo on a clean background
- **Shot / camera:** full-frame graphic / static
- **Acceptance:** min_duration_s=2.945; aspect_ratio=16:9; resolution=[1920, 1080]; fps=25; frame_rate_mode=constant; no_audio_track_after_conform=True; no_text_or_logos=False; must_not_stretch=True
- **Source:** unresolved
- **Blockers:**
  - no asset provided yet
  - brand logo_asset_path missing
  - brand typography missing
  - brand brand_colors missing

## Brand profile (verbatim)

- brand_name: Prudential Health India
- logo_asset_path: REQUIRES_USER_INPUT
- brand_colors: REQUIRES_USER_INPUT
- typography: REQUIRES_USER_INPUT
- tone: REQUIRES_USER_INPUT
- visual_restrictions: REQUIRES_USER_INPUT
- required_disclaimers: REQUIRES_USER_INPUT
- prohibited_visual_elements: REQUIRES_USER_INPUT
