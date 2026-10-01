# Scene plan: job_20261001_103028_eb8ca2

- Audio: 6.760 s; target: 8.0 s. audio is 6.76 s vs target 8.0 s (-1.24 s); audio is not stretched. Any intro/outro padding is an editorial decision.
- Alignment: whisper-cross-attention-dtw (openai/whisper-base.en, transformers) — coverage 100.0%, 0 uncertain/unaligned word(s)
- Validation: PASS; scenes needing review: S01, S02
- Missing inputs: logo_asset_path, brand_colors, typography, tone, visual_restrictions, required_disclaimers, prohibited_visual_elements, presenter.consent_confirmed

| Scene | Start (s) | End (s) | Duration (s) | Words | Mode | Speaker | Narration |
|---|---|---|---|---|---|---|---|
| S01 | 0.000 | 3.990 | 3.990 | 0-8 | talking_head | yes | Hello, this is my first local voice cloning test. |
| S02 | 3.990 | 6.760 | 2.770 | 9-13 | text_graphic | no | Welcome to Prudential Health India! |

## S01 — talking_head (0.000–3.990 s)

- **Narration** (0.150–3.640 s): Hello, this is my first local voice cloning test.
- **Visual** (rule: greeting with available presenter): Presenter (the narrator (owner of the reference voice), seated, facing camera) speaking directly to camera
- **Shot / camera:** medium close-up, eye level / static
- **Transition out:** cross-dissolve 0.35 s centred in the 0.70 s pause (in: fade in from black)
- **Assets:** presenter_video [REQUIRES_USER_INPUT]; lip_sync_audio [available]; brand_colors [REQUIRES_USER_INPUT]
- **Prompt:** Presenter (the narrator (owner of the reference voice), seated, facing camera) speaking directly to camera. Shot: medium close-up, eye level; camera: static. Style: clean, bright, modern corporate; natural light; uncluttered backgrounds. Aspect ratio 16:9, duration 3.99 s. No on-screen text, captions or logos. Mouth clearly visible, minimal head motion (lip-sync applied later).
- **Status:** needs_review — missing input: presenter_video; missing input: brand_colors

## S02 — text_graphic (3.990–6.760 s)

- **Narration** (4.340–6.450 s): Welcome to Prudential Health India!
- **Visual** (rule: brand-name card): Brand card: Prudential Health India name and logo on a clean background
- **Shot / camera:** full-frame graphic / static
- **Transition out:** fade out to black (after narration ends)
- **Assets:** logo [REQUIRES_USER_INPUT]; typography [REQUIRES_USER_INPUT]; brand_colors [REQUIRES_USER_INPUT]
- **Prompt:** Brand card: Prudential Health India name and logo on a clean background. Shot: full-frame graphic; camera: static. Style: clean, bright, modern corporate; natural light; uncluttered backgrounds. Aspect ratio 16:9, duration 2.77 s.
- **Status:** needs_review — missing input: logo; missing input: typography; missing input: brand_colors

Timing precision: Timing resolution 20 ms (Whisper encoder frames), edges snapped on 10 ms energy frames. Accuracy vs ground truth not measured; *_ms values are formatting, not ms precision.
