# IndicF5 Local Voice Test

A local proof of concept that uses **AI4Bharat IndicF5** to generate speech conditioned on a short recording of my own voice.
Everything runs on this Mac. The voice recording never leaves the machine; only the model weights are downloaded from Hugging Face.

- Model: https://huggingface.co/ai4bharat/IndicF5 (snapshot `ba85abed`)
- Code: https://github.com/ai4bharat/IndicF5 (commit `13f7c4d`)

## Status

| Phase | Status |
|---|---|
| 1. Environment validation | Done |
| 2. Isolated installation | Done: `scripts/environment_check.py` passes |
| 3. Pipeline implementation | Done: CLI, UI, vocab audit, audio validation, test sequence |
| 4. HF auth + model download | **Waiting for you:** `hf auth login` |
| 5. Voice recording | **Waiting for you:** record `reference/my_voice.wav` |
| 6. First generation + report | After 4 and 5 |

## Quick start (after setup)

```bash
cd ~/Desktop/indicf5-voice-test
.venv/bin/hf auth login                     # once; accept the model terms on huggingface.co first
.venv/bin/python scripts/download_model.py  # once; ~1.4 GB into ~/.cache/huggingface
# record reference/my_voice.wav (see "Record your voice")
zsh scripts/run_test_sequence.sh            # steps A–J + a Hindi control run
afplay outputs/first_test.wav
```

One-off generation:

```bash
.venv/bin/python scripts/generate.py \
  --reference reference/my_voice.wav \
  --transcript reference/my_voice.txt \
  --text "Hello, this is my first local voice cloning test." \
  --output outputs/first_test.wav
```

Useful flags: `--device cpu` (forces the direct backend), `--backend direct`, `--seed N` (default 42),
`--nfe-step 16` (faster, rougher), `--strict-vocab` (abort on unknown characters), `--overwrite`,
`--benchmark-runs 2` (extra timed runs to measure warm speed). Each run writes `reports/runs/<name>.json`.

Local UI (binds to `127.0.0.1:8501` only, telemetry off via `.streamlit/config.toml`):

```bash
.venv/bin/streamlit run scripts/app.py
```

## Voice generation module (`scripts/voicegen.py`, pipeline Phase 1)

```bash
.venv/bin/python scripts/voicegen.py --input texts/voicegen_test.txt --display-text texts/voicegen_test.en.txt
```

Each run writes `outputs/job_<YYYYmmdd_HHMMSS>_<hex>/` containing `input.txt`, `speech.wav` (24 kHz mono PCM16), `metadata.json` and `generation.log`. The exit code is 0 only if every job succeeds.

- **Input** is passed to IndicF5 verbatim. For English, write the Devanagari phonetic respelling (see `texts/pronunciation_dictionary.json`), and pass the real English wording with `--display-text`. Phase 2 alignment should use `metadata.input.display_text` when it's present.
- **Settings** (`--seed --nfe-step --cfg-strength --speed --sway-sampling-coef --sentence-pause-s --paragraph-pause-s --peak-ceiling-dbtp --strict-vocab`) default to the known-good values: 42 / 32 / 2.0 / 1.0 / −1.0.
- **Chunking:** sentences (after `। ! ? .`) and paragraphs (blank lines) are generated separately, with no cross-fade. They're joined with explicit pauses (0.45 s / 0.70 s).
- **Post-processing:** edge trim and fades outside speech, plus one static gain only if the true peak would exceed −1 dBTP. There's no limiter, compression or denoising.
- **One model load per process.** Pass several `--input` files to reuse it. Python API: `VoiceGenerator(...).generate(text, output_dir, settings, display_text)`.
- **Validation** (`metadata.validation`) is technical only: readable, non-empty, finite, not silent, mono 24 kHz PCM16, no clipping, alignment-ready. It does not verify pronunciation.
- **Errors** produce a job folder with `status: "failed"` and `error.code`. The codes are:
  - Input: `INPUT_MISSING`, `INPUT_EMPTY`, `INPUT_UNKNOWN_TOKENS`, `INVALID_SETTINGS`
  - Reference: `REFERENCE_MISSING`, `REFERENCE_INVALID_FORMAT`, `REFERENCE_BAD_DURATION`, `REFERENCE_TEXT_MISSING`, `REFERENCE_TEXT_EMPTY`
  - Model: `MODEL_INIT_FAILED`, `MODEL_INIT_OOM`
  - Generation: `GENERATION_FAILED`, `GENERATION_OOM`, `GENERATION_INVALID_OUTPUT`
  - Output and environment: `OUTPUT_INVALID`, `DISK_FULL`, `JOB_EXISTS`
- **`metadata.chunks[].start_s/end_s`** are exact chunk placements from assembly, **not** word timestamps.

## Word alignment (`scripts/align_words.py`, pipeline Phase 2)

```bash
.venv/bin/python scripts/align_words.py --job <JOB_ID> [--transcript english.txt] [--overwrite]
```

This writes `outputs/<JOB_ID>/alignment/`:
- `words.json`: per word, its start and end in s and ms, confidence, status and flags.
- `words.csv`
- `subtitles.srt`
- `alignment_report.json`: validation, coverage, gaps and overlaps.

`speech.wav` and `metadata.json` are only read, never changed.

- **Engine:** transcript-constrained alignment with the already-cached `openai/whisper-base.en`. The known English text is fed to the decoder, and its attention is aligned to the audio using the model's published alignment heads (dynamic time warping, the OpenAI `word_timestamps` method). Word edges are then snapped to voiced 10 ms frames. It runs offline on the CPU in a few seconds.
- **Transcript:** `metadata.input.display_text`, or `--transcript`. Devanagari input is rejected. Each sentence is aligned inside its exact Phase 1 chunk span.
- **Confidence:** the geometric-mean probability of the word's spoken (non-punctuation) tokens. Below 0.30, a word is `uncertain`. A word that can't be placed is `unaligned`, with null times rather than invented ones.
- **Precision:** 20 ms resolution (10 ms at word edges). Accuracy against ground truth is **not measured**, and the `*_ms` fields are a format, not a precision claim. A phonetic-level aligner (for example torchaudio MMS_FA) would be more precise, but it needs a model download.

## Scene planning (`scripts/scene_planner.py`, pipeline Phase 3)

```bash
.venv/bin/python scripts/scene_planner.py --job <JOB_ID> --brief texts/briefs/sample_brief.json [--overwrite]
```

This writes `outputs/<JOB_ID>/scene_plan/scene_plan.json` and `scene_plan.md`. It's deterministic and local, with no models and no network. All times come from the Phase 2 word timestamps.

- **Quality gate.** The planner re-validates Phase 2's word times and stops (`ALIGNMENT_INSUFFICIENT`) if:
  - any word has invalid or non-chronological times
  - timestamp coverage is below 95%
  - more than 20% of words are uncertain
  - the first or last word has no timestamp
- **Scenes.** They start as Phase 2 sentences:
  - a sentence longer than `max_scene_s` (8 s) is split at its longest internal pause, never mid-word
  - a unit shorter than `min_scene_s` (1.5 s) is merged with a neighbour
- **Timeline.** Scenes cover the audio contiguously, cutting at the midpoint of the pause between them. `narration_start_s`/`narration_end_s` give the exact first and last word times.
- **Visual mode, in priority order:**
  1. `brief.scene_overrides`
  2. a brand-name card, which becomes `text_graphic`
  3. a greeting when a presenter is available, which becomes `talking_head` (limited by `presenter.max_on_camera_ratio`)
  4. `brief.keyword_visuals`
  5. otherwise a `voiceover_broll` template, flagged for review
- **Brand data.** It's read from the brand profile (`texts/briefs/prudential_brand_profile.json`) and never invented. Missing fields are `REQUIRES_USER_INPUT`, and the affected scene's status becomes `missing_inputs`.
- **Status** (`validation.status`) and exit code:
  - `valid` or `valid_with_missing_inputs` → exit 0
  - `needs_review` → exit 2: uncertain words, a cut with no real silence gap, a scene over `max_scene_s` that can't be split at a word boundary, or template direction
  - `failed` → exit 1: a hard timeline error; **no plan files are written**
  - `validation.ok` means the timeline is valid. Missing brand inputs never make a timeline invalid.
- **`--dry-run`** validates without writing anything. Each scene has `validation_issues` (severity `error`, `review` or `missing_input`), and each boundary between scenes has `boundary_after` (`time_s`, `gap_s`, `safe`, `reason`).

## Scripts

| Script | Step | What it does |
|---|---|---|
| `environment_check.py` | A | versions, imports, MPS ops, memory (offline) |
| `download_model.py` | B, C | checks auth + gated access, downloads pinned snapshot + Vocos |
| `inspect_vocabulary.py` | E | runs the real upstream tokenizer on English, writes `reports/vocabulary_audit.md` |
| `validate_audio.py` | F, I | checks the reference (or `--output-check` a result); `--import` converts m4a/aiff to 24 kHz mono WAV |
| `generate.py` | D, G, H, J | loads the model, generates, saves WAV, reports timing/memory/device |
| `app.py` | — | Streamlit UI: record/upload/select reference, generate, play, download |
| `run_test_sequence.sh` | A–J | everything in order, plus a Hindi control run with upstream's example prompt |

### Two backends, same upstream code

- **official** (default) is the README usage: `AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)`
  then `model(text, ref_audio_path=..., ref_text=...)`. It's pinned to revision `ba85abed` so the remote `model.py` can't change underneath us.
  **Finding (model.py reviewed):** it selects `cuda` or `cpu` only, never MPS. So on this Mac the official path always runs on the CPU. It also wraps the models in `torch.compile`, and afterwards removes silences and normalizes the output to −20 dBFS.
  **For MPS, use `--backend direct --device mps`.** The first real run used that setup.
- **direct** builds the same thing from the pinned `f5_tts` package (`load_vocoder`, `load_model`,
  `preprocess_ref_audio_text`, `infer_process`) with the same `model.safetensors` (strict key match).
  It exists because it lets us choose the device (CPU fallback) and sampling parameters.
  Upstream's own `api.py` can't be used: it passes `ckpt_file` into `load_model`'s `mel_spec_type` slot, and `load_model` never loads weights (`load_checkpoint` is commented out).

Inference runs with `HF_HUB_OFFLINE=1`, so after the download nothing touches the network.
Your reference is converted to 24 kHz mono 16-bit in a private temp folder. That's necessary because pydub, which upstream uses, can't read float WAVs without FFmpeg. The trimmed copy that upstream writes to the system temp dir is redirected there too and deleted afterwards.
Only one generation runs at a time (a file lock is shared by the CLI and the UI).

## ⚠️ English is outside the model's documented languages

IndicF5 officially supports 11 Indian languages: Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil and Telugu.
**English is not on that list.** Using an English reference and English target text is a *controlled compatibility experiment*, and output quality is unknown.

Why this matters technically: IndicF5 turns text into characters and looks each one up in `checkpoints/vocab.txt`.
Any character missing from the vocab is silently mapped to index 0 (`f5_tts/model/utils.py:93`).
If the vocab lacks Latin letters, English text will produce garbled speech.
The vocab is inside the gated repo, so this can only be checked after authentication (the planned first step of Phase 4).

English text is never translated automatically by this project.

## Setup (already done, and reproducible)

Requires [uv](https://docs.astral.sh/uv/) and Homebrew Python 3.12.

```bash
cd ~/Desktop/indicf5-voice-test
uv venv --python /opt/homebrew/bin/python3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/environment_check.py
```

To rebuild with the exact same versions of every package, use `requirements.lock.txt` instead of `requirements.txt`.

Nothing is installed globally. FFmpeg is **not** required (see pins below).
You may see a `pydub` warning, "Couldn't find ffmpeg"; that's expected, because pydub reads WAV files without FFmpeg.

### Why these pins

| Pin | Reason |
|---|---|
| `transformers==4.49.0` | Transformers 4.50 and later breaks IndicF5's custom model loading (meta-tensor init). Upstream `requirements.txt` says `<4.50`, but `setup.py` doesn't pin it, so a plain install gets a broken version. |
| `huggingface-hub<1.0` | Required by transformers 4.49. |
| `torch==2.8.0`, `torchaudio==2.8.0` | From torchaudio 2.9 on, `torchaudio.load()` goes through TorchCodec, which needs FFmpeg. Version 2.8 uses the `soundfile` backend. |
| `numpy<=1.26.4` | Upstream constraint. |
| IndicF5 at commit `13f7c4d` | Reproducible: this is the code that was reviewed. |

## Apple Silicon (MPS) notes

- Upstream picks the device automatically: `cuda`, then `mps`, then `cpu`. On this Mac it chooses `mps`.
- Upstream tries to enable CPU fallback but misspells the variable (`PYTOCH_ENABLE_MPS_FALLBACK`), so it has no effect.
  Our scripts set the correct `PYTORCH_ENABLE_MPS_FALLBACK=1` in code **before importing torch**. Upstream code is not modified.
- float64 is unsupported on MPS. IndicF5 inference uses float32.

There's no `.env` file. The only setting is the variable above, which the scripts set themselves, and the Hugging Face token is managed by the official CLI.

## Hugging Face access (you do this yourself)

The model is gated. Before Phase 4:

1. Log in at huggingface.co, open https://huggingface.co/ai4bharat/IndicF5, and accept the terms.
2. Create a **Read** token at https://huggingface.co/settings/tokens.
3. In your own terminal, run:
   ```bash
   ~/Desktop/indicf5-voice-test/.venv/bin/hf auth login
   ```
   Paste the token at the hidden prompt. Answer **n** to "Add token as git credential?".
   The token is saved to `~/.cache/huggingface/token`, never inside this project.
4. Check with `~/Desktop/indicf5-voice-test/.venv/bin/hf auth whoami`.

Never put the token in source code, in this README, or in a chat.

## Record your voice

Target: **10–15 s**, one speaker, quiet room, no music, natural pace, little silence at either end.
Upstream **clips any reference longer than 15 s**, and the transcript must match what's left.

Read this script, which is already saved in `reference/my_voice.txt`. If you change even one word while speaking, edit the file to match:

> Hello, my name is Vishwak. I am recording this sample to test local artificial intelligence voice cloning. I want the generated speech to sound natural and consistent with my original voice.

**Option 1: browser (easiest).** Run `.venv/bin/streamlit run scripts/app.py`, choose **Record now**, record, then click **Save as reference/my_voice.wav**. Audio stays on localhost.

**Option 2: QuickTime Player** → File → New Audio Recording → record → save (e.g. `~/Desktop/my_voice.m4a`), then:

```bash
.venv/bin/python scripts/validate_audio.py --import ~/Desktop/my_voice.m4a
```

This converts the file to `reference/my_voice.wav` (24 kHz, mono, 16-bit) with macOS's built-in `afconvert` and validates it. It refuses to overwrite an existing recording unless you pass `--force`. Voice Memos exports (m4a) work the same way.

Check any time with `.venv/bin/python scripts/validate_audio.py`.

## Layout

```
reference/     my_voice.wav + my_voice.txt (git-ignored)
outputs/       generated WAVs (git-ignored; never overwritten without --overwrite)
reports/       vocabulary_audit.md/json; runs/*.json per generation (runs/ git-ignored)
scripts/       see table above; common.py holds shared helpers
.streamlit/    localhost-only UI config
requirements.txt        direct pins (+ streamlit for the optional UI)
requirements.lock.txt   exact resolved environment
```
