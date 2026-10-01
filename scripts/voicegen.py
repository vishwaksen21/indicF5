"""Reusable local IndicF5 voice generation (Phase 1 of the video pipeline).

CLI (one process, model loaded once, one job folder per input file):

    .venv/bin/python scripts/voicegen.py --input texts/my_script.txt \\
        [--display-text texts/my_script.en.txt] [--output-dir outputs]

Python:

    from voicegen import VoiceGenerator, GenerationSettings
    gen = VoiceGenerator("reference/my_voice_trimmed.wav", "reference/my_voice_trimmed_devanagari.txt")
    result = gen.generate("हेलो, दिस इज़ ए टेस्ट।", output_dir="outputs", display_text="Hello, this is a test.")
    result.job_dir / "speech.wav"

Each job writes outputs/job_<id>/{input.txt, speech.wav, metadata.json, generation.log}.

Input text is passed to IndicF5 verbatim. For English, it must be the Devanagari phonetic
respelling (Latin-script English is unintelligible with this checkpoint, see README);
pass the real English wording as display_text so Phase 2 alignment/captions can use it.

Generation: the text is split at sentence ends (। ! ? .) and blank lines; every sentence is
generated independently with upstream infer_batch_process (no cross-fade) and joined with
explicit pauses. Post-processing is minimal: edge trim with margins + fades outside speech,
and a single static gain only if the true peak would exceed the ceiling. No limiter,
compression, EQ or denoising.
"""

import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"          # never download: weights must already be cached
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402  (sets PYTORCH_ENABLE_MPS_FALLBACK before torch loads)

import argparse  # noqa: E402
import contextlib  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

SR = common.TARGET_SR
SCHEMA_VERSION = 1
MIN_FREE_DISK_BYTES = 500 * 1024**2
REF_MIN_S, REF_MAX_S = 3.0, 15.0            # upstream clips references at 15 s
SENTENCE_END = re.compile(r"(?<=[।!?.])\s+")
_SECRET = re.compile(r"hf_[A-Za-z0-9]{10,}")


class VoiceGenError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def redact(text: str) -> str:
    return _SECRET.sub("hf_***", str(text))


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclasses.dataclass
class GenerationSettings:
    seed: int = 42
    nfe_step: int = 32
    cfg_strength: float = 2.0
    speed: float = 1.0
    sway_sampling_coef: float = -1.0
    sentence_pause_s: float = 0.45
    paragraph_pause_s: float = 0.70
    clause_pause_s: float = 0.20             # only used if one sentence is too long for one chunk
    peak_ceiling_dbtp: float = -1.0
    strict_vocab: bool = False

    def validate(self):
        checks = [(1 <= self.nfe_step <= 128, "nfe_step must be 1-128"),
                  (0.0 <= self.cfg_strength <= 10.0, "cfg_strength must be 0-10"),
                  (0.5 <= self.speed <= 2.0, "speed must be 0.5-2.0"),
                  (all(0.0 <= p <= 3.0 for p in (self.sentence_pause_s, self.paragraph_pause_s, self.clause_pause_s)),
                   "pauses must be 0-3 s"),
                  (-12.0 <= self.peak_ceiling_dbtp <= 0.0, "peak_ceiling_dbtp must be -12..0")]
        for ok, msg in checks:
            if not ok:
                raise VoiceGenError("INVALID_SETTINGS", msg)


@dataclasses.dataclass
class JobResult:
    job_id: str
    job_dir: Path
    status: str
    metadata: dict


# ---------------------------------------------------------------- text


def split_text(text: str, max_bytes: int) -> list[dict]:
    """Paragraphs (blank lines) -> sentences -> chunks. Returns [{text, pause_after}] where
    pause_after is 'sentence' | 'paragraph' | 'clause' | None (last)."""
    from f5_tts.infer.utils_infer import chunk_text

    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    for pi, para in enumerate(paragraphs):
        sentences = [s for s in SENTENCE_END.split(para) if s.strip()]
        for si, s in enumerate(sentences):
            parts = chunk_text(s, max_chars=max_bytes) if len(s.encode()) > max_bytes else [s]
            for k, part in enumerate(parts):
                last_part = k == len(parts) - 1
                if not last_part:
                    kind = "clause"
                elif si < len(sentences) - 1:
                    kind = "sentence"
                elif pi < len(paragraphs) - 1:
                    kind = "paragraph"
                else:
                    kind = None
                chunks.append({"text": part, "pause_after": kind})
    return chunks


# ---------------------------------------------------------------- audio checks


def validate_wav(path: Path) -> dict:
    """Technical checks only; says nothing about pronunciation."""
    checks = {}
    try:
        info = sf.info(str(path))
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        checks["readable"] = True
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "checks": {"readable": False}, "error": redact(e)}
    mono = data.mean(axis=1) if data.size else np.zeros(0, np.float32)
    frame = int(sr * 0.02)
    n = len(mono) // frame if frame else 0
    fdb = 20 * np.log10(np.sqrt((mono[: n * frame].reshape(n, frame) ** 2).mean(1)) + 1e-10) if n else np.array([])
    voiced = float((fdb > -45).mean()) if n else 0.0
    rms_db = float(20 * np.log10(np.sqrt(np.mean(mono**2)) + 1e-10)) if mono.size else -200.0
    peak = float(np.abs(mono).max()) if mono.size else 0.0
    checks.update({
        "non_empty": info.frames > 0,
        "duration_positive": info.duration > 0,
        "sample_rate_24k": sr == SR,
        "mono": info.channels == 1,
        "pcm16": info.subtype == "PCM_16",
        "finite_values": bool(np.isfinite(data).all()),
        "not_silent": rms_db > -60.0 and peak > 10 ** (-50 / 20),
        "has_speech_frames": voiced >= 0.10,
        "no_clipping": float(np.mean(np.abs(data) >= 0.999)) < 0.001 if data.size else False,
    })
    checks["alignment_ready"] = all(checks[k] for k in ("readable", "non_empty", "finite_values", "not_silent",
                                                         "has_speech_frames", "mono")) and sr >= 16000
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "measurements": {"duration_s": round(info.duration, 3), "sample_rate": sr, "channels": info.channels,
                         "subtype": info.subtype, "frames": info.frames, "rms_dbfs": round(rms_db, 2),
                         "peak_dbfs": round(20 * np.log10(peak + 1e-10), 2), "voiced_fraction": round(voiced, 3)},
    }


# ---------------------------------------------------------------- generator


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(redact(s))

    def flush(self):
        for st in self.streams:
            st.flush()


class VoiceGenerator:
    """Holds one loaded IndicF5 model + one prepared reference; generate() can be called repeatedly."""

    def __init__(self, reference_audio, reference_text, device: str = "mps"):
        self.reference_audio = Path(reference_audio)
        self.reference_text_path = Path(reference_text) if Path(str(reference_text)).suffix == ".txt" else None
        self._reference_text_arg = reference_text
        self.device_pref = device
        self.engine = None
        self._ref = None            # (audio tensor, sr, final ref text, clip seconds)
        self.model_load_seconds = None

    # -- validation that needs no model
    def _check_reference(self, log):
        if not self.reference_audio.is_file():
            raise VoiceGenError("REFERENCE_MISSING", f"Reference audio not found: {self.reference_audio}")
        try:
            info = sf.info(str(self.reference_audio))
        except Exception as e:  # noqa: BLE001
            raise VoiceGenError("REFERENCE_INVALID_FORMAT",
                                f"Reference audio is not a readable audio file ({self.reference_audio.name}): {e}")
        if not REF_MIN_S <= info.duration <= REF_MAX_S + 0.5:
            raise VoiceGenError("REFERENCE_BAD_DURATION",
                                f"Reference is {info.duration:.1f} s; need {REF_MIN_S}-{REF_MAX_S} s "
                                "(upstream clips longer references, breaking transcript alignment)")
        if self.reference_text_path is not None and not self.reference_text_path.is_file():
            raise VoiceGenError("REFERENCE_TEXT_MISSING", f"Reference transcript not found: {self.reference_text_path}")
        text = common.read_transcript(str(self._reference_text_arg))
        if not text:
            raise VoiceGenError("REFERENCE_TEXT_EMPTY",
                                "Reference transcript is empty (automatic transcription is deliberately disabled)")
        log.info("reference: %s (%.2f s, %d Hz, %d ch)", self.reference_audio.name, info.duration,
                 info.samplerate, info.channels)
        return text, info

    def _ensure_model(self, log):
        if self.engine is not None:
            return
        from generate import IndicF5Engine

        try:
            self.engine = IndicF5Engine(self.device_pref, "direct", log=log.info)
        except Exception as e:  # noqa: BLE001
            if common.is_oom_error(e):
                raise VoiceGenError("MODEL_INIT_OOM", f"Out of memory while loading IndicF5: {e}")
            raise VoiceGenError("MODEL_INIT_FAILED", f"IndicF5 failed to initialise: {type(e).__name__}: {e}")
        self.model_load_seconds = self.engine.load_seconds

    def _ensure_reference(self, ref_text, log):
        if self._ref is not None:
            return
        import torchaudio
        from f5_tts.infer.utils_infer import preprocess_ref_audio_text

        with tempfile.TemporaryDirectory(prefix="voicegen_") as tmp:
            prepared = Path(tmp) / "ref.wav"
            data, sr = sf.read(str(self.reference_audio), dtype="float32", always_2d=True)
            mono = data.mean(axis=1)
            if sr != SR:
                import soxr

                mono = soxr.resample(mono, sr, SR, quality="VHQ")
            sf.write(str(prepared), np.clip(mono, -1, 1), SR, subtype="PCM_16")
            saved, tempfile.tempdir = tempfile.tempdir, tmp   # keep upstream's temp copy of the voice private
            try:
                clip, final_text = preprocess_ref_audio_text(str(prepared), ref_text, show_info=log.info)
                audio, a_sr = torchaudio.load(clip)
            finally:
                tempfile.tempdir = saved
        self._ref = (audio, a_sr, final_text, audio.shape[-1] / a_sr)

    def generate(self, text: str, output_dir="outputs", settings: GenerationSettings | None = None,
                 display_text: str | None = None, job_id: str | None = None) -> JobResult:
        settings = settings or GenerationSettings()
        job_id = job_id or f"job_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        job_dir = output_dir / job_id
        if job_dir.exists():
            raise VoiceGenError("JOB_EXISTS", f"{job_dir} already exists")
        job_dir.mkdir()

        log = logging.getLogger(f"voicegen.{job_id}")
        log.setLevel(logging.INFO)
        log.propagate = False
        fh = logging.FileHandler(job_dir / "generation.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        log.addHandler(fh)
        log.addHandler(sh)

        started = datetime.now(timezone.utc)
        t_job = time.perf_counter()
        meta = {
            "schema_version": SCHEMA_VERSION, "job_id": job_id, "status": "failed",
            "created_at": started.astimezone().isoformat(timespec="seconds"),
            "input": {"file": "input.txt", "text": text, "display_text": display_text,
                      "alignment_text_source": "display_text" if display_text else "input.txt"},
            "reference": {"audio_file": self.reference_audio.name, "audio_path": str(self.reference_audio),
                          "transcript_file": self.reference_text_path.name if self.reference_text_path else None},
            "output": {"file": "speech.wav"},
            "model": {"id": "ai4bharat/IndicF5", "revision": common.REVISION, "vocoder": common.VOCODER_REPO,
                      "backend": "direct", "device": None},
            "settings": dataclasses.asdict(settings),
            "environment": {"python": platform.python_version(), "machine": platform.machine(),
                            "macos": platform.mac_ver()[0]},
        }
        try:
            with open(job_dir / "generation.log", "a", encoding="utf-8") as logf, \
                    contextlib.redirect_stdout(_Tee(sys.__stdout__, logf)), \
                    contextlib.redirect_stderr(_Tee(sys.__stderr__, logf)):
                self._run(text, settings, job_dir, meta, log)
            meta["status"] = "success" if meta["validation"]["ok"] else "failed_validation"
            if meta["status"] != "success":
                meta["error"] = {"code": "OUTPUT_INVALID",
                                 "message": "speech.wav failed: " + ", ".join(
                                     k for k, v in meta["validation"]["checks"].items() if not v)}
        except VoiceGenError as e:
            meta["error"] = {"code": e.code, "message": redact(e)}
        except Exception as e:  # noqa: BLE001
            code = "GENERATION_OOM" if common.is_oom_error(e) else "GENERATION_FAILED"
            meta["error"] = {"code": code, "message": redact(f"{type(e).__name__}: {e}")}
        meta["timings"] = {**meta.get("timings", {}), "job_total_seconds": round(time.perf_counter() - t_job, 2)}
        if "error" in meta:
            log.error("%s: %s", meta["error"]["code"], meta["error"]["message"])
        else:
            log.info("success: %s (%.2f s)", job_dir / "speech.wav", meta["output"]["duration_s"])
        (job_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        log.removeHandler(fh)
        log.removeHandler(sh)
        fh.close()
        return JobResult(job_id, job_dir, meta["status"], meta)

    def fail_job(self, output_dir, code: str, message: str) -> JobResult:
        """Record a job that failed before generation could start (e.g. missing input file)."""
        job_id = f"job_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        job_dir = Path(output_dir) / job_id
        job_dir.mkdir(parents=True)
        stamp = datetime.now(timezone.utc).astimezone()
        (job_dir / "generation.log").write_text(f"{stamp.isoformat(timespec='seconds')} ERROR {code}: {redact(message)}\n")
        meta = {"schema_version": SCHEMA_VERSION, "job_id": job_id, "status": "failed",
                "created_at": stamp.isoformat(timespec="seconds"), "error": {"code": code, "message": redact(message)}}
        (job_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ERROR] {code}: {redact(message)}")
        return JobResult(job_id, job_dir, "failed", meta)

    def _run(self, text, settings, job_dir, meta, log):
        import torch
        from f5_tts.infer.utils_infer import infer_batch_process
        from f5_tts.model.utils import convert_char_to_pinyin, seed_everything
        from generate import _sync
        from produce_voiceover import trim, true_peak_db

        settings.validate()
        if not text or not text.strip():
            raise VoiceGenError("INPUT_EMPTY", "Input script is empty")
        (job_dir / "input.txt").write_text(text, encoding="utf-8")
        if shutil.disk_usage(job_dir).free < MIN_FREE_DISK_BYTES:
            raise VoiceGenError("DISK_FULL", f"Less than {MIN_FREE_DISK_BYTES // 1024**2} MB free in {job_dir}")
        ref_text, _ = self._check_reference(log)
        meta["reference"]["audio_sha256"] = sha256(self.reference_audio)
        if self.reference_text_path:
            meta["reference"]["transcript_sha256"] = sha256(self.reference_text_path)

        loaded_now = self.engine is None
        self._ensure_model(log)
        self._ensure_reference(ref_text, log)
        meta["model"]["device"] = self.engine.device
        audio, a_sr, ref_final, ref_s = self._ref
        meta["reference"]["seconds_used"] = round(ref_s, 2)

        # Upstream infer_process chunk-size formula (UTF-8 bytes).
        max_bytes = int(len(ref_final.encode()) / ref_s * (25 - ref_s))
        chunks = split_text(text, max_bytes)
        unknown = sorted({t for c in chunks for t in convert_char_to_pinyin([c["text"]])[0]
                          if t not in self.engine.vocab_map})
        meta["input"]["unknown_tokens"] = unknown
        if unknown:
            log.warning("tokens not in the model vocab (read as index 0): %s", unknown)
            if settings.strict_vocab:
                raise VoiceGenError("INPUT_UNKNOWN_TOKENS", f"Unknown tokens with strict_vocab: {unknown}")
        log.info("%d chunk(s), limit %d bytes", len(chunks), max_bytes)

        params = dict(nfe_step=settings.nfe_step, cfg_strength=settings.cfg_strength, speed=settings.speed,
                      sway_sampling_coef=settings.sway_sampling_coef, cross_fade_duration=0.0)
        waves = []
        with common.inference_lock():
            seed_everything(settings.seed)
            t0 = time.perf_counter()
            with common.PeakMemorySampler() as sampler, torch.inference_mode():
                for i, c in enumerate(chunks, 1):
                    tc = time.perf_counter()
                    w, _, _ = infer_batch_process((audio, a_sr), ref_final, [c["text"]], self.engine.model,
                                                  self.engine.vocoder, mel_spec_type="vocos",
                                                  device=self.engine.device, **params)
                    _sync(self.engine.device)
                    w = np.asarray(w, dtype=np.float32).reshape(-1)
                    if w.size == 0 or not np.isfinite(w).all():
                        raise VoiceGenError("GENERATION_INVALID_OUTPUT", f"chunk {i} produced empty or non-finite audio")
                    c["generation_seconds"] = round(time.perf_counter() - tc, 2)
                    waves.append(w)
                    log.info("chunk %d/%d: %.2f s audio in %.2f s", i, len(chunks), len(w) / SR,
                             c["generation_seconds"])
            gen_seconds = time.perf_counter() - t0

        pauses = {"sentence": settings.sentence_pause_s, "paragraph": settings.paragraph_pause_s,
                  "clause": settings.clause_pause_s, None: 0.0}
        from produce_voiceover import HEAD_MARGIN_S, TAIL_MARGIN_S

        margin = HEAD_MARGIN_S + TAIL_MARGIN_S
        parts, pos = [np.zeros(int(0.10 * SR), np.float32)], int(0.10 * SR)
        for c, w in zip(chunks, waves):
            y, info = trim(w)
            c["start_s"] = round((pos + int(HEAD_MARGIN_S * SR)) / SR, 3)
            parts.append(y)
            pos += len(y)
            c["end_s"] = round((pos - int(TAIL_MARGIN_S * SR)) / SR, 3)
            c["model_audio_ended_abruptly"] = info["possible_truncated_ending"]
            c["pause_after_s"] = pauses[c["pause_after"]]
            gap = int(max(0.0, c["pause_after_s"] - margin) * SR) if c["pause_after"] else 0
            parts.append(np.zeros(gap, np.float32))
            pos += gap
        parts.append(np.zeros(int(max(0.0, 0.30 - TAIL_MARGIN_S) * SR), np.float32))
        mix = np.concatenate(parts)

        tp = true_peak_db(mix)
        gain_db = 0.0
        if tp > settings.peak_ceiling_dbtp:          # single static gain: no limiter, no compression
            gain_db = settings.peak_ceiling_dbtp - tp - 0.05
            mix = (mix * 10 ** (gain_db / 20)).astype(np.float32)
        if not np.isfinite(mix).all() or np.abs(mix).max() >= 1.0:
            raise VoiceGenError("OUTPUT_INVALID", "assembled audio is non-finite or exceeds full scale")

        tmp_path = job_dir / ".speech.tmp.wav"
        sf.write(str(tmp_path), mix, SR, subtype="PCM_16")
        os.replace(tmp_path, job_dir / "speech.wav")

        v = validate_wav(job_dir / "speech.wav")
        meta["validation"] = v
        meta["output"].update({k: v["measurements"][k] for k in ("sample_rate", "channels", "duration_s", "subtype")})
        meta["output"]["bytes"] = (job_dir / "speech.wav").stat().st_size
        meta["output"]["true_peak_dbtp"] = round(true_peak_db(mix), 2)
        meta["post_processing"] = {"chunk_join": "independent sentence chunks, no cross-fade, explicit pauses",
                                   "trim_margins_s": [HEAD_MARGIN_S, TAIL_MARGIN_S],
                                   "static_gain_db": round(gain_db, 2), "limiter": False,
                                   "compression": False, "denoise": False}
        meta["chunks"] = [{"index": i, "text": c["text"], "start_s": c["start_s"], "end_s": c["end_s"],
                           "pause_after_s": c["pause_after_s"], "generation_seconds": c["generation_seconds"],
                           "model_audio_ended_abruptly": c["model_audio_ended_abruptly"]}
                          for i, c in enumerate(chunks, 1)]
        meta["chunks_note"] = ("start_s/end_s are exact chunk placements from assembly (speech span incl. no "
                               "margins). They are NOT word-level timestamps; Phase 2 must align words itself.")
        meta["timings"] = {"model_load_seconds": self.model_load_seconds if loaded_now else 0.0,
                           "model_loaded_in_this_job": loaded_now, "generation_seconds": round(gen_seconds, 2)}
        meta["memory"] = sampler.summary()


# ---------------------------------------------------------------- CLI


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, action="append", required=True,
                    help="UTF-8 script file (repeatable; model is loaded once for all)")
    ap.add_argument("--display-text", type=Path, help="English wording of the script (single --input only)")
    ap.add_argument("--reference", type=Path, default=common.REFERENCE_DIR / "my_voice_trimmed.wav")
    ap.add_argument("--reference-text", default=str(common.REFERENCE_DIR / "my_voice_trimmed_devanagari.txt"))
    ap.add_argument("--output-dir", type=Path, default=common.OUTPUTS_DIR)
    ap.add_argument("--device", choices=["mps", "cpu"], default="mps")
    d = GenerationSettings()
    for f in dataclasses.fields(GenerationSettings):
        name = "--" + f.name.replace("_", "-")
        if f.type in (bool, "bool"):
            ap.add_argument(name, action="store_true")
        else:
            ap.add_argument(name, type=type(getattr(d, f.name)), default=getattr(d, f.name))
    args = ap.parse_args(argv)
    if args.display_text and len(args.input) > 1:
        ap.error("--display-text works with a single --input")

    settings = GenerationSettings(**{f.name: getattr(args, f.name) for f in dataclasses.fields(GenerationSettings)})
    gen = VoiceGenerator(args.reference, args.reference_text, device=args.device)
    results = []
    for inp in args.input:
        text, display = None, None
        if inp.is_file():
            text = inp.read_text(encoding="utf-8")
        if args.display_text:
            display = args.display_text.read_text(encoding="utf-8").strip() if args.display_text.is_file() else None
        if text is None:
            res = gen.fail_job(args.output_dir, "INPUT_MISSING", f"Input script not found: {inp}")
            results.append(res)
            continue
        if args.display_text and display is None:
            print(f"[ERROR] display text not found: {args.display_text}")
            sys.exit(2)
        results.append(gen.generate(text, args.output_dir, settings, display_text=display))
    for r in results:
        print(f"{r.status.upper():18} {r.job_dir}")
    sys.exit(0 if all(r.status == "success" for r in results) else 1)


if __name__ == "__main__":
    main()
