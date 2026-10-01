"""Generate speech in the reference voice with IndicF5, fully locally.

    .venv/bin/python scripts/generate.py \\
        --reference reference/my_voice.wav \\
        --transcript reference/my_voice.txt \\
        --text "Hello, this is my first local voice cloning test." \\
        --output outputs/first_test.wav

EXPERIMENTAL for English: English is not one of IndicF5's documented languages.
Your text is passed to the model exactly as typed. It is never translated or rewritten.

Backends:
  official (default)  The documented IndicF5 usage from the upstream README:
                      AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)
                      audio = model(text, ref_audio_path=..., ref_text=...)
                      pinned to the reviewed snapshot revision; device chosen by upstream model.py.
  direct              Same upstream f5_tts functions (load_vocoder, load_model, preprocess_ref_audio_text,
                      infer_process) with the same checkpoint, but with explicit device choice and
                      sampling parameters. Used for --device cpu and for automatic CPU fallback.

Runs offline: weights must already be downloaded by scripts/download_model.py.
"""

import os
import sys
from pathlib import Path

# Inference never touches the network; download_model.py is the only online step.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402  (sets PYTORCH_ENABLE_MPS_FALLBACK before torch loads)
from common import (  # noqa: E402
    OUTPUTS_DIR,
    REPO_ID,
    REPORTS_DIR,
    REVISION,
    TARGET_SR,
    PeakMemorySampler,
    check_memory_before_load,
    inference_lock,
    is_oom_error,
    local_snapshot,
    memory_snapshot,
    pick_device,
    read_transcript,
    timestamp,
    vocoder_dir,
)

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

# F5TTS_Base architecture, identical to upstream api.py / infer_cli.py.
MODEL_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
# Upstream defaults (f5_tts/infer/utils_infer.py).
DEFAULTS = dict(nfe_step=32, cfg_strength=2.0, speed=1.0, sway_sampling_coef=-1.0, cross_fade_duration=0.15)


def _sync(device):
    import torch

    if device == "mps":
        torch.mps.synchronize()


class IndicF5Engine:
    """Loads IndicF5 once; generate() can then be called repeatedly."""

    def __init__(self, device: str = "auto", backend: str = "official", log=print):
        from f5_tts.model.utils import get_tokenizer

        self.log = log
        # CPU is only controllable through the direct backend.
        self.backend = "direct" if device == "cpu" else backend
        self.mem_before_load = check_memory_before_load(log)
        snap = local_snapshot()
        self.vocab_path = snap / "checkpoints" / "vocab.txt"
        self.vocab_map, _ = get_tokenizer(str(self.vocab_path), "custom")

        t0 = time.perf_counter()
        with PeakMemorySampler() as sampler:
            if self.backend == "official":
                self._load_official()
            else:
                self._load_direct(pick_device(device))
            _sync(self.device)
        self.load_seconds = round(time.perf_counter() - t0, 2)
        self.load_memory = sampler.summary()
        log(f"[{self.backend}] model loaded on {self.device} in {self.load_seconds} s ({self.weights_info})")

    def _load_official(self):
        from transformers import AutoModel

        # Exactly the README call, plus the pinned revision so remote code can't change underneath us.
        self.model = AutoModel.from_pretrained(REPO_ID, revision=REVISION, trust_remote_code=True)
        self.model.eval()
        self.vocoder = None
        self.device = str(next(self.model.parameters()).device).split(":")[0]
        self.weights_info = f"AutoModel {type(self.model).__name__}"

    def _load_direct(self, device: str):
        import torch
        from f5_tts.infer.utils_infer import load_model, load_vocoder
        from f5_tts.model import DiT
        from safetensors.torch import load_file

        self.device = device
        snap = local_snapshot()
        self.vocoder = load_vocoder("vocos", is_local=True, local_path=str(vocoder_dir()), device="cpu")
        # Build on CPU, fill weights, then move once: avoids two full copies on the GPU.
        self.model = load_model(DiT, MODEL_CFG, mel_spec_type="vocos", vocab_file=str(self.vocab_path),
                                device="cpu")
        state = load_file(str(snap / "model.safetensors"), device="cpu")
        self.weights_info = self._load_weights(state)
        del state
        gc.collect()
        self.model = self.model.to(torch.float32).to(self.device).eval()
        self.vocoder = self.vocoder.to(self.device).eval()

    def _load_weights(self, state: dict) -> str:
        """Split model.safetensors into the CFM (ema_model.*) and Vocos (vocoder.*) parts.

        The HF repo stores INF5Model's state dict, i.e. its two submodules. Loads with
        strict=True so any key mismatch fails loudly instead of running a half-loaded model.
        """
        cfm, voc, other = {}, {}, []
        for k, v in state.items():
            k2 = k.replace("_orig_mod.", "")  # torch.compile wrapper prefix, if present
            if k2.startswith("ema_model."):
                cfm[k2[len("ema_model."):]] = v
            elif k2.startswith("vocoder."):
                voc[k2[len("vocoder."):]] = v
            else:
                other.append(k)
        if other:
            raise RuntimeError(f"Unrecognised keys in model.safetensors (first 5): {other[:5]}")
        # Same backward-compat patch as upstream load_checkpoint(): these are recomputed buffers.
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            cfm.pop(key, None)
        self.model.load_state_dict(cfm, strict=True)
        if voc:
            self.vocoder.load_state_dict(voc, strict=True)
        return f"{len(cfm)} model tensors, " + (f"{len(voc)} vocoder tensors from checkpoint" if voc
                                                  else "vocoder from charactr/vocos-mel-24khz")

    def unknown_tokens(self, text: str) -> dict:
        from f5_tts.model.utils import convert_char_to_pinyin

        tokens = convert_char_to_pinyin([text])[0]
        unk = [t for t in tokens if t not in self.vocab_map]
        return {"n_tokens": len(tokens), "n_unknown": len(unk), "unknown": sorted(set(unk))}

    def generate(self, ref_audio: Path, ref_text: str, gen_text: str, seed: int = 42,
                 remove_silence: bool = False, **params) -> tuple[np.ndarray, dict]:
        import torch
        from f5_tts.infer.utils_infer import remove_silence_for_generated_wav
        from f5_tts.model.utils import seed_everything

        if not ref_text.strip():
            # Upstream would otherwise download Whisper and transcribe. We require the exact transcript.
            raise ValueError("Reference transcript is empty. Provide the exact words spoken in the recording.")
        if not gen_text.strip():
            raise ValueError("Text to generate is empty.")
        p = {**DEFAULTS, **{k: v for k, v in params.items() if v is not None}}
        if self.backend == "official" and any(p[k] != DEFAULTS[k] for k in DEFAULTS):
            self.log("Note: the official backend uses upstream defaults; sampling overrides need --backend direct.")
            p = dict(DEFAULTS)

        with tempfile.TemporaryDirectory(prefix="indicf5_") as tmp:
            # Normalise any WAV (float, 24-bit, stereo, 48 kHz...) to 24 kHz mono PCM16, which
            # pydub (used by upstream preprocessing) can read without FFmpeg.
            prepared = Path(tmp) / "ref.wav"
            data, sr = sf.read(str(ref_audio), dtype="float32", always_2d=True)
            mono = data.mean(axis=1)
            if sr != TARGET_SR:
                import soxr

                mono = soxr.resample(mono, sr, TARGET_SR, quality="VHQ")
            sf.write(str(prepared), np.clip(mono, -1, 1), TARGET_SR, subtype="PCM_16")

            # Upstream preprocessing writes a trimmed copy of the reference with
            # NamedTemporaryFile(delete=False). Point tempfile at our private dir so that copy
            # of your voice is deleted with it instead of lingering in the system temp dir.
            saved_tempdir, tempfile.tempdir = tempfile.tempdir, tmp
            try:
                seed_everything(seed)
                _sync(self.device)
                t0 = time.perf_counter()
                with PeakMemorySampler() as sampler, torch.inference_mode():
                    if self.backend == "official":
                        wave = self.model(gen_text, ref_audio_path=str(prepared), ref_text=ref_text)
                        out_sr, ref_text_final = TARGET_SR, ref_text  # README: saved at 24000 Hz
                    else:
                        wave, out_sr, ref_text_final = self._infer_direct(prepared, ref_text, gen_text, p)
                    _sync(self.device)
                gen_seconds = time.perf_counter() - t0
            finally:
                tempfile.tempdir = saved_tempdir

            wave = np.asarray(wave)
            if wave.dtype == np.int16:  # README normalisation step
                wave = wave.astype(np.float32) / 32768.0
            wave = wave.astype(np.float32).squeeze()

            if remove_silence:
                out_tmp = Path(tmp) / "out.wav"
                sf.write(str(out_tmp), np.clip(wave, -1, 1), out_sr, subtype="PCM_16")
                remove_silence_for_generated_wav(str(out_tmp))
                wave, out_sr = sf.read(str(out_tmp), dtype="float32")

        audio_s = len(wave) / out_sr
        meta = {
            "backend": self.backend,
            "device": self.device,
            "seed": seed,
            "params": p,
            "remove_silence": remove_silence,
            "reference_text_used": ref_text_final,
            "generation_seconds": round(gen_seconds, 2),
            "output_audio_seconds": round(audio_s, 2),
            "real_time_factor": round(gen_seconds / audio_s, 3) if audio_s else None,
            "memory_during_generation": sampler.summary(),
        }
        return wave, meta

    def _infer_direct(self, prepared: Path, ref_text: str, gen_text: str, p: dict):
        from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

        ref_clip, ref_text_final = preprocess_ref_audio_text(str(prepared), ref_text, show_info=self.log)
        wave, out_sr, _ = infer_process(ref_clip, ref_text_final, gen_text, self.model, self.vocoder,
                                        mel_spec_type="vocos", show_info=self.log, device=self.device, **p)
        return wave, out_sr, ref_text_final

    def release(self):
        import torch

        self.model = self.vocoder = None
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()


def save_wav(wave: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(wave, -1, 1), TARGET_SR, subtype="PCM_16")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, default=common.DEFAULT_REF_AUDIO, help="reference WAV (any rate)")
    ap.add_argument("--transcript", default=str(common.DEFAULT_REF_TEXT),
                    help="path to .txt with the exact words spoken, or the transcript itself")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="text to speak (passed verbatim)")
    g.add_argument("--text-file", type=Path, help="read text to speak from a UTF-8 file")
    ap.add_argument("--output", type=Path, help="output WAV (default outputs/gen-<timestamp>.wav)")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing an existing output file")
    ap.add_argument("--backend", choices=["official", "direct"], default="official",
                    help="official = README AutoModel usage (default); direct = same upstream functions, "
                         "explicit device/params")
    ap.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto",
                    help="auto = MPS, falling back to CPU (direct backend) if generation fails; "
                         "cpu implies --backend direct")
    ap.add_argument("--seed", type=int, default=42, help="fixed by default for comparable runs")
    ap.add_argument("--nfe-step", type=int, default=DEFAULTS["nfe_step"], help="sampling steps (upstream: 32)")
    ap.add_argument("--cfg-strength", type=float, default=DEFAULTS["cfg_strength"])
    ap.add_argument("--speed", type=float, default=DEFAULTS["speed"])
    ap.add_argument("--remove-silence", action="store_true", help="upstream post-trim of long pauses")
    ap.add_argument("--strict-vocab", action="store_true", help="abort if any character is unknown to the vocab")
    ap.add_argument("--benchmark-runs", type=int, default=0,
                    help="extra timed runs after the first (not saved); measures warm MPS speed")
    args = ap.parse_args()

    gen_text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8").strip()
    ref_text = read_transcript(args.transcript)
    out = args.output or OUTPUTS_DIR / f"gen-{timestamp()}.wav"
    if out.exists() and not args.overwrite:
        sys.exit(f"Refusing to overwrite {out}. Pass --overwrite or choose another --output.")
    if not args.reference.is_file():
        sys.exit(f"Reference audio not found: {args.reference}\nRecord it first (README: 'Record your voice').")

    print("EXPERIMENTAL: English is not a documented IndicF5 language; quality is unknown.")
    print(f"Text (verbatim): {gen_text!r}")
    try:
        with inference_lock():
            engine = IndicF5Engine(args.device, args.backend)
            for label, t in (("reference transcript", ref_text), ("text to generate", gen_text)):
                u = engine.unknown_tokens(t)
                if u["n_unknown"]:
                    print(f"WARNING: {u['n_unknown']}/{u['n_tokens']} tokens in the {label} are not in the vocab "
                          f"and will be read as index 0: {u['unknown']}")
                    if args.strict_vocab:
                        sys.exit("Aborted (--strict-vocab). Input was not modified.")
            params = dict(nfe_step=args.nfe_step, cfg_strength=args.cfg_strength, speed=args.speed)
            fallback_reason = None
            try:
                wave, meta = engine.generate(args.reference, ref_text, gen_text, args.seed, args.remove_silence,
                                             **params)
            except Exception as e:
                if engine.device != "mps" or args.device != "auto":
                    raise
                fallback_reason = f"[{engine.backend}/mps] {type(e).__name__}: {e}"
                print(f"MPS generation failed ({fallback_reason[:300]}).\nRetrying once on CPU (direct backend)...")
                engine.release()
                engine = IndicF5Engine("cpu", "direct")
                wave, meta = engine.generate(args.reference, ref_text, gen_text, args.seed, args.remove_silence,
                                             **params)
            save_wav(wave, out)

            bench = []
            for i in range(args.benchmark_runs):
                _, m = engine.generate(args.reference, ref_text, gen_text, args.seed, args.remove_silence, **params)
                bench.append(m["generation_seconds"])
                print(f"  benchmark run {i + 1}: {m['generation_seconds']} s")
    except (common.InferenceBusy, FileNotFoundError, PermissionError) as e:
        sys.exit(str(e))
    except Exception as e:
        if is_oom_error(e):
            sys.exit(f"OUT OF MEMORY: {e}\nClose other apps, try a shorter --text, or use --device cpu.")
        raise

    from validate_audio import audio_stats

    stats = audio_stats(out)
    report = {
        "output": str(out),
        "text": gen_text,
        "reference": str(args.reference),
        "model": f"ai4bharat/IndicF5@{REVISION[:8]}",
        "experimental_english": True,
        "fallback_to_cpu_reason": fallback_reason,
        "model_load_seconds": engine.load_seconds,
        "memory_before_load": engine.mem_before_load,
        "memory_during_load": engine.load_memory,
        "memory_after": memory_snapshot(),
        **meta,
        "benchmark_generation_seconds": bench,
        "output_file": stats,
        "host": {"machine": platform.machine(), "macos": platform.mac_ver()[0]},
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "runs").mkdir(exist_ok=True)
    report_path = REPORTS_DIR / "runs" / f"{out.stem}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print(f"\nSaved {out}")
    print(f"  backend           {meta['backend']}")
    print(f"  device            {meta['device']}{'  (after MPS failure)' if fallback_reason else ''}")
    print(f"  model load        {engine.load_seconds} s")
    print(f"  generation        {meta['generation_seconds']} s for {meta['output_audio_seconds']} s of audio "
          f"(RTF {meta['real_time_factor']}; first run includes MPS warm-up)")
    m = meta["memory_during_generation"]
    print(f"  peak memory       RSS {m['peak_process_rss_gb']} GB, MPS driver {m['peak_mps_driver_gb']} GB, "
          f"min system free {m['min_system_available_gb']} GB")
    print(f"  output            {stats['sample_rate']} Hz, {stats['channels']} ch, {stats['duration_s']} s, "
          f"{stats['file_size_bytes']} bytes, non-silent={stats['non_silent']}")
    print(f"  report            {report_path}")
    print(f"\nListen:  afplay {out}")


if __name__ == "__main__":
    main()
