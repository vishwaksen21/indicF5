"""Like generate.py, but chunks the text only at sentence ends.

Upstream infer_process -> chunk_text splits after , ; : . ! ? but never after the Devanagari
danda (।), so long Devanagari text gets cut mid-sentence at commas. This script groups whole
sentences (split after । ! ? .) into chunks within upstream's own size limit, then calls
upstream infer_batch_process with those chunks. Everything else matches generate.py
(direct backend, same weights, reference handling, cross-fade, silence removal, report).

    .venv/bin/python scripts/generate_chunks.py \\
        --reference reference/my_voice_trimmed.wav \\
        --transcript reference/my_voice_trimmed_devanagari.txt \\
        --text-file script.txt --output outputs/narration.wav
"""

import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402  (sets PYTORCH_ENABLE_MPS_FALLBACK before torch loads)
from generate import DEFAULTS, IndicF5Engine, _sync, save_wav  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

SENTENCE_END = re.compile(r"(?<=[।!?.])\s+")


def sentence_chunks(text: str, max_bytes: int) -> tuple[list[str], list[str]]:
    """Greedily pack whole sentences into chunks of at most max_bytes (UTF-8).
    A single sentence longer than the limit falls back to upstream chunk_text."""
    from f5_tts.infer.utils_infer import chunk_text

    chunks, notes, cur = [], [], ""
    for s in SENTENCE_END.split(" ".join(text.split())):
        if len(s.encode()) > max_bytes:
            if cur:
                chunks.append(cur)
                cur = ""
            parts = chunk_text(s, max_chars=max_bytes)
            notes.append(f"sentence over {max_bytes} B split by upstream chunk_text into {len(parts)} parts")
            chunks.extend(parts)
            continue
        cand = f"{cur} {s}".strip()
        if cur and len(cand.encode()) > max_bytes:
            chunks.append(cur)
            cur = s
        else:
            cur = cand
    if cur:
        chunks.append(cur)
    return chunks, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, default=common.DEFAULT_REF_AUDIO)
    ap.add_argument("--transcript", default=str(common.DEFAULT_REF_TEXT))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--text-file", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", choices=["mps", "cpu"], default="mps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nfe-step", type=int, default=DEFAULTS["nfe_step"])
    ap.add_argument("--cfg-strength", type=float, default=DEFAULTS["cfg_strength"])
    ap.add_argument("--speed", type=float, default=DEFAULTS["speed"])
    ap.add_argument("--remove-silence", action="store_true")
    args = ap.parse_args()

    if args.output.exists():
        sys.exit(f"Refusing to overwrite {args.output}.")
    gen_text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
    ref_text = common.read_transcript(args.transcript)
    p = {**DEFAULTS, "nfe_step": args.nfe_step, "cfg_strength": args.cfg_strength, "speed": args.speed}

    import torch
    import torchaudio
    from f5_tts.infer.utils_infer import infer_batch_process, preprocess_ref_audio_text, remove_silence_for_generated_wav
    from f5_tts.model.utils import seed_everything

    with common.inference_lock():
        engine = IndicF5Engine(args.device, "direct")
        unk = engine.unknown_tokens(" ".join(gen_text.split()))
        if unk["n_unknown"]:
            print(f"WARNING: {unk['n_unknown']} unknown tokens (read as index 0): {unk['unknown']}")

        with tempfile.TemporaryDirectory(prefix="indicf5_") as tmp:
            # Same reference normalisation and temp-file containment as generate.py.
            prepared = Path(tmp) / "ref.wav"
            data, sr = sf.read(str(args.reference), dtype="float32", always_2d=True)
            mono = data.mean(axis=1)
            if sr != common.TARGET_SR:
                import soxr

                mono = soxr.resample(mono, sr, common.TARGET_SR, quality="VHQ")
            sf.write(str(prepared), np.clip(mono, -1, 1), common.TARGET_SR, subtype="PCM_16")
            saved_tempdir, tempfile.tempdir = tempfile.tempdir, tmp
            try:
                ref_clip, ref_text_final = preprocess_ref_audio_text(str(prepared), ref_text)
                audio, a_sr = torchaudio.load(ref_clip)
                # Upstream's chunk-size formula from infer_process().
                max_bytes = int(len(ref_text_final.encode()) / (audio.shape[-1] / a_sr) * (25 - audio.shape[-1] / a_sr))
                chunks, notes = sentence_chunks(gen_text, max_bytes)
                print(f"{len(chunks)} sentence-aligned chunks (limit {max_bytes} B):")
                for i, c in enumerate(chunks, 1):
                    print(f"  #{i} [{len(c.encode())} B] {c}")
                for n in notes:
                    print(f"  note: {n}")

                seed_everything(args.seed)
                _sync(engine.device)
                t0 = time.perf_counter()
                with common.PeakMemorySampler() as sampler, torch.inference_mode():
                    wave, out_sr, _ = infer_batch_process(
                        (audio, a_sr), ref_text_final, chunks, engine.model, engine.vocoder,
                        mel_spec_type="vocos", device=engine.device, **p,
                    )
                    _sync(engine.device)
                gen_seconds = time.perf_counter() - t0
            finally:
                tempfile.tempdir = saved_tempdir

            if args.remove_silence:
                out_tmp = Path(tmp) / "out.wav"
                sf.write(str(out_tmp), np.clip(wave, -1, 1), out_sr, subtype="PCM_16")
                remove_silence_for_generated_wav(str(out_tmp))
                wave, out_sr = sf.read(str(out_tmp), dtype="float32")

    save_wav(np.asarray(wave, dtype=np.float32), args.output)
    from validate_audio import audio_stats

    stats = audio_stats(args.output)
    audio_s = stats["duration_s"]
    report = {
        "output": str(args.output),
        "text": gen_text,
        "chunks": chunks,
        "chunking": "sentence-aligned (scripts/generate_chunks.py)",
        "chunking_notes": notes,
        "reference": str(args.reference),
        "reference_text_used": ref_text_final,
        "model": f"ai4bharat/IndicF5@{common.REVISION[:8]}",
        "backend": "direct",
        "device": engine.device,
        "seed": args.seed,
        "params": p,
        "remove_silence": args.remove_silence,
        "model_load_seconds": engine.load_seconds,
        "generation_seconds": round(gen_seconds, 2),
        "output_audio_seconds": audio_s,
        "real_time_factor": round(gen_seconds / audio_s, 3) if audio_s else None,
        "memory_before_load": engine.mem_before_load,
        "memory_during_load": engine.load_memory,
        "memory_during_generation": sampler.summary(),
        "memory_after": common.memory_snapshot(),
        "output_file": stats,
    }
    (common.REPORTS_DIR / "runs").mkdir(parents=True, exist_ok=True)
    report_path = common.REPORTS_DIR / "runs" / f"{args.output.stem}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    m = report["memory_during_generation"]
    print(f"\nSaved {args.output}")
    print(f"  device       {engine.device} (direct)   model load {engine.load_seconds} s")
    print(f"  generation   {report['generation_seconds']} s for {audio_s} s of audio (RTF {report['real_time_factor']})")
    print(f"  peak memory  RSS {m['peak_process_rss_gb']} GB, MPS driver {m['peak_mps_driver_gb']} GB, "
          f"min system free {m['min_system_available_gb']} GB")
    print(f"  report       {report_path}")


if __name__ == "__main__":
    main()
