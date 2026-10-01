"""Produce a narration from a sentence-chunk plan (texts/*_plan.json) with IndicF5.

    .venv/bin/python scripts/produce_voiceover.py --plan texts/prudential_40sec_v2_plan.json --preflight-only
    .venv/bin/python scripts/produce_voiceover.py --plan texts/prudential_40sec_v2_plan.json \\
        --output outputs/prudential_40sec_v2.wav

One job = one model load; each chunk is generated independently with upstream
infer_batch_process (a single-chunk batch, so no cross-fade), then:
  trim edge silence with margins -> short edge fades -> per-chunk level match (<= +/-3 dB)
  -> join with the plan's explicit pauses -> global gain + soft look-ahead peak limiter
  -> true-peak check (4x oversampled) -> lead-in / tail -> 24 kHz mono PCM16.
Raw chunks are kept next to the output so audio fixes never need a regeneration.
"""

import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402  (sets PYTORCH_ENABLE_MPS_FALLBACK before torch loads)

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

SR = common.TARGET_SR
FRAME = int(SR * 0.01)
HEAD_MARGIN_S, TAIL_MARGIN_S = 0.05, 0.15   # kept around detected speech so onsets/releases survive
FADE_IN_S, FADE_OUT_S = 0.04, 0.08          # fades live inside the margins, never on detected speech
EDGE_FADE_IN_S, EDGE_FADE_OUT_S = 0.005, 0.015  # only where the model's own audio starts/ends abruptly
TARGET_ACTIVE_RMS_DB = -18.0                # speech-frame RMS of the final mix
LIMIT_THRESHOLD_DB = -1.5                   # sample-peak ceiling for the limiter
TRUE_PEAK_MAX_DB = -1.0
MAX_CHUNK_GAIN_DB = 3.0


def db(x):
    return 20 * np.log10(np.maximum(x, 1e-10))


def frame_db(x):
    n = len(x) // FRAME
    return db(np.sqrt((x[: n * FRAME].reshape(n, FRAME) ** 2).mean(1)))


# ---------------------------------------------------------------- preflight


def preflight(plan: dict, vocab: set) -> dict:
    from f5_tts.model.utils import convert_char_to_pinyin

    approved = " ".join(plan["approved_english_script"].split())
    joined = " ".join(c["english"] for c in plan["chunks"])
    words = lambda s: re.findall(r"[\w']+", s)  # noqa: E731
    checks, per_chunk = [], []
    checks.append({"check": "chunks reproduce approved English script exactly", "ok": joined == approved})
    for c in plan["chunks"]:
        dev = unicodedata.normalize("NFC", c["devanagari"])
        toks = convert_char_to_pinyin([dev])[0]
        unk = sorted({t for t in toks if t not in vocab})
        en_w, dv_w = words(c["english"]), dev.replace("।", " ").split()
        dv_w = [w.strip(",!:।") for w in dv_w if w.strip(",!:।")]
        plain_pha = [w for w in dv_w if re.search("फ(?!़)", w)]  # फ without nukta = aspirated p
        per_chunk.append({
            "id": c["id"], "english": c["english"], "devanagari": dev,
            "english_words": len(en_w), "devanagari_words": len(dv_w),
            "word_count_match": len(en_w) == len(dv_w),
            "tokens": len(toks), "unknown_tokens": unk,
            "ends_with_sentence_punct": dev.rstrip()[-1] in "।!",
            "plain_pha_words": plain_pha,
            "pause_after_s": c["pause_after_s"],
        })
    checks.append({"check": "every chunk has a 1:1 word mapping", "ok": all(p["word_count_match"] for p in per_chunk)})
    checks.append({"check": "no unknown tokens", "ok": all(not p["unknown_tokens"] for p in per_chunk)})
    checks.append({"check": "every chunk ends with । or !", "ok": all(p["ends_with_sentence_punct"] for p in per_chunk)})
    checks.append({"check": "no plain फ (aspirated p) where f is intended", "ok": all(not p["plain_pha_words"] for p in per_chunk)})
    brand = [p["devanagari"].count("प्रूडेंशियल") for p in per_chunk]
    en_brand = [p["english"].count("Prudential") for p in per_chunk]
    checks.append({"check": "brand spelled identically wherever it occurs", "ok": brand == en_brand})
    art = [(p["id"], p["english"].split().count("a"), p["devanagari"].split().count("ए")) for p in per_chunk]
    checks.append({"check": "every article 'a' maps to ए", "ok": all(a == e for _, a, e in art)})
    checks.append({"check": "no lone अ word left", "ok": all("अ" not in p["devanagari"].split() for p in per_chunk)})
    return {"ok": all(c["ok"] for c in checks), "checks": checks, "chunks": per_chunk}


# ---------------------------------------------------------------- audio processing


def trim(x: np.ndarray) -> tuple[np.ndarray, dict]:
    e = frame_db(x)
    thr = max(float(np.percentile(e, 95)) - 35.0, -55.0)
    voiced = np.where(e > thr)[0]
    if not len(voiced):
        raise RuntimeError("chunk is silent")
    on, off = voiced[0] * FRAME, (voiced[-1] + 1) * FRAME
    tail_room = (len(x) - off) / SR
    a = max(0, on - int(HEAD_MARGIN_S * SR))
    b = min(len(x), off + int(TAIL_MARGIN_S * SR))
    y = x[a:b].copy()
    # If the model's audio itself starts/ends abruptly (no margin left), fade the real audio edge
    # so the join to padded silence is not a hard step (v2 audit: chunk 4 ended at speech level).
    if a == 0:
        n = int(EDGE_FADE_IN_S * SR)
        y[:n] *= np.linspace(0, 1, n, dtype=np.float32)
    if b == len(x):
        n = int(EDGE_FADE_OUT_S * SR)
        y[-n:] *= 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, n, dtype=np.float32))
    # pad with silence if the model gave us less margin than we want
    pre = int(HEAD_MARGIN_S * SR) - (on - a)
    post = int(TAIL_MARGIN_S * SR) - (b - off)
    y = np.concatenate([np.zeros(max(pre, 0), np.float32), y, np.zeros(max(post, 0), np.float32)])
    fi, fo = int(FADE_IN_S * SR), int(FADE_OUT_S * SR)
    y[:fi] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fi))
    y[-fo:] *= 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, fo))
    act = e[e > thr]
    return y, {
        "raw_s": round(len(x) / SR, 3), "trimmed_s": round(len(y) / SR, 3),
        "speech_s": round((off - on) / SR, 3), "threshold_db": round(thr, 1),
        "model_tail_after_speech_s": round(tail_room, 3),
        "possible_truncated_ending": bool(tail_room < 0.03),
        "end_level_re_speech_db": round(float(e[-1] - np.percentile(e, 95)), 1),
        "speech_level_cut_at_end": bool(tail_room < 0.03 and e[-1] > np.percentile(e, 95) - 20),
        "edge_fade_applied": {"start": bool(a == 0), "end": bool(b == len(x))},
        "active_rms_db": round(float(10 * np.log10(np.mean(10 ** (act / 10)))), 2),
    }


def soft_limit(x: np.ndarray, ceiling_db: float) -> tuple[np.ndarray, int]:
    from scipy.ndimage import minimum_filter1d, uniform_filter1d

    ceil = 10 ** (ceiling_db / 20)
    need = np.minimum(1.0, ceil / np.maximum(np.abs(x), 1e-9))
    la = int(0.004 * SR)  # 4 ms look-ahead/hold
    g = minimum_filter1d(need, size=2 * la + 1)
    g = uniform_filter1d(g, size=la)  # smoothing narrower than the hold: still <= need at every peak
    rel = np.exp(-1.0 / (0.06 * SR))  # 60 ms release
    out = np.empty_like(g)
    cur = 1.0
    for i, v in enumerate(g):
        cur = v if v < cur else v + (cur - v) * rel
        out[i] = cur
    return (x * out).astype(np.float32), int((out < 0.999).sum())


def true_peak_db(x: np.ndarray) -> float:
    import soxr

    return float(db(np.abs(soxr.resample(x, SR, SR * 4, quality="VHQ")).max()))


def mix_chunks(plan: dict, pf_chunks: list, chunk_meta: list, raw_chunks: list):
    """Deterministic post-processing of raw chunks into the final mix. Returns (mix, post_processing)."""
    trimmed = []
    for m, w in zip(chunk_meta, raw_chunks):
        y, info = trim(w)
        m.update(info)
        trimmed.append(y)
    ref_level = float(np.median([m["active_rms_db"] for m in chunk_meta]))
    for m, y in zip(chunk_meta, trimmed):
        gdb = float(np.clip(ref_level - m["active_rms_db"], -MAX_CHUNK_GAIN_DB, MAX_CHUNK_GAIN_DB))
        m["level_match_gain_db"] = round(gdb, 2)
        y *= 10 ** (gdb / 20)

    margin = HEAD_MARGIN_S + TAIL_MARGIN_S
    parts = [np.zeros(int(plan["lead_in_s"] * SR), np.float32)]
    for c, m, y in zip(pf_chunks, chunk_meta, trimmed):
        m["start_s"] = round(sum(len(p) for p in parts) / SR + HEAD_MARGIN_S, 3)
        parts.append(y)
        m["end_s"] = round(sum(len(p) for p in parts) / SR - TAIL_MARGIN_S, 3)
        if c["pause_after_s"] > 0:
            parts.append(np.zeros(int(max(0.0, c["pause_after_s"] - margin) * SR), np.float32))
    parts.append(np.zeros(int(max(0.0, plan["tail_s"] - TAIL_MARGIN_S) * SR), np.float32))
    mix = np.concatenate(parts)

    e = frame_db(mix)
    act = e[e > np.percentile(e, 95) - 35]
    mix_active = float(10 * np.log10(np.mean(10 ** (act / 10))))
    global_gain_db = TARGET_ACTIVE_RMS_DB - mix_active
    mix = (mix * 10 ** (global_gain_db / 20)).astype(np.float32)
    mix, limited = soft_limit(mix, LIMIT_THRESHOLD_DB)
    tp = true_peak_db(mix)
    extra = 0.0
    if tp > TRUE_PEAK_MAX_DB:  # deterministic trim for inter-sample overs
        extra = TRUE_PEAK_MAX_DB - tp - 0.05
        mix = (mix * 10 ** (extra / 20)).astype(np.float32)
        tp = true_peak_db(mix)
    if np.abs(mix).max() >= 1.0:
        raise RuntimeError("mix still exceeds full scale after limiting")

    return mix, {
        "chunk_join": "independent chunks, no cross-fade, explicit pauses",
        "trim_margins_s": [HEAD_MARGIN_S, TAIL_MARGIN_S], "edge_fades_s": [FADE_IN_S, FADE_OUT_S],
        "abrupt_edge_fades_s": [EDGE_FADE_IN_S, EDGE_FADE_OUT_S],
        "chunk_level_reference_db": round(ref_level, 2), "max_chunk_gain_db": MAX_CHUNK_GAIN_DB,
        "global_gain_db": round(global_gain_db, 2), "limiter_ceiling_db": LIMIT_THRESHOLD_DB,
        "limiter_active_samples": limited, "true_peak_trim_db": round(extra, 2),
        "final_true_peak_dbtp": round(tp, 2), "upstream_remove_silence": False,
    }


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--reference", type=Path, default=common.REFERENCE_DIR / "my_voice_trimmed.wav")
    ap.add_argument("--transcript", default=str(common.REFERENCE_DIR / "my_voice_trimmed_devanagari.txt"))
    ap.add_argument("--output", type=Path)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nfe-step", type=int, default=32)
    ap.add_argument("--cfg-strength", type=float, default=2.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    args = ap.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    snap = common.local_snapshot()
    vocab = {l[:-1] for l in open(snap / "checkpoints" / "vocab.txt", encoding="utf-8")}
    pf = preflight(plan, vocab)
    pf_path = common.REPORTS_DIR / f"{plan['name']}_preflight.json"
    pf_path.write_text(json.dumps(pf, ensure_ascii=False, indent=2))
    for c in pf["checks"]:
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['check']}")
    print(f"Preflight {'OK' if pf['ok'] else 'FAILED'} -> {pf_path}")
    if not pf["ok"]:
        sys.exit("Stopping before generation: preflight failed.")
    if args.preflight_only:
        return
    if not args.output:
        sys.exit("--output is required for generation")
    if args.output.exists():
        sys.exit(f"Refusing to overwrite {args.output}")
    chunk_dir = args.output.with_name(args.output.stem + "_chunks")
    if chunk_dir.exists():
        sys.exit(f"Refusing to overwrite {chunk_dir}")

    import torch
    import torchaudio
    from f5_tts.infer.utils_infer import infer_batch_process, preprocess_ref_audio_text
    from f5_tts.model.utils import seed_everything
    from generate import IndicF5Engine, _sync

    params = dict(nfe_step=args.nfe_step, cfg_strength=args.cfg_strength, speed=args.speed,
                  sway_sampling_coef=args.sway_sampling_coef, cross_fade_duration=0.0)
    ref_text = common.read_transcript(args.transcript)
    raw_chunks, chunk_meta = [], []
    with common.inference_lock():
        engine = IndicF5Engine("mps", "direct")
        with tempfile.TemporaryDirectory(prefix="indicf5_") as tmp:
            prepared = Path(tmp) / "ref.wav"
            data, sr = sf.read(str(args.reference), dtype="float32", always_2d=True)
            mono = data.mean(axis=1)
            if sr != SR:
                import soxr

                mono = soxr.resample(mono, sr, SR, quality="VHQ")
            sf.write(str(prepared), np.clip(mono, -1, 1), SR, subtype="PCM_16")
            saved, tempfile.tempdir = tempfile.tempdir, tmp
            try:
                ref_clip, ref_text_final = preprocess_ref_audio_text(str(prepared), ref_text)
                ref_audio = torchaudio.load(ref_clip)
                seed_everything(args.seed)  # once: the chunk sequence is deterministic for this plan
                t_all = time.perf_counter()
                with common.PeakMemorySampler() as sampler, torch.inference_mode():
                    for c in pf["chunks"]:
                        t0 = time.perf_counter()
                        wave, _, _ = infer_batch_process(ref_audio, ref_text_final, [c["devanagari"]],
                                                         engine.model, engine.vocoder, mel_spec_type="vocos",
                                                         device=engine.device, **params)
                        _sync(engine.device)
                        raw_chunks.append(np.asarray(wave, dtype=np.float32))
                        chunk_meta.append({"id": c["id"], "generation_seconds": round(time.perf_counter() - t0, 2)})
                        print(f"  chunk {c['id']}: {chunk_meta[-1]['generation_seconds']} s -> {len(wave) / SR:.2f} s audio")
                gen_seconds = time.perf_counter() - t_all
            finally:
                tempfile.tempdir = saved

    # keep raw model output (pre-processing) for deterministic re-mixing without regeneration
    chunk_dir.mkdir(parents=True)
    for m, w in zip(chunk_meta, raw_chunks):
        sf.write(str(chunk_dir / f"raw_{m['id']:02d}.wav"), w, SR, subtype="FLOAT")

    mix, post = mix_chunks(plan, pf["chunks"], chunk_meta, raw_chunks)
    tp = post["final_true_peak_dbtp"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), mix, SR, subtype="PCM_16")

    from validate_audio import audio_stats

    stats = audio_stats(args.output)
    report = {
        "output": str(args.output),
        "plan": str(args.plan),
        "approved_english_script": plan["approved_english_script"],
        "reference": str(args.reference),
        "reference_text_used": ref_text_final,
        "model": f"ai4bharat/IndicF5@{common.REVISION[:8]}",
        "backend": "direct", "device": engine.device, "seed": args.seed,
        "params": params,
        "post_processing": post,
        "model_load_seconds": engine.load_seconds,
        "generation_seconds": round(gen_seconds, 2),
        "chunks": [{**c, **m} for c, m in zip(pf["chunks"], chunk_meta)],
        "memory_before_load": engine.mem_before_load,
        "memory_during_generation": sampler.summary(),
        "output_file": stats,
        "raw_chunks_dir": str(chunk_dir),
        "preflight_report": str(pf_path),
    }
    rp = common.REPORTS_DIR / "runs" / f"{args.output.stem}.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    m = sampler.summary()
    print(f"\nSaved {args.output}  ({stats['duration_s']} s, true peak {tp:.2f} dBTP)")
    print(f"  generation {gen_seconds:.1f} s total, model load {engine.load_seconds} s")
    print(f"  peak memory RSS {m['peak_process_rss_gb']} GB, MPS driver {m['peak_mps_driver_gb']} GB, "
          f"min free {m['min_system_available_gb']} GB")
    print(f"  report {rp}")


if __name__ == "__main__":
    main()
