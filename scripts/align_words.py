"""Phase 2: word-level forced alignment of a Phase 1 voicegen job (local only, no downloads).

    .venv/bin/python scripts/align_words.py --job job_20261001_103028_eb8ca2
    .venv/bin/python scripts/align_words.py --job <JOB_ID> --transcript my_script.en.txt --overwrite

Engine: transcript-constrained Whisper alignment with the cached openai/whisper-base.en.
The known English transcript is teacher-forced through the decoder; the cross-attention of
the model's published alignment heads is normalised, median-filtered and aligned to audio
frames with dynamic time warping (the method of OpenAI Whisper's word_timestamps). Word
edges are then snapped to voiced frames (10 ms energy) so they don't sit in pauses.
Per-word confidence = geometric mean of the forced tokens' probabilities given the audio.

Resolution: 20 ms (Whisper encoder frames), 10 ms after edge snapping. Accuracy against
ground truth is NOT measured; *_ms fields are a format, not a precision claim.

Writes <job>/alignment/{words.json, words.csv, subtitles.srt, alignment_report.json}.
Never modifies speech.wav or metadata.json.
"""

import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"           # cached model only, never download
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

ASR_MODEL = "openai/whisper-base.en"
WHISPER_SR = 16000
FRAME_S = 0.02                 # Whisper encoder frame (2 x 10 ms mel hop)
MAX_WINDOW_S = 30.0
MEDIAN_WIDTH = 7
CONF_UNCERTAIN = 0.30
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
SRT_LINE_CHARS, SRT_MAX_CUE_S = 42, 7.0


class AlignError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------- text


def split_display(text: str) -> list[str]:
    """Paragraphs (blank lines) -> sentences, mirroring voicegen.split_text's structure."""
    out = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        out += [s for s in SENTENCE_END.split(para) if s.strip()]
    return out


# ---------------------------------------------------------------- DTW alignment


def dtw_path(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Monotonic DTW over (tokens x frames); returns (token_idx, frame_idx) path."""
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    trace = np.zeros((n + 1, m + 1), np.int8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = (acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])
            k = int(np.argmin(c))
            acc[i, j] = cost[i - 1, j - 1] + c[k]
            trace[i, j] = k
    i, j, ti, fj = n, m, [], []
    while i > 0 and j > 0:
        ti.append(i - 1)
        fj.append(j - 1)
        k = trace[i, j]
        if k == 0:
            i, j = i - 1, j - 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    return np.array(ti[::-1]), np.array(fj[::-1])


class WhisperAligner:
    name = f"whisper-cross-attention-dtw ({ASR_MODEL}, transformers)"

    def __init__(self, device="cpu"):
        import torch
        from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor

        self.torch = torch
        try:
            self.proc = WhisperProcessor.from_pretrained(ASR_MODEL, local_files_only=True)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                ASR_MODEL, local_files_only=True, attn_implementation="eager").eval().to(device)
            heads = GenerationConfig.from_pretrained(ASR_MODEL, local_files_only=True).alignment_heads
        except Exception as e:  # noqa: BLE001
            raise AlignError("ALIGNER_UNAVAILABLE", f"Cached {ASR_MODEL} could not be loaded offline: {e}")
        if not heads:
            raise AlignError("ALIGNER_UNAVAILABLE", f"{ASR_MODEL} has no alignment_heads in its generation config")
        self.heads, self.device = heads, device
        tok = self.proc.tokenizer
        tok.set_prefix_tokens(predict_timestamps=False)
        self.prefix = tok.prefix_tokens            # [<|startoftranscript|>, <|notimestamps|>] for .en
        self.eot = tok.eos_token_id

    def align(self, audio16: np.ndarray, text: str) -> list[dict]:
        """Align `text` inside a <=30 s window. Returns words with window-relative times."""
        torch = self.torch
        tok = self.proc.tokenizer
        text_ids = tok.encode(" " + text, add_special_tokens=False)
        ids = self.prefix + text_ids + [self.eot]
        feats = self.proc(audio16, sampling_rate=WHISPER_SR, return_tensors="pt").input_features.to(self.device)
        with torch.no_grad():
            out = self.model(input_features=feats, decoder_input_ids=torch.tensor([ids], device=self.device),
                             output_attentions=True)
        n_frames = min(int(np.ceil(len(audio16) / (WHISPER_SR * FRAME_S))), 1500)
        # Query row k attends to the audio used to predict token k+1: rows [<|notimestamps|> .. last text]
        rows = slice(len(self.prefix) - 1, len(ids) - 1)
        w = torch.stack([out.cross_attentions[l][0, h] for l, h in self.heads])[:, rows, :n_frames].float()
        std, mean = torch.std_mean(w, dim=-2, keepdim=True, unbiased=False)
        w = ((w - mean) / (std + 1e-8)).cpu().numpy()
        from scipy.ndimage import median_filter

        w = median_filter(w, size=(1, 1, MEDIAN_WIDTH))
        matrix = w.mean(axis=0)                     # (n_text + 1, frames): predicts text tokens + eot
        ti, fj = dtw_path(-matrix)
        jumps = np.concatenate([[True], np.diff(ti) > 0])
        token_start_frame = fj[jumps]               # one per row
        # forced-token probabilities (logits[k] predicts ids[k+1])
        logp = torch.log_softmax(out.logits[0].float(), -1)
        tgt = torch.tensor(ids[1:], device=logp.device)
        tok_logp = logp[:-1].gather(-1, tgt[:, None]).squeeze(-1).cpu().numpy()[len(self.prefix) - 1:]

        # group BPE tokens into whitespace words (punctuation attaches to the preceding word)
        words, cur = [], None
        for k, tid in enumerate(text_ids):
            piece = tok.decode([tid])
            if cur is None or piece.startswith(" "):
                cur = {"word": piece.strip(), "tok": [k]}
                words.append(cur)
            else:
                cur["word"] += piece
                cur["tok"].append(k)
        res = []
        for wd in words:
            a, b = wd["tok"][0], wd["tok"][-1] + 1          # rows a..b-1; b = next token's row
            s, e = token_start_frame[a] * FRAME_S, token_start_frame[b] * FRAME_S
            # confidence over spoken tokens only: punctuation is not audible, so its probability
            # (e.g. '!' vs '.') says nothing about whether the word was spoken
            spoken = [k for k in range(a, b) if re.search(r"\w", tok.decode([text_ids[k]]))] or list(range(a, b))
            res.append({"word": wd["word"], "dtw_start_s": float(s), "dtw_end_s": float(e),
                        "confidence": float(np.exp(tok_logp[spoken].mean())),
                        "confidence_incl_punctuation": float(np.exp(tok_logp[a:b].mean())), "n_tokens": b - a})
        return res


# ---------------------------------------------------------------- energy refinement


class Energy:
    def __init__(self, x: np.ndarray, sr: int):
        self.fs = 0.01
        fr = int(sr * self.fs)
        n = len(x) // fr
        self.db = 20 * np.log10(np.sqrt((x[: n * fr].reshape(n, fr) ** 2).mean(1)) + 1e-10)
        self.thr = max(float(np.percentile(self.db, 95)) - 35.0, -55.0)
        self.voiced = self.db > self.thr

    def snap(self, s: float, e: float) -> tuple[float, float, float]:
        """Move edges that lie in silence onto the nearest voiced frame inside [s, e]."""
        i0, i1 = int(round(s / self.fs)), max(int(round(e / self.fs)), int(round(s / self.fs)) + 1)
        v = np.where(self.voiced[i0:i1])[0]
        if not len(v):
            return s, e, 0.0
        ns = (i0 + v[0]) * self.fs if not self.voiced[min(i0, len(self.voiced) - 1)] else s
        last = i1 - 1
        ne = (i0 + v[-1] + 1) * self.fs if last >= len(self.voiced) or not self.voiced[last] else e
        return ns, max(ne, ns + self.fs), float(len(v) / (i1 - i0))


# ---------------------------------------------------------------- outputs


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_cues(words: list[dict]) -> list[dict]:
    cues = []
    for si in sorted({w["sentence_index"] for w in words}):
        ws = [w for w in words if w["sentence_index"] == si and w["start_s"] is not None]
        group = []
        for w in ws:
            text = " ".join(x["word"] for x in group + [w])
            too_long = len(text) > 2 * SRT_LINE_CHARS or (group and w["end_s"] - group[0]["start_s"] > SRT_MAX_CUE_S)
            if group and too_long:
                cues.append(group)
                group = []
            group.append(w)
        if group:
            cues.append(group)
    out = []
    for g in cues:
        text = " ".join(w["word"] for w in g)
        if len(text) > SRT_LINE_CHARS:  # wrap to two lines at the word nearest the middle
            cut = min(range(1, len(g)), key=lambda k: abs(len(" ".join(w["word"] for w in g[:k])) - len(text) / 2))
            text = " ".join(w["word"] for w in g[:cut]) + "\n" + " ".join(w["word"] for w in g[cut:])
        out.append({"start_s": g[0]["start_s"], "end_s": g[-1]["end_s"], "text": text})
    return out


SRT_TS = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$")


def check_srt(path: Path, duration: float) -> dict:
    blocks = [b for b in path.read_text(encoding="utf-8").strip().split("\n\n") if b.strip()]
    bad, prev_end = [], 0.0
    to_s = lambda t: int(t[:2]) * 3600 + int(t[3:5]) * 60 + int(t[6:8]) + int(t[9:]) / 1000  # noqa: E731
    for k, b in enumerate(blocks, 1):
        lines = b.split("\n")
        if len(lines) < 3 or lines[0] != str(k) or not SRT_TS.match(lines[1]):
            bad.append(k)
            continue
        a, z = to_s(lines[1][:12]), to_s(lines[1][17:])
        if not (0 <= a < z <= duration + 0.001) or a < prev_end - 0.001:
            bad.append(k)
        prev_end = z
    return {"cues": len(blocks), "malformed_or_out_of_order": bad, "ok": not bad and len(blocks) > 0}


# ---------------------------------------------------------------- main


def align_job(job_dir: Path, transcript_path: Path | None = None, overwrite=False, device="cpu") -> dict:
    t0 = time.perf_counter()
    meta_path, wav = job_dir / "metadata.json", job_dir / "speech.wav"
    if not meta_path.is_file():
        raise AlignError("JOB_NOT_FOUND", f"No metadata.json in {job_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") != "success":
        raise AlignError("JOB_NOT_SUCCESSFUL", f"Job status is {meta.get('status')!r}; nothing to align")
    if not wav.is_file():
        raise AlignError("AUDIO_MISSING", f"{wav} not found")
    out_dir = job_dir / "alignment"
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise AlignError("ALIGNMENT_EXISTS", f"{out_dir} already has results; pass --overwrite to redo alignment")

    # transcript: explicit file > metadata display_text; never the Devanagari model input
    if transcript_path:
        transcript, source = transcript_path.read_text(encoding="utf-8").strip(), str(transcript_path)
    else:
        transcript, source = (meta.get("input", {}).get("display_text") or "").strip(), "metadata.input.display_text"
    if not transcript:
        raise AlignError("TRANSCRIPT_MISSING", "No English display transcript in metadata; pass --transcript FILE")
    if re.search(r"[ऀ-෿]", transcript):
        raise AlignError("TRANSCRIPT_NOT_ENGLISH", "Transcript contains Indic script; supply the English display text")

    # audio validation
    try:
        info = sf.info(str(wav))
        x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    except Exception as e:  # noqa: BLE001
        raise AlignError("AUDIO_UNREADABLE", f"{wav}: {e}")
    x = x.mean(axis=1)
    duration = len(x) / sr
    audio_checks = {"readable": True, "finite": bool(np.isfinite(x).all()), "non_empty": len(x) > 0,
                    "mono": info.channels == 1, "duration_s": round(duration, 3),
                    "metadata_duration_s": meta["output"].get("duration_s"),
                    "duration_matches_metadata": abs(duration - meta["output"].get("duration_s", -1)) < 0.001}
    if not (audio_checks["finite"] and audio_checks["non_empty"]):
        raise AlignError("AUDIO_INVALID", "speech.wav is empty or contains non-finite samples")

    sentences = split_display(transcript)
    chunks = meta.get("chunks") or []
    if len(sentences) == len(chunks) and chunks:
        # windows: each chunk's exact span extended to the middle of the neighbouring pauses
        windows = []
        for k, c in enumerate(chunks):
            lo = 0.0 if k == 0 else (chunks[k - 1]["end_s"] + c["start_s"]) / 2
            hi = duration if k == len(chunks) - 1 else (c["end_s"] + chunks[k + 1]["start_s"]) / 2
            windows.append((lo, hi))
        segmentation = "phase1_chunk_spans"
    elif duration <= MAX_WINDOW_S:
        windows, sentences, segmentation = [(0.0, duration)], [" ".join(sentences)], "whole_audio"
    else:
        raise AlignError("SEGMENT_MISMATCH",
                         f"{len(sentences)} transcript sentences vs {len(chunks)} audio chunks and audio > 30 s; "
                         "make the display transcript's sentences match the generated chunks")
    if any(hi - lo > MAX_WINDOW_S for lo, hi in windows):
        raise AlignError("WINDOW_TOO_LONG", "A sentence window exceeds Whisper's 30 s limit")

    import soxr

    aligner = WhisperAligner(device)
    energy = Energy(x, sr)
    words = []
    for si, ((lo, hi), sent) in enumerate(zip(windows, sentences)):
        seg = x[int(lo * sr):int(hi * sr)]
        seg16 = soxr.resample(seg, sr, WHISPER_SR, quality="HQ").astype(np.float32)
        for w in aligner.align(seg16, sent):
            s, e = lo + w["dtw_start_s"], lo + w["dtw_end_s"]
            flags = []
            if e - s < FRAME_S / 2:
                flags.append("dtw_zero_duration")
            rs, re_, voiced = energy.snap(s, max(e, s + FRAME_S))
            rs, re_ = max(0.0, min(rs, duration)), max(0.0, min(re_, duration))
            if w["confidence"] < CONF_UNCERTAIN:
                flags.append("low_confidence")
            if voiced < 0.2:
                flags.append("mostly_silent_span")
            if re_ - rs < 0.05:
                flags.append("very_short")
            if re_ - rs > 2.0:
                flags.append("very_long")
            status = "aligned" if not flags else "uncertain"
            if "dtw_zero_duration" in flags and voiced == 0.0:
                status, rs, re_ = "unaligned", None, None
            words.append({"index": len(words), "sentence_index": si, "word": w["word"],
                          "start_s": None if rs is None else round(rs, 3),
                          "end_s": None if re_ is None else round(re_, 3),
                          "start_ms": None if rs is None else int(round(rs * 1000)),
                          "end_ms": None if re_ is None else int(round(re_ * 1000)),
                          "confidence": round(w["confidence"], 4),
                          "confidence_incl_punctuation": round(w["confidence_incl_punctuation"], 4),
                          "status": status, "flags": flags,
                          "dtw_start_s": round(s, 3), "dtw_end_s": round(e, 3)})

    # ---- validation
    display_words = transcript.split()
    aligned = [w for w in words if w["start_s"] is not None]
    v = {
        "word_sequence_matches_transcript": [w["word"] for w in words] == display_words,
        "within_audio_bounds": all(0 <= w["start_s"] and w["end_s"] <= duration + 1e-6 for w in aligned),
        "start_before_end": all(w["start_s"] < w["end_s"] for w in aligned),
        "no_negative_times": all(w["start_s"] >= 0 for w in aligned),
        "chronological_starts": all(b["start_s"] >= a["start_s"] for a, b in zip(aligned, aligned[1:])),
        "unaligned_words_have_no_times": all(w["start_s"] is None for w in words if w["status"] == "unaligned"),
    }
    overlaps = [{"words": [a["word"], b["word"]], "overlap_s": round(a["end_s"] - b["start_s"], 3)}
                for a, b in zip(aligned, aligned[1:]) if a["end_s"] > b["start_s"] + 1e-6]
    gaps = [{"after": a["word"], "before": b["word"], "gap_s": round(b["start_s"] - a["end_s"], 3)}
            for a, b in zip(aligned, aligned[1:]) if b["start_s"] - a["end_s"] >= 0.15]
    sent_vs_chunk = []
    if segmentation == "phase1_chunk_spans":
        for si, c in enumerate(chunks):
            ws = [w for w in aligned if w["sentence_index"] == si]
            if ws:
                sent_vs_chunk.append({"sentence_index": si, "first_word_start_minus_chunk_start_s":
                                      round(ws[0]["start_s"] - c["start_s"], 3),
                                      "last_word_end_minus_chunk_end_s": round(ws[-1]["end_s"] - c["end_s"], 3)})

    out_dir.mkdir(exist_ok=True)
    precision = ("Timing resolution 20 ms (Whisper encoder frames), edges snapped on 10 ms energy frames. "
                 "Accuracy vs ground truth not measured; *_ms values are formatting, not ms precision.")
    words_json = {"audio": "speech.wav", "job_id": meta["job_id"], "transcript": transcript,
                  "transcript_source": source, "duration_s": round(duration, 3),
                  "alignment_engine": WhisperAligner.name, "segmentation": segmentation,
                  "precision_note": precision,
                  "confidence_note": "geometric-mean probability of the forced spoken (non-punctuation) tokens given the audio "
                                     f"(0-1); < {CONF_UNCERTAIN} flagged low_confidence",
                  "words": words}
    (out_dir / "words.json").write_text(json.dumps(words_json, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_dir / "words.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["index", "sentence_index", "word", "start_s", "end_s", "start_ms", "end_ms",
                     "confidence", "status", "flags"])
        for w in words:
            wr.writerow([w["index"], w["sentence_index"], w["word"], w["start_s"], w["end_s"], w["start_ms"],
                         w["end_ms"], w["confidence"], w["status"], ";".join(w["flags"])])
    cues = build_cues(words)
    (out_dir / "subtitles.srt").write_text(
        "".join(f"{k}\n{srt_time(c['start_s'])} --> {srt_time(c['end_s'])}\n{c['text']}\n\n"
                for k, c in enumerate(cues, 1)), encoding="utf-8")
    v["srt"] = check_srt(out_dir / "subtitles.srt", duration)

    n = len(words)
    report = {
        "job_id": meta["job_id"], "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "alignment_engine": WhisperAligner.name, "device": device, "downloads": "none (offline mode enforced)",
        "segmentation": segmentation, "audio_checks": audio_checks,
        "counts": {"transcript_words": len(display_words), "aligned": sum(w["status"] == "aligned" for w in words),
                   "uncertain": sum(w["status"] == "uncertain" for w in words),
                   "unaligned": sum(w["status"] == "unaligned" for w in words)},
        "coverage_percent": round(100 * len(aligned) / len(display_words), 1) if display_words else 0.0,
        "uncertain_or_unaligned": [{k: w[k] for k in ("index", "word", "status", "flags", "confidence")}
                                   for w in words if w["status"] != "aligned"],
        "validation": v, "validation_ok": all(val if isinstance(val, bool) else val["ok"] for val in v.values()),
        "overlaps": overlaps, "gaps_ge_150ms": gaps, "sentence_span_vs_phase1_chunk": sent_vs_chunk,
        "precision_note": precision, "runtime_seconds": round(time.perf_counter() - t0, 2),
    }
    (out_dir / "alignment_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True, help="job id (folder name under --outputs-dir) or path to the job folder")
    ap.add_argument("--outputs-dir", type=Path, default=common.OUTPUTS_DIR)
    ap.add_argument("--transcript", type=Path, help="English display transcript (default: metadata display_text)")
    ap.add_argument("--overwrite", action="store_true", help="replace existing alignment/ results")
    ap.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = ap.parse_args()
    job_dir = Path(args.job) if Path(args.job).is_dir() else args.outputs_dir / args.job
    if args.transcript and not args.transcript.is_file():
        sys.exit(f"[ERROR] TRANSCRIPT_MISSING: {args.transcript} not found")
    try:
        r = align_job(job_dir, args.transcript, args.overwrite, args.device)
    except AlignError as e:
        sys.exit(f"[ERROR] {e.code}: {e}")
    c = r["counts"]
    print(f"{'OK' if r['validation_ok'] else 'VALIDATION ISSUES'}  {job_dir / 'alignment'}")
    print(f"  words {c['transcript_words']}: aligned {c['aligned']}, uncertain {c['uncertain']}, "
          f"unaligned {c['unaligned']}  (coverage {r['coverage_percent']}%)  in {r['runtime_seconds']} s")
    for w in r["uncertain_or_unaligned"]:
        print(f"  {w['status']:9} {w['word']!r} {w['flags']} conf={w['confidence']}")
    sys.exit(0 if r["validation_ok"] else 1)


if __name__ == "__main__":
    main()
