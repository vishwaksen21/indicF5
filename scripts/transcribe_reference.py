"""Draft the reference transcript locally with Whisper (openai/whisper-base.en, ~294 MB, cached once).

    .venv/bin/python scripts/transcribe_reference.py [--audio reference/my_voice_trimmed.wav]

Runs in-process via transformers; audio never leaves the machine. Writes <audio>.txt
(refuses to overwrite). Listen once to confirm: any ASR error becomes a transcript mismatch.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402,F401

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

MODEL = "openai/whisper-base.en"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path, default=common.REFERENCE_DIR / "my_voice_trimmed.wav")
    ap.add_argument("--force", action="store_true", help="overwrite an existing transcript")
    args = ap.parse_args()
    out = args.audio.with_suffix(".txt")
    if out.exists() and not args.force:
        sys.exit(f"Refusing to overwrite {out}. Pass --force.")

    import soxr
    from transformers import pipeline

    audio, sr = sf.read(str(args.audio), dtype="float32", always_2d=True)
    audio = soxr.resample(audio.mean(axis=1), sr, 16000, quality="VHQ").astype(np.float32)
    # CPU: a 74M-parameter model on 13 s of audio is fast, and it leaves MPS memory for IndicF5.
    asr = pipeline("automatic-speech-recognition", model=MODEL, device="cpu", torch_dtype="float32")
    res = asr({"raw": audio, "sampling_rate": 16000}, return_timestamps=True,
              generate_kwargs={"num_beams": 5})
    text = " ".join(res["text"].split())
    for c in res.get("chunks", []):
        a, b = c["timestamp"]
        print(f"  {a if a is not None else '?':>5} - {b if b is not None else '?':>5} s  {c['text'].strip()}")
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\nTranscript ({len(text.split())} words): {text}\nSaved {out}")


if __name__ == "__main__":
    main()
