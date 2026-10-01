"""Cut a reference segment out of a longer recording (the source is only read, never modified).

    .venv/bin/python scripts/trim_reference.py --source linkedin-video.wav \\
        --start 0.25 --end 13.56 --output reference/my_voice_trimmed.wav

Downmixes to mono, resamples to 24 kHz (soxr VHQ), applies short fades to avoid clicks,
writes PCM16, and records provenance (source SHA-256, times) next to the output as .json.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TARGET_SR  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--start", type=float, required=True, help="seconds")
    ap.add_argument("--end", type=float, required=True, help="seconds")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fade-ms", type=float, default=10.0)
    ap.add_argument("--force", action="store_true", help="allow overwriting --output")
    args = ap.parse_args()

    if args.output.resolve() == args.source.resolve():
        sys.exit("Output must differ from source.")
    if args.output.exists() and not args.force:
        sys.exit(f"Refusing to overwrite {args.output}. Pass --force to replace it.")

    src_hash = sha256(args.source)
    info = sf.info(str(args.source))
    if not 0 <= args.start < args.end <= info.duration:
        sys.exit(f"Need 0 <= start < end <= {info.duration:.3f}")
    a, b = int(round(args.start * info.samplerate)), int(round(args.end * info.samplerate))
    data, sr = sf.read(str(args.source), start=a, stop=b, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != TARGET_SR:
        import soxr

        mono = soxr.resample(mono, sr, TARGET_SR, quality="VHQ")
    n_fade = int(TARGET_SR * args.fade_ms / 1000)
    if n_fade:
        ramp = np.linspace(0, 1, n_fade, dtype=np.float32)
        mono[:n_fade] *= ramp
        mono[-n_fade:] *= ramp[::-1]
    args.output.parent.mkdir(exist_ok=True)
    sf.write(str(args.output), np.clip(mono, -1, 1), TARGET_SR, subtype="PCM_16")

    assert sha256(args.source) == src_hash, "source changed during trimming?!"
    prov = {
        "source": str(args.source.resolve()),
        "source_sha256": src_hash,
        "source_format": f"{info.channels} ch, {info.samplerate} Hz, {info.subtype}",
        "start_s": args.start,
        "end_s": args.end,
        "duration_s": round(args.end - args.start, 3),
        "processing": f"mean downmix, soxr VHQ -> {TARGET_SR} Hz, {args.fade_ms} ms fades, PCM_16",
    }
    args.output.with_suffix(".json").write_text(json.dumps(prov, indent=2))
    print(f"Wrote {args.output} ({prov['duration_s']} s from {args.start}-{args.end} s of {args.source.name})")
    print(f"Source unchanged: sha256 {src_hash[:16]}...")


if __name__ == "__main__":
    main()
