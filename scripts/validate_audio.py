"""Validate (and optionally import) the reference recording, or check a generated WAV.

Validate the reference recording and transcript:
    .venv/bin/python scripts/validate_audio.py

Import a recording made in QuickTime / Voice Memos (m4a, aiff, caf, wav, mp3...)
into reference/my_voice.wav as 24 kHz mono 16-bit WAV (uses macOS's built-in afconvert):
    .venv/bin/python scripts/validate_audio.py --import ~/Desktop/recording.m4a

Check a generated output file:
    .venv/bin/python scripts/validate_audio.py --output-check outputs/first_test.wav

Nothing here uses the network.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_REF_AUDIO, DEFAULT_REF_TEXT, TARGET_SR, read_transcript  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

SILENCE_DBFS = -40.0  # frame counts as silence below this level
FRAME_S = 0.02


def db(x: float) -> float:
    return float(20 * np.log10(max(x, 1e-10)))


def audio_stats(path: Path) -> dict:
    info = sf.info(str(path))
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    frame = max(1, int(sr * FRAME_S))
    n = len(mono) // frame
    frame_db = np.array([db(np.sqrt(np.mean(mono[i * frame : (i + 1) * frame] ** 2))) for i in range(n)])
    voiced = np.where(frame_db > SILENCE_DBFS)[0]
    if len(voiced):
        lead = voiced[0] * FRAME_S
        trail = (n - 1 - voiced[-1]) * FRAME_S
        # longest internal pause
        gaps = np.diff(voiced) - 1
        longest_pause = float(gaps.max() * FRAME_S) if len(gaps) else 0.0
    else:
        lead = trail = len(mono) / sr
        longest_pause = 0.0
    return {
        "path": str(path),
        "duration_s": round(len(mono) / sr, 3),
        "sample_rate": sr,
        "channels": info.channels,
        "subtype": info.subtype,
        "file_size_bytes": path.stat().st_size,
        "peak_dbfs": round(db(float(np.abs(mono).max()) if len(mono) else 0.0), 1),
        "rms_dbfs": round(db(float(np.sqrt(np.mean(mono**2))) if len(mono) else 0.0), 1),
        "clipped_fraction": round(float(np.mean(np.abs(data) >= 0.999)), 5),
        "voiced_fraction": round(len(voiced) / n, 3) if n else 0.0,
        "leading_silence_s": round(lead, 2),
        "trailing_silence_s": round(trail, 2),
        "longest_internal_pause_s": round(longest_pause, 2),
        "non_silent": bool(len(voiced)) and float(np.abs(mono).max()) > 1e-3,
    }


def import_recording(src: Path, dst: Path, force: bool) -> None:
    """Convert any macOS-readable audio file to 24 kHz mono PCM16 WAV."""
    if dst.exists() and not force:
        sys.exit(f"Refusing to overwrite existing {dst}. Re-run with --force to replace it.")
    if not src.is_file():
        sys.exit(f"Source file not found: {src}")
    with tempfile.TemporaryDirectory() as tmp:
        decoded = Path(tmp) / "decoded.wav"
        # afconvert (built into macOS) decodes m4a/aac/caf/aiff/mp3; keep native rate & channels,
        # then downmix and resample ourselves so the conversion is explicit.
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEF32", str(src), str(decoded)], check=True)
        data, sr = sf.read(str(decoded), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != TARGET_SR:
        import soxr

        mono = soxr.resample(mono, sr, TARGET_SR, quality="VHQ")
    dst.parent.mkdir(exist_ok=True)
    sf.write(str(dst), np.clip(mono, -1, 1), TARGET_SR, subtype="PCM_16")
    print(f"Imported {src.name}: {data.shape[1]} ch @ {sr} Hz -> {dst} (1 ch @ {TARGET_SR} Hz, PCM_16)")


def print_stats(s: dict) -> None:
    for k, v in s.items():
        print(f"  {k:26} {v}")


def validate_reference(audio: Path, transcript: Path) -> int:
    problems, warnings = [], []
    print(f"== Reference audio: {audio}")
    if not audio.is_file():
        print("  missing. Record it first (see README 'Record your voice').")
        return 1
    s = audio_stats(audio)
    print_stats(s)

    d = s["duration_s"]
    speech = d - s["leading_silence_s"] - s["trailing_silence_s"]
    if d > 15.0:
        warnings.append(
            f"{d:.1f} s is over 15 s. Upstream clips the reference to about 15 s, so the transcript "
            "would no longer match the audio. Re-record 10–15 s, or trim."
        )
    if speech < 6.0:
        problems.append(f"only ~{speech:.1f} s of speech; need about 10–15 s.")
    elif speech < 9.0:
        warnings.append(f"~{speech:.1f} s of speech; 10–15 s gives better voice similarity.")
    if s["channels"] != 1:
        warnings.append(f"{s['channels']} channels; will be downmixed to mono (fine).")
    if s["sample_rate"] != TARGET_SR:
        warnings.append(f"{s['sample_rate']} Hz; will be resampled to {TARGET_SR} Hz (fine).")
    if s["clipped_fraction"] > 0.001:
        problems.append("clipping detected; record further from the mic or lower input gain.")
    if s["rms_dbfs"] < -35:
        warnings.append("recording is quiet (RMS < -35 dBFS); speak closer to the mic.")
    if s["leading_silence_s"] > 1.0 or s["trailing_silence_s"] > 1.0:
        warnings.append("more than 1 s of silence at the start or end; upstream trims it (fine).")
    if s["longest_internal_pause_s"] > 1.0:
        warnings.append(
            f"a {s['longest_internal_pause_s']} s pause mid-recording; long pauses can be learned as pacing."
        )

    print(f"\n== Transcript: {transcript}")
    if not transcript.is_file() or not read_transcript(str(transcript)):
        problems.append("transcript missing or empty.")
    else:
        text = read_transcript(str(transcript))
        words = len(text.split())
        print(f"  text          {text!r}")
        print(f"  words/chars   {words} / {len(text)}")
        if speech > 0:
            wps = words / speech
            print(f"  words per sec {wps:.2f} (typical conversational English: ~2–3.5)")
            if not 1.3 <= wps <= 4.5:
                warnings.append(
                    f"{wps:.2f} words/s is unusual; make sure the transcript matches EXACTLY what you said."
                )

    print()
    for w in warnings:
        print(f"  [WARN] {w}")
    for p in problems:
        print(f"  [FAIL] {p}")
    print(f"== Result: {'OK' if not problems else 'NOT READY'}")
    return 1 if problems else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", type=Path, default=DEFAULT_REF_AUDIO)
    ap.add_argument("--transcript", type=Path, default=DEFAULT_REF_TEXT)
    ap.add_argument("--import", dest="import_src", type=Path, help="convert a recording into --audio")
    ap.add_argument("--force", action="store_true", help="allow --import to overwrite --audio")
    ap.add_argument("--output-check", type=Path, help="report stats for a generated WAV instead")
    args = ap.parse_args()

    if args.output_check:
        print(f"== Output check: {args.output_check}")
        s = audio_stats(args.output_check)
        print_stats(s)
        sys.exit(0 if s["non_silent"] else 1)
    if args.import_src:
        import_recording(args.import_src.expanduser(), args.audio, args.force)
    sys.exit(validate_reference(args.audio, args.transcript))


if __name__ == "__main__":
    main()
