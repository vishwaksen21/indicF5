"""Phase 4B-3: read-only media helpers (integrity, image probing, video probing, decode smoke tests).

Nothing in this module writes, converts, resizes, recompresses or otherwise alters media. Video
probing uses the installed ffprobe/ffmpeg as subprocesses; a decode smoke test decodes every frame
to FFmpeg's null muxer (no output file). ffprobe metadata alone is never treated as proof of decodability.
"""

import hashlib
import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path

IMAGE_EXTS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP", ".avif": "AVIF"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v"}
SUPPORTED_EXTS = set(IMAGE_EXTS) | VIDEO_EXTS
CFR_TOLERANCE_S = 0.0005   # max deviation of a frame interval from the median to still count as constant


class MediaError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------- integrity


def safe_path(path, allowed_root) -> Path:
    """Resolve `path` and require a regular file inside `allowed_root` (after resolving symlinks and '..')."""
    root = Path(allowed_root).resolve()
    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.exists() and not p.is_symlink():
        raise MediaError("FILE_MISSING", f"{path} does not exist")
    real = p.resolve()
    try:
        real.relative_to(root)
    except ValueError:
        kind = "symlink target" if p.is_symlink() else "path"
        raise MediaError("PATH_OUTSIDE_ALLOWED_DIR", f"{kind} {real} is outside the permitted directory {root}")
    if p.is_symlink():
        raise MediaError("SYMLINK_REJECTED", f"{path} is a symlink; register the real file instead")
    if real.is_dir():
        raise MediaError("NOT_A_FILE", f"{path} is a directory")
    if not real.is_file():
        raise MediaError("NOT_A_FILE", f"{path} is not a regular file")
    if real.suffix.lower() not in SUPPORTED_EXTS:
        raise MediaError("UNSUPPORTED_EXTENSION", f"{real.suffix or '(none)'} is not supported "
                                                  f"(supported: {', '.join(sorted(SUPPORTED_EXTS))})")
    return real


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_info(path: Path) -> dict:
    return {"sha256": sha256(path), "bytes": path.stat().st_size, "extension": path.suffix.lower(),
            "kind": "image" if path.suffix.lower() in IMAGE_EXTS else "video"}


# ---------------------------------------------------------------- images


def probe_image(path: Path) -> dict:
    """Pillow probe: header, integrity verify() and a full pixel decode (load()). Never re-saves."""
    from PIL import Image, UnidentifiedImageError, features

    ext = path.suffix.lower()
    out = {"readable": False, "decode_ok": False, "format": None, "width": None, "height": None, "mode": None,
           "has_alpha": None, "exif_orientation": None, "errors": []}
    if ext == ".avif" and not features.check("avif"):
        out["errors"].append("this Pillow build has no AVIF support")
        return out
    try:
        with Image.open(path) as im:
            out.update(format=im.format, width=im.width, height=im.height, mode=im.mode)
            out["has_alpha"] = im.mode in ("RGBA", "LA", "PA", "La", "RGBa") or "transparency" in im.info
            try:
                out["exif_orientation"] = im.getexif().get(0x0112)
            except Exception:  # noqa: BLE001  (some formats have no EXIF support)
                out["exif_orientation"] = None
            im.verify()                       # structural integrity (consumes the image object)
        out["readable"] = True
        with Image.open(path) as im:
            im.load()                         # full decode of the pixel data
        out["decode_ok"] = True
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        out["errors"].append(f"{type(e).__name__}: {e}")
    if out["format"] and IMAGE_EXTS.get(ext) != out["format"]:
        out["errors"].append(f"extension {ext} does not match detected format {out['format']}")
        out["format_matches_extension"] = False
    else:
        out["format_matches_extension"] = out["format"] is not None
    return out


# ---------------------------------------------------------------- video


def _run(cmd: list, timeout: int = 600) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise MediaError("TOOL_MISSING", f"{cmd[0]} not found")
    except subprocess.TimeoutExpired:
        raise MediaError("TOOL_TIMEOUT", f"{cmd[0]} timed out after {timeout}s")


def _rate(s):
    try:
        f = Fraction(s)
        return None if f == 0 else f
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _float(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def probe_video(path: Path, ffprobe: str = "ffprobe", ffmpeg: str = "ffmpeg") -> dict:
    out = {"readable": False, "container": None, "video_codec": None, "width": None, "height": None, "pix_fmt": None,
           "avg_fps": None, "nominal_fps": None, "duration_s": None, "frame_count_metadata": None, "rotation": 0,
           "color_space": None, "color_transfer": None, "color_primaries": None, "color_range": None,
           "frame_rate_mode": None, "has_audio": None, "audio_codec": None, "audio_streams": 0,
           "decode": None, "errors": [], "warnings": []}
    r = _run([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])
    if r.returncode != 0:
        out["errors"].append(f"ffprobe failed: {r.stderr.strip()[-300:] or 'unknown error'}")
        return out
    try:
        info = json.loads(r.stdout or "{}")
    except json.JSONDecodeError:
        out["errors"].append("ffprobe returned invalid JSON")
        return out
    streams = info.get("streams") or []
    v = next((s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")), None)
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    out["container"] = (info.get("format") or {}).get("format_name")
    out["has_audio"], out["audio_streams"] = bool(audio), len(audio)
    out["audio_codec"] = audio[0].get("codec_name") if audio else None
    if v is None:
        out["errors"].append("no video stream")
        return out
    out["readable"] = True
    avg, nom = _rate(v.get("avg_frame_rate")), _rate(v.get("r_frame_rate"))
    out.update(video_codec=v.get("codec_name"), width=v.get("width"), height=v.get("height"), pix_fmt=v.get("pix_fmt"),
               avg_fps=float(avg) if avg else None, nominal_fps=float(nom) if nom else None,
               avg_fps_fraction=str(avg) if avg else None, nominal_fps_fraction=str(nom) if nom else None,
               color_space=v.get("color_space"), color_transfer=v.get("color_transfer"),
               color_primaries=v.get("color_primaries"), color_range=v.get("color_range"))
    out["duration_s"] = _float(v.get("duration")) or _float((info.get("format") or {}).get("duration"))
    if out["duration_s"] is None:
        out["warnings"].append("duration missing from metadata")
    try:
        out["frame_count_metadata"] = int(v["nb_frames"]) if v.get("nb_frames") not in (None, "N/A") else None
    except (TypeError, ValueError):
        out["frame_count_metadata"] = None
    rot = None
    for sd in v.get("side_data_list") or []:
        if "rotation" in sd:
            rot = sd["rotation"]
    if rot is None and (v.get("tags") or {}).get("rotate"):
        rot = (v.get("tags") or {}).get("rotate")
    try:
        out["rotation"] = int(float(rot or 0)) % 360
    except (TypeError, ValueError):
        out["warnings"].append(f"unparseable rotation {rot!r}")

    # frame-rate mode from packet timestamps (no decode); sorted because B-frames reorder packets
    p = _run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time", "-of", "csv=p=0",
              str(path)])
    pts = sorted(t for t in (_float(x.strip().rstrip(",")) for x in p.stdout.splitlines()) if t is not None)
    if len(pts) >= 3:
        d = [b - a for a, b in zip(pts, pts[1:])]
        med = sorted(d)[len(d) // 2]
        out["frame_interval_median_s"] = round(med, 6)
        out["frame_rate_mode"] = "constant" if all(abs(x - med) <= CFR_TOLERANCE_S for x in d) else "variable"
    else:
        out["frame_rate_mode"] = "unknown"
        out["warnings"].append("too few packets to assess constant vs variable frame rate")
    out["decode"] = decode_smoke_test(path, ffmpeg)
    if not out["decode"]["ok"]:
        out["errors"].append("decode smoke test failed: " + "; ".join(out["decode"]["errors"])[:300])
    return out


def decode_smoke_test(path: Path, ffmpeg: str = "ffmpeg") -> dict:
    """Decode all video (and audio) frames to the null muxer, stopping at the first decode error.
    Produces no file. Returns frames decoded and any errors."""
    r = _run([ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-xerror", "-err_detect", "explode",
              "-i", str(path), "-map", "0:v:0", "-map", "0:a?", "-f", "null", "-progress", "pipe:1", "-nostats", "-"])
    frames = None
    for ln in r.stdout.splitlines():
        if ln.startswith("frame="):
            try:
                frames = int(ln.split("=", 1)[1])
            except ValueError:
                pass
    errs = [ln for ln in r.stderr.splitlines() if ln.strip()]
    return {"ok": r.returncode == 0 and not errs and (frames or 0) > 0, "frames_decoded": frames,
            "returncode": r.returncode, "errors": errs[:10]}


def probe(path: Path, ffprobe: str = "ffprobe", ffmpeg: str = "ffmpeg") -> dict:
    """Integrity + format-specific probe. `path` must already have passed safe_path()."""
    info = file_info(path)
    info["probe"] = probe_image(path) if info["kind"] == "image" else probe_video(path, ffprobe, ffmpeg)
    return info
