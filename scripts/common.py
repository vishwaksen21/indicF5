"""Shared helpers for the local IndicF5 voice test.

Import this module BEFORE torch (directly or indirectly): it sets
PYTORCH_ENABLE_MPS_FALLBACK=1, which torch only reads at import time.
Upstream f5_tts sets a misspelled variable (PYTOCH_...) that has no effect.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Keep third-party telemetry off; everything here is local.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("WANDB_MODE", "disabled")

import contextlib
import fcntl
import resource
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = PROJECT_ROOT / "reference"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_REF_AUDIO = REFERENCE_DIR / "my_voice.wav"
DEFAULT_REF_TEXT = REFERENCE_DIR / "my_voice.txt"

REPO_ID = "ai4bharat/IndicF5"
# Snapshot reviewed for this project. Pinning the revision means a silent
# upstream change to the weights or vocab can't alter results.
REVISION = "ba85abedf18dc479a447eaa0eccbd76ab78a47d5"
# What inference needs, plus small files used only for diagnosis: upstream's example
# prompts (an English one and an Indic one) and the English dataset-prep script.
MODEL_FILES = [
    "model.safetensors",
    "checkpoints/vocab.txt",
    "config.json",
    "model.py",
    "README.md",
    "prompts/PAN_F_HAPPY_00001.wav",
    "f5_tts/infer/examples/basic/basic_ref_en.wav",
    "f5_tts/infer/examples/basic/basic.toml",
    "f5_tts/train/datasets/prepare_in22_en_10k.py",
]
VOCODER_REPO = "charactr/vocos-mel-24khz"

TARGET_SR = 24000  # f5_tts.infer.utils_infer.target_sample_rate

# Rough working-set needs for the model: fp32 weights are 1.4 GB, plus
# vocoder, activations and the Python/torch runtime. Guideline, not a measurement.
MIN_AVAILABLE_GB = 3.0
COMFORT_AVAILABLE_GB = 5.0


# ---------------------------------------------------------------- Hugging Face


def hf_token_present() -> bool:
    """True if a token is configured. The token itself is never read into output."""
    from huggingface_hub import get_token

    return get_token() is not None


AUTH_HELP = (
    "Hugging Face authentication is missing. In your own terminal run:\n"
    f"    {PROJECT_ROOT}/.venv/bin/hf auth login\n"
    "and paste a Read token at the hidden prompt (answer 'n' to git credential).\n"
    "Also accept the model terms at https://huggingface.co/ai4bharat/IndicF5"
)


def local_snapshot(allow_download: bool = False) -> Path:
    """Path to the pinned IndicF5 snapshot. Downloads only if allowed and authenticated."""
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(REPO_ID, revision=REVISION, allow_patterns=MODEL_FILES, local_files_only=True))
    except Exception:
        if not allow_download:
            raise FileNotFoundError(
                "IndicF5 weights are not downloaded yet. Run:\n"
                f"    {PROJECT_ROOT}/.venv/bin/python scripts/download_model.py"
            )
    if not hf_token_present():
        raise PermissionError(AUTH_HELP)
    return Path(snapshot_download(REPO_ID, revision=REVISION, allow_patterns=MODEL_FILES))


def vocoder_dir(allow_download: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            VOCODER_REPO, allow_patterns=["config.yaml", "pytorch_model.bin"], local_files_only=not allow_download
        )
    )


# ---------------------------------------------------------------- device & memory


def pick_device(preference: str = "auto") -> str:
    import torch

    if preference == "cpu":
        return "cpu"
    if preference in ("auto", "mps") and torch.backends.mps.is_available():
        return "mps"
    if preference == "mps":
        raise RuntimeError("MPS requested but not available on this machine.")
    return "cpu"


def gb(n_bytes: float) -> float:
    return round(n_bytes / 1024**3, 2)


def memory_snapshot() -> dict:
    import psutil

    vm = psutil.virtual_memory()
    snap = {
        "system_total_gb": gb(vm.total),
        "system_available_gb": gb(vm.available),
        "process_rss_gb": gb(psutil.Process().memory_info().rss),
    }
    try:
        import torch

        if torch.backends.mps.is_available():
            snap["mps_allocated_gb"] = gb(torch.mps.current_allocated_memory())
            snap["mps_driver_gb"] = gb(torch.mps.driver_allocated_memory())
    except Exception:
        pass
    return snap


def check_memory_before_load(log=print) -> dict:
    """Warn (don't block) when free memory is low. Returns the snapshot."""
    snap = memory_snapshot()
    avail = snap["system_available_gb"]
    if avail < MIN_AVAILABLE_GB:
        log(
            f"WARNING: only {avail} GB memory available (want >= {MIN_AVAILABLE_GB} GB). "
            "Close browsers/Electron apps first, or loading may swap heavily or fail."
        )
    elif avail < COMFORT_AVAILABLE_GB:
        log(f"Note: {avail} GB memory available; closing other apps will help.")
    return snap


class PeakMemorySampler:
    """Samples process RSS, MPS driver memory and system availability in a thread.

    On Apple Silicon, GPU buffers live in unified memory and are not fully counted
    in RSS, so MPS driver memory is reported separately.
    """

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.peak_rss = 0
        self.peak_mps = 0
        self.min_available = float("inf")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        import psutil
        import torch

        proc = psutil.Process()
        has_mps = torch.backends.mps.is_available()
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, proc.memory_info().rss)
            self.min_available = min(self.min_available, psutil.virtual_memory().available)
            if has_mps:
                self.peak_mps = max(self.peak_mps, torch.mps.driver_allocated_memory())
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def summary(self) -> dict:
        return {
            "peak_process_rss_gb": gb(self.peak_rss),
            "peak_mps_driver_gb": gb(self.peak_mps),
            "min_system_available_gb": gb(self.min_available) if self.min_available != float("inf") else None,
            # macOS reports ru_maxrss in bytes.
            "lifetime_max_rss_gb": gb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        }


def is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, MemoryError) or "out of memory" in msg or "failed to allocate" in msg


# ---------------------------------------------------------------- single-job lock


class InferenceBusy(RuntimeError):
    pass


@contextlib.contextmanager
def inference_lock():
    """Cross-process lock so CLI and UI never run two generations at once."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    lock_path = OUTPUTS_DIR / ".inference.lock"
    with open(lock_path, "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise InferenceBusy("Another generation is already running (CLI or UI). Wait for it to finish.")
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ---------------------------------------------------------------- misc


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def read_transcript(path_or_text: str) -> str:
    """Accept a path to a .txt file or literal transcript text. Preserved verbatim except
    that line breaks become spaces and outer whitespace is stripped."""
    p = Path(path_or_text).expanduser()
    text = p.read_text(encoding="utf-8") if p.suffix == ".txt" or p.is_file() else path_or_text
    return " ".join(text.splitlines()).strip()
