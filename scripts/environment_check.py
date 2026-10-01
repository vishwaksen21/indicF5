"""Phase 2 environment check for the local IndicF5 voice test.

Verifies package versions, imports, MPS availability and basic MPS ops.
It does NOT download model weights and does NOT run inference.

Run from the project root:
    .venv/bin/python scripts/environment_check.py
"""

import os

# Must be set before torch is imported. Upstream f5_tts sets a misspelled
# variable (PYTOCH_ENABLE_MPS_FALLBACK) that has no effect, so we set the real one.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Guarantee this check never touches the network.
os.environ["HF_HUB_OFFLINE"] = "1"

import platform
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPECTED = {
    "torch": "2.8.0",
    "torchaudio": "2.8.0",
    "transformers": "4.49.0",
}

failures = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{': ' + detail if detail else ''}")
    if not ok:
        failures.append(label)


print("== System")
print(f"  Python      {sys.version.split()[0]} ({sys.executable})")
print(f"  Platform    {platform.platform()} / {platform.machine()}")
check("Running inside project venv", sys.prefix != sys.base_prefix and str(PROJECT_ROOT) in sys.prefix)
check("Python 3.12", sys.version_info[:2] == (3, 12))

print("\n== Pinned package versions")
for pkg, want in EXPECTED.items():
    got = version(pkg)
    check(f"{pkg}=={want}", got == want, f"installed {got}")
np_ver = version("numpy")
check("numpy<=1.26.4", tuple(map(int, np_ver.split(".")[:3])) <= (1, 26, 4), f"installed {np_ver}")
hub_ver = version("huggingface-hub")
check("huggingface-hub<1.0", int(hub_ver.split(".")[0]) < 1, f"installed {hub_ver}")
print(f"  f5-tts (IndicF5) {version('f5-tts')}")

print("\n== Imports")
import torch

check("torch import", True, torch.__version__)
check("PYTORCH_ENABLE_MPS_FALLBACK=1 set before torch import", os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1")

import torchaudio

backends = torchaudio.list_audio_backends()
check("torchaudio import", True, f"backends: {backends}")
check("torchaudio has a non-FFmpeg backend (soundfile)", "soundfile" in backends)

import transformers

check("transformers import", True, transformers.__version__)

import vocos  # noqa: F401
from f5_tts.infer import utils_infer

check("f5_tts (IndicF5) import", True, f"upstream default device = {utils_infer.device!r}")

print("\n== MPS (Apple GPU)")
check("MPS built into torch", torch.backends.mps.is_built())
mps_ok = torch.backends.mps.is_available()
check("MPS available", mps_ok)
if mps_ok:
    try:
        a = torch.randn(256, 256, device="mps")
        check("MPS matmul (float32)", torch.allclose((a @ a).cpu(), a.cpu() @ a.cpu(), atol=1e-2))
        x = torch.randn(1, 24000, device="mps")
        win = torch.hann_window(1024, device="mps")
        spec = torch.stft(x, n_fft=1024, hop_length=256, window=win, return_complex=True)
        y = torch.istft(spec, n_fft=1024, hop_length=256, window=win, length=24000)
        check("MPS stft/istft round trip (used by mel/vocoder)", torch.allclose(x.cpu(), y.cpu(), atol=1e-3))
    except Exception as e:  # report, don't crash
        check("MPS basic ops", False, f"{type(e).__name__}: {e}")
    try:
        torch.zeros(1, dtype=torch.float64, device="mps")
        print("  [INFO] float64 on MPS: allowed")
    except (TypeError, RuntimeError):
        print("  [INFO] float64 on MPS: unsupported (expected; IndicF5 inference uses float32)")

print("\n== Hugging Face auth (local check only, token never printed)")
from huggingface_hub import get_token

print(f"  Token configured: {'yes' if get_token() else 'no  -> run `hf auth login` yourself before Phase 4'}")

print("\n== Resources")
total_gb = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 1024**3
try:
    import psutil

    avail = f"{psutil.virtual_memory().available / 1024**3:.1f} GB available"
except ImportError:
    avail = "availability unknown (psutil missing)"
print(f"  Memory      {total_gb:.0f} GB total, {avail}")
disk = shutil.disk_usage(PROJECT_ROOT)
print(f"  Disk        {disk.free / 1024**3:.0f} GB free")
venv_size = subprocess.check_output(["du", "-sh", str(PROJECT_ROOT / ".venv")], text=True).split()[0]
print(f"  .venv size  {venv_size}")

print(f"\n== Result: {'ALL CHECKS PASSED' if not failures else f'{len(failures)} FAILED: ' + ', '.join(failures)}")
sys.exit(1 if failures else 0)
