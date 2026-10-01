"""Steps B + C: confirm Hugging Face auth and gated access, then download the pinned
IndicF5 snapshot and the Vocos vocoder into the standard HF cache (~/.cache/huggingface).

    .venv/bin/python scripts/download_model.py

The token is never printed; only the account name is shown.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AUTH_HELP, MODEL_FILES, REPO_ID, REVISION, gb, hf_token_present, vocoder_dir  # noqa: E402

from huggingface_hub import HfApi, hf_hub_download, snapshot_download  # noqa: E402
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError  # noqa: E402

print("== B. Hugging Face authentication")
if not hf_token_present():
    sys.exit(AUTH_HELP)
try:
    print(f"  [PASS] logged in as: {HfApi().whoami().get('name')}")
except HfHubHTTPError as e:
    sys.exit(f"  [FAIL] token rejected ({e.response.status_code if e.response is not None else e}).\n{AUTH_HELP}")

try:
    hf_hub_download(REPO_ID, "config.json", revision=REVISION)
    print(f"  [PASS] gated access to {REPO_ID} granted")
except GatedRepoError:
    sys.exit(
        f"  [FAIL] your account hasn't been granted access to {REPO_ID}.\n"
        "  Open https://huggingface.co/ai4bharat/IndicF5, accept the terms, then re-run this script."
    )

print(f"\n== C. Download {REPO_ID} @ {REVISION[:8]} (about 1.4 GB, resumable)")
snap = Path(snapshot_download(REPO_ID, revision=REVISION, allow_patterns=MODEL_FILES))
for f in MODEL_FILES:
    p = snap / f
    print(f"  [{'PASS' if p.is_file() else 'FAIL'}] {f:48} {gb(p.stat().st_size) if p.is_file() else '-'} GB")
print(f"  snapshot: {snap}")

print(f"\n== C. Download vocoder (charactr/vocos-mel-24khz, public)")
voc = vocoder_dir(allow_download=True)
for f in ("config.yaml", "pytorch_model.bin"):
    print(f"  [{'PASS' if (voc / f).is_file() else 'FAIL'}] {f}")

missing = [f for f in MODEL_FILES if not (snap / f).is_file()]
print(f"\n== Result: {'DOWNLOAD COMPLETE' if not missing else 'MISSING: ' + ', '.join(missing)}")
sys.exit(1 if missing else 0)
