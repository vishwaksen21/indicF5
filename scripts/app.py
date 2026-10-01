"""Local Streamlit UI for the IndicF5 voice test (localhost only).

    .venv/bin/streamlit run scripts/app.py

Server address, telemetry-off and upload limits come from .streamlit/config.toml.
"""

import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402  (sets PYTORCH_ENABLE_MPS_FALLBACK before torch loads)

import tempfile  # noqa: E402

import streamlit as st  # noqa: E402

from common import (  # noqa: E402
    DEFAULT_REF_AUDIO,
    DEFAULT_REF_TEXT,
    OUTPUTS_DIR,
    InferenceBusy,
    inference_lock,
    is_oom_error,
    memory_snapshot,
    read_transcript,
    timestamp,
)
from validate_audio import audio_stats  # noqa: E402

FIRST_TEST = "Hello, this is my first local voice cloning test."

st.set_page_config(page_title="IndicF5 Voice Test", page_icon="🎙️", layout="centered")
st.title("IndicF5 local voice test")
st.warning(
    "**Experimental:** English is not one of IndicF5's documented languages (11 Indian languages). "
    "Your text is sent to the model exactly as typed and is never translated. Everything runs on this Mac."
)


@st.cache_resource(show_spinner="Loading IndicF5 (first time takes a while)...")
def get_engine(device: str, backend: str):
    from generate import IndicF5Engine

    return IndicF5Engine(device, backend, log=lambda m: None)


def tmp_dir() -> Path:
    if "tmp" not in st.session_state:
        st.session_state.tmp = tempfile.mkdtemp(prefix="indicf5_ui_")
    return Path(st.session_state.tmp)


# ---------------------------------------------------------------- sidebar: system
with st.sidebar:
    st.header("System")
    import torch

    st.write(f"MPS available: **{torch.backends.mps.is_available()}**")
    m = memory_snapshot()
    st.write(f"Memory free: **{m['system_available_gb']} / {m['system_total_gb']} GB**")
    if m["system_available_gb"] < common.MIN_AVAILABLE_GB:
        st.error("Low memory. Close other apps before loading the model.")
    backend = st.selectbox("Backend", ["official", "direct"],
                           help="official = README AutoModel usage. direct = same upstream functions, explicit device.")
    device = st.selectbox("Device", ["auto", "mps", "cpu"], help="cpu forces the direct backend")
    with st.expander("Sampling (direct backend only)"):
        seed = st.number_input("Seed", value=42, step=1)
        nfe = st.slider("NFE steps", 8, 64, 32, help="upstream default 32; fewer = faster, rougher")
        cfg = st.slider("CFG strength", 0.5, 4.0, 2.0, 0.1)
        speed = st.slider("Speed", 0.6, 1.5, 1.0, 0.05)
    remove_sil = st.checkbox("Trim long silences in output", value=False)

# ---------------------------------------------------------------- 1. reference audio
st.subheader("1. Reference voice (10–15 s)")
source = st.radio("Source", ["Saved reference/my_voice.wav", "Record now", "Upload WAV"], horizontal=True)
ref_path = None
if source.startswith("Saved"):
    if DEFAULT_REF_AUDIO.is_file():
        ref_path = DEFAULT_REF_AUDIO
        st.audio(str(ref_path))
    else:
        st.info("No saved recording yet. Choose **Record now**.")
elif source == "Record now":
    st.caption("Quiet room, natural pace, 10–15 s. Read the script in the transcript box below.")
    rec = st.audio_input("Record", sample_rate=24000)
    if rec is not None:
        ref_path = tmp_dir() / "recorded.wav"
        ref_path.write_bytes(rec.getvalue())
        exists = DEFAULT_REF_AUDIO.is_file()
        ok_overwrite = st.checkbox("Replace the existing reference/my_voice.wav", value=False) if exists else True
        if st.button("Save as reference/my_voice.wav", disabled=not ok_overwrite):
            DEFAULT_REF_AUDIO.parent.mkdir(exist_ok=True)
            DEFAULT_REF_AUDIO.write_bytes(rec.getvalue())
            st.success(f"Saved {DEFAULT_REF_AUDIO}")
else:
    up = st.file_uploader("WAV file", type=["wav"])
    if up is not None:
        ref_path = tmp_dir() / "uploaded.wav"
        ref_path.write_bytes(up.getvalue())
        st.audio(str(ref_path))

if ref_path is not None:
    s = audio_stats(ref_path)
    c1, c2, c3 = st.columns(3)
    c1.metric("Duration", f"{s['duration_s']} s")
    c2.metric("Sample rate", f"{s['sample_rate']} Hz")
    c3.metric("Level (RMS)", f"{s['rms_dbfs']} dBFS")
    if s["duration_s"] > 15:
        st.warning("Over 15 s: upstream clips the reference, so the transcript may no longer match.")
    elif s["duration_s"] < 8:
        st.warning("Under ~8 s: voice similarity usually improves with 10–15 s.")
    if s["clipped_fraction"] > 0.001:
        st.warning("Clipping detected. Record a little further from the mic.")

# ---------------------------------------------------------------- 2. transcript
st.subheader("2. Exact transcript of the reference")
saved_text = read_transcript(str(DEFAULT_REF_TEXT)) if DEFAULT_REF_TEXT.is_file() else ""
ref_text = st.text_area("Must match the recording word for word", value=saved_text, height=100)
if ref_text.strip() and " ".join(ref_text.split()) != saved_text:
    ok_t = st.checkbox("Replace reference/my_voice.txt", value=False) if saved_text else True
    if st.button("Save transcript", disabled=not ok_t):
        DEFAULT_REF_TEXT.write_text(" ".join(ref_text.split()) + "\n", encoding="utf-8")
        st.success(f"Saved {DEFAULT_REF_TEXT}")

# ---------------------------------------------------------------- 3. generate
st.subheader("3. Text to generate")
gen_text = st.text_area("English text (sent verbatim)", value=FIRST_TEST, height=100)

if st.button("Generate", type="primary", disabled=ref_path is None or not ref_text.strip() or not gen_text.strip()):
    try:
        with inference_lock():
            engine = get_engine(device, backend)
            unk = engine.unknown_tokens(gen_text)
            if unk["n_unknown"]:
                st.warning(f"{unk['n_unknown']} character(s) are not in the model vocab and will be read as "
                           f"index 0: {unk['unknown']}. Your text was not changed.")
            with st.spinner(f"Generating on {engine.device}..."):
                wave, meta = engine.generate(ref_path, " ".join(ref_text.split()), gen_text, int(seed),
                                             remove_sil, nfe_step=nfe, cfg_strength=cfg, speed=speed)
        from generate import save_wav

        out = OUTPUTS_DIR / f"ui-{timestamp()}.wav"  # timestamped: never overwrites
        save_wav(wave, out)
        st.session_state.last = {"path": str(out), "meta": meta, "load": engine.load_seconds}
    except InferenceBusy as e:
        st.error(str(e))
    except FileNotFoundError as e:
        st.error(str(e))
    except Exception as e:  # show every failure in the UI rather than a stack trace
        if is_oom_error(e):
            st.error(f"Out of memory: {e}\n\nClose other apps, shorten the text, or switch Device to cpu.")
        else:
            st.exception(e)

if "last" in st.session_state:
    last = st.session_state.last
    meta = last["meta"]
    st.subheader("Result")
    st.audio(last["path"])
    with open(last["path"], "rb") as f:
        st.download_button("Download WAV", f.read(), file_name=Path(last["path"]).name, mime="audio/wav")
    st.caption(f"Saved to {last['path']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Device", f"{meta['device']} ({meta['backend']})")
    c2.metric("Generation", f"{meta['generation_seconds']} s")
    c3.metric("Audio", f"{meta['output_audio_seconds']} s")
    c4.metric("RTF", meta["real_time_factor"])
    mem = meta["memory_during_generation"]
    st.caption(f"Model load (once per session): {last['load']} s · Peak RSS {mem['peak_process_rss_gb']} GB · "
               f"MPS driver {mem['peak_mps_driver_gb']} GB · min free {mem['min_system_available_gb']} GB")
