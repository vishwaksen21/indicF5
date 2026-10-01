"""Step E: audit whether IndicF5's tokenizer can represent English text.

Runs the exact upstream text path used at inference time:
    text -> f5_tts.model.utils.convert_char_to_pinyin -> vocab lookup (list_str_to_idx),
where any token missing from vocab.txt silently becomes index 0.

    .venv/bin/python scripts/inspect_vocabulary.py [--text "extra sentence to test"]

Writes reports/vocabulary_audit.md and reports/vocabulary_audit.json. No network.
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import json
import string
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_REF_TEXT, REPORTS_DIR, REVISION, local_snapshot, read_transcript  # noqa: E402

SAMPLE_SCRIPT = (
    "Hello, my name is Vishwak. I am recording this sample to test local artificial intelligence "
    "voice cloning. I want the generated speech to sound natural and consistent with my original voice."
)
FIRST_TEST = "Hello, this is my first local voice cloning test."
PANGRAM = "The quick brown fox jumps over the lazy dog. 0123456789 (it's 5:30, isn't it?) \"Yes!\" - OK; fine."


def script_of(ch: str) -> str:
    if ch == " ":
        return "SPACE"
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return "UNNAMED"
    if len(ch) > 1:
        return "MULTI-CHAR TOKEN"
    first = name.split()[0]
    return first if first in {"LATIN", "DEVANAGARI", "BENGALI", "GUJARATI", "GURMUKHI", "KANNADA",
                               "MALAYALAM", "ORIYA", "TAMIL", "TELUGU", "DIGIT", "CJK"} else unicodedata.category(ch)


def load_vocab(path: Path) -> list[str]:
    # Same parsing as f5_tts.model.utils.get_tokenizer(..., "custom"): line[:-1] per line.
    with open(path, encoding="utf-8") as f:
        return [line[:-1] for line in f]


def tokenize(text: str, vocab_map: dict) -> dict:
    from f5_tts.model.utils import convert_char_to_pinyin

    tokens = convert_char_to_pinyin([text])[0]
    ids = [vocab_map.get(t, 0) for t in tokens]  # identical to list_str_to_idx
    unknown = [t for t in tokens if t not in vocab_map]
    return {
        "input": text,
        "tokens_after_upstream_preprocessing": "".join(tokens),
        "preprocessing_changed_text": "".join(tokens) != text,
        "n_tokens": len(tokens),
        "n_unknown": len(unknown),
        "unknown_fraction": round(len(unknown) / max(1, len(tokens)), 4),
        "unknown_tokens": dict(Counter(unknown)),
        "ids_preview": ids[:40],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=Path, help="default: checkpoints/vocab.txt from the pinned snapshot")
    ap.add_argument("--text", action="append", default=[], help="extra text to test (repeatable)")
    args = ap.parse_args()

    try:
        vocab_path = args.vocab or local_snapshot() / "checkpoints" / "vocab.txt"
    except FileNotFoundError as e:
        sys.exit(str(e))
    vocab = load_vocab(vocab_path)
    vocab_map = {tok: i for i, tok in enumerate(vocab)}  # later duplicates win, as upstream
    dupes = [t for t, c in Counter(vocab).items() if c > 1]

    by_script = Counter(script_of(t) if len(t) == 1 else "MULTI-CHAR TOKEN" for t in vocab)
    groups = {
        "lowercase a-z": string.ascii_lowercase,
        "uppercase A-Z": string.ascii_uppercase,
        "digits 0-9": string.digits,
        "punctuation .,!?'\"-:;()": ".,!?'\"-:;()",
    }
    coverage = {
        name: {"present": sum(c in vocab_map for c in chars), "total": len(chars),
               "missing": "".join(c for c in chars if c not in vocab_map)}
        for name, chars in groups.items()
    }

    tests = {"first_test_sentence": FIRST_TEST, "sample_recording_script": SAMPLE_SCRIPT, "pangram": PANGRAM}
    if DEFAULT_REF_TEXT.is_file() and read_transcript(str(DEFAULT_REF_TEXT)):
        tests["your_reference_transcript"] = read_transcript(str(DEFAULT_REF_TEXT))
    for i, t in enumerate(args.text):
        tests[f"extra_{i + 1}"] = t
    results = {k: tokenize(v, vocab_map) for k, v in tests.items()}

    worst = max(r["unknown_fraction"] for r in results.values())
    letters_ok = coverage["lowercase a-z"]["missing"] == "" and coverage["uppercase A-Z"]["missing"] == ""
    if letters_ok and worst == 0:
        verdict = ("TECHNICALLY PROCESSABLE: every character of the English test strings maps to a real vocab "
                   "entry. This says nothing about pronunciation quality; English is not a documented "
                   "IndicF5 language.")
    elif worst < 0.05:
        verdict = ("MOSTLY PROCESSABLE: a few characters are unknown and will be read as index 0 "
                   f"({vocab[0]!r}). Generation can proceed as an experiment.")
    else:
        verdict = ("NOT ADEQUATELY REPRESENTABLE: a large share of English characters map to the unknown "
                   "index. English output is expected to be garbled. Input is preserved, not rewritten.")

    audit = {
        "vocab_file": str(vocab_path),
        "snapshot_revision": REVISION,
        "vocab_size": len(vocab),
        "index_0_token": vocab[0],
        "index_0_note": "upstream maps every unknown token to index 0 (f5_tts/model/utils.py list_str_to_idx)",
        "duplicate_tokens": dupes,
        "tokens_by_script": dict(by_script.most_common()),
        "english_coverage": coverage,
        "latin_extras_in_vocab": "".join(sorted(t for t in vocab if len(t) == 1 and script_of(t) == "LATIN"
                                                and t not in string.ascii_letters)),
        "tests": results,
        "verdict": verdict,
    }

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "vocabulary_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    md = [
        "# IndicF5 English vocabulary audit", "",
        f"- Vocab file: `checkpoints/vocab.txt` @ snapshot `{REVISION[:8]}`",
        f"- Vocab size: **{len(vocab)}** tokens; index 0 is {vocab[0]!r} (every unknown token becomes this)",
        f"- Duplicate entries: {dupes or 'none'}", "",
        "## Tokens by script", "", "| Script / category | Count |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in by_script.most_common()], "",
        "## English character coverage", "", "| Group | Present | Missing |", "|---|---|---|",
        *[f"| {k} | {v['present']}/{v['total']} | `{v['missing'] or '-'}` |" for k, v in coverage.items()], "",
        "## Test strings (real upstream preprocessing + lookup)", "",
        "| Test | Tokens | Unknown | Unknown chars | Preprocessing changed text? |", "|---|---|---|---|---|",
        *[f"| {k} | {r['n_tokens']} | {r['n_unknown']} ({r['unknown_fraction']:.1%}) | "
          f"`{''.join(r['unknown_tokens']) or '-'}` | {'yes' if r['preprocessing_changed_text'] else 'no'} |"
          for k, r in results.items()], "",
        "## Verdict", "", verdict, "",
        "Coverage is necessary but not sufficient: it only proves English text is not dropped. Whether the "
        "model learned English letter-to-sound mappings depends on its training data, which this audit "
        "cannot see. Only listening to generated audio answers that.",
    ]
    (REPORTS_DIR / "vocabulary_audit.md").write_text("\n".join(md) + "\n")

    print("\n".join(md))
    print(f"\nSaved reports/vocabulary_audit.md and .json")


if __name__ == "__main__":
    main()
