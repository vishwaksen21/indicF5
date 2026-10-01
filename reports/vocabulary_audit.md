# IndicF5 English vocabulary audit

- Vocab file: `checkpoints/vocab.txt` @ snapshot `ba85abed`
- Vocab size: **2545** tokens; index 0 is ' ' (every unknown token becomes this)
- Duplicate entries: none

## Tokens by script

| Script / category | Count |
|---|---|
| MULTI-CHAR TOKEN | 1326 |
| LATIN | 176 |
| Lo | 143 |
| DEVANAGARI | 104 |
| BENGALI | 82 |
| KANNADA | 79 |
| ORIYA | 75 |
| MALAYALAM | 73 |
| TELUGU | 72 |
| Ll | 70 |
| GUJARATI | 70 |
| GURMUKHI | 66 |
| TAMIL | 49 |
| Lu | 48 |
| Po | 22 |
| DIGIT | 10 |
| Mn | 10 |
| Sm | 9 |
| Lm | 9 |
| UNNAMED | 9 |
| Sk | 7 |
| No | 6 |
| So | 5 |
| Cf | 5 |
| Sc | 4 |
| Ps | 3 |
| Pe | 3 |
| Zs | 3 |
| Pf | 2 |
| SPACE | 1 |
| Pd | 1 |
| Pc | 1 |
| Pi | 1 |
| CJK | 1 |

## English character coverage

| Group | Present | Missing |
|---|---|---|
| lowercase a-z | 26/26 | `-` |
| uppercase A-Z | 26/26 | `-` |
| digits 0-9 | 10/10 | `-` |
| punctuation .,!?'"-:;() | 11/11 | `-` |

## Test strings (real upstream preprocessing + lookup)

| Test | Tokens | Unknown | Unknown chars | Preprocessing changed text? |
|---|---|---|---|---|
| first_test_sentence | 49 | 0 (0.0%) | `-` | no |
| sample_recording_script | 190 | 0 (0.0%) | `-` | no |
| pangram | 98 | 0 (0.0%) | `-` | yes |
| your_reference_transcript | 190 | 0 (0.0%) | `-` | no |

## Verdict

TECHNICALLY PROCESSABLE: every character of the English test strings maps to a real vocab entry. This says nothing about pronunciation quality; English is not a documented IndicF5 language.

Coverage is necessary but not sufficient: it only proves English text is not dropped. Whether the model learned English letter-to-sound mappings depends on its training data, which this audit cannot see. Only listening to generated audio answers that.
