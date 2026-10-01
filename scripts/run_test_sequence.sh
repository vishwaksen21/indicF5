#!/bin/zsh
# Test sequence A–J. Stops at the first failing step.
#   zsh scripts/run_test_sequence.sh
set -e
cd "${0:A:h}/.."
PY=.venv/bin/python
TS=$(date +%Y%m%d-%H%M%S)
EN_OUT=outputs/first_test.wav
[[ -e $EN_OUT ]] && EN_OUT=outputs/first_test-$TS.wav   # never overwrite a previous result

echo "\n######## A. Environment validation";           $PY scripts/environment_check.py
echo "\n######## B+C. HF auth + model download";      $PY scripts/download_model.py
echo "\n######## E. English vocabulary audit";        $PY scripts/inspect_vocabulary.py
echo "\n######## F. Reference recording validation";  $PY scripts/validate_audio.py
echo "\n######## D+G+H+J. Load model, first English inference, save, measure"
$PY scripts/generate.py --text "Hello, this is my first local voice cloning test." --output $EN_OUT
echo "\n######## I. Output verification";              $PY scripts/validate_audio.py --output-check $EN_OUT

# Control: the upstream README example (Punjabi prompt -> Hindi text), a documented language pair.
# If this sounds right but English doesn't, the pipeline works and the limitation is English itself.
SNAP=$($PY -c "import sys; sys.path.insert(0,'scripts'); from common import local_snapshot; print(local_snapshot())")
CTRL_OUT=outputs/control_hindi-$TS.wav
echo "\n######## Control. Documented-language sanity check (upstream README example)"
$PY scripts/generate.py \
  --reference "$SNAP/prompts/PAN_F_HAPPY_00001.wav" \
  --transcript "ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।" \
  --text "नमस्ते! संगीत की तरह जीवन भी खूबसूरत होता है, बस इसे सही ताल में जीना आना चाहिए." \
  --output $CTRL_OUT

echo "\n######## Done. Listen:"
echo "  afplay $EN_OUT       # your voice, English (experimental)"
echo "  afplay $CTRL_OUT     # upstream control, Hindi"
