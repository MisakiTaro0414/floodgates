#!/usr/bin/env bash
# Download the base models of the paper from Hugging Face into $LOOPSPEC_DIR/models/<name>, where the scripts read them
# (default ../looped_refs/loopspec/models, the layout of a LoopSpec checkout).
#   bash scripts/download_models.sh                         # all seven
#   bash scripts/download_models.sh ByteDance/Ouro-1.4B
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
DIR="${LOOPSPEC_DIR:-../looped_refs/loopspec}/models"
ALL="ByteDance/Ouro-1.4B ByteDance/Ouro-2.6B ByteDance/Ouro-1.4B-Thinking ByteDance/Ouro-2.6B-Thinking
     smcleish/Recurrent-Llama-3.2-train-recurrence-32 smcleish/Recurrent-TinyLlama-3T-train-recurrence-32 tomg-group-umd/huginn-0125"
for m in ${@:-$ALL}; do
  hf download "$m" --local-dir "$DIR/${m#*/}"
done
