#!/usr/bin/env bash
# Self-generated training data for a prelude-core-coda model: the model's greedy continuations on the four domains of the paper
# (GSM8K train, MATH train, CodeSearchNet function signatures, FineWeb-Edu prefixes), then the four-domain mix.
#   MODEL=Recurrent-Llama-3.2-train-recurrence-32 TAG=rl32 bash scripts/gen_data.sh
# For the Ouro models the continuations come from an SGLang server with the LoopSpec sglang_recurrent backend:
#   datagen/gen_data.py (GSM8K, MATH) and datagen/gen_data_chat.py (the Thinking models), see their docstrings.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODEL="${MODEL:-Recurrent-Llama-3.2-train-recurrence-32}"; TAG="${TAG:-rl32}"
MODEL_DIR="${MODEL_DIR:-${LOOPSPEC_DIR:-../looped_refs/loopspec}/models/$MODEL}"
mkdir -p data
for t in gsm8k math code general; do
  if [ "$t" = math ]; then export GEN_MAXNEW=1024; else unset GEN_MAXNEW; fi     # the paper's MATH continuations: at most 1,024 tokens
  case "$MODEL:$t" in huginn*) unset GEN_BOS ;; *:gsm8k|*:math) export GEN_BOS=1 ;; *) unset GEN_BOS ;; esac   # BOS as in the paper's data
  GEN_TASK=$t GEN_MODEL="$MODEL_DIR" python datagen/gen_raven_data2.py "data/${t}_train_${TAG}_greedy.jsonl" "${BATCH:-32}"
done
python datagen/build_mix.py "$TAG"     # data/mix4c_<tag>_greedy.jsonl, the file that scripts/train_*.sh take as DATA
