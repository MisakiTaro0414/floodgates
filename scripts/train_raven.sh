#!/usr/bin/env bash
# Train the Floodgates drafter of a prelude-core-coda model (Recurrent-Llama-3.2, Recurrent-TinyLlama, Huginn-3.5B) with the
# paper's recipe (Appendix A).  --shift is the next-token readout of these models: k mask tokens give k + 1 drafts.
#   DATA=data/mix4c_rl32_greedy.jsonl bash scripts/train_raven.sh                                         # Recurrent-Llama-3.2
#   MODEL=Recurrent-TinyLlama-3T-train-recurrence-32 DATA=data/mix4c_tl32_greedy.jsonl bash scripts/train_raven.sh
#   MODEL=huginn-0125 DATA=data/mix4c_huginn_greedy.jsonl bash scripts/train_raven.sh
# Extra arguments are passed to train_raven.py.  Writes OUT/drafter.pt, OUT/args.json and OUT/train_log.jsonl.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODEL="${MODEL:-Recurrent-Llama-3.2-train-recurrence-32}"
MODEL_DIR="${MODEL_DIR:-${LOOPSPEC_DIR:-../looped_refs/loopspec}/models/$MODEL}"
DATA="${DATA:?Set DATA to the self-generated training mix (see scripts/gen_data.sh)}"
OUT="${OUT:-runs/floodgates-$MODEL}"

python train_raven.py --model "$MODEL_DIR" --data "$DATA" --out "$OUT" --shift \
  --k "${K:-12}" --rank "${RANK:-64}" --anchors "${ANCHORS:-64}" \
  --steps "${STEPS:-32000}" --batch "${BATCH:-8}" --lr "${LR:-4e-4}" --warmup "${WARMUP:-100}" --save-every "${SAVE_EVERY:-1000}" "$@"
