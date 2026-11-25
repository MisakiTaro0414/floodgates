#!/usr/bin/env bash
# Train the Floodgates drafter of an Ouro model with the paper's recipe (Appendix A).
#   DATA=data/mix4c_ouro14_greedy.jsonl bash scripts/train_ouro.sh                      # Ouro-1.4B
#   MODEL=Ouro-2.6B DATA=data/mix4c_ouro26_greedy.jsonl bash scripts/train_ouro.sh
#   MODEL=Ouro-1.4B-Thinking K=8 STEPS=16000 DATA=data/mix_chat_ouro14t_greedy.jsonl bash scripts/train_ouro.sh
# Extra arguments are passed to train.py.  Writes OUT/drafter.pt, OUT/args.json and OUT/train_log.jsonl.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODEL="${MODEL:-Ouro-1.4B}"
MODEL_DIR="${MODEL_DIR:-${LOOPSPEC_DIR:-../looped_refs/loopspec}/models/$MODEL}"
DATA="${DATA:?Set DATA to the self-generated training mix (see scripts/gen_data.sh)}"
OUT="${OUT:-runs/floodgates-$MODEL}"

python train.py --model "$MODEL_DIR" --data "$DATA" --out "$OUT" \
  --k "${K:-12}" --rank "${RANK:-64}" --anchors "${ANCHORS:-64}" --no-state --no-noise --rounds-max 1 \
  --steps "${STEPS:-32000}" --batch "${BATCH:-8}" --lr "${LR:-4e-4}" --save-every "${SAVE_EVERY:-2000}" "$@"
