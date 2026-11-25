#!/usr/bin/env bash
# Download the trained drafters of the paper into runs/<name>/ (drafter.pt + args.json each).
#   bash scripts/download_checkpoints.sh                 # all seven
#   bash scripts/download_checkpoints.sh floodgates-Ouro-1.4B
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
HF_REPO="${HF_REPO:-MisakiTaro0414/floodgates}"
ALL="floodgates-Ouro-1.4B floodgates-Ouro-2.6B floodgates-Ouro-1.4B-Thinking floodgates-Ouro-2.6B-Thinking floodgates-Recurrent-Llama-3.2-train-recurrence-32 floodgates-Recurrent-TinyLlama-3T-train-recurrence-32 floodgates-huginn-0125"
for name in ${@:-$ALL}; do
  hf download "$HF_REPO" --include "$name/*" --local-dir runs/
done
