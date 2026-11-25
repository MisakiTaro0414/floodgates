#!/usr/bin/env bash
# Batched serving with continuous batching: system throughput of plain decoding and of Floodgates with the confidence-based
# verification budget ("cost"), across batch sizes, on the full GSM8K test set (1,319 prompts) as in the paper.
#   RUN=runs/floodgates-Ouro-1.4B bash scripts/bench_serving.sh                                                   # Ouro
#   RUN=runs/floodgates-Recurrent-Llama-3.2-train-recurrence-32 BATCHES=1,2,4,8,16,32,64,128 bash scripts/bench_serving.sh
# MODES selects what runs (default plain,cost; raven also: fixed, ref<J>, ref<J>c, wave<J>, das<r>_<g>, pld<n>_<g>; Ouro: fixed, costu).
# Results: $OUT/$TAG/b<B>_cont.json and _outputs.jsonl (OUT defaults to runs/batch).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODEL="${MODEL:-$(python scripts/model_name.py "${RUN:-}")}"
case "$MODEL" in Recurrent-*|huginn-*) FAMILY="${FAMILY:-raven}" ;; *) FAMILY="${FAMILY:-ouro}" ;; esac   # the family of the base model
MODEL="${MODEL:-$([ "$FAMILY" = ouro ] && echo Ouro-1.4B || echo Recurrent-Llama-3.2-train-recurrence-32)}"; BATCHES="${BATCHES:-1,2,4,8,16,32}"; MODES="${MODES:-plain,cost}"
export TRACE="${TRACE:-traces/$MODEL-gsm8k-full.jsonl}" N_PROBLEMS="${N_PROBLEMS:-1319}" SCHED=cont PREFIX="${PREFIX:-1}" TMAX="${TMAX:-1088}" DET="${DET:-1}" OUT="${OUT:-runs/batch}"
if [ "$FAMILY" = ouro ]; then
  TAG="${TAG:-$(basename "${RUN:?Set RUN}")_gsm8k}" python benchmarks/bench_engine_batch.py "$RUN" "$BATCHES" "$MODES"
else
  RUN="${RUN:?Set RUN}" TAG="${TAG:-$(basename "$RUN")_gsm8k}" PREFILL=graph REFCMP=0 python benchmarks/bench_raven_engine_batch.py "$BATCHES" "$MODES"
fi
