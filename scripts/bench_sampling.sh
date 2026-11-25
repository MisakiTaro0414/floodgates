#!/usr/bin/env bash
# Lossless speculative sampling at batch size one (temperature 1, top-p 0.7 in the paper), Floodgates and the training-free
# baselines against plain sampling in the same engine.  Task prompts, stops and limits are set from TASK inside the benchmarks.
#   RUN=runs/floodgates-Ouro-1.4B TASK=gsm8k bash scripts/bench_sampling.sh
#   RUN=runs/floodgates-Ouro-1.4B-Thinking TASK=gsm8kchat bash scripts/bench_sampling.sh
#   RUN=runs/floodgates-Recurrent-Llama-3.2-train-recurrence-32 TASK=gsm8k PASSES=24 bash scripts/bench_sampling.sh
#   SPEC=wave:24,1,32:rl32 TASK=gsm8k bash scripts/bench_sampling.sh
# PASSES, LOOPS_PER_PASS, WINDOW and MODEL as in scripts/bench_greedy.sh.  Results go to $OUT (default runs/sampling_full):
# <label>.json and <label>_outputs.jsonl; the paper's speedup, with the tokens counted up to the stop string, is printed at the end.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PASSES="${PASSES:-0}"
export TASK="${TASK:-gsm8k}" TEMP="${TEMP:-1.0}" TOP_P="${TOP_P:-0.7}"
MODEL="${MODEL:-$(python scripts/model_name.py "${SPEC:-${RUN:-}}")}"
case "$MODEL" in Recurrent-*|huginn-*) FAMILY="${FAMILY:-raven}" ;; *) FAMILY="${FAMILY:-ouro}" ;; esac   # the family of the base model
MODEL="${MODEL:-$([ "$FAMILY" = ouro ] && echo Ouro-1.4B || echo Recurrent-Llama-3.2-train-recurrence-32)}"
LPP="${LOOPS_PER_PASS:-$([ "$MODEL" = huginn-0125 ] && echo 2 || echo 1)}"
export TRACE="${TRACE:-traces/$MODEL-$TASK.jsonl}"    # prompts, as in traces/; the stops and limits of the task are set inside the benchmark
STAMP=$(mktemp)
if [ "$FAMILY" = ouro ]; then
  B=benchmarks/bench_ouro_sample_all.py; case $TASK in *chat) B=benchmarks/bench_ouro_sample_think.py ;; esac
  python $B "${SPEC:-lal:1:${RUN:?Set RUN to a drafter run directory}}"
elif [ -n "${SPEC:-}" ]; then
  python benchmarks/bench_raven_sample_all.py "$SPEC"
elif [ "$PASSES" -gt 0 ]; then
  python benchmarks/bench_raven_sample_all.py "hybrid:$PASSES,$LPP,${WINDOW:-32}:${RUN:?Set RUN}"
else
  python benchmarks/bench_raven_sample_all.py "${RUN:?Set RUN to a drafter run directory}" 1
fi
# the paper counts the tokens up to the stop string: recount the result files this run wrote
find "${FLOODGATES_RUNS:-runs}" ${RUN:+"$RUN"} -name "*.json" -newer "$STAMP" 2>/dev/null | sort -u | while read -r f; do
  python evaluation/stopcut.py "$f" || true
done || true
rm -f "$STAMP"
