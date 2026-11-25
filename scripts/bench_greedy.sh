#!/usr/bin/env bash
# Greedy decoding at batch size one: Floodgates against plain decoding in the same engine, on one benchmark, with the prompts,
# stop strings and generation limits of the paper (prompt traces of the LoopSpec harness, shipped under traces/).
#   RUN=runs/floodgates-Ouro-1.4B TASK=gsm8k bash scripts/bench_greedy.sh                                   # one drafting pass
#   RUN=runs/floodgates-Ouro-1.4B-Thinking TASK=gsm8kchat bash scripts/bench_greedy.sh                      # a Thinking model
#   RUN=runs/floodgates-Recurrent-Llama-3.2-train-recurrence-32 TASK=math500 PASSES=24 bash scripts/bench_greedy.sh
#   RUN=runs/floodgates-huginn-0125 TASK=gsm8k PASSES=24 bash scripts/bench_greedy.sh
#   SPEC=das:2,8:rl32 TASK=gsm8k bash scripts/bench_greedy.sh                                               # a training-free baseline
# TASK: gsm8k | math500 | humaneval | mbpp | bbh (Thinking models: gsm8kchat | math500chat).  The model, its family (ouro | raven)
# and its prompts follow RUN or SPEC (MODEL / FAMILY override them).  PASSES: the elastic drafting budget E (raven only; 0 = a single
# drafting pass); each pass runs LOOPS_PER_PASS loops (1; Huginn: 2) over WINDOW draft positions (32).
# SPEC overrides the run: das:r,gamma:<model>, wave:J,k,gamma:<model>, jacobi:J,k,gamma:<model>, pld:n,gamma:<model>, or base:<model_dir>
# for plain decoding only (see the docstrings of benchmarks/bench_raven_engine.py and benchmarks/bench_ouro_baselines.py).
# The benchmark prints its own speedup; the paper's speedup, with the tokens counted up to the stop string, is printed at the end.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
TASK="${TASK:-gsm8k}"; PASSES="${PASSES:-0}"
MODEL="${MODEL:-$(python scripts/model_name.py "${SPEC:-${RUN:-}}")}"
case "$MODEL" in Recurrent-*|huginn-*) FAMILY="${FAMILY:-raven}" ;; *) FAMILY="${FAMILY:-ouro}" ;; esac   # the family of the base model
NL=$'\n'
CODE_STOPS="<|endoftext|>;;<|endofmask|>;;</s>;;${NL}if __name__;;${NL}def main(;;${NL}print(;;${NL}# Test;;${NL}assert ;;${NL}\`\`\`"
case $TASK in
  gsm8k)     N=100; D=100; MAXN=512;  STOPS='Q:;;</s>;;<|im_end|>' ;;
  math500)   N=100; D=100; MAXN=2048; STOPS='Problem:' ;;
  humaneval) N=164; D=164; MAXN=1024; STOPS=$CODE_STOPS ;;
  mbpp)      N=378; D=378; MAXN=1024; STOPS=$CODE_STOPS ;;
  bbh)       N=540; D=20;  MAXN=1024; STOPS="</s>;;Q;;${NL}${NL}" ;;
  gsm8kchat)   N=100; D=100; MAXN=4096; STOPS='<|im_end|>'; export TMAX="${TMAX:-16384}" ;;
  math500chat) N=100; D=100; MAXN=8192; STOPS='<|im_end|>'; export TMAX="${TMAX:-16384}" ;;
  *) echo "unknown task $TASK"; exit 1 ;;
esac
STAMP=$(mktemp)
if [ "$FAMILY" = ouro ]; then
  MODEL="${MODEL:-Ouro-1.4B}"
  export N_PROBLEMS="${N_PROBLEMS:-$N}" MAX_NEW="${MAX_NEW:-$MAXN}" STOP_SEP=';;' STOP_STRS="${STOP_STRS:-$STOPS}" EVAL_TAG="${EVAL_TAG:-_$TASK}"
  export TRACE="${TRACE:-traces/$MODEL-$TASK.jsonl}"
  if [ -n "${SPEC:-}" ]; then python benchmarks/bench_ouro_baselines.py "$SPEC"; else python benchmarks/bench_engine.py "${RUN:?Set RUN to a drafter run directory}" 1; fi
else
  MODEL="${MODEL:-Recurrent-Llama-3.2-train-recurrence-32}"
  LPP="${LOOPS_PER_PASS:-$([ "$MODEL" = huginn-0125 ] && echo 2 || echo 1)}"
  export N_PROBLEMS="${N_PROBLEMS:-$N}" MAX_NEW="${MAX_NEW:-$MAXN}" STOP_SEP=';;' STOP_STRS="${STOP_STRS:-$STOPS}" EVAL_TAG="${EVAL_TAG:-_$TASK}" TF_TAU=0
  export TRACE="${TRACE:-traces/$MODEL-$TASK.jsonl}"
  if [ -n "${SPEC:-}" ]; then python benchmarks/bench_raven_engine.py "$SPEC" 1
  elif [ "$PASSES" -gt 0 ]; then WAVE_CARRY=1 python benchmarks/bench_raven_engine.py "hybrid:$PASSES,$LPP,${WINDOW:-32}:${RUN:?Set RUN}" 1
  else python benchmarks/bench_raven_engine.py "${RUN:?Set RUN to a drafter run directory}" 1; fi
fi
# the paper counts the tokens up to the stop string: recount the result files this run wrote
find "${FLOODGATES_RUNS:-runs}" ${RUN:+"$RUN"} -name "*.json" -newer "$STAMP" 2>/dev/null | sort -u | while read -r f; do
  python evaluation/stopcut.py "$f" || true
done || true
rm -f "$STAMP"
