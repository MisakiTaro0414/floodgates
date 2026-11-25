"""Print the base-model name of a drafter run, a benchmark spec (<kind>:<cfg>:<model or run>) or a model directory, so that the
scripts read the prompts traces/<model>-<task>.jsonl of the right model; prints nothing when it cannot tell.
    python scripts/model_name.py <run | spec>"""
import json, os, sys
SHORT = {"rl32": "Recurrent-Llama-3.2-train-recurrence-32", "tl32": "Recurrent-TinyLlama-3T-train-recurrence-32", "huginn": "huginn-0125"}
x = (sys.argv[1] if len(sys.argv) > 1 else "").rsplit(":", 1)[-1].rstrip("/")
if x and os.path.exists(os.path.join(x, "args.json")):
    print(os.path.basename(json.load(open(os.path.join(x, "args.json")))["model"].rstrip("/")))
elif x in SHORT:
    print(SHORT[x])
elif x and os.path.isdir(x):
    print(os.path.basename(x))
