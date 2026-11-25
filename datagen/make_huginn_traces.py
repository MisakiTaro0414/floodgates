"""The prompts of the Recurrent-Llama traces (the LoopSpec harness prompts) re-tokenized for Huginn: traces/huginn-0125-<task>.jsonl
(no BOS, as the harness prompts of every model).  The GSM8K file comes from gen_raven_trace.py instead.
    python datagen/make_huginn_traces.py math500 humaneval mbpp bbh"""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import json, sys
from transformers import PreTrainedTokenizerFast
from floodgates.paths import LOOPSPEC, trace
MD = LOOPSPEC + '/models'
tl = PreTrainedTokenizerFast(tokenizer_file=f"{MD}/Recurrent-Llama-3.2-train-recurrence-32/tokenizer.json")
th = PreTrainedTokenizerFast(tokenizer_file=f"{MD}/huginn-0125/tokenizer.json")
for t in sys.argv[1:]:
    rows = [json.loads(l) for l in open(trace("Recurrent-Llama-3.2-train-recurrence-32", t))]
    out = []
    for i, r in enumerate(rows):
        text = tl.decode(r["input"], skip_special_tokens=True)
        ids = th(text, add_special_tokens=False)["input_ids"]
        assert th.decode(ids, skip_special_tokens=True) == text, (t, i)
        out.append({"i": i, "input": ids, "output_ids": [], "completion_tokens": r.get("completion_tokens")})   # no Huginn reference outputs: "output_ids" is only an equality statistic in the bench
    with open(trace("huginn-0125", t), "w") as f:
        for r in out: f.write(json.dumps(r) + "\n")
    print(t, len(out), "prompts; huginn tokens max", max(len(r["input"]) for r in out), "first ids", out[0]["input"][:4])
