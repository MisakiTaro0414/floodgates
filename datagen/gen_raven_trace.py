"""Reference traces for a Raven-family model: greedy GSM8K-CoT (3-shot, LoopSpec format, stop 'Q:', no BOS) with RavenMin.generate.
  python gen_raven_trace.py <model_dir> <out.jsonl> [N=100] [split=test]   (resumable; one JSON line per problem)"""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
from floodgates.paths import ROOT
import json, os, sys, time
from floodgates.models.raven_min import RavenMin
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
M, OUT = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 100
split = sys.argv[4] if len(sys.argv) > 4 else "test"
tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(M, "tokenizer.json"))
m = RavenMin(M)
prefix = open(os.path.join(ROOT, "datagen", "prompts", "gsm8k_cot_3shot_prefix.txt")).read()
qs = load_dataset("openai/gsm8k", "main", split=split)["question"][:N]
done = {json.loads(l)["i"] for l in open(OUT)} if os.path.exists(OUT) else set()
eos = m.c.get("eos_token_id")
stop = lambda out: "Q:" in tok.decode(out[-8:]) or out[-1] == eos
with open(OUT, "a") as f:
    for i, q in enumerate(qs):
        if i in done:
            continue
        t = time.time()
        inp = tok(prefix + "Q: " + q + "\nA:", add_special_tokens=False)["input_ids"]
        o = m.generate(inp, 512, stop)
        f.write(json.dumps({"i": i, "input": inp, "output_ids": o, "text": tok.decode(o)}) + "\n"); f.flush()
        print(f"{i}: {len(o)} tokens {time.time() - t:.1f}s", flush=True)
