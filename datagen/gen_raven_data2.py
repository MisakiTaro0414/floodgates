"""Self-distillation data for the prelude-core-coda models in any of gen_data.py's domains (env GEN_TASK = code | general |
gsm8k | math), generated greedily with RavenMin.generate_batch; same output format as gen_data.py (i, prompt_ids, output_ids,
text), resumable by i.   GEN_TASK=code GEN_MODEL=<model_dir> python gen_raven_data2.py <out.jsonl> [batch=32] [limit]
GEN_BOS=1 starts every prompt with the tokenizer's BOS token, as in the paper's GSM8K and MATH data of Recurrent-Llama and
Recurrent-TinyLlama (made by an earlier version of this script); their code and FineWeb-Edu data and all Huginn data have none."""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import json, os, sys, time
import torch
OUT = sys.argv[1]; BS = int(sys.argv[2]) if len(sys.argv) > 2 else 32; LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else None
sys.argv = [sys.argv[0], OUT]                       # gen_data reads argv[1] as OUT
import gen_data as G                                # builds G.qs, G.prefix, G.FMT, G.STOP, G.MAXNEW, G.tok, G.MODEL
from floodgates.models.raven_min import RavenMin

m = RavenMin(G.MODEL); tok = G.tok
_tc = json.load(open(os.path.join(G.MODEL, "tokenizer_config.json")))          # eos as the bench does (the config's id is Huginn's)
_e = _tc.get("eos_token"); _e = _e["content"] if isinstance(_e, dict) else _e
eos = tok.convert_tokens_to_ids(_e) if _e else tok.eos_token_id
print("eos", _e, eos, flush=True)
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: done.add(json.loads(l)["i"])
        except Exception: pass
qs = G.qs[:LIMIT] if LIMIT else G.qs
SHARD, NSHARD = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))      # SHARD/NSHARD: prompts i with i % NSHARD == SHARD
MAXNEW = int(os.environ.get("GEN_MAXNEW", G.MAXNEW))
BOS = os.environ.get("GEN_BOS", "0") == "1"
todo = [(i, tok(G.prefix + G.FMT(q), add_special_tokens=BOS)["input_ids"]) for i, q in enumerate(qs) if i not in done and i % NSHARD == SHARD]
todo = [(i, p) for i, p in todo if len(p) + MAXNEW < 4000]
todo.sort(key=lambda x: len(x[1]))
print(f"{len(todo)} prompts to generate ({len(done)} done, shard {SHARD}/{NSHARD}), max_new {MAXNEW}, stops {G.STOP}, R={m.R}", flush=True)
STOP_LB = max(8, max(len(tok(x, add_special_tokens=False)["input_ids"]) for x in G.STOP) + 2)
def stopped(out):
    if out and out[-1] == eos: return True
    return any(x in tok.decode(out[-STOP_LB:]) for x in G.STOP)
t0 = time.time(); n = 0
with open(OUT, "a") as f:
    for k in range(0, len(todo), BS):
        chunk = todo[k:k + BS]
        outs = m.generate_batch([p for _, p in chunk], MAXNEW, stopped)
        for (i, p), o in zip(chunk, outs):
            f.write(json.dumps({"i": i, "prompt_ids": p, "output_ids": o, "text": tok.decode(o)}) + "\n"); n += len(o)
        f.flush()
        print(f"{k + len(chunk)}/{len(todo)} prompts, {n} tokens, {n / (time.time() - t0):.0f} tok/s", flush=True)
print("GEN_DONE", len(qs), flush=True)
