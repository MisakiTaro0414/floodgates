"""[TASK=gsm8k|math500|humaneval|mbpp|bbh STRIDE=1] Our reimplementation of the diffusion-forcing sampler of Geiping et al. (training-free, not lossless) vs plain decoding in the
same eager RavenMin code (only the ratio is comparable with the CUDA-graph engine).  Wall-clock tok/s, accuracy, passes per token.
  OUT=<json> MAX_NEW=512 python df_eval.py <model_dir> <trace.jsonl> <N> [inner=4] [eps=0.03] [eta=0.1]
Cost model (batch-1 memory-bound layer reads): AR = pre + core*R + post per token; a pass = pre + core*inner + post."""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import json, os, re, sys, time
import torch
import os
from floodgates.paths import LOOPSPEC
from floodgates.models.raven_min import RavenMin
from transformers import PreTrainedTokenizerFast
M, TR, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
inner = int(sys.argv[4]) if len(sys.argv) > 4 else 4; eps = float(sys.argv[5]) if len(sys.argv) > 5 else 0.03
eta = float(sys.argv[6]) if len(sys.argv) > 6 else 0.1
tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(M, "tokenizer.json"))
m = RavenMin(M); c = m.c
pre, core, post = c["n_layers_in_prelude"], c["n_layers_in_recurrent_block"], c["n_layers_in_coda"]
import glob
gold = {}                                                                   # offline gold answers (question text -> answer)
for f in glob.glob(LOOPSPEC + '/runs/Ouro-1.4B-gsm8k-n100/baseline/samples_*.jsonl'):
    for l in open(f):
        d = json.loads(l)["doc"]; gold[d["question"].strip()] = d["answer"].split("####")[-1].strip().replace(",", "")
num = lambda s: (re.findall(r"answer is\s*\$?(-?[\d,]*\.?\d+)", s) or [None])[0]
TASK = os.environ.get("TASK", "gsm8k"); STRIDE = int(os.environ.get("STRIDE", 1))
CODE_STOPS = ["<|endoftext|>", "<|endofmask|>", "</s>", "\nif __name__", "\ndef main(", "\nprint(", "\n# Test", "\nassert ", "\n```"]
STOPS, MAXN = {"gsm8k": (["Q:", "</s>"], 512), "math500": (["Problem:"], 2048), "humaneval": (CODE_STOPS, 1024), "mbpp": (CODE_STOPS, 1024), "bbh": (["</s>", "Q", "\n\n"], 1024)}[TASK]
MAXN = int(os.environ.get("MAX_NEW", MAXN))
_tc = json.load(open(os.path.join(M, "tokenizer_config.json"))); _e = _tc.get("eos_token"); _e = _e["content"] if isinstance(_e, dict) else _e
eos = tok.convert_tokens_to_ids(_e) if _e else c.get("eos_token_id")
LB = max(12, max(len(tok(x, add_special_tokens=False)["input_ids"]) for x in STOPS) + 2)
stop = lambda out: any(x in tok.decode(out[-LB:]) for x in STOPS) or out[-1] == eos
rows = [json.loads(l) for l in open(TR)][::STRIDE][:N]
print(f"task {TASK}: {len(rows)} prompts (stride {STRIDE}), max_new {MAXN}, eos {_e} {eos}", flush=True)
def correct(o, q):
    a = num(tok.decode(o)); g = gold.get(q)
    return int(a is not None and g is not None and a.replace(",", "").rstrip(".") == g)
t0 = time.time(); res = {}
for name in ("ar", "df"):
    acc = ntok = npass = nit = 0; secs = 0.0
    for i, r in enumerate(rows):
        q = re.findall(r"Q: (.*?)\nA:", tok.decode(r["input"]), re.S)[-1].strip() if TASK == "gsm8k" else None
        torch.cuda.synchronize(); t = time.time()
        if name == "df":
            o, passes, its = m.df_generate(r["input"], MAXN, stop, inner=inner, eps=eps, eta=eta); npass += passes; nit += its
        else:
            o = m.generate(r["input"], MAXN, stop); o = o[0] if isinstance(o, tuple) else o
        torch.cuda.synchronize(); secs += time.time() - t
        acc += correct(o, q) if TASK == "gsm8k" else 0; ntok += len(o)
        if i % 20 == 19: print(f"{name}: {i + 1}/{len(rows)} {ntok / secs:.1f} tok/s acc {acc / (i + 1):.2f}", flush=True)
    res[name] = {"tok_s": ntok / secs, "acc": acc / len(rows), "tokens": ntok, "sec": secs}
    if name == "df": res[name].update({"passes_per_token": npass / ntok, "iters_per_token": nit / ntok, "inner": inner, "eps": eps, "eta": eta})
res["ratio"] = res["df"]["tok_s"] / res["ar"]["tok_s"]
print(json.dumps(res), f"({time.time() - t0:.0f}s)", flush=True)
json.dump(res, open(os.environ.get("OUT", "df_eval_out.json"), "w"), indent=1)
