"""Smoke / correctness test of engine_batch.BatchEngine against the batch-1 engine (engine.py) on a few GSM8K prompts.
  python test_engine_batch.py [B=4] [max_new=96] [modes=plain,fixed,cost,costu]
Checks: batched plain rows == batch-1 plain outputs (up to bf16 near-ties); LoopAhead modes == batched plain; per-mode
decode time; cost-model table."""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import json, os, sys, time
import torch
import os
from floodgates.paths import RUNS, model_dir, trace
from floodgates.models.ouro_min import OuroMin
from floodgates.drafters.drafter import LoopFlowDrafter
from floodgates.engines.engine import Engine
from floodgates.engines.engine_batch import BatchEngine
from floodgates.tokenizer import load_tokenizer

B = int(sys.argv[1]) if len(sys.argv) > 1 else 4
MAX_NEW = int(sys.argv[2]) if len(sys.argv) > 2 else 96
MODES = (sys.argv[3] if len(sys.argv) > 3 else "plain,fixed,cost,costu").split(",")
RUN = os.environ.get("RUN", os.path.join(RUNS, "floodgates-Ouro-1.4B"))
TR = os.environ.get("TRACE", trace("Ouro-1.4B", "gsm8k"))
a = json.load(open(os.path.join(RUN, "args.json")))
ouro = OuroMin(model_dir(a["model"]))
dr = LoopFlowDrafter(ouro, k=a["k"], rank=a["rank"], kappa=a["kappa"], use_state=not a["no_state"], use_noise=not a["no_noise"],
                     flowmap=a.get("flowmap", False), shift=a.get("shift", False), feedback=a.get("feedback", False))
missing, unexpected = dr.load_state_dict(torch.load(os.path.join(RUN, "drafter.pt")), strict=False)
assert not unexpected and all(m.startswith("ouro.") or m == "draft_emb" for m in missing)
dr.eval().requires_grad_(False)
tok = load_tokenizer(model_dir(a["model"]))
STOPS = ["Q:", "</s>", "<|im_end|>"]
LB = max(12, max(len(tok(x, add_special_tokens=False)["input_ids"]) for x in STOPS) + dr.k + 2)
stop = lambda out: any(x in tok.decode(out[-LB:]) for x in STOPS) or tok.eos_token_id in out[-dr.k - 1:]
rows = [json.loads(l) for l in open(TR)][:B]
prompts = [r["input"] for r in rows]

# batch-1 reference (engine.py plain), outputs cut at the stop like engine_batch
e1 = Engine(ouro, None, tmax=1024)
e1.warmup(1024)
ref = []
for p in prompts:
    o, _ = e1.generate_plain(p, MAX_NEW, stop)
    n_stop = min(len(o), MAX_NEW)
    for n in range(1, len(o) + 1):
        if stop(o[:n]):
            n_stop = n; break
    ref.append(o[:n_stop])
del e1; torch.cuda.empty_cache()

eng = BatchEngine(ouro, dr, B=B, tmax=1024)
t = time.time(); eng.prepare(B, MODES); print(f"graphs captured in {time.time() - t:.0f}s", flush=True)
if B in eng.cost:
    c = eng.cost[B]
    print("cost model (ms):", json.dumps({"plain": round(c["plain_ms"] or -1, 3), "draft": round(c["draft_ms"], 3), "policy": round(c["policy_ms"], 3),
                                          "verify": {k: round(v, 3) for k, v in c["verify_ms"].items()}}), flush=True)


def common(x, y):
    i = 0
    while i < min(len(x), len(y)) and x[i] == y[i]:
        i += 1
    return i


outs = {}
for mode in MODES:
    eng.run_wave(prompts, mode, 16, stop)                      # warm-up (host paths)
    r = eng.run_wave(prompts, mode, MAX_NEW, stop)
    outs[mode] = r["outs"]
    ntok = sum(len(o) - 1 for o in r["outs"])
    rec = [x for c in r["cycles"] for x in c]
    tau = sum(x[1] for x in rec) / max(len(rec), 1)
    kb = sum(x[2] for x in rec) / max(len(rec), 1)
    same_ref = sum(int(o == q) for o, q in zip(r["outs"], ref))
    agree = sum(common(o, q) for o, q in zip(r["outs"], ref)) / sum(len(q) for q in ref)
    line = f"{mode:6s}: {ntok} tok in {r['t_decode'] * 1e3:.1f} ms = {ntok / r['t_decode']:.1f} tok/s; prefill {r['t_prefill'] * 1e3:.1f} ms; " \
           f"tau {tau:.2f} mean k_b {kb:.2f}; identical to batch-1 plain {same_ref}/{B} (token agreement {agree:.3f})"
    if mode != "plain" and "plain" in outs:
        same_p = sum(int(o == q) for o, q in zip(r["outs"], outs["plain"]))
        line += f"; identical to batched plain {same_p}/{B}"
    if mode in ("cost", "costu"):
        line += f"; E[acc] pred {sum(x[3] for x in rec) / max(len(rec), 1):.2f} real {sum(x[1] - 1 for x in rec) / max(len(rec), 1):.2f}"
    if r["choices"]:
        from collections import Counter
        bl = eng.bucket_list(B)
        line += f"; chosen N {dict(Counter(bl[c][0] for c in r['choices']))}"
    print(line, flush=True)
for b in range(min(B, 2)):
    print(f"row {b} plain:", repr(tok.decode(outs.get("plain", ref)[b][:60])))
    for mode in MODES[1:]:
        o, q = outs[mode][b], outs.get("plain", ref)[b]
        if o != q:
            i = common(o, q)
            print(f"  {mode} diverges at {i}: {tok.decode(o[i:i + 8])!r} vs {tok.decode(q[i:i + 8])!r}")
