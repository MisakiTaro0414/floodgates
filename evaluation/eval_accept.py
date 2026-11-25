"""Offline accepted length of greedy speculative decoding with the looped-flow drafter.
Reference = the SGLang server's deterministic greedy outputs (LoopSpec baseline trace, 100 GSM8K test problems).  A cycle at
emitted token y_m: draft k tokens from the context (< P+m) + y_m; accept the longest prefix equal to the reference; the
verifier then emits one more reference token -> tokens per verification tau = accepted + 1.  Loop-equivalent tokens per
full 4-loop step: gamma' = tau * R / (R + n_rounds) (a draft round costs one loop at batch 1).
  python eval_accept.py <run_dir> [n_rounds ...]
External-drafter runs (args.json ext_layers > 0, drafter_ext.py): one pass per cycle (rounds default to 1), the drafter's
context K/V computed once per sequence; a draft pass costs ext_layers / 24 loops in the loop-equivalent number."""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import json, os, sys
import torch
import os
from floodgates.paths import model_dir, trace
from floodgates.models.ouro_min import OuroMin
from floodgates.drafters.drafter import LoopFlowDrafter, draft_commit
from floodgates.drafters.drafter_ext import ExtDrafter
run = sys.argv[1]
TR = os.environ.get("TRACE", trace("Ouro-1.4B", "gsm8k"))
TAG = os.environ.get("EVAL_TAG", "")   # e.g. "_math500" -> output files carry the benchmark
a = json.load(open(os.path.join(run, "args.json")))
ext = a.get("ext_layers", 0) > 0
rounds_list = [int(x) for x in sys.argv[2:]] or ([1] if ext else [1, 2, 3])
ouro = OuroMin(model_dir(a["model"]))
if ext:
    dr = ExtDrafter(ouro, k=a["k"], n_layers=a["ext_layers"], init=a.get("ext_init", "random"))
else:
    dr = LoopFlowDrafter(ouro, k=a["k"], rank=a["rank"], kappa=a["kappa"], use_state=not a["no_state"], use_noise=not a["no_noise"], flowmap=a.get("flowmap", False), shift=a.get("shift", False), ctx_loop=a.get("ctx_loop"), feedback=a.get("feedback", False), markov_rank=a.get("markov_rank", 0))
sd = torch.load(os.path.join(run, os.environ.get("DRAFTER_PT", "drafter.pt")))   # DRAFTER_PT=drafter_step4000.pt for a checkpoint
missing, unexpected = dr.load_state_dict(sd, strict=False)
assert not unexpected and all(m.startswith("ouro.") or m == "draft_emb" for m in missing), (unexpected, [m for m in missing if not m.startswith("ouro.")])
dr.eval()
rows = [json.loads(l) for l in open(TR)][: int(os.environ.get('N_PROBLEMS', 100))]
res = {}
for n in rounds_list:
    gen = torch.Generator(device="cuda").manual_seed(0)
    K = a["k"] + (1 if a.get("shift") else 0)
    taus, pos_acc = [], [[0, 0] for _ in range(K)]
    for r in rows:
        p, o = r["input"], r["output_ids"]
        ids = torch.tensor([p + o], device="cuda")
        kv = dr.prepare(ids) if ext else ouro.forward_context(ids, keep_kv_loop=a.get("ctx_loop"), logits=False)[1]
        m = 0
        while m < len(o) - 1:
            d = (draft_commit(dr, kv, len(p) + m, o[m], n_rounds=n, gen=gen) if a.get("commit") else dr.draft(kv, len(p) + m, o[m], n_rounds=n, gen=gen))
            ref = o[m + 1: m + 1 + K]
            acc = 0
            for j, (x, y) in enumerate(zip(d, ref)):
                pos_acc[j][1] += 1; pos_acc[j][0] += int(x == y)
            while acc < len(ref) and d[acc] == ref[acc]:
                acc += 1
            taus.append(acc + 1); m += acc + 1
    tau = sum(taus) / len(taus)
    cost = a["ext_layers"] / ouro.L if ext else n          # draft cost in loops: ext = one pass of ext_layers layers
    res[n] = {"tau": tau, "gamma_loop_equiv": tau * 4 / (4 + cost), "cycles": len(taus),
              "pos_acc": [round(c / t, 3) if t else None for c, t in pos_acc]}
    print(f"rounds={n}: tokens per verification tau={tau:.3f}, loop-equivalent per 4-loop step={tau * 4 / (4 + cost):.3f}, "
          f"cycles={len(taus)}, per-slot acc={res[n]['pos_acc']}", flush=True)
json.dump(res, open(os.path.join(run, f"accept{TAG}.json"), "w"), indent=1)
