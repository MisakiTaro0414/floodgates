"""Batched wall-clock of the training-free baselines (sequential DAS, prompt lookup) in the Ouro batched engine
(engine_batch_baselines.BatchBaselineEngine), continuous batching, greedy, on the prompts of a LoopSpec trace; the plain
batched decoding of the same engine is the reference.
  python bench_engine_batch_baselines.py <run_dir> <B,B,...> [modes=plain,das1_8,das2_8,pld3_12,pld2_8]
env: as bench_engine_batch.py (N_PROBLEMS, MAX_NEW, STOP_STRS, TRACE, TMAX, TAG, OUT, PREFIX, DET, OUTSFX), B1 (batch-1
     engine_baselines outputs prefix, default runs/ouro-baselines/<kind>_<cfg>_<model>_gsm8k_pp, for eq_b1).
Outputs: <OUT>/<TAG>/b<B>_baselines<OUTSFX>.json ('modes' as bench_engine_batch.py) + _outputs.jsonl."""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import gc, json, os, sys, time
import torch
import os
from floodgates.paths import model_dir, trace
from floodgates.models.ouro_min import OuroMin
from floodgates.drafters.drafter import LoopFlowDrafter
from floodgates.engines.engine_batch_baselines import BatchBaselineEngine, parse_bmode
from floodgates.tokenizer import load_tokenizer

run = sys.argv[1]
BATCHES = [int(x) for x in sys.argv[2].split(",")]
MODES = (sys.argv[3] if len(sys.argv) > 3 else "plain,das1_8,das2_8,pld3_12,pld2_8").split(",")
assert MODES[0] == "plain"
N = int(os.environ.get("N_PROBLEMS", 100)); MAX_NEW = int(os.environ.get("MAX_NEW", 512))
STOPS = [x for x in os.environ.get("STOP_STRS", "Q:;;</s>;;<|im_end|>").split(";;") if x]
TR = os.environ.get("TRACE", trace("Ouro-1.4B", "gsm8k"))
TMAX = int(os.environ.get("TMAX", 1024))
TAG = os.environ.get("TAG", "ouro14_gsm8k")
OUT = os.path.join(os.environ.get("OUT", "runs/batch"), TAG)
os.makedirs(OUT, exist_ok=True)

a = json.load(open(os.path.join(run, "args.json")))
ouro = OuroMin(model_dir(a["model"]))
dr = LoopFlowDrafter(ouro, k=a["k"], rank=a["rank"], kappa=a["kappa"], use_state=not a["no_state"], use_noise=not a["no_noise"],
                     flowmap=a.get("flowmap", False), shift=a.get("shift", False), feedback=a.get("feedback", False))
missing, unexpected = dr.load_state_dict(torch.load(os.path.join(run, "drafter.pt")), strict=False)
assert not unexpected and all(m.startswith("ouro.") or m == "draft_emb" for m in missing)
dr.eval().requires_grad_(False)
tok = load_tokenizer(model_dir(a["model"]))
LB = max(12, max(len(tok(x, add_special_tokens=False)["input_ids"]) for x in STOPS) + dr.k + 2)
stop = lambda out: any(x in tok.decode(out[-LB:]) for x in STOPS) or tok.eos_token_id in out[-dr.k - 1:]
rows = [json.loads(l) for l in open(TR)][:N]
prompts = [r["input"] for r in rows]
assert max(len(p) for p in prompts) + MAX_NEW + 3 * (dr.k + 1) <= TMAX, "TMAX too small"


def cut_at_stop(o):
    for q in range(1, min(len(o), MAX_NEW) + 1):
        if stop(o[:q]):
            return o[:q]
    return o[:MAX_NEW]


def common(x, y):
    i = 0
    while i < min(len(x), len(y)) and x[i] == y[i]:
        i += 1
    return i


mname = os.path.basename(model_dir(a["model"]).rstrip("/"))
ref1 = {}
for md in MODES:
    if md == "plain":
        continue
    kind, cfg = md[:3], md[3:].replace("_", "-")
    stem = os.environ.get("B1", "runs/ouro-baselines") + f"/{kind}_{cfg}_{mname}_gsm8k_pp"
    if os.path.exists(stem + "_outputs.jsonl") and "gsm8k" in TR and MAX_NEW == 512:
        R = [json.loads(l) for l in open(stem + "_outputs.jsonl")]
        js = json.load(open(stem + ".json"))
        ref1[md] = dict(stem=stem, outs=[cut_at_stop(r["spec"]) for r in R][:N], plain=[cut_at_stop(r["plain"]) for r in R][:N],
                        speedup=js["speedup"], tau=js["tau"])
print(json.dumps({"gpu": torch.cuda.get_device_name(), "run": run, "batches": BATCHES, "modes": MODES, "n": N,
                  "b1_refs": sorted(ref1), "det": os.environ.get("DET", "0"), "prefix": os.environ.get("PREFIX", "0")}), flush=True)
for B in BATCHES:
    t = time.time()
    eng = BatchBaselineEngine(ouro, dr, B=B, tmax=TMAX, gmax=max(parse_bmode(m).get("nd", 0) for m in MODES))   # buffers bound the largest block
    for Bs in eng.ladder():
        eng.prepare(Bs, MODES)
    t_prep = time.time() - t
    print(f"[B={B}] {len(eng.graphs)} graphs in {t_prep:.0f}s", flush=True)
    for md in MODES:
        eng.run_continuous(prompts[:min(N, 2 * B)], md, 16, stop)
    acc = {}
    for md in MODES:
        acc[md] = eng.run_continuous(prompts, md, MAX_NEW, stop)
    res = {"B": B, "n": N, "gpu": torch.cuda.get_device_name(), "sched": "cont", "det": os.environ.get("DET", "0") == "1",
           "prefix_cache": os.environ.get("PREFIX", "0") == "1", "t_prepare_s": t_prep, "modes": {}}
    plain = acc["plain"]["outs"]
    for md in MODES:
        r = acc[md]
        rc = [x for c in r["cycles"] for x in c]
        dec_tok = sum(len(o) - 1 for o in r["outs"]); all_tok = sum(len(o) for o in r["outs"])
        d = {"decode_tok_s": dec_tok / r["t_decode"], "e2e_tok_s": all_tok / (r["t_decode"] + r["t_prefill"]), "decode_s": r["t_decode"],
             "prefill_s": r["t_prefill"], "tokens": all_tok, "decode_tokens": dec_tok, "cycles": len(r["cycles"]),
             "tau": sum(x[1] for x in rc) / max(len(rc), 1), "mean_k_b": sum(x[2] for x in rc) / max(len(rc), 1),
             "eq_plain": sum(int(o == p) for o, p in zip(r["outs"], plain)),
             "tok_agree_plain": sum(common(o, p) for o, p in zip(r["outs"], plain)) / sum(len(p) for p in plain),
             "eq_trace": sum(int(o == r_["output_ids"]) for o, r_ in zip(r["outs"], rows)),
             "n_compactions": r["n_compactions"], "mean_active_rows": sum(len(c) for c in r["cycles"]) / max(len(r["cycles"]), 1)}
        if md in ref1:
            d["eq_b1"] = sum(int(o == q) for o, q in zip(r["outs"], ref1[md]["outs"]))
            d["b1_speedup_raw"] = ref1[md]["speedup"]; d["b1_tau"] = ref1[md]["tau"]; d["b1_ref"] = ref1[md]["stem"]
        if parse_bmode(md)["cat"] == "pld":
            d["match_rate"] = sum(int(x[2] > 0) for x in rc) / max(len(rc), 1)
        res["modes"][md] = d
    for md in MODES:
        res["modes"][md]["speedup_decode"] = res["modes"][md]["decode_tok_s"] / res["modes"]["plain"]["decode_tok_s"]
        res["modes"][md]["speedup_e2e"] = res["modes"][md]["e2e_tok_s"] / res["modes"]["plain"]["e2e_tok_s"]
    keep = ("decode_tok_s", "e2e_tok_s", "speedup_decode", "tau", "eq_plain", "eq_b1", "b1_speedup_raw", "match_rate")
    print(json.dumps({"B": B, **{md: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["modes"][md].items() if k in keep}
                                 for md in MODES}}), flush=True)
    sfx = "_baselines" + os.environ.get("OUTSFX", "")
    json.dump(res, open(os.path.join(OUT, f"b{B}{sfx}.json"), "w"), indent=1)
    with open(os.path.join(OUT, f"b{B}{sfx}_outputs.jsonl"), "w") as f:
        for i in range(N):
            f.write(json.dumps({md: acc[md]["outs"][i] for md in MODES}) + "\n")
    del eng, acc; gc.collect(); torch.cuda.empty_cache()
print("ALL DONE", flush=True)
