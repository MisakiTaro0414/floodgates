"""Real wall-clock of continuous-batching decoding of a Raven-family model (raven_engine_batch.RavenBatchEngine): plain batched
decoding vs LoopAhead (one pass: fixed block / per-sequence cost rule; with refinement: J loops over the window, fixed or
cost rule), DAS-Wave, sequential DAS and prompt lookup, greedy, on the prompts of a LoopSpec trace.
  python bench_raven_engine_batch.py <B,B,...> [modes]
  modes (default plain,fixed,cost,ref8,ref24,ref8c,ref24c,wave8,wave24,das2_8,pld3_12,pld2_8; plain must come first):
    plain | fixed (LAL one pass, all k+1 = 13 drafts) | cost (LAL one pass, per-sequence block length by the cost rule) |
    ref<J> (LAL + J of the model's loops over the 32-draft window, carried states, all 32 verified) | ref<J>c (same, cost
    rule) | wave<J> (DAS-Wave J,1,32) | das<r>_<g> (sequential DAS r,g) | pld<n>_<g> (prompt lookup n,g)
env: RUN (drafter run dir, runs/floodgates-Recurrent-Llama-3.2-train-recurrence-32), N_PROBLEMS (100), MAX_NEW (512), STOP_STRS ('Q:;;</s>', separator ';;'),
     TRACE (traces/Recurrent-Llama-3.2-train-recurrence-32-gsm8k.jsonl), TMAX (1024), TAG (rl32_gsm8k), OUT (runs/batch), OUTSFX,
     DET (0; 1 = batch-invariant target steps), ELT (auto: inductor at B = 1 without DET, else triton), REFCMP (1: compare
     with the batch-1 engine's outputs of each mode), PREFILL (eager = the batch-1 engine's prefill | graph), PREFIX (0; 1 =
     shared-prefix K/V cache, needs PREFILL=graph).
Outputs: <OUT>/<TAG>/b<B>_cont<OUTSFX>.json ('modes': decode_tok_s, e2e_tok_s, tau, eq_plain, eq_b1, speedup_decode, ...)
and _outputs.jsonl.  Decode throughput = generated tokens after the first (prefill) token / decode wall time (prefill
excluded); end-to-end = all generated tokens / (prefill + decode); tokens counted through the stop (look-ahead tokens past
the stop are not counted).  Each mode runs over the whole prompt list on the same engine (same GPU); graphs and the cost
model are prepared first (not timed); a short warm-up run per mode precedes the timed runs."""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import gc, json, os, sys, time
from collections import Counter
import torch
import os
from floodgates.paths import model_dir, RUNS, ROOT, trace
from floodgates.models.raven_min import RavenMin
from floodgates.drafters.drafter_raven import RavenSelfDrafter
from floodgates.engines.raven_engine_batch import RavenBatchEngine, parse_mode
from transformers import PreTrainedTokenizerFast

HERE = ROOT   # repository root
BATCHES = [int(x) for x in sys.argv[1].split(",")]
MODES = (sys.argv[2] if len(sys.argv) > 2 else "plain,fixed,cost,ref8,ref24,ref8c,ref24c,wave8,wave24,das2_8,pld3_12,pld2_8").split(",")
assert MODES[0] == "plain"
RUN = os.environ.get("RUN", os.path.join(RUNS, "floodgates-Recurrent-Llama-3.2-train-recurrence-32"))
N = int(os.environ.get("N_PROBLEMS", 100)); MAX_NEW = int(os.environ.get("MAX_NEW", 512))
STOPS = [x for x in os.environ.get("STOP_STRS", "Q:;;</s>").split(";;") if x]
TR = os.environ.get("TRACE", trace("Recurrent-Llama-3.2-train-recurrence-32", "gsm8k"))
TMAX = int(os.environ.get("TMAX", 1024))
TAG = os.environ.get("TAG", "rl32_gsm8k")
DET = os.environ.get("DET", "0") == "1"
OUT = os.path.join(os.environ.get("OUT", os.path.join(RUNS, "batch")), TAG)
os.makedirs(OUT, exist_ok=True)

a = json.load(open(os.path.join(RUN, "args.json")))
model = model_dir(a["model"])
m = RavenMin(model)
dr = RavenSelfDrafter(m, k=a["k"], rank=a["rank"], shift=a.get("shift", False), feedback=a.get("feedback", False),
                      iters=a.get("iters", 1), carry=a.get("carry", False), causal=a.get("causal", False))
missing, unexpected = dr.load_state_dict(torch.load(os.path.join(RUN, "drafter.pt")), strict=False)
assert not unexpected and all(x.startswith("m.") or x == "draft_emb" for x in missing)
dr.eval().requires_grad_(False)
tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model, "tokenizer.json"))
tc = json.load(open(os.path.join(model, "tokenizer_config.json")))
eos_tok = tc.get("eos_token"); eos_tok = eos_tok["content"] if isinstance(eos_tok, dict) else eos_tok
eos = tok.convert_tokens_to_ids(eos_tok) if eos_tok else None
KMAX = 33
LB = max(12, max(len(tok(x, add_special_tokens=False)["input_ids"]) for x in STOPS) + KMAX + 2)
stop = lambda out: any(x in tok.decode(out[-LB:]) for x in STOPS) or (eos is not None and eos in out[-KMAX - 1:])
rows = [json.loads(l) for l in open(TR)][:N]
prompts = [r["input"] for r in rows]
assert max(len(p) for p in prompts) + MAX_NEW + 3 * KMAX <= TMAX, "TMAX too small"


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


# batch-1 engine (raven_engine.py, A100) runs of every mode: outputs (cut at the stop like here) and speedups
W = os.path.basename(model.rstrip("/")); RD = os.path.join(RUNS, "raven"); DE = os.path.join(RD, "das-engine")
B1 = {"plain": (f"{RD}/ld-rl32-mix4-k12/engine_n1_gsm8k_pp4", "plain"), "fixed": (f"{RD}/ld-rl32-mix4-k12/engine_n1_gsm8k_pp4", "spec"),
      "cost": (f"{RD}/ld-rl32-mix4-k12/engine_n1_gsm8k_pp4", "spec"),
      "das2_8": (f"{DE}/das_2-8_{W}_gsm8k_pp", "spec"), "pld3_12": (f"{DE}/pld_3-12_{W}_gsm8k_pp", "spec"),
      "pld2_8": (f"{DE}/pld_2-8_{W}_gsm8k_pp", "spec")}
for J in (8, 24):
    for c in ("", "c"):
        B1[f"ref{J}{c}"] = (f"{DE}/hybrid_{J}-1-32_{W}_gsm8k_pp4_carry", "spec")
        B1[f"wave{J}{c}"] = (f"{DE}/wave_{J}-1-32_{W}_gsm8k_pp", "spec")
ref1 = {}
if os.environ.get("REFCMP", "1") == "1" and "ld-rl32-mix4-k12" in RUN and "gsm8k" in TR and MAX_NEW == 512:
    for md, (stem, key) in B1.items():
        if md in MODES and os.path.exists(stem + "_outputs.jsonl"):
            R = [json.loads(l) for l in open(stem + "_outputs.jsonl")]
            js = json.load(open(stem + ".json"))
            sp = [cut_at_stop(r[key]) for r in R]; pl = [cut_at_stop(r["plain"]) for r in R]
            if key == "spec":
                cut_speedup = (sum(len(o) - 1 for o in sp) / sum(r["dt_spec"] for r in R)) / (sum(len(o) - 1 for o in pl) / sum(r["dt_plain"] for r in R))
                pl_ts, sp_ts = js["plain_tok_s"], js["spec_tok_s"]
            else:
                cut_speedup, pl_ts, sp_ts = 1.0, js["plain_tok_s"], js["plain_tok_s"]
            ref1[md] = dict(stem=os.path.relpath(stem, HERE), outs=sp[:N], speedup_raw=js["speedup"] if key == "spec" else 1.0,
                            speedup_cut=cut_speedup, tok_s=sp_ts, plain_tok_s=pl_ts, tau=js.get("tau") if key == "spec" else 1.0)

print(json.dumps({"gpu": torch.cuda.get_device_name(), "run": RUN, "batches": BATCHES, "modes": MODES, "n": N, "max_new": MAX_NEW,
                  "stops": STOPS, "det": DET, "elt": os.environ.get("ELT", "auto"), "b1_refs": sorted(ref1)}), flush=True)
for B in BATCHES:
    t = time.time()
    eng = RavenBatchEngine(m, dr, B=B, tmax=TMAX, det=DET)
    for Bs in eng.ladder():
        eng.prepare(Bs, MODES)
    t_prep = time.time() - t
    print(f"[B={B}] {len(eng.graphs)} graphs + cost model in {t_prep:.0f}s; widths {eng.ladder()}; elt {eng.elt}", flush=True)
    step_ms = eng.time_steps(B, MODES)
    for md in MODES:                                               # warm-up (host paths, allocator), not timed
        eng.run_continuous(prompts[:min(N, 2 * B)], md, 16, stop)
    acc = {}
    for md in MODES:
        r = eng.run_continuous(prompts, md, MAX_NEW, stop)
        acc[md] = r
        rc = [x for c in r["cycles"] for x in c]
        dec_tok = sum(len(o) - 1 for o in r["outs"])
        print(f"[B={B}] {md}: {dec_tok / r['t_decode']:.1f} tok/s decode, tau {sum(x[1] for x in rc) / max(len(rc), 1):.2f}, "
              f"{r['t_decode']:.1f}s decode + {r['t_prefill']:.1f}s prefill", flush=True)
    res = {"B": B, "n": N, "gpu": torch.cuda.get_device_name(), "det": DET, "elt": eng.elt, "sched": "cont", "prefix_cache": os.environ.get("PREFIX", "0") == "1",
           "prefill": os.environ.get("PREFILL", "eager"), "prefix_len": getattr(eng, "pref_len", 0),
           "t_prepare_s": t_prep, "n_graphs": len(eng.graphs), "step_ms_at_L400": step_ms,
           "cost_model_ms": {f"{k[0]}|{k[1]}": v for k, v in eng.cost.items()}, "modes": {}}
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
             "t_total": r["t_total"], "n_prefill_calls": r["n_prefill_calls"], "n_compactions": r["n_compactions"],
             "mean_width": sum(r["widths"]) / max(len(r["widths"]), 1),
             "mean_active_rows": sum(len(c) for c in r["cycles"]) / max(len(r["cycles"]), 1), "desc": md,
             "calib": r.get("calib"), "share": eng.share}
        if md in ref1:
            R1 = ref1[md]
            d["eq_b1"] = sum(int(o == q) for o, q in zip(r["outs"], R1["outs"]))
            d["tok_agree_b1"] = sum(common(o, q) for o, q in zip(r["outs"], R1["outs"])) / max(sum(len(q) for q in R1["outs"]), 1)
            d["b1_ref"] = R1["stem"]; d["b1_speedup_raw"] = R1["speedup_raw"]; d["b1_speedup_cut"] = R1["speedup_cut"]
            d["b1_tok_s"] = R1["tok_s"]; d["b1_tau"] = R1["tau"]
        if "plain" in ref1:
            d["eq_b1_plain"] = sum(int(o == q) for o, q in zip(r["outs"], ref1["plain"]["outs"]))
        mi = parse_mode(md, eng.nd_dr)
        if mi["cat"] == "cost":
            d["pred_acc_per_row_cycle"] = sum(x[3] for x in rc) / max(len(rc), 1)
            d["real_acc_per_row_cycle"] = sum(x[1] - 1 for x in rc) / max(len(rc), 1)
            d["chosen_drafts_per_row"] = dict(sorted(Counter(round(c - 1, 2) for c in r["choices"]).items()))
        if mi["cat"] == "pld":
            d["match_rate"] = sum(int(x[2] > 0) for x in rc) / max(len(rc), 1)
        res["modes"][md] = d
    for md in MODES:
        res["modes"][md]["speedup_decode"] = res["modes"][md]["decode_tok_s"] / res["modes"]["plain"]["decode_tok_s"]
        res["modes"][md]["speedup_e2e"] = res["modes"][md]["e2e_tok_s"] / res["modes"]["plain"]["e2e_tok_s"]
    keep = ("decode_tok_s", "e2e_tok_s", "speedup_decode", "speedup_e2e", "tau", "mean_k_b", "eq_plain", "eq_b1", "eq_b1_plain",
            "b1_speedup_cut", "eq_trace", "pred_acc_per_row_cycle", "real_acc_per_row_cycle", "match_rate", "n_compactions")
    print(json.dumps({"B": B, **{md: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["modes"][md].items() if k in keep}
                                 for md in MODES}}), flush=True)
    sfx = "_cont" + os.environ.get("OUTSFX", "")
    json.dump(res, open(os.path.join(OUT, f"b{B}{sfx}.json"), "w"), indent=1)
    with open(os.path.join(OUT, f"b{B}{sfx}_outputs.jsonl"), "w") as f:
        for i in range(N):
            f.write(json.dumps({md: acc[md]["outs"][i] for md in MODES}) + "\n")
    del eng, acc; gc.collect(); torch.cuda.empty_cache()
print("ALL DONE", flush=True)
