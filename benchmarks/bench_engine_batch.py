"""Real wall-clock: plain batched decoding vs LoopAhead (fixed k / cost-aware per-sequence block length / cost-aware one k
per batch) in engine_batch.BatchEngine, static waves of B prompts, greedy, on the prompts of a LoopSpec trace.
  python bench_engine_batch.py <run_dir> <B,B,...> [modes=plain,fixed,cost,costu]
env: SCHED (static | cont), N_PROBLEMS (100), MAX_NEW (512), STOP_STRS ("Q:;;</s>;;<|im_end|>", separator ";;"), TRACE (LoopSpec Ouro-1.4B GSM8K
     baseline trace), TMAX (1024), TAG (output subdir, default "ouro14_gsm8k"), REF (batch-1 engine outputs jsonl for the
     identity check), MERGE_LORA (1), FP32P (1), GEMM_TMAX (64), PREFIX (0; 1 = shared-prefix K/V cache, SCHED=cont),
     DET (0; 1 = batch-invariant kernels), OUTSFX (suffix of the output files), OUT (runs/batch).
Outputs: <OUT>/<TAG>/b<B>[_cont]<OUTSFX>.json (+ _outputs.jsonl with every mode's output ids per problem).
Per B: all graphs captured and the cost model T(B, N) measured first (not timed).
SCHED=static: waves of B prompts decoded together until all finish (finished rows idle; a last partial wave runs with its
own Bs-row graphs); the modes run back to back on the SAME wave (plain, fixed, cost, costu), so clock drift hits them alike.
SCHED=cont: continuous batching (BatchEngine.run_continuous: freed rows refilled from the queue, tail compaction to B/2,
B/4, ... rows), each mode over the whole prompt list.
Decode throughput = generated tokens after the first (prefill) token / decode wall time (prefill excluded, as
bench_engine.py and SGLang's decode tok/s); end-to-end = all generated tokens / (prefill + decode) wall time.  Tokens are
counted through the stop string (= SGLang completion_tokens); look-ahead tokens past the stop are not counted."""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import gc, json, os, sys, time
from collections import Counter
import torch
import os
from floodgates.paths import model_dir, trace
from floodgates.models.ouro_min import OuroMin
from floodgates.drafters.drafter import LoopFlowDrafter
from floodgates.engines.engine_batch import BatchEngine
from floodgates.tokenizer import load_tokenizer

run = sys.argv[1]
BATCHES = [int(x) for x in sys.argv[2].split(",")]
MODES = (sys.argv[3] if len(sys.argv) > 3 else "plain,fixed,cost,costu").split(",")
assert MODES[0] == "plain"
N = int(os.environ.get("N_PROBLEMS", 100)); MAX_NEW = int(os.environ.get("MAX_NEW", 512))
STOPS = [x for x in os.environ.get("STOP_STRS", "Q:;;</s>;;<|im_end|>").split(";;") if x]
TR = os.environ.get("TRACE", trace("Ouro-1.4B", "gsm8k"))
TMAX = int(os.environ.get("TMAX", 1024))
TAG = os.environ.get("TAG", "ouro14_gsm8k")
SCHED = os.environ.get("SCHED", "static")
REF = os.environ.get("REF", os.path.join(run, "engine_n1_gsm8k_outputs.jsonl"))
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
    n_stop = min(len(o), MAX_NEW)
    for n in range(1, min(len(o), MAX_NEW) + 1):
        if stop(o[:n]):
            return o[:n]
    return o[:n_stop]


ref_b1 = None
if os.path.exists(REF):
    ref_b1 = [cut_at_stop(json.loads(l)["plain"]) for l in open(REF)][:N]


def common(x, y):
    i = 0
    while i < min(len(x), len(y)) and x[i] == y[i]:
        i += 1
    return i


print(json.dumps({"gpu": torch.cuda.get_device_name(), "sched": SCHED, "run": run, "batches": BATCHES, "modes": MODES, "n": N, "max_new": MAX_NEW,
                  "stops": STOPS, "merge_lora": os.environ.get("MERGE_LORA", "1"), "fp32p": os.environ.get("FP32P", "1")}), flush=True)
for B in BATCHES:
    t = time.time()
    eng = BatchEngine(ouro, dr, B=B, tmax=TMAX)
    sizes = sorted({len(prompts[i:i + B]) for i in range(0, N, B)}, reverse=True) if SCHED == "static" else eng.ladder()
    for Bs in sizes:
        eng.prepare(Bs, MODES)
    print(f"[B={B}] graphs + cost model in {time.time() - t:.0f}s; widths {sizes}", flush=True)
    acc = {m: dict(outs=[None] * N, t_dec=0.0, t_pre=0.0, cycles=0, rows_cyc=[], choices=[], extra={}) for m in MODES}
    if SCHED == "static":
        for m in MODES:                                             # warm-up (host paths, allocator), not timed
            eng.run_wave(prompts[:B], m, 16, stop)
        for w0 in range(0, N, B):
            wave = prompts[w0:w0 + B]
            for m in MODES:
                r = eng.run_wave(wave, m, MAX_NEW, stop)
                A = acc[m]
                for i, o in enumerate(r["outs"]):
                    A["outs"][w0 + i] = o
                A["t_dec"] += r["t_decode"]; A["t_pre"] += r["t_prefill"]; A["cycles"] += len(r["cycles"])
                A["rows_cyc"] += [x for c in r["cycles"] for x in c]
                bl = eng.bucket_list(len(wave)) if r["choices"] else None
                A["choices"] += [bl[c][0] / len(wave) for c in r["choices"]]
    else:
        for m in MODES:
            eng.run_continuous(prompts[:min(N, 2 * B)], m, 16, stop)
        for m in MODES:
            r = eng.run_continuous(prompts, m, MAX_NEW, stop)
            A = acc[m]
            A["outs"] = r["outs"]; A["t_dec"] = r["t_decode"]; A["t_pre"] = r["t_prefill"]; A["cycles"] = len(r["cycles"])
            A["rows_cyc"] = [x for c in r["cycles"] for x in c]; A["choices"] = r["choices"]
            A["extra"] = {"t_total": r["t_total"], "n_prefill_calls": r["n_prefill_calls"], "n_compactions": r["n_compactions"],
                          "mean_width": sum(r["widths"]) / max(len(r["widths"]), 1),
                          "mean_active_rows": sum(len(c) for c in r["cycles"]) / max(len(r["cycles"]), 1),
                          "calib": r.get("calib")}
    res = {"B": B, "n": N, "gpu": torch.cuda.get_device_name(), "cost_model_ms": {str(k): v for k, v in eng.cost.items()}, "modes": {}}
    plain = acc["plain"]["outs"]
    for m in MODES:
        A = acc[m]
        dec_tok = sum(len(o) - 1 for o in A["outs"]); all_tok = sum(len(o) for o in A["outs"])
        rc = A["rows_cyc"]
        d = {"decode_tok_s": dec_tok / A["t_dec"], "e2e_tok_s": all_tok / (A["t_dec"] + A["t_pre"]), "decode_s": A["t_dec"],
             "prefill_s": A["t_pre"], "tokens": all_tok, "decode_tokens": dec_tok, "cycles": A["cycles"],
             "tau": sum(x[1] for x in rc) / max(len(rc), 1), "mean_k_b": sum(x[2] for x in rc) / max(len(rc), 1),
             "eq_plain": sum(int(o == p) for o, p in zip(A["outs"], plain)),
             "tok_agree_plain": sum(common(o, p) for o, p in zip(A["outs"], plain)) / sum(len(p) for p in plain),
             "eq_trace": sum(int(o == r_["output_ids"][:len(o)] and len(o) == len(r_["output_ids"])) for o, r_ in zip(A["outs"], rows))}
        if ref_b1:
            d["eq_b1_plain"] = sum(int(o == p) for o, p in zip(A["outs"], ref_b1))
        d.update(A["extra"])
        if m in ("cost", "costu"):
            d["pred_acc_per_row_cycle"] = sum(x[3] for x in rc) / max(len(rc), 1)
            d["real_acc_per_row_cycle"] = sum(x[1] - 1 for x in rc) / max(len(rc), 1)
            d["chosen_drafts_per_row"] = dict(sorted(Counter(round(c - 1, 2) for c in A["choices"]).items()))
        res["modes"][m] = d
    for m in MODES:
        res["modes"][m]["speedup_decode"] = res["modes"][m]["decode_tok_s"] / res["modes"]["plain"]["decode_tok_s"]
        res["modes"][m]["speedup_e2e"] = res["modes"][m]["e2e_tok_s"] / res["modes"]["plain"]["e2e_tok_s"]
    print(json.dumps({"B": B, "sched": SCHED, **{m: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["modes"][m].items()
                                       if k in ("decode_tok_s", "e2e_tok_s", "speedup_decode", "speedup_e2e", "tau", "mean_k_b",
                                                "eq_plain", "eq_b1_plain", "eq_trace", "pred_acc_per_row_cycle", "real_acc_per_row_cycle",
                                                "mean_active_rows", "n_compactions")}
                                  for m in MODES}}), flush=True)
    res["sched"] = SCHED; res["prefix_cache"] = os.environ.get("PREFIX", "0") == "1"; res["det"] = os.environ.get("DET", "0") == "1"
    sfx = ("" if SCHED == "static" else "_cont") + os.environ.get("OUTSFX", "")
    json.dump(res, open(os.path.join(OUT, f"b{B}{sfx}.json"), "w"), indent=1)
    with open(os.path.join(OUT, f"b{B}{sfx}_outputs.jsonl"), "w") as f:
        for i in range(N):
            f.write(json.dumps({m: acc[m]["outs"][i] for m in MODES}) + "\n")
    del eng; gc.collect(); torch.cuda.empty_cache()
print("ALL DONE", flush=True)
