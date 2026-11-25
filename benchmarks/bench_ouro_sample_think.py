"""[bench_ouro_sample_think.py = bench_ouro_sample_all.py + the two chat tasks of the Thinking models]
Wall-clock: plain SAMPLING vs lossless speculative SAMPLING in the same batch-1 Ouro CUDA-graph engine for every method of the
greedy Ouro table (bench_ouro_baselines.py / bench_own.py / bench_engine.py are the greedy versions), on the prompts, stops and
limits of scripts/bench_greedy.sh.  Decode throughput = generated tokens / decode time (prefill excluded; the first token
comes from the prefill), tokens = len(ids) - 1 as in the greedy benches.
  python bench_ouro_sample_all.py <spec>
    spec: das:r,gamma:<model_dir>          sequential DAS (aligned reads)                    engine_baselines_sample.OuroBaselineSampleEngine
          wave:J,k,gamma:<model_dir>       DAS-Wave; with WAVE_GROW=0 WAVE_NOCARRY=1 (k = R) Jacobi decoding           (same engine)
          pld:n,gamma:<model_dir>          prompt lookup decoding                                                      (same engine)
          hybrid:J,k,gamma:<drafter run>   LAL + J own-readout refinement loops (WAVE_CARRY=1: from the drafter's slot states)
                                                                                             engine_baselines_sample.OuroOwnSampleEngine
          lal:n:<drafter run>              LAL, n drafter passes (n = 1: one pass)          engine_sample.OuroSampleEngine (unchanged)
    proposals (DAS_Q overrides): das / pld / Jacobi = the greedy drafts with one-hot q; DAS-Wave / hybrid / lal = d ~ q = softmax(last
    readouts / T), truncated to its own top-p nucleus (Q_TOPP=1).
  env: TASK (gsm8k | math500 | humaneval | mbpp | bbh: defaults of scripts/bench_greedy.sh for N_PROBLEMS 100/100/164/378/540, MAX_NEW
       512/2048/1024/1024/1024, STOP_STRS, TRACE = traces/<Model>-<task>.jsonl),
       N_PROBLEMS, MAX_NEW, STOP_STRS (STOP_SEP-separated, default ';;'), STOP_SEP, TRACE, TMAX (default max(4096, prompt + MAX_NEW
       rounded up to 256) = the greedy runs' 4096 whenever it fits), EVAL_TAG (e.g. _sh0of4), TEMP (1.0), TOP_P (0.7), Q_TOPP (1),
       DAS_Q (sample | argmax), TOPP_CAND (4096), SEED (1000; problem i: torch.cuda.manual_seed(SEED + i) before the plain AND before
       the speculative generation, i = the problem's index in the task's full trace when TRACE holds a subset of it, so a sharded
       run reproduces the unsharded one), WAVE_GROW / WAVE_NOCARRY / WAVE_CARRY / WAVE_READS / SEED_DELTA (as the greedy bench),
       OUT (runs/sampling_full), RESUME (1: skip a finished label, continue from <label>.partial.jsonl written on the same GPU type),
       GRAPHS (1), PIPE (1).
  out: <OUT>/<label>.json + <label>_outputs.jsonl, label = <kind>_<cfg with , -> ->_<model>_<task>_T<T>_p<top_p><EVAL_TAG><suffix>
       (suffix: _carry / _nodelta / _nogrow / _nocarry / _<reads> as the greedy bench, _q<mode> if DAS_Q is not the default,
       _noqtopp if Q_TOPP=0).  Per problem: idx, gidx, seed, plain / spec ids, tau (per-cycle accepted+1), dt_plain / dt_spec,
       end reasons.  "Capped" = ended at MAX_NEW without a stop string / EOS; aggregates over all problems and without capped
       generations (each side's own, and paired)."""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import hashlib, json, math, os, sys, time
import torch
import os
from floodgates.paths import model_dir, RUNS, ROOT, trace
from floodgates.models.ouro_min import OuroMin
from floodgates.tokenizer import load_tokenizer

HERE = ROOT   # repository root
CODE_STOPS = "<|endoftext|>;;<|endofmask|>;;</s>;;\nif __name__;;\ndef main(;;\nprint(;;\n# Test;;\nassert ;;\n```"
TASKS = {"gsm8k": (100, 100, 512, "Q:;;</s>"), "math500": (100, 100, 2048, "Problem:"), "humaneval": (164, 164, 1024, CODE_STOPS),
         "mbpp": (378, 378, 1024, CODE_STOPS), "bbh": (540, 20, 1024, "</s>;;Q;;\n\n")}          # N, trace n<D>, MAX_NEW, stops
TASKS.update({"gsm8kchat": (100, 100, 4096, "<|im_end|>"), "math500chat": (100, 100, 8192, "<|im_end|>")})   # Thinking models (chat template, thinking on): the limits and stop of the greedy Thinking baselines

spec = sys.argv[1]
kind, cfg, path = spec.split(":", 2)
assert kind in ("das", "wave", "hybrid", "pld", "lal"), kind
TASK = os.environ.get("TASK", "gsm8k"); assert TASK in TASKS, TASK
N0, D, MAXN0, STOPS0 = TASKS[TASK]
N = int(os.environ.get("N_PROBLEMS", N0)); MAX_NEW = int(os.environ.get("MAX_NEW", MAXN0)); TAG = os.environ.get("EVAL_TAG", "")
TEMP = float(os.environ.get("TEMP", 1.0)); TOP_P = float(os.environ.get("TOP_P", 0.7)); Q_TOPP = os.environ.get("Q_TOPP", "1") == "1"
SEED = int(os.environ.get("SEED", 1000)); RESUME = os.environ.get("RESUME", "1") == "1"
OUT = os.environ.get("OUT", os.path.join(RUNS, "sampling_full"))
v = [int(x) for x in cfg.split(",")]

# ---- method spec (as bench_ouro_baselines.py) ------------------------------------------------------------------------------
das, run_dr = None, None
if kind == "lal":
    n_rounds = v[0]; run_dr = path
elif kind in ("wave", "hybrid"):
    das = {"mode": "wave", "J": v[0], "k": v[1], "gamma": v[2]}
elif kind == "pld":
    das = {"mode": "pld", "n": v[0], "gamma": v[1]}
else:
    das = {"mode": "seq", "r": v[0], "gamma": v[1]}
if kind == "hybrid":
    das["init"] = "loopahead"; run_dr = path
    if int(os.environ.get("WAVE_CARRY", 0)):
        das["carry"] = True
        if os.environ.get("SEED_DELTA") == "0":
            das["seed_delta"] = False
if das is not None and das["mode"] == "wave":
    if os.environ.get("WAVE_GROW") == "0":
        das["grow"] = False
    if os.environ.get("WAVE_NOCARRY") == "1":
        das["carry_rounds"] = False
    if os.environ.get("WAVE_READS") in ("depth", "final"):
        das["reads"] = os.environ["WAVE_READS"]
if run_dr is not None:
    a = json.load(open(os.path.join(run_dr, "args.json"))); model = model_dir(a["model"])
else:
    model = path
jacobi = kind == "wave" and das.get("grow") is False and das.get("carry_rounds") is False
q_default = "sample" if (kind in ("hybrid", "lal") or (kind == "wave" and not jacobi)) else "argmax"
Q_MODE = os.environ.get("DAS_Q", q_default); assert Q_MODE in ("sample", "argmax"), Q_MODE
qt = Q_TOPP and Q_MODE == "sample"
mname = os.path.basename(model.rstrip("/"))
TR_FULL = trace(mname, TASK)
TR = os.environ.get("TRACE", TR_FULL)
suffix = ""
if das is not None:
    suffix = (("_carry" if das.get("carry") else "") + ("_nodelta" if das.get("seed_delta") is False else "")
              + ("_nogrow" if das.get("grow") is False else "") + ("_nocarry" if das.get("carry_rounds") is False else "")
              + (f"_{das['reads']}" if "reads" in das else ""))
suffix += (f"_q{Q_MODE}" if Q_MODE != q_default else "") + ("_noqtopp" if (Q_MODE == "sample" and not Q_TOPP) else "")
label = f"{kind}_{('n' + cfg) if kind == 'lal' else cfg.replace(',', '-')}_{mname}_{TASK}_T{TEMP}_p{TOP_P}{TAG}{suffix}"
os.makedirs(OUT, exist_ok=True); base = os.path.join(OUT, label)
if RESUME and os.path.exists(base + ".json"):
    print(f"{base}.json exists, skipped (RESUME=0 to rerun)", flush=True); sys.exit(0)

# ---- model, drafter, engine ------------------------------------------------------------------------------------------------
ouro = OuroMin(model)
dr = None
if run_dr is not None:
    from floodgates.drafters.drafter import LoopFlowDrafter
    dr = LoopFlowDrafter(ouro, k=a["k"], rank=a["rank"], kappa=a["kappa"], use_state=not a["no_state"], use_noise=not a["no_noise"],
                         flowmap=a.get("flowmap", False), feedback=a.get("feedback", False), markov_rank=a.get("markov_rank", 0))
    missing, unexpected = dr.load_state_dict(torch.load(os.path.join(run_dr, "drafter.pt")), strict=False)
    assert not unexpected and all(m.startswith("ouro.") or m == "draft_emb" for m in missing)
    dr.eval().requires_grad_(False)
    assert not dr.feedback and not dr.markov_rank and not dr.shift, "the one-pass drafter / hybrid seed take the plain drafter"
K = dr.k if kind == "lal" else das["gamma"]                                    # drafts per cycle
tok = load_tokenizer(model)
eos = tok.eos_token_id
STOPS = [x for x in os.environ.get("STOP_STRS", STOPS0).split(os.environ.get("STOP_SEP", ";;")) if x]
LB = max(12, max(len(tok(x, add_special_tokens=False)["input_ids"]) for x in STOPS) + K + 2)
stop = lambda out: any(x in tok.decode(out[-LB:]) for x in STOPS) or eos in out[-K - 1:]


def end_reason(out):
    if any(x in tok.decode(out) for x in STOPS):
        return "stop"
    if eos in out:
        return "eos"
    return "max" if len(out) >= MAX_NEW else "other"


rows = [json.loads(l) for l in open(TR)][:N]
# problem index for the seed: position in the task's full trace (sharded traces = subsets of it), else the local index
gidx, seed_index = list(range(len(rows))), "local"
if os.path.abspath(TR) != os.path.abspath(TR_FULL) and os.path.exists(TR_FULL):
    where = {}
    for j, l in enumerate(open(TR_FULL)):
        where.setdefault(tuple(json.loads(l)["input"]), []).append(j)
    g_ = []
    for r in rows:
        js = where.get(tuple(r["input"]), [])
        g_.append(js.pop(0) if js else None)
    if all(x is not None for x in g_):
        gidx, seed_index = g_, "global"
    else:
        print(f"WARNING: {sum(x is None for x in g_)} prompts of {TR} are not in {TR_FULL}: seeds use the local problem index", flush=True)
elif os.path.abspath(TR) == os.path.abspath(TR_FULL):
    seed_index = "global"
max_len = max(len(r["input"]) for r in rows) + MAX_NEW + K + 2
TMAX = int(os.environ.get("TMAX", max(4096, 256 * math.ceil(max_len / 256))))
assert TMAX >= max_len, (TMAX, max_len)
graphs = os.environ.get("GRAPHS", "1") == "1"
if kind == "lal":
    from floodgates.engines.engine_sample import OuroSampleEngine
    eng = OuroSampleEngine(ouro, dr, n_rounds=n_rounds, temp=TEMP, top_p=TOP_P, q_topp=Q_TOPP, q_mode=Q_MODE, tmax=TMAX, graphs=graphs)
else:
    from floodgates.engines.engine_baselines_sample import OuroBaselineSampleEngine, OuroOwnSampleEngine
    E = OuroOwnSampleEngine if kind == "hybrid" else OuroBaselineSampleEngine
    eng = E(ouro, das, dr, temp=TEMP, top_p=TOP_P, q_topp=Q_TOPP, q_mode=Q_MODE, tmax=TMAX, graphs=graphs)
print(f"{label}: engine {type(eng).__name__} das {das} model {mname} task {TASK} N {len(rows)} MAX_NEW {MAX_NEW} stops {STOPS} "
      f"T {TEMP} top-p {TOP_P} q_mode {Q_MODE} q_topp {qt} drafts/cycle {K} seed {SEED} ({seed_index} index) tmax {TMAX} trace {TR}",
      flush=True)
t = time.time(); eng.warmup(max_len)
print(f"graphs captured in {time.time() - t:.0f}s (attn {eng.attn_kind}, gemm {eng.gemm_kind}, pipe {eng.pipe}, fused {eng.fused})", flush=True)
torch.cuda.manual_seed(0)
eng.generate_plain(rows[0]["input"], 32, stop); eng.generate_spec(rows[0]["input"], 32, stop)       # warm-up runs (excluded)
if hasattr(eng, "n_nodraft"):
    eng.n_nodraft = 0
eng.uncov.zero_()

# ---- the timed runs --------------------------------------------------------------------------------------------------------
GPU = torch.cuda.get_device_name(); NODE = os.environ.get("HOSTNAME", "")
recs = []; part = base + ".partial.jsonl"
if RESUME and os.path.exists(part):
    for l in open(part):
        try:
            x = json.loads(l)
        except Exception:
            break
        if len(recs) >= len(rows) or x["idx"] != len(recs) or x.get("gpu") != GPU or x.get("gidx") != gidx[len(recs)]:
            break
        recs.append(x)
    print(f"resuming after {len(recs)} problems from {part}", flush=True)
with open(part, "w") as fpart:
    for x in recs:
        fpart.write(json.dumps(x) + "\n")
fpart = open(part, "a"); t_start = time.time(); nd0 = 0
for i, r in enumerate(rows):
    if i < len(recs):
        continue
    s_i = SEED + gidx[i]
    torch.cuda.manual_seed(s_i)
    o_p, dt_p = eng.generate_plain(r["input"], MAX_NEW, stop)
    torch.cuda.manual_seed(s_i)
    nd0 = getattr(eng, "n_nodraft", 0)
    o_s, tau, dt_s = eng.generate_spec(r["input"], MAX_NEW, stop)
    recs.append({"idx": i, "gidx": gidx[i], "seed": s_i, "plain": o_p, "spec": o_s, "tau": tau, "dt_plain": dt_p, "dt_spec": dt_s,
                 "end_plain": end_reason(o_p), "end_spec": end_reason(o_s), "gpu": GPU, "node": NODE,
                 "cycles_nodraft": (eng.n_nodraft - nd0) if kind == "pld" else None})
    fpart.write(json.dumps(recs[-1]) + "\n"); fpart.flush()
    if (i + 1) % 10 == 0 or i + 1 == len(rows):
        P = sum(len(x["plain"]) - 1 for x in recs) / sum(x["dt_plain"] for x in recs)
        S_ = sum(len(x["spec"]) - 1 for x in recs) / sum(x["dt_spec"] for x in recs)
        ts = [t_ for x in recs for t_ in x["tau"]]
        print(f"{i + 1}/{len(rows)} plain {P:.1f} tok/s  spec {S_:.1f} tok/s  speedup {S_ / P:.2f}  tau {sum(ts) / len(ts):.2f}  "
              f"capped plain {sum(x['end_plain'] == 'max' for x in recs)} spec {sum(x['end_spec'] == 'max' for x in recs)}  "
              f"({time.time() - t_start:.0f}s)", flush=True)


def agg(sel_p, sel_s):
    """Throughput / tau over the selected plain and speculative generations."""
    rp = [x for x in recs if sel_p(x)]; rs = [x for x in recs if sel_s(x)]
    tp = sum(x["dt_plain"] for x in rp); ts = sum(x["dt_spec"] for x in rs)
    np_ = sum(len(x["plain"]) - 1 for x in rp); ns = sum(len(x["spec"]) - 1 for x in rs)
    taus = [t_ for x in rs for t_ in x["tau"]]
    pt = np_ / tp if tp else None; st = ns / ts if ts else None
    return {"n_plain": len(rp), "n_spec": len(rs), "plain_tok_s": pt, "spec_tok_s": st, "speedup": st / pt if pt and st else None,
            "tau": sum(taus) / len(taus) if taus else None, "cycles": len(taus), "plain_tokens": np_, "spec_tokens": ns,
            "plain_seconds": tp, "spec_seconds": ts, "ms_per_plain_step": 1000 * tp / max(np_, 1),
            "ms_per_cycle": 1000 * ts / max(len(taus), 1)}


capped = lambda side: (lambda x: x[f"end_{side}"] == "max")
res = {"label": label, "kind": kind, "cfg": cfg, "engine": type(eng).__name__, "das": das, "drafter": run_dr, "model": mname,
       "model_dir": model, "task": TASK, "temp": TEMP, "top_p": TOP_P, "q_mode": Q_MODE, "q_topp": qt,
       "readout": "own" if kind == "hybrid" else None, "rounds": n_rounds if kind == "lal" else None, "k": K,
       "all": agg(lambda x: True, lambda x: True),
       "excl_capped": agg(lambda x: not capped("plain")(x), lambda x: not capped("spec")(x)),
       "excl_capped_paired": agg(lambda x: not (capped("plain")(x) or capped("spec")(x)), lambda x: not (capped("plain")(x) or capped("spec")(x))),
       "n_capped_plain": sum(capped("plain")(x) for x in recs), "n_capped_spec": sum(capped("spec")(x) for x in recs),
       "ends_plain": {k: sum(x["end_plain"] == k for x in recs) for k in ("stop", "eos", "max", "other")},
       "ends_spec": {k: sum(x["end_spec"] == k for x in recs) for k in ("stop", "eos", "max", "other")},
       "mean_len_plain": sum(len(x["plain"]) for x in recs) / len(recs), "mean_len_spec": sum(len(x["spec"]) for x in recs) / len(recs),
       "plain_outputs_sha1": hashlib.sha1(json.dumps([x["plain"] for x in recs]).encode()).hexdigest(),
       "cycles_nodraft": sum(x["cycles_nodraft"] for x in recs) if kind == "pld" else None,
       "topp_cand": eng.k_cand, "uncovered_rows": dict(zip(("plain_and_prefill", "verify_target", "proposal"), eng.uncov.tolist())),
       "n": len(recs), "seed": f"torch.cuda.manual_seed({SEED} + i) before plain and before spec, i = {seed_index} problem index",
       "seed_base": SEED, "seed_index": seed_index,
       "gpu": GPU, "torch": torch.__version__, "fused": eng.fused, "attn": eng.attn_kind, "gemm": eng.gemm_kind, "pipe": eng.pipe,
       "fp32p": getattr(eng.att, "fp32p", None), "graphs": eng.use_graphs, "tmax": TMAX, "max_new": MAX_NEW, "stops": STOPS,
       "trace": TR, "wall_seconds": time.time() - t_start, "nodes": sorted({x.get("node", "") for x in recs})}
res.update({k: res["all"][k] for k in ("plain_tok_s", "spec_tok_s", "speedup", "tau", "cycles", "plain_tokens", "spec_tokens",
                                       "ms_per_plain_step", "ms_per_cycle")})
print(json.dumps(res), flush=True)
json.dump(res, open(base + ".json", "w"), indent=1)
with open(base + "_outputs.jsonl", "w") as f:
    for x in recs:
        f.write(json.dumps(x) + "\n")
fpart.close(); os.remove(part)
print("wrote", base + ".json", flush=True)
