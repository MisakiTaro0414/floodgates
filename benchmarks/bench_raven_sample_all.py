"""Wall-clock: plain SAMPLING vs lossless speculative SAMPLING in the same batch-1 CUDA-graph engine
(raven_engine_sample_all.RavenSampleEngineAll = raven_engine_sample.RavenSampleEngine + sequential DAS / PLD) for every greedy
baseline of the 32-loop table, on the prompts / stops / limits of the greedy runs (scripts/bench_greedy.sh).
  python bench_raven_sample_all.py <spec> [n_rounds]
  spec:  das:r,gamma:<model>          sequential DAS (Mature-Spec), argmax drafts, one-hot q
         wave:J,k,gamma:<model>       DAS-Wave; DAS_Q (sample: q = last-round readout softmax, top-p truncated with Q_TOPP=1 |
                                      argmax: one-hot); WAVE_GROW=0 + WAVE_NOCARRY=1 = Jacobi decoding (q defaults to argmax)
         jacobi:J,k,gamma:<model>     Jacobi decoding (= wave with WAVE_GROW=0 WAVE_NOCARRY=1), argmax drafts, one-hot q
         pld:n,gamma:<model>          prompt lookup decoding, one-hot q (no n-gram match -> a plain sampling step)
         hybrid:J,k,gamma:<run_dir>   LAL + J of the model's own loops (DAS_Q sample default; WAVE_CARRY=1 default = carried
                                      slot states, 0 = restart; WAVE_STOP as bench_raven_engine.py)
         <run_dir>                    LAL (trained drafter), n_rounds passes (default 1), q = final-pass softmax (Q_TOPP=1)
         <model> = a model directory or rl32 | tl32 | huginn.
  env:   TASK (gsm8k | math500 | humaneval | mbpp | bbh; defaults = scripts/bench_greedy.sh: N_PROBLEMS 100/100/164/378/540, MAX_NEW
         512/2048/1024/1024/1024 (TinyLlama MATH-500: 1024; TinyLlama BBH refused), STOP_STRS, TRACE per model and task),
         N_PROBLEMS, MAX_NEW, STOP_STRS (STOP_SEP-separated, default ';;' as scripts/bench_greedy.sh), TRACE, EVAL_TAG,
         SHARD=k/n (rows k::n of the first N_PROBLEMS; problem seeds stay those of the unsharded run; EVAL_TAG default
         _sh<k>of<n>), IDX_OFFSET / IDX_STRIDE (global index = IDX_OFFSET + IDX_STRIDE * i for a hand-made subset TRACE;
         default 0 / 1), TEMP (1.0; 0 = exact greedy), TOP_P (0.7), Q_TOPP (1), TOPP_CAND (4096), SEED (1000: problem with
         global index g: torch.cuda.manual_seed(SEED + g) before the plain AND before the speculative generation), OUT
         (runs/sampling_full), BUCKET (256), GRAPHS (1).
Writes <OUT>/<label>.json and <label>_outputs.jsonl, label = <method>_<cfg>_<model>_<task>_T<T>_p<top_p><EVAL_TAG><suffixes>
(method das | daswave | jacobi | pld | lal | lalref).  Speedup = (sum spec tokens / sum spec seconds) / (sum plain tokens /
sum plain seconds), tokens = len(ids) - 1 (the first token comes from the untimed prefill), as bench_raven_engine.py.
"Capped" = the generation reached MAX_NEW without a stop string / EOS; aggregates also without the capped generations."""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import hashlib, json, os, re, sys, time
import torch
import os
from floodgates.paths import LOOPSPEC, RUNS, ROOT, trace
from floodgates.paths import model_dir as paths_model_dir
from floodgates.models.raven_min import RavenMin
from floodgates.drafters.drafter_raven import RavenSelfDrafter
from floodgates.engines.raven_engine_sample_all import RavenSampleEngineAll
from transformers import PreTrainedTokenizerFast

HERE = ROOT   # repository root
L = LOOPSPEC
MODELS = {"rl32": "Recurrent-Llama-3.2-train-recurrence-32", "tl32": "Recurrent-TinyLlama-3T-train-recurrence-32", "huginn": "huginn-0125"}
SHORT = {v: k for k, v in MODELS.items()}
CODE_STOPS = "<|endoftext|>;;<|endofmask|>;;</s>;;\nif __name__;;\ndef main(;;\nprint(;;\n# Test;;\nassert ;;\n```"
TASKS = {"gsm8k": (100, 100, 512, "Q:;;</s>"), "math500": (100, 100, 2048, "Problem:"),       # (N, harness -n<D>, MAX_NEW, stops)
         "humaneval": (164, 164, 1024, CODE_STOPS), "mbpp": (378, 378, 1024, CODE_STOPS), "bbh": (540, 20, 1024, "</s>;;Q;;\n\n")}


def model_dir(x):                                    # rl32 | tl32 | huginn, a model directory, or the model of a drafter's args.json
    return f"{L}/models/{MODELS[x]}" if x in MODELS else paths_model_dir(x.rstrip("/"))


spec = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
TASK = os.environ.get("TASK", "gsm8k"); TN, TD, TMAX_NEW, TSTOPS = TASKS[TASK]
TEMP = float(os.environ.get("TEMP", 1.0)); TOP_P = float(os.environ.get("TOP_P", 0.7)); Q_TOPP = os.environ.get("Q_TOPP", "1") == "1"
SEED = int(os.environ.get("SEED", 1000)); BUCKET = int(os.environ.get("BUCKET", 256)); K_CAND = int(os.environ.get("TOPP_CAND", 4096))
OUT = os.environ.get("OUT", os.path.join(RUNS, "sampling_full"))
das, run_dr, a, suffix = None, None, None, ""
kind = spec.split(":", 1)[0] if ":" in spec and spec.split(":", 1)[0] in ("das", "wave", "jacobi", "pld", "hybrid") else "lal"
if kind == "lal":
    run_dr = spec.rstrip("/")
    a = json.load(open(os.path.join(run_dr, "args.json"))); model = model_dir(a["model"])
    method, cfg = "lal", f"{os.path.basename(run_dr)}-n{n}"
    if os.environ.get("ADAPT_MAX"):
        suffix += f"_adapt{os.environ['ADAPT_MAX']}{os.environ.get('ADAPT_RULE', 'conf')}{os.environ.get('ADAPT_DELTA', '0.8')}"
    qt = Q_TOPP
else:
    _, c, target = spec.split(":", 2); v = [int(x) for x in c.split(",")]
    cfg = c.replace(",", "-")
    if kind == "das":
        das = {"mode": "seq", "r": v[0], "gamma": v[1], "q": "argmax"}; method = "das"; model = model_dir(target)
    elif kind == "pld":
        das = {"mode": "pld", "n": v[0], "gamma": v[1], "q": "argmax"}; method = "pld"; model = model_dir(target)
    else:
        das = {"mode": "wave", "J": v[0], "k": v[1], "gamma": v[2]}
        if kind == "jacobi" or (os.environ.get("WAVE_GROW") == "0" and os.environ.get("WAVE_NOCARRY") == "1"):
            das.update(grow=False, carry_rounds=False); method = "jacobi"; model = model_dir(target)
            das["q"] = os.environ.get("DAS_Q", "argmax")
            suffix += "" if das["q"] == "argmax" else f"_q{das['q']}"
        elif kind == "wave":
            method = "daswave"; model = model_dir(target)
            if os.environ.get("WAVE_GROW") == "0":
                das["grow"] = False; suffix += "_nogrow"
            if os.environ.get("WAVE_NOCARRY") == "1":
                das["carry_rounds"] = False; suffix += "_nocarry"
            das["q"] = os.environ.get("DAS_Q", "sample")
        else:                                                                   # hybrid: drafting pass + the model's own loops
            method = "lalref"; run_dr = target.rstrip("/")
            a = json.load(open(os.path.join(run_dr, "args.json"))); model = model_dir(a["model"])
            cfg += f"-{os.path.basename(run_dr)}"
            das.update(init="loopahead", q=os.environ.get("DAS_Q", "sample"))
            if os.environ.get("WAVE_CARRY", "1") == "1":
                das["carry"] = True
        if method in ("daswave", "lalref"):
            suffix = f"_q{das['q']}" + ("-full" if das["q"] == "sample" and not Q_TOPP else "") + suffix
        if method == "lalref":
            suffix += "_carry" if das.get("carry") else "_restart"
        if kind in ("wave", "hybrid") and int(os.environ.get("WAVE_STOP", 0)):
            das["stop"] = int(os.environ["WAVE_STOP"]); suffix += f"_stop{das['stop']}"
    qt = Q_TOPP and das.get("q") == "sample"
MS = SHORT.get(os.path.basename(model.rstrip("/")), os.path.basename(model.rstrip("/")))
assert not (MS == "tl32" and TASK == "bbh"), "TinyLlama BBH: 3-shot prompts up to 2012 tokens + 1024 new tokens exceed its 2048 context"
N = int(os.environ.get("N_PROBLEMS", TN))
MAX_NEW = int(os.environ.get("MAX_NEW", 1024 if (MS == "tl32" and TASK == "math500") else TMAX_NEW))
TR = os.environ.get("TRACE", trace(model, TASK))
rows = [json.loads(l) for l in open(TR)][:N]
gidx = [int(os.environ.get("IDX_OFFSET", 0)) + int(os.environ.get("IDX_STRIDE", 1)) * i for i in range(len(rows))]
SHARD = os.environ.get("SHARD")
if SHARD:
    sk, sn = (int(x) for x in SHARD.split("/"))
    rows, gidx = rows[sk::sn], gidx[sk::sn]
TAG = os.environ.get("EVAL_TAG", f"_sh{sk}of{sn}" if SHARD else "")
suffix += (f"_cand{K_CAND}" if K_CAND != 4096 else "") + (f"_seed{SEED}" if SEED != 1000 else "")
label = f"{method}_{cfg}_{MS}_{TASK}_T{TEMP}_p{TOP_P}{TAG}{suffix}"
os.makedirs(OUT, exist_ok=True); base = os.path.join(OUT, label)

m = RavenMin(model)
dr = None
if run_dr is not None:
    dr = RavenSelfDrafter(m, k=a["k"], rank=a["rank"], shift=a.get("shift", False), feedback=a.get("feedback", False), iters=a.get("iters", 1),
                          carry=a.get("carry", False), causal=a.get("causal", False))
    missing, unexpected = dr.load_state_dict(torch.load(os.path.join(run_dr, "drafter.pt")), strict=False)
    assert not unexpected and all(x.startswith("m.") or x == "draft_emb" for x in missing), (unexpected, [x for x in missing if not x.startswith("m.")])
    dr.eval().requires_grad_(False)
K = das["gamma"] if das is not None else a["k"] + (1 if a.get("shift") else 0)
tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model, "tokenizer.json"))
tc = json.load(open(os.path.join(model, "tokenizer_config.json")))
eos_tok = tc.get("eos_token"); eos_tok = eos_tok["content"] if isinstance(eos_tok, dict) else eos_tok
eos = tok.convert_tokens_to_ids(eos_tok) if eos_tok else None
STOPS = [x for x in os.environ.get("STOP_STRS", TSTOPS).split(os.environ.get("STOP_SEP", ";;")) if x]
LB = max(12, max(len(tok(x, add_special_tokens=False)["input_ids"]) for x in STOPS) + K + 2)
stop = lambda out: any(x in tok.decode(out[-LB:]) for x in STOPS) or (eos is not None and eos in out[-K - 1:])


def end_reason(out):
    if any(x in tok.decode(out) for x in STOPS):
        return "stop"
    if eos is not None and eos in out:
        return "eos"
    return "max" if len(out) >= MAX_NEW else "other"


max_len = max(len(r["input"]) for r in rows) + MAX_NEW + K + 2
tmax = min(4096, BUCKET * (max_len // BUCKET + 1))
eng = RavenSampleEngineAll(m, dr, n_rounds=n, temp=TEMP, top_p=TOP_P, q_topp=Q_TOPP, tmax=tmax, bucket=BUCKET,
                           graphs=os.environ.get("GRAPHS", "1") == "1", das=das)
print(f"{label}: spec {spec} rounds {n} task {TASK} N {len(rows)} (global idx {gidx[0]}..{gidx[-1]}) MAX_NEW {MAX_NEW} stops {STOPS} "
      f"T {TEMP} top-p {TOP_P} q_topp {qt} q_mode {eng.q_mode} drafts/cycle {eng.nd} das {das} pre_coda_norm {eng.pre_coda_norm} "
      f"trace {TR}", flush=True)
t = time.time(); eng.warmup(max_len); print(f"graphs captured in {time.time() - t:.0f}s (tmax {tmax}, fused {eng.fused}, attn {eng.attn_kind})", flush=True)
torch.cuda.manual_seed(0)
eng.generate_plain(rows[0]["input"], 32, stop); eng.generate_spec(rows[0]["input"], 32, stop)       # warm-up runs (excluded)
eng.passes_used.clear(); eng.rounds_used.clear(); eng.uncov.zero_(); eng.pld_nomatch = 0
recs = []; t_start = time.time()
for i, (g, r) in enumerate(zip(gidx, rows)):
    torch.cuda.manual_seed(SEED + g)
    o_p, dt_p = eng.generate_plain(r["input"], MAX_NEW, stop)
    torch.cuda.manual_seed(SEED + g)
    np0, nr0, nm0 = len(eng.passes_used), len(eng.rounds_used), eng.pld_nomatch
    o_s, tau, dt_s = eng.generate_spec(r["input"], MAX_NEW, stop)
    recs.append({"idx": g, "seed": SEED + g, "plain": o_p, "spec": o_s, "tau": tau, "dt_plain": dt_p, "dt_spec": dt_s,
                 "end_plain": end_reason(o_p), "end_spec": end_reason(o_s),
                 "passes": eng.passes_used[np0:] if eng.adapt_max else None,
                 "rounds": eng.rounds_used[nr0:] if eng.wave_stop else None,
                 "pld_nomatch": eng.pld_nomatch - nm0 if (das is not None and das["mode"] == "pld") else None})
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
sha = lambda key: hashlib.sha1(json.dumps([x[key] for x in recs]).encode()).hexdigest()
res = {"label": label, "method": method, "cfg": cfg, "model": MS, "model_dir": model, "spec": spec, "drafter": run_dr, "task": TASK,
       "temp": TEMP, "top_p": TOP_P, "q_topp": qt, "q_mode": eng.q_mode, "das": das, "rounds": n, "adapt_max": eng.adapt_max,
       "adapt_rule": getattr(eng, "adapt_rule", None), "adapt_delta": getattr(eng, "adapt_delta", None),
       "mean_passes": (sum(eng.passes_used) / len(eng.passes_used)) if eng.passes_used else None,
       "mean_rounds": (sum(eng.rounds_used) / len(eng.rounds_used)) if eng.rounds_used else None,
       "pld_nomatch_cycles": eng.pld_nomatch if (das is not None and das["mode"] == "pld") else None,
       "all": agg(lambda x: True, lambda x: True),
       "excl_capped": agg(lambda x: not capped("plain")(x), lambda x: not capped("spec")(x)),
       "excl_capped_paired": agg(lambda x: not (capped("plain")(x) or capped("spec")(x)), lambda x: not (capped("plain")(x) or capped("spec")(x))),
       "n_capped_plain": sum(capped("plain")(x) for x in recs), "n_capped_spec": sum(capped("spec")(x) for x in recs),
       "ends_plain": {k: sum(x["end_plain"] == k for x in recs) for k in ("stop", "eos", "max", "other")},
       "ends_spec": {k: sum(x["end_spec"] == k for x in recs) for k in ("stop", "eos", "max", "other")},
       "mean_len_plain": sum(len(x["plain"]) for x in recs) / len(recs), "mean_len_spec": sum(len(x["spec"]) for x in recs) / len(recs),
       "plain_outputs_sha1": sha("plain"), "spec_outputs_sha1": sha("spec"),
       "topp_cand": eng.k_cand, "uncovered_rows": dict(zip(("plain_and_prefill", "verify_target", "proposal"), eng.uncov.tolist())),
       "rows": {"plain_and_prefill": sum(len(x["plain"]) - 1 for x in recs) + 2 * len(recs),
                "verify_target": sum(len(x["tau"]) for x in recs) * (eng.nd + 1),
                "proposal": sum(len(x["tau"]) for x in recs) * eng.nd if eng.q_mode == "sample" else 0},
       "n": len(recs), "idx": gidx, "shard": SHARD, "seed": f"torch.cuda.manual_seed({SEED} + global idx) before plain and before spec",
       "drafts_per_cycle": eng.nd, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "fused": eng.fused,
       "attn": eng.attn_kind, "graphs": eng.use_graphs, "bucket": BUCKET, "tmax": tmax, "max_new": MAX_NEW, "stops": STOPS,
       "trace": TR, "pre_coda_norm": eng.pre_coda_norm, "wall_seconds": time.time() - t_start}
res.update({k: res["all"][k] for k in ("plain_tok_s", "spec_tok_s", "speedup", "tau", "cycles")})
print(json.dumps({k: v for k, v in res.items() if k != "idx"}), flush=True)
json.dump(res, open(base + ".json", "w"), indent=1)
with open(base + "_outputs.jsonl", "w") as f:
    for x in recs:
        f.write(json.dumps(x) + "\n")
print(f"wrote {base}.json", flush=True)
