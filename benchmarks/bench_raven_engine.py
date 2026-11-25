"""Wall-clock: plain decoding vs self-drafting in the same batch-1 engine (raven_engine.py) for a Raven-family model, greedy,
on the prompts of a reference trace.  Decode throughput = generated tokens / decode time (prefill excluded).
  python bench_raven_engine.py <run_dir | base:<model> | das:r,gamma:<model> | wave:J,k,gamma:<model> | jacobi:J,k,gamma:<model> |
                               pld:n,gamma:<model> | hybrid:J,k,gamma:<run_dir>> <n_rounds>     (<model>: rl32 | tl32 | huginn | a model directory)
  env: N_PROBLEMS (100), MAX_NEW (512), STOP_STRS ("Q:|</s>", '|'-separated), TRACE (jsonl with "input" ids and
       "output_ids"), EVAL_TAG, GRAPHS (1), FUSED (1), ATTN (triton | gqa), GEMM (triton | cublas), PROFILE (0), BUCKET (256),
       TF_TAU (1: also the teacher-forced tau of eval_accept_raven.py's setting, "tau_tf", not timed)
  base:<model_dir> benchmarks plain decoding only (JSON -> runs/raven/plain_<model><TAG>.json)."""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import json, os, sys, time
import torch
import os
from floodgates.paths import model_dir, RUNS, trace
from floodgates.models.raven_min import RavenMin
from floodgates.drafters.drafter_raven import RavenSelfDrafter
from floodgates.engines.raven_engine import RavenEngine
from transformers import PreTrainedTokenizerFast

run, n = sys.argv[1], int(sys.argv[2])
N = int(os.environ.get("N_PROBLEMS", 100)); MAX_NEW = int(os.environ.get("MAX_NEW", 512))
TAG = os.environ.get("EVAL_TAG", ""); BUCKET = int(os.environ.get("BUCKET", 256))
TR = os.environ.get("TRACE", trace("Recurrent-Llama-3.2-train-recurrence-32", "gsm8k"))
dr, K, das = None, 0, None
SHORT_MODELS = {"rl32": "Recurrent-Llama-3.2-train-recurrence-32", "tl32": "Recurrent-TinyLlama-3T-train-recurrence-32", "huginn": "huginn-0125"}
if run.startswith("base:"):
    model = model_dir(SHORT_MODELS.get(run[5:], run[5:]))
    base = os.path.join(RUNS, "raven", f"plain_{os.path.basename(model.rstrip('/'))}{TAG}")
elif run.split(":", 1)[0] in ("wave", "das", "hybrid", "pld", "jacobi"):   # wave:J,k,gamma:<model_dir> | das:r,gamma:<model_dir> | hybrid:J,k,gamma:<drafter run_dir> | pld:n,gamma:<model_dir>
    kind, cfg, model = run.split(":", 2); v = [int(x) for x in cfg.split(",")]
    jacobi = kind == "jacobi"; kind = "wave" if jacobi else kind               # Jacobi decoding = DAS-Wave without growth and without carried rounds
    if kind != "hybrid":
        model = model_dir(SHORT_MODELS.get(model, model))
    das = ({"mode": "wave", "J": v[0], "k": v[1], "gamma": v[2]} if kind in ("wave", "hybrid") else
           {"mode": "pld", "n": v[0], "gamma": v[1]} if kind == "pld" else {"mode": "seq", "r": v[0], "gamma": v[1]})
    if kind in ("hybrid", "wave") and int(os.environ.get("WAVE_STOP", 0)):   # convergence-stopped rounds, chunks of WAVE_STOP rounds
        das["stop"] = int(os.environ["WAVE_STOP"])
    if kind == "hybrid":
        das["init"] = "loopahead"; run_dr = model
        if int(os.environ.get("WAVE_CARRY", 0)):                            # rounds start from the drafting pass's slot states
            das["carry"] = True
        a = json.load(open(os.path.join(run_dr, "args.json"))); model = model_dir(a["model"])
    if os.environ.get("WAVE_GROW") == "0" or jacobi:                        # full window from the first round (no DAS-Wave growth)
        das["grow"] = False
    if os.environ.get("WAVE_NOCARRY") == "1" or jacobi:                     # rounds restart from the initial state (Jacobi decoding)
        das["carry_rounds"] = False
    if os.environ.get("CHEAP"):                                             # CHEAP=topk:F candidate-vocabulary readouts in the rounds
        das["cheap"] = [int(x) for x in os.environ["CHEAP"].split(":")]
    K = das["gamma"]
    base = os.path.join(RUNS, "raven", "das-engine", f"{kind}_{cfg.replace(',', '-')}_{os.path.basename(model.rstrip('/'))}{TAG}" + (f"_stop{das['stop']}" if das.get("stop") else "") + (f"_cheap{das['cheap'][0]}x{das['cheap'][1]}a" if das.get("cheap") else "") + ("_carry" if das.get("carry") else "") + ("_nogrow" if das.get("grow") is False else "") + ("_nocarry" if das.get("carry_rounds") is False else ""))
    os.makedirs(os.path.dirname(base), exist_ok=True)
else:
    a = json.load(open(os.path.join(run, "args.json")))
    model = model_dir(a["model"])
    base = os.path.join(run, f"engine_n{n}{TAG}" + (f"_adapt{os.environ['ADAPT_MAX']}{os.environ.get('ADAPT_RULE', 'conf')}{os.environ.get('ADAPT_DELTA', '0.8')}" if os.environ.get("ADAPT_MAX") else ""))
m = RavenMin(model)
if not run.startswith("base:") and (das is None or das.get("init") == "loopahead"):
    dr = RavenSelfDrafter(m, k=a["k"], rank=a["rank"], shift=a.get("shift", False), feedback=a.get("feedback", False), iters=a.get("iters", 1), carry=a.get("carry", False), causal=a.get("causal", False))
    missing, unexpected = dr.load_state_dict(torch.load(os.path.join(run if das is None else run_dr, "drafter.pt")), strict=False)
    assert not unexpected and all(x.startswith("m.") or x == "draft_emb" for x in missing), (unexpected, [x for x in missing if not x.startswith("m.")])
    dr.eval().requires_grad_(False)
    if das is None:
        K = a["k"] + (1 if a.get("shift") else 0)
tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model, "tokenizer.json"))       # (no remote-code prompt)
tc = json.load(open(os.path.join(model, "tokenizer_config.json")))
eos_tok = tc.get("eos_token"); eos_tok = eos_tok["content"] if isinstance(eos_tok, dict) else eos_tok
eos = tok.convert_tokens_to_ids(eos_tok) if eos_tok else None
STOPS = [x for x in os.environ.get("STOP_STRS", "Q:|</s>").split(os.environ.get("STOP_SEP", "|")) if x]       # task stop strings (lm-eval 'until')   # STOP_SEP=";;" when a stop contains "|" (e.g. <|endoftext|>)
LB = max(12, max(len(tok(x, add_special_tokens=False)["input_ids"]) for x in STOPS) + K + 2)
stop = lambda out: any(x in tok.decode(out[-LB:]) for x in STOPS) or (eos is not None and eos in out[-K - 1:])
rows = [json.loads(l) for l in open(TR)][:N]
max_len = max(len(r["input"]) for r in rows) + MAX_NEW + K + 2
tmax = min(4096, BUCKET * (max_len // BUCKET + 1))
eng = RavenEngine(m, dr, n_rounds=n, tmax=tmax, bucket=BUCKET, graphs=os.environ.get("GRAPHS", "1") == "1", das=das)
SPEC = dr is not None or das is not None
t = time.time(); eng.warmup(max_len); print(f"graphs captured in {time.time() - t:.0f}s (tmax {tmax}, fused {eng.fused}, attn {eng.attn_kind})", flush=True)
eng.generate_plain(rows[0]["input"], 32, stop)                                    # warm-up runs (excluded)
if SPEC:
    eng.generate_spec(rows[0]["input"], 32, stop)
    if eng.profile:
        eng.t_draft = eng.t_verify = 0.0; eng.n_prof = 0
TF = os.environ.get("TF_TAU", "1") == "1" and dr is not None      # also the offline-style (teacher-forced) tau, not timed
tp = ts = 0.0; np_ = ns = 0; taus = []; taus_tf = []; same = same_ref = 0; tok_ref = tok_tot = 0; outs = []


def common(a, b):
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i


t_start = time.time()
for i, r in enumerate(rows):
    o_p, dt_p = eng.generate_plain(r["input"], MAX_NEW, stop)
    tp += dt_p; np_ += len(o_p) - 1; ref = r["output_ids"]
    same_ref += int(o_p[:len(ref)] == ref[:len(o_p)])
    tok_ref += common(o_p, ref); tok_tot += len(ref)                         # reference tokens reproduced before the first divergence
    rec = {"plain": o_p, "dt_plain": dt_p}
    if SPEC:
        o_s, tau, dt_s = eng.generate_spec(r["input"], MAX_NEW, stop)
        ts += dt_s; ns += len(o_s) - 1; taus += tau
        same += int(o_s[:len(o_p)] == o_p[:len(o_s)])
        rec.update({"spec": o_s, "tau": tau, "dt_spec": dt_s})
        if TF:
            taus_tf += eng.tau_teacher_forced(r["input"], ref)
    outs.append(rec)
    if (i + 1) % 10 == 0:
        msg = f"{i + 1}/{len(rows)} plain {np_ / tp:.1f} tok/s"
        if SPEC:
            msg += f"  spec {ns / ts:.1f} tok/s  speedup {(ns / ts) / (np_ / tp):.2f}  tau {sum(taus) / len(taus):.2f}  spec==plain {same}  plain==ref {same_ref}"
        print(msg + f"  ({time.time() - t_start:.0f}s)", flush=True)
res = {"plain_tok_s": np_ / tp, "spec_tok_s": ns / ts if SPEC else None,
       "speedup": (ns / ts) / (np_ / tp) if SPEC else None, "tau": sum(taus) / len(taus) if taus else None,
       "spec_eq_plain": same if SPEC else None, "das": das, "adapt_max": eng.adapt_max, "adapt_rule": getattr(eng, "adapt_rule", None), "adapt_delta": getattr(eng, "adapt_delta", None),
       "mean_passes": (sum(eng.passes_used) / len(eng.passes_used)) if eng.passes_used else None,
       "mean_rounds": (sum(eng.rounds_used) / len(eng.rounds_used)) if getattr(eng, "rounds_used", None) else None, "plain_eq_server": same_ref, "plain_server_tok_agree": (tok_ref / tok_tot) if tok_tot else None,   # None: the trace has no reference outputs (Huginn non-GSM8K prompts)
       "n": len(rows), "rounds": n, "gpu": torch.cuda.get_device_name(), "fused": eng.fused, "attn": eng.attn_kind,
       "graphs": eng.use_graphs, "bucket": BUCKET, "max_new": MAX_NEW, "k": K, "plain_tokens": np_, "spec_tokens": ns,
       "ms_per_plain_step": 1000 * tp / max(np_, 1), "ms_per_cycle": 1000 * ts / max(len(taus), 1) if taus else None,
       "tau_tf": sum(taus_tf) / len(taus_tf) if taus_tf else None, "cycles_tf": len(taus_tf)}
if (dr is not None or das is not None) and eng.profile and eng.n_prof:
    res.update({"profile_cycles": eng.n_prof, "ms_draft_per_cycle": eng.t_draft / eng.n_prof, "ms_verify_per_cycle": eng.t_verify / eng.n_prof})
print(json.dumps(res), flush=True)
json.dump(res, open(base + ".json", "w"), indent=1)
with open(base + "_outputs.jsonl", "w") as f:
    for o in outs:
        f.write(json.dumps(o) + "\n")
