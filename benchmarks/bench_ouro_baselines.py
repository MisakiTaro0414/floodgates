"""Wall-clock: plain decoding vs a training-free baseline (engine_baselines.OuroBaselineEngine) in the Ouro engine, greedy,
on the prompts of a LoopSpec trace; the spec strings, env knobs, JSON keys and output rows of bench_raven_engine.py.
  python bench_ouro_baselines.py <spec> [n_rounds]
    spec: das:r,gamma:<model_dir>          sequential DAS (aligned reads), gamma drafts of r loops each
          wave:J,k,gamma:<model_dir>       DAS-Wave (J rounds of k loops over a window of gamma positions)
                                           WAVE_GROW=0 (full window from the first round), WAVE_NOCARRY=1 (rounds restart from
                                           the embeddings); both with k = R (= 4) = Jacobi decoding; WAVE_READS=depth|final
                                           (prefix read at the window's loop depth | at the final loop; default: depth for Jacobi)
          hybrid:J,k,gamma:<drafter run>   the trained one-pass drafter's block seeds the window (LAL + J refinement loops);
                                           WAVE_CARRY=1: the seeded positions start from the drafter's slot states
                                           (SEED_DELTA=0: without the embedding delta); J = 0 = the one-pass drafter
          pld:n,gamma:<model_dir>          prompt lookup decoding (last n tokens, gamma proposals)
          jacobi:J,k,gamma:<model_dir>     Jacobi decoding (= wave with WAVE_GROW=0 WAVE_NOCARRY=1; k = R)
  env: N_PROBLEMS (100), MAX_NEW (512), STOP_STRS ("Q:"), STOP_SEP ("|"), TRACE, EVAL_TAG, GRAPHS (1), TMAX (4096), PIPE (1)
  out: runs/ouro-baselines/<kind>_<cfg with ->_<model><EVAL_TAG>[_carry][_nogrow][_nocarry].json + _outputs.jsonl"""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import json, os, sys, time
import torch
import os
from floodgates.paths import model_dir, RUNS, trace
from floodgates.models.ouro_min import OuroMin
from floodgates.engines.engine_baselines import OuroBaselineEngine
from floodgates.tokenizer import load_tokenizer

run = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
N = int(os.environ.get("N_PROBLEMS", 100)); MAX_NEW = int(os.environ.get("MAX_NEW", 512)); TAG = os.environ.get("EVAL_TAG", "")
TR = os.environ.get("TRACE", trace("Ouro-1.4B", "gsm8k"))
kind, cfg, model = run.split(":", 2); v = [int(x) for x in cfg.split(",")]
jacobi = kind == "jacobi"; kind = "wave" if jacobi else kind                   # Jacobi decoding = DAS-Wave without growth and without carried rounds
assert kind in ("das", "wave", "hybrid", "pld"), kind
das = ({"mode": "wave", "J": v[0], "k": v[1], "gamma": v[2]} if kind in ("wave", "hybrid") else
       {"mode": "pld", "n": v[0], "gamma": v[1]} if kind == "pld" else {"mode": "seq", "r": v[0], "gamma": v[1]})
run_dr = None
if kind == "hybrid":
    das["init"] = "loopahead"; run_dr = model
    if int(os.environ.get("WAVE_CARRY", 0)):                               # rounds start from the drafting pass's slot states
        das["carry"] = True
        if os.environ.get("SEED_DELTA") == "0":
            das["seed_delta"] = False
    a = json.load(open(os.path.join(run_dr, "args.json"))); model = model_dir(a["model"])
if os.environ.get("WAVE_GROW") == "0" or jacobi:                            # full window from the first round (no DAS-Wave growth)
    das["grow"] = False
if os.environ.get("WAVE_NOCARRY") == "1" or jacobi:                         # rounds restart from the embeddings (Jacobi decoding)
    das["carry_rounds"] = False
if os.environ.get("WAVE_READS") in ("depth", "final"):                      # prefix reads of a wave round (default: depth iff Jacobi)
    das["reads"] = os.environ["WAVE_READS"]
K = das["gamma"]
suffix = (("_carry" if das.get("carry") else "") + ("_nodelta" if das.get("seed_delta") is False else "") + ("_nogrow" if das.get("grow") is False else "")
          + ("_nocarry" if das.get("carry_rounds") is False else "") + (f"_{das['reads']}" if "reads" in das else ""))
base = os.path.join(RUNS, "ouro-baselines",
                    f"{kind}_{cfg.replace(',', '-')}_{os.path.basename(model.rstrip('/'))}{TAG}{suffix}")
os.makedirs(os.path.dirname(base), exist_ok=True)
ouro = OuroMin(model)
dr = None
if run_dr is not None:
    from floodgates.drafters.drafter import LoopFlowDrafter
    dr = LoopFlowDrafter(ouro, k=a["k"], rank=a["rank"], kappa=a["kappa"], use_state=not a["no_state"], use_noise=not a["no_noise"],
                         flowmap=a.get("flowmap", False), feedback=a.get("feedback", False), markov_rank=a.get("markov_rank", 0))
    missing, unexpected = dr.load_state_dict(torch.load(os.path.join(run_dr, "drafter.pt")), strict=False)
    assert not unexpected and all(m.startswith("ouro.") or m == "draft_emb" for m in missing)
    dr.eval().requires_grad_(False)
    assert not dr.feedback and not dr.markov_rank and not dr.shift, "the hybrid seed takes the plain one-pass drafter"
tok = load_tokenizer(model)
STOPS = [x for x in os.environ.get("STOP_STRS", os.environ.get("STOP_STR", "Q:")).split(os.environ.get("STOP_SEP", "|")) if x]
LB = max(12, max(len(tok(x, add_special_tokens=False)["input_ids"]) for x in STOPS) + K + 2)
stop = lambda out: any(x in tok.decode(out[-LB:]) for x in STOPS) or tok.eos_token_id in out[-K - 1:]
rows = [json.loads(l) for l in open(TR)][:N]
max_len = max(len(r["input"]) for r in rows) + MAX_NEW + K + 2
eng = OuroBaselineEngine(ouro, das, dr, tmax=int(os.environ.get("TMAX", 4096)), graphs=os.environ.get("GRAPHS", "1") == "1")
t = time.time(); eng.warmup(max_len); print(f"graphs captured in {time.time() - t:.0f}s (das {das}, attn {eng.attn_kind}, gemm {eng.gemm_kind}, pipe {eng.pipe})", flush=True)
gen = torch.Generator(device="cuda").manual_seed(0)
eng.generate_plain(rows[0]["input"], 32, stop); eng.generate_spec(rows[0]["input"], 32, stop, gen)      # warm-up runs (excluded)
eng.n_nodraft = 0
tp = ts = 0.0; np_ = ns = 0; taus = []; same = same_ref = 0; tok_ref = tok_tot = 0; outs = []


def common(a, b):
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i


t_start = time.time()
for i, r in enumerate(rows):
    o_p, dt_p = eng.generate_plain(r["input"], MAX_NEW, stop)
    o_s, tau, dt_s = eng.generate_spec(r["input"], MAX_NEW, stop, gen)
    tp += dt_p; ts += dt_s; np_ += len(o_p) - 1; ns += len(o_s) - 1; taus += tau
    same += int(o_s[:len(o_p)] == o_p[:len(o_s)]); ref = r["output_ids"]
    same_ref += int(o_p[:len(ref)] == ref[:len(o_p)])
    tok_ref += common(o_p, ref); tok_tot += len(ref)                         # reference tokens reproduced before the first divergence
    outs.append({"plain": o_p, "dt_plain": dt_p, "spec": o_s, "tau": tau, "dt_spec": dt_s})
    if (i + 1) % 10 == 0:
        print(f"{i + 1}/{len(rows)} plain {np_ / tp:.1f} tok/s  spec {ns / ts:.1f} tok/s  speedup {(ns / ts) / (np_ / tp):.2f}  "
              f"tau {sum(taus) / len(taus):.2f}  spec==plain {same}  plain==ref {same_ref}  ({time.time() - t_start:.0f}s)", flush=True)
res = {"plain_tok_s": np_ / tp, "spec_tok_s": ns / ts, "speedup": (ns / ts) / (np_ / tp), "tau": sum(taus) / len(taus),
       "spec_eq_plain": same, "das": das, "adapt_max": 0, "adapt_rule": None, "adapt_delta": None, "mean_passes": None, "mean_rounds": None,
       "plain_eq_server": same_ref, "plain_server_tok_agree": tok_ref / tok_tot, "n": len(rows), "rounds": n,
       "gpu": torch.cuda.get_device_name(), "fused": eng.fused, "attn": eng.attn_kind, "fp32p": getattr(eng.att, "fp32p", None),
       "gemm": eng.gemm_kind, "pipe": eng.pipe, "graphs": eng.use_graphs, "bucket": None, "max_new": MAX_NEW, "k": K,
       "plain_tokens": np_, "spec_tokens": ns, "ms_per_plain_step": 1000 * tp / max(np_, 1), "ms_per_cycle": 1000 * ts / max(len(taus), 1),
       "cycles": len(taus), "cycles_nodraft": eng.n_nodraft if das["mode"] == "pld" else None, "tau_tf": None, "cycles_tf": 0,
       "model": model, "drafter": run_dr, "trace": TR}
print(json.dumps(res), flush=True)
json.dump(res, open(base + ".json", "w"), indent=1)
with open(base + "_outputs.jsonl", "w") as f:
    for o in outs:
        f.write(json.dumps(o) + "\n")
print("wrote", base + ".json", flush=True)
