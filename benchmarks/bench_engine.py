"""Wall-clock: plain decoding vs self-drafting in the same batch-1 engine (engine.py), greedy, on the prompts of a LoopSpec
trace.  Decode throughput = generated tokens / decode time (prefill excluded, as SGLang's decode tok/s).
  python bench_engine.py <run_dir> <n_rounds>     env: N_PROBLEMS (100), MAX_NEW (512), STOP_STR ("Q:"), TRACE, EVAL_TAG,
                                                  GRAPHS (1), TREE_S (1: draft candidates per cycle, tree verification),
                                                  ATTN (triton | sdpa), PIPE (1: launch-ahead, 0: serial) -- see engine.py"""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import json, os, sys, time
import torch
import os
from floodgates.paths import model_dir, trace
from floodgates.models.ouro_min import OuroMin
from floodgates.drafters.drafter import LoopFlowDrafter
from floodgates.engines.engine import Engine
from floodgates.tokenizer import load_tokenizer

run, n = sys.argv[1], int(sys.argv[2])
N = int(os.environ.get("N_PROBLEMS", 100)); MAX_NEW = int(os.environ.get("MAX_NEW", 512))
STOP_STR = os.environ.get("STOP_STR", "Q:"); TAG = os.environ.get("EVAL_TAG", ""); S = int(os.environ.get("TREE_S", 1))
TR = os.environ.get("TRACE", trace("Ouro-1.4B", "gsm8k"))
a = json.load(open(os.path.join(run, "args.json")))
ouro = OuroMin(model_dir(a["model"]))
dr = LoopFlowDrafter(ouro, k=a["k"], rank=a["rank"], kappa=a["kappa"], use_state=not a["no_state"], use_noise=not a["no_noise"],
                     flowmap=a.get("flowmap", False), feedback=a.get("feedback", False), markov_rank=a.get("markov_rank", 0),
                     ctx_loop=a.get("ctx_loop"))                                 # None: final-loop cache (paper); 0: first-loop ablation
missing, unexpected = dr.load_state_dict(torch.load(os.path.join(run, "drafter.pt")), strict=False)
assert not unexpected and all(m.startswith("ouro.") or m == "draft_emb" for m in missing)
dr.eval().requires_grad_(False)
tok = load_tokenizer(model_dir(a["model"]))
STOPS = [x for x in os.environ.get("STOP_STRS", STOP_STR).split(os.environ.get("STOP_SEP", "|")) if x]        # task stop strings (lm-eval 'until')   # STOP_SEP=";;" when a stop contains "|" (e.g. <|endoftext|>)
LB = max(12, max(len(tok(x, add_special_tokens=False)["input_ids"]) for x in STOPS) + dr.k + 2)
stop = lambda out: any(x in tok.decode(out[-LB:]) for x in STOPS) or tok.eos_token_id in out[-dr.k - 1:]
rows = [json.loads(l) for l in open(TR)][:N]
eng = Engine(ouro, dr, n_rounds=n, tmax=int(os.environ.get("TMAX", 4096)), graphs=os.environ.get("GRAPHS", "1") == "1", S=S)
t = time.time(); eng.warmup(max(len(r["input"]) for r in rows) + MAX_NEW + S * (dr.k + 1) + 2); print(f"graphs captured in {time.time() - t:.0f}s", flush=True)
gen = torch.Generator(device="cuda").manual_seed(0)
# warm-up runs (excluded)
eng.generate_plain(rows[0]["input"], 32, stop); eng.generate_spec(rows[0]["input"], 32, stop, gen)
tp = ts = 0.0; np_ = ns = 0; taus = []; same = same_ref = 0; tok_ref = tok_tot = 0; outs = []
def common(a, b):
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i
for r in rows:
    o_p, dt_p = eng.generate_plain(r["input"], MAX_NEW, stop)
    o_s, tau, dt_s = eng.generate_spec(r["input"], MAX_NEW, stop, gen)
    tp += dt_p; ts += dt_s; np_ += len(o_p) - 1; ns += len(o_s) - 1; taus += tau
    same += int(o_s[:len(o_p)] == o_p[:len(o_s)]); ref = r["output_ids"]
    same_ref += int(o_p[:len(ref)] == ref[:len(o_p)])
    tok_ref += common(o_p, ref); tok_tot += len(ref)                         # server tokens reproduced before the first divergence
    outs.append({"plain": o_p, "spec": o_s, "tau": tau, "dt_plain": dt_p, "dt_spec": dt_s})
res = {"plain_tok_s": np_ / tp, "spec_tok_s": ns / ts, "speedup": (ns / ts) / (np_ / tp), "tau": sum(taus) / len(taus),
       "spec_eq_plain": same, "plain_eq_server": same_ref, "plain_server_tok_agree": tok_ref / tok_tot, "n": len(rows), "rounds": n,
       "tree_S": S, "attn": eng.attn_kind, "fp32p": getattr(eng.att, "fp32p", None), "gemm": eng.gemm_kind, "pipe": eng.pipe,
       "ctx_loop": a.get("ctx_loop"), "gpu": torch.cuda.get_device_name(), "adapt_max": eng.adapt_max, "adapt_delta": eng.adapt_delta if eng.adapt_max else None,
       "mean_passes": (sum(eng.passes_used) / len(eng.passes_used)) if eng.passes_used else None}
print(json.dumps(res), flush=True)
base = os.path.join(run, f"engine_n{n}{'_S%d' % S if S > 1 else ''}{TAG}" + (f"_adapt{eng.adapt_max}d{eng.adapt_delta}" if eng.adapt_max else ""))
json.dump(res, open(base + ".json", "w"), indent=1)
with open(base + "_outputs.jsonl", "w") as f:
    for o in outs:
        f.write(json.dumps(o) + "\n")
