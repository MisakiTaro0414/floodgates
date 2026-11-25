"""Correctness checks of bench_raven_sample_all.py / raven_engine_sample_all.py (GSM8K prompts; TEST_MODEL rl32 (default) | tl32 |
huginn, KCORE (1) = core iterations per DAS-Wave / refinement round).
  PART=greedy  T=0 (the exact greedy switch of raven_engine_sample): RavenSampleEngineAll's plain and speculative outputs vs the
               GREEDY engine (raven_engine.RavenEngine) built in the same process, for every method of the sampling sweep, on
               the first N_PROBLEMS (5) GSM8K prompts with the greedy runs' stops / limits (scripts/bench_greedy.sh); also vs the stored
               greedy outputs (runs/raven/das-engine/*_outputs.jsonl) where they exist.
  PART=dist    T=1, top-p 0.7 (tie nucleus, TOPP_CAND 4096): for sequential DAS and PLD (one-hot q) at two contexts (GSM8K
               prompts 0 and 1 + reference prefix; the first position whose nucleus has >= 2 tokens, max <= 0.8, whose second-token
               nucleus after the argmax has >= 2 tokens, and where the method's first draft d_1 has 0.1 <= p(d_1) <= 0.9):
               N_SAMPLES draws of (t1, t2) from spec cycles from the same state vs p_1 and p_2(.|t1) from the engine's own target
               pass (chi-square), the acceptance rate of d_1 vs p(d_1), plain sampling draws as a reference, and a NEGATIVE
               CONTROL (BrokenVerifyEngine: emits from p at a rejection) that the test must reject.
  PART=diag    large-N first-cycle check at the dist contexts: verification-row target vs plain-row target, accept rate of d_1 and
               t1 against both (das 2,8, pld 3,12, jacobi 2,32,32).
  PART=seed    T=1: same seed -> identical plain and speculative ids (twice per method, 2 prompts, MAX_NEW 128), a different seed
               changes them, and the plain outputs are identical across all methods' engines.
Writes runs/sampling_full/sbr_tests/test_<part><EVAL_TAG>.json (OUT)."""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import json, os, sys, time
import torch
import os
from floodgates.paths import LOOPSPEC, RUNS, ROOT, trace
from scipy import stats
from floodgates.engines.raven_engine_sample_all import RavenSampleEngineAll, BrokenVerifyEngine

PART = os.environ.get("PART", "greedy")
HERE = ROOT   # repository root
OUT = os.environ.get("OUT", os.path.join(RUNS, "sampling_full", "sbr_tests")); os.makedirs(OUT, exist_ok=True)
L = LOOPSPEC
MK = os.environ.get("TEST_MODEL", "rl32")                               # rl32 (default) | tl32 | huginn
W = {"rl32": "Recurrent-Llama-3.2-train-recurrence-32", "tl32": "Recurrent-TinyLlama-3T-train-recurrence-32", "huginn": "huginn-0125"}[MK]
MODEL = f"{L}/models/{W}"
RD = os.path.join(RUNS, "raven")
TRACE = trace(W, "gsm8k")
DR = os.environ.get("RUN", os.path.join(RUNS, f"floodgates-{W}"))
STOPS = ["Q:", "</s>"]                                                  # scripts/bench_greedy.sh gsm8k
R_ = 4 if MK == "huginn" else 2; KC = int(os.environ.get("KCORE", 1))
ST = (lambda x: None) if MK == "huginn" else (lambda x: x)              # stored Huginn greedy runs predate the pre-coda norm fix
METHODS = [  # (name, drafter run or None, das, stored greedy outputs (relative to runs/raven) or None)
    (f"das {R_},8", None, {"mode": "seq", "r": R_, "gamma": 8, "q": "argmax"}, ST(f"das-engine/das_{R_}-8_{W}_gsm8k_pp")),
    ("jacobi 2,32,32", None, {"mode": "wave", "J": 2, "k": 32, "gamma": 32, "grow": False, "carry_rounds": False, "q": "argmax"},
     ST(f"das-engine/wave_2-32-32_{W}_gsm8k_pp_nogrow_nocarry")),
    ("pld 3,12", None, {"mode": "pld", "n": 3, "gamma": 12, "q": "argmax"}, ST(f"das-engine/pld_3-12_{W}_gsm8k_pp")),
    (f"daswave 8,{KC},32 q=sample", None, {"mode": "wave", "J": 8, "k": KC, "gamma": 32, "q": "sample"}, None),
    (f"daswave 24,{KC},32 q=sample", None, {"mode": "wave", "J": 24, "k": KC, "gamma": 32, "q": "sample"},
     ST(f"das-engine/wave_24-{KC}-32_{W}_gsm8k_pp")),
    ("floodgates n1", DR, None, None),
    (f"lalref 8,{KC},32 carry q=sample", DR, {"mode": "wave", "J": 8, "k": KC, "gamma": 32, "init": "loopahead", "carry": True, "q": "sample"}, None),
    (f"lalref 24,{KC},32 carry q=sample", DR, {"mode": "wave", "J": 24, "k": KC, "gamma": 32, "init": "loopahead", "carry": True, "q": "sample"},
     ST(f"das-engine/hybrid_24-{KC}-32_{W}_gsm8k_pp4_carry")),
]
if os.environ.get("ONLY"):
    METHODS = [x for x in METHODS if any(s in x[0] for s in os.environ["ONLY"].split(","))]


def chi2(obs, exp_p):
    """As test_raven_sample.chi2: bins with expected >= 5, rest pooled; out-of-support observations reported (must be 0)."""
    N = sum(obs.values()); oos = sum(v for k, v in obs.items() if exp_p.get(k, 0.0) == 0.0)
    big = [k for k, p in exp_p.items() if p * N >= 5]; small = [k for k in exp_p if k not in big]
    fo = [obs.get(k, 0) for k in big]; fe = [exp_p[k] * N for k in big]
    if small and sum(exp_p[k] for k in small) * N >= 5:
        fo.append(sum(obs.get(k, 0) for k in small)); fe.append(sum(exp_p[k] for k in small) * N)
    elif small and fo:
        fo[-1] += sum(obs.get(k, 0) for k in small); fe[-1] += sum(exp_p[k] for k in small) * N
    if fo:
        fo[-1] += oos
    tv = 0.5 * sum(abs(obs.get(k, 0) / max(N, 1) - exp_p.get(k, 0.0)) for k in set(obs) | set(exp_p))
    if len(fo) < 2:
        return {"N": N, "bins": len(fo), "chi2": 0.0, "p_value": 1.0, "tv": tv, "out_of_support": oos}
    c, pv = stats.chisquare(fo, [x * sum(fo) / sum(fe) for x in fe])
    return {"N": N, "bins": len(fo), "chi2": float(c), "p_value": float(pv), "tv": tv, "out_of_support": oos}


def load_common():
    from floodgates.models.raven_min import RavenMin
    from transformers import PreTrainedTokenizerFast
    return RavenMin(MODEL), PreTrainedTokenizerFast(tokenizer_file=os.path.join(MODEL, "tokenizer.json"))


def make_dr(m, run):
    from floodgates.drafters.drafter_raven import RavenSelfDrafter
    a = json.load(open(os.path.join(run, "args.json")))
    dr = RavenSelfDrafter(m, k=a["k"], rank=a["rank"], shift=a.get("shift", False), feedback=a.get("feedback", False), iters=a.get("iters", 1),
                          carry=a.get("carry", False), causal=a.get("causal", False))
    missing, unexpected = dr.load_state_dict(torch.load(os.path.join(run, "drafter.pt")), strict=False)
    assert not unexpected and all(x.startswith("m.") or x == "draft_emb" for x in missing)
    return dr.eval().requires_grad_(False)


def stop_fn(tok, K):
    """bench_raven_engine.py / bench_raven_sample_all.py stop rule."""
    tc = json.load(open(os.path.join(MODEL, "tokenizer_config.json")))
    eos_tok = tc.get("eos_token"); eos_tok = eos_tok["content"] if isinstance(eos_tok, dict) else eos_tok
    eos = tok.convert_tokens_to_ids(eos_tok) if eos_tok else None
    LB = max(12, max(len(tok(x, add_special_tokens=False)["input_ids"]) for x in STOPS) + K + 2)
    return lambda out: any(x in tok.decode(out[-LB:]) for x in STOPS) or (eos is not None and eos in out[-K - 1:])


def n_drafts(dr, das):
    return (dr.k + (1 if dr.shift else 0)) if das is None else das["gamma"]


# ---------------------------------------------------------------------------------------------------------------------
def part_greedy():
    from floodgates.engines.raven_engine import RavenEngine
    m, tok = load_common()
    NP = int(os.environ.get("N_PROBLEMS", 5)); MAX_NEW = int(os.environ.get("MAX_NEW", 512))
    rows = [json.loads(l) for l in open(TRACE)][:NP]
    res = {}
    for name, run, das, stored in METHODS:
        dr = make_dr(m, run) if run else None
        K = n_drafts(dr, das); stop = stop_fn(tok, K)
        max_len = max(len(r["input"]) for r in rows) + MAX_NEW + K + 2; tmax = min(4096, 256 * (max_len // 256 + 1))
        outs = {}
        for kind in ("greedy", "sample_T0"):
            if kind == "greedy":
                eng = RavenEngine(m, dr, n_rounds=1, tmax=tmax, das=None if das is None else {k: v for k, v in das.items() if k != "q"})
            else:
                eng = RavenSampleEngineAll(m, dr, n_rounds=1, temp=0.0, top_p=0.7, q_topp=True, tmax=tmax, das=das)
            t0 = time.time(); eng.warmup(max_len); tc = time.time() - t0
            o = []
            for r in rows:
                torch.cuda.manual_seed(123)
                op, _ = eng.generate_plain(r["input"], MAX_NEW, stop)
                os_, taus, _ = eng.generate_spec(r["input"], MAX_NEW, stop)
                o.append((op, os_, taus))
            outs[kind] = o
            print(f"  {name} {kind}: captured in {tc:.0f}s; lens plain {[len(x[0]) for x in o]} spec {[len(x[1]) for x in o]} "
                  f"tau {[round(sum(x[2]) / len(x[2]), 2) for x in o]}", flush=True)
            del eng; torch.cuda.empty_cache()
        g, s = outs["greedy"], outs["sample_T0"]
        r_ = {"n": len(rows), "plain_identical": sum(a[0] == b[0] for a, b in zip(g, s)), "spec_identical": sum(a[1] == b[1] for a, b in zip(g, s)),
              "taus_identical": sum(a[2] == b[2] for a, b in zip(g, s)),
              "greedy_spec_eq_greedy_plain": sum(a[1][:len(a[0])] == a[0][:len(a[1])] for a in g),
              "spec_tokens": sum(len(a[1]) for a in g), "tau_T0": sum(sum(b[2]) for b in s) / sum(len(b[2]) for b in s)}
        if stored and os.path.exists(os.path.join(RD, stored + "_outputs.jsonl")):
            st = [json.loads(l) for l in open(os.path.join(RD, stored + "_outputs.jsonl"))][:len(rows)]
            r_.update({"stored": stored, "plain_eq_stored": sum(b[0] == x["plain"] for b, x in zip(s, st)),
                       "spec_eq_stored": sum(b[1] == x["spec"] for b, x in zip(s, st)),
                       "taus_eq_stored": sum(b[2] == x.get("tau") for b, x in zip(s, st))})
        res[name] = r_
        print(f"GREEDY CHECK {name}: {r_}", flush=True)
        del dr; torch.cuda.empty_cache()
    res["summary"] = {"all_identical_to_greedy_engine": all(v["plain_identical"] == v["n"] and v["spec_identical"] == v["n"] and v["taus_identical"] == v["n"]
                                                              for v in res.values()),
                      "all_identical_to_stored": all(v.get("spec_eq_stored", v["n"]) == v["n"] and v.get("plain_eq_stored", v["n"]) == v["n"]
                                                     for v in res.values() if isinstance(v, dict) and "n" in v)}
    print("GREEDY SUMMARY", res["summary"], flush=True)
    return res


# ---------------------------------------------------------------------------------------------------------------------
DIST_METHODS = [("das 2,8", {"mode": "seq", "r": 2, "gamma": 8, "q": "argmax"}, RavenSampleEngineAll),
                ("pld 3,12", {"mode": "pld", "n": 3, "gamma": 12, "q": "argmax"}, RavenSampleEngineAll),
                ("NEGATIVE CONTROL das 2,8 broken residual", {"mode": "seq", "r": 2, "gamma": 8, "q": "argmax"}, BrokenVerifyEngine)]


def find_context(eng, trace, pi):
    """The first qualifying position in prompt pi (see the docstring) or None; leaves the cache rows < cut at the reference."""
    row = trace[pi]; prompt, ref = row["input"], row["output_ids"]; P = len(prompt)
    torch.cuda.manual_seed(7)
    eng.prefill(prompt)                                            # rows 0..P-1
    for t in range(min(len(ref) - 1, 200)):                        # force the reference tokens
        _, p = eng.plain_from(P + t, ref[t])                        # p(. | prompt, ref[:t+1]); row P+t = ref[t]
        if t < 5 or int((p > 0).sum()) < 2 or float(p.max()) > 0.8:
            continue
        seq = prompt + ref[:t + 1]
        d = eng.drafts_at(P + t, ref[t], seq)                        # (overwrites rows >= P+t; restored below)
        eng.plain_from(P + t, ref[t])
        if not d or not (0.1 <= float(p[d[0]]) <= 0.9):
            continue
        p1d = {int(i): float(p[i]) for i in torch.nonzero(p > 0)[:, 0]}
        p2 = {}
        for a in p1d:                                               # p_2(. | a) from the engine's own target pass
            eng.plain_from(P + t, ref[t]); _, pa = eng.plain_from(P + t + 1, a)
            p2[a] = {int(i): float(pa[i]) for i in torch.nonzero(pa > 0)[:, 0]}
        eng.plain_from(P + t, ref[t])
        if len(p2[max(p1d, key=p1d.get)]) >= 2:
            return prompt, ref, P, P + t, t, d, p1d, p2
    return None


def part_dist():
    m, tok = load_common()
    NS = int(os.environ.get("N_SAMPLES", 3000)); TOP_P = 0.7
    NCTX = int(os.environ.get("N_CONTEXTS", 2))
    trace = [json.loads(l) for l in open(TRACE)]
    res = {}

    for mi, (name, das, Eng) in enumerate(DIST_METHODS):
        if os.environ.get("ONLY") and not any(s in name for s in os.environ["ONLY"].split(",")):
            continue
        eng = Eng(m, None, n_rounds=1, temp=1.0, top_p=TOP_P, q_topp=True, keep_p=True, tmax=1024, das=das)
        eng.warmup(max(len(r["input"]) for r in trace[:20]) + 260)
        pi = -1
        for ci in range(NCTX if Eng is RavenSampleEngineAll else 1):
            found = None
            while found is None:
                pi += 1; assert pi < 20, f"no context found for {name}"
                found = find_context(eng, trace, pi)
            prompt, ref, P, cut, ctx_t, d0, p1d, p2 = found
            y = ref[ctx_t]; seq0 = prompt + ref[:ctx_t + 1]
            joint = {(a, b): p1d[a] * pb for a in p1d for b, pb in p2[a].items()}
            acc1 = [0]

            def spec_draw():
                new = eng.cycle_from(cut, y, seq0)
                acc1[0] += int(len(new) >= 2)                           # the first cycle accepted d_1
                if len(new) < 2:
                    new = new + eng.cycle_from(cut + 1, new[0], seq0 + [new[0]])
                return new[0], new[1]

            def plain_draw():
                t1, _ = eng.plain_from(cut, y); t2, _ = eng.plain_from(cut + 1, t1)
                return t1, t2

            ctx = {"prompt": pi, "context_tokens": P + ctx_t + 1, "cut": cut, "y": tok.decode([y]), "drafts": tok.decode(d0),
                   "d1": tok.decode([d0[0]]), "p_d1": p1d.get(d0[0], 0.0),
                   "nucleus_p1": {tok.decode([k]): round(v, 4) for k, v in p1d.items()},
                   "nucleus_p2": {tok.decode([a]): {tok.decode([b]): round(v, 4) for b, v in p2[a].items()} for a in p1d}}
            todo = ([("plain sampling", plain_draw)] if Eng is RavenSampleEngineAll else []) + [(name, spec_draw)]
            for nm, fn in todo:
                seed = 11 + 1000 * mi + 100 * ci + (0 if nm == "plain sampling" else 1)
                torch.cuda.manual_seed(seed); acc1[0] = 0
                t0 = time.time(); c1, c2 = {}, {}
                for _ in range(NS):
                    t1, t2 = fn()
                    c1[t1] = c1.get(t1, 0) + 1; c2.setdefault(t1, {}); c2[t1][t2] = c2[t1].get(t2, 0) + 1
                cj = {(a, b): v for a in c2 for b, v in c2[a].items()}
                r = {"context": ctx, "seed": seed, "t1": chi2(c1, p1d), "joint_t1_t2": chi2(cj, joint), "seconds": time.time() - t0}
                for a in [k for k in sorted(c2, key=lambda k: -c1[k]) if k in p2][:2]:
                    r[f"t2|t1={tok.decode([a])!r}"] = chi2(c2[a], p2[a])
                if nm != "plain sampling":                              # P(accept d_1) must be p(d_1) for one-hot q
                    k1 = acc1[0]; pd1 = ctx["p_d1"]
                    r["accept_d1"] = {"observed": k1 / NS, "expected": pd1, "binom_p_value": float(stats.binomtest(k1, NS, pd1).pvalue)}
                key = f"{nm} | prompt {pi}" + (f" (context of {name})" if nm == "plain sampling" else "")
                res[key] = r
                print(f"DIST {key}: {json.dumps(r)}", flush=True)
        del eng; torch.cuda.empty_cache()
    real = {k: v for k, v in res.items() if not k.startswith("NEGATIVE")}
    neg = {k: v for k, v in res.items() if k.startswith("NEGATIVE")}
    pv = lambda rs: [v["p_value"] for r in rs.values() for v in r.values() if isinstance(v, dict) and "p_value" in v] + \
                    [r["accept_d1"]["binom_p_value"] for r in rs.values() if "accept_d1" in r]
    res["summary"] = {"n_tests": len(pv(real)), "min_p_value": min(pv(real)), "n_p_below_0.01": sum(x < 0.01 for x in pv(real)),
                      "out_of_support": sum(v["out_of_support"] for r in real.values() for v in r.values() if isinstance(v, dict) and "out_of_support" in v),
                      "negative_control_min_p_value": min(pv(neg)) if neg else None}
    print("DIST SUMMARY", res["summary"], flush=True)
    return res


# ---------------------------------------------------------------------------------------------------------------------
def part_seed():
    m, tok = load_common()
    rows = [json.loads(l) for l in open(TRACE)][:2]; MAX_NEW = int(os.environ.get("MAX_NEW", 128))
    res = {}; plains = {}
    for name, run, das, _ in METHODS:
        dr = make_dr(m, run) if run else None
        K = n_drafts(dr, das); stop = stop_fn(tok, K)
        max_len = max(len(r["input"]) for r in rows) + MAX_NEW + K + 2; tmax = min(4096, 256 * (max_len // 256 + 1))
        eng = RavenSampleEngineAll(m, dr, n_rounds=1, temp=1.0, top_p=0.7, q_topp=True, tmax=tmax, das=das)
        eng.warmup(max_len)
        o = {}
        for rep, s0 in (("a", 1000), ("b", 1000), ("c", 5000)):
            o[rep] = []
            for i, r in enumerate(rows):
                torch.cuda.manual_seed(s0 + i); op, _ = eng.generate_plain(r["input"], MAX_NEW, stop)
                torch.cuda.manual_seed(s0 + i); os_, taus, _ = eng.generate_spec(r["input"], MAX_NEW, stop)
                o[rep].append((op, os_, taus))
        plains[name] = [x[0] for x in o["a"]]
        r_ = {"same_seed_plain_identical": sum(x[0] == y[0] for x, y in zip(o["a"], o["b"])),
              "same_seed_spec_identical": sum(x[1] == y[1] for x, y in zip(o["a"], o["b"])),
              "same_seed_taus_identical": sum(x[2] == y[2] for x, y in zip(o["a"], o["b"])),
              "other_seed_spec_differs": sum(x[1] != y[1] for x, y in zip(o["a"], o["c"])), "n": len(rows),
              "tau": sum(sum(x[2]) for x in o["a"]) / sum(len(x[2]) for x in o["a"])}
        res[name] = r_
        print(f"SEED CHECK {name}: {r_}", flush=True)
        del eng, dr; torch.cuda.empty_cache()
    ref = next(iter(plains.values()))
    res["summary"] = {"all_same_seed_identical": all(v["same_seed_plain_identical"] == v["n"] and v["same_seed_spec_identical"] == v["n"] for v in res.values()),
                      "plain_identical_across_methods": all(v == ref for v in plains.values())}
    print("SEED SUMMARY", res["summary"], flush=True)
    return res


def part_diag():
    """Large-N first-cycle check at the dist contexts: the verification pass's own target row p_v (row 0 = the token after y, read
    from pbuf) vs the plain step's row p_p (TV and |p_v(d_1) - p_p(d_1)|), then N_DIAG first cycles: P(accept d_1) vs p_v(d_1) and
    p_p(d_1) (binomial), t1 vs p_v and p_p (chi-square).  Lossless = t1 ~ p_v exactly; p_v vs p_p differ only by the kernels'
    rounding at different row counts."""
    m, tok = load_common()
    NS = int(os.environ.get("N_DIAG", 15000)); NCTX = int(os.environ.get("N_CONTEXTS", 2))
    trace = [json.loads(l) for l in open(TRACE)]
    meths = [("das 2,8", {"mode": "seq", "r": 2, "gamma": 8, "q": "argmax"}), ("pld 3,12", {"mode": "pld", "n": 3, "gamma": 12, "q": "argmax"}),
             ("jacobi 2,32,32", {"mode": "wave", "J": 2, "k": 32, "gamma": 32, "grow": False, "carry_rounds": False, "q": "argmax"})]
    res = {}
    for mi, (name, das) in enumerate(meths):
        if os.environ.get("ONLY") and not any(x in name for x in os.environ["ONLY"].split(",")):
            continue
        eng = RavenSampleEngineAll(m, None, n_rounds=1, temp=1.0, top_p=0.7, q_topp=True, keep_p=True, tmax=1024, das=das)
        eng.warmup(max(len(r["input"]) for r in trace[:20]) + 260)
        pi = -1
        for ci in range(NCTX):
            found = None
            while found is None:
                pi += 1; assert pi < 20
                found = find_context(eng, trace, pi) if name != "jacobi 2,32,32" else find_context_any(eng, trace, pi)
            prompt, ref, P, cut, ctx_t, d0, p1d, p2 = found
            y = ref[ctx_t]; seq0 = prompt + ref[:ctx_t + 1]
            _, pp = eng.plain_from(cut, y)                              # plain row (1-row pass)
            torch.cuda.manual_seed(99); eng.cycle_from(cut, y, seq0)
            d1 = int(eng.ids[0, 1]) if das["mode"] != "pld" else d0[0]
            pv = eng.pbuf[0].clone()                                     # verification row 0 (nd+1-row pass)
            assert d1 == d0[0]
            torch.cuda.manual_seed(5000 + 100 * mi + ci)
            c1 = {}; acc = 0
            for _ in range(NS):
                new = eng.cycle_from(cut, y, seq0)
                c1[new[0]] = c1.get(new[0], 0) + 1; acc += int(len(new) >= 2)
            dp = lambda p: {int(i): float(p[i]) for i in torch.nonzero(p > 0)[:, 0]}
            r = {"prompt": pi, "cut": cut, "y": tok.decode([y]), "d1": tok.decode([d1]), "verify_rows": eng.nd + 1,
                 "tv_plain_verify_row0": 0.5 * float((pp - pv).abs().sum()), "p_plain_d1": float(pp[d1]), "p_verify_d1": float(pv[d1]),
                 "N": NS, "accept_rate": acc / NS,
                 "accept_vs_verify_p": float(stats.binomtest(acc, NS, float(pv[d1])).pvalue),
                 "accept_vs_plain_p": float(stats.binomtest(acc, NS, float(pp[d1])).pvalue),
                 "t1_vs_verify_p": chi2(c1, dp(pv)), "t1_vs_plain_p": chi2(c1, dp(pp))}
            res[f"{name} | prompt {pi}"] = r
            print(f"DIAG {name} | prompt {pi}: {json.dumps(r)}", flush=True)
        del eng; torch.cuda.empty_cache()
    return res


def find_context_any(eng, trace, pi):
    """As find_context without the d_1 condition (Jacobi's drafts need a graph run per position; first nucleus >= 2 position)."""
    row = trace[pi]; prompt, ref = row["input"], row["output_ids"]; P = len(prompt)
    torch.cuda.manual_seed(7); eng.prefill(prompt)
    for t in range(min(len(ref) - 1, 200)):
        _, p = eng.plain_from(P + t, ref[t])
        if t < 5 or int((p > 0).sum()) < 2 or float(p.max()) > 0.8:
            continue
        d = eng.drafts_at(P + t, ref[t]); eng.plain_from(P + t, ref[t])
        if 0.1 <= float(p[d[0]]) <= 0.9:
            return prompt, ref, P, P + t, t, d, None, None
    return None


if __name__ == "__main__":
    print(f"PART {PART} torch {torch.__version__} gpu {torch.cuda.get_device_name() if torch.cuda.is_available() else None}", flush=True)
    r = {"greedy": part_greedy, "dist": part_dist, "seed": part_seed, "diag": part_diag}[PART]()
    json.dump(r, open(os.path.join(OUT, f"test_{PART}{os.environ.get('EVAL_TAG', '')}.json"), "w"), indent=1, default=str)
    print("ALL DONE", flush=True)
