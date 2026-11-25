"""Correctness checks of engine_sample.py (lossless speculative sampling in the Ouro CUDA-graph engine); mirror of test_raven_sample.py.
  PART=toy     verify() / sample() (the exact functions the engine's graphs run) on a toy Markov target with a parallel drafter
               (q_j fixed before the draws, like LoopAhead), many full generations; empirical marginals / pair joints of the emitted
               sequence vs the exact chain (chi-square) + a negative control (residual replaced by p: must fail). CPU is fine.
  PART=greedy  T=0 (exact greedy switch): OuroSampleEngine's plain and speculative outputs vs the GREEDY engine (engine.Engine)
               token for token, on the first N_PROBLEMS prompts of each task, for every model (MODELS); also: two T>0 runs with
               the same seed give identical outputs (reproducibility of the device RNG through the CUDA graphs).
  PART=dist    on the real models at fixed contexts (prompt 0 + reference prefix, first position whose nucleus has >= 3 tokens,
               max prob <= 0.75 and a random second token): N_SAMPLES draws of (t1, t2) by speculative cycles from the same state
               and by plain sampling, for each sampling setting (SETTINGS); t1 vs p_1, (t1, t2) vs p_1 p_2, t2 | t1=a vs p_2(.|a)
               (references from the engine's own 1-row plain step) by chi-square.
  PART=dist2   two-sample test (homogeneity) plain vs LoopAhead over whole SEQ_LEN-token continuations that cross draft-cycle
               boundaries (per-position token distributions + 3-token prefixes), fixed context, N_SEQ each.
Env: MODELS (comma list of 14,26,thinking), OUT (runs/sampling_ouro/tests), EVAL_TAG; toy: TOY_N, TOY_SEED, TOY_ONLY, TOY_DEV.  Writes <OUT>/test_<part><tag>.json."""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import json, os, sys, time
import torch
import os
from floodgates.paths import RUNS, model_dir, trace
from scipy import stats
from floodgates.engines.sampling_ops import verify, probs, sample

PART = os.environ.get("PART", "toy")
OUT = os.environ.get("OUT", os.path.join(RUNS, "sampling_ouro", "tests")); os.makedirs(OUT, exist_ok=True)
MODELS = {  # name -> (drafter run, [(task, trace, n_problems, max_new, stops)])
    "14": (f"{RUNS}/floodgates-Ouro-1.4B", [("gsm8k", trace("Ouro-1.4B", "gsm8k"), 10, 512, ["Q:", "</s>", "<|im_end|>"]),
                                           ("math500", trace("Ouro-1.4B", "math500"), 3, 768, ["Problem:"])]),
    "26": (f"{RUNS}/floodgates-Ouro-2.6B", [("gsm8k", trace("Ouro-2.6B", "gsm8k"), 10, 512, ["Q:", "</s>", "<|im_end|>"]),
                                           ("math500", trace("Ouro-2.6B", "math500"), 3, 768, ["Problem:"])]),
    "thinking": (f"{RUNS}/floodgates-Ouro-1.4B-Thinking", [("gsm8kchat", trace("Ouro-1.4B-Thinking", "gsm8kchat"), 4, 1024,
                                            ["Q:", "</s>", "<|im_end|>"])]),
}
SEL = os.environ.get("MODELS", "14,26,thinking").split(",")
SETTINGS = [(1.0, 0.7), (0.6, 0.95)]


def chi2(obs, exp_p):
    """obs: counts dict token -> n; exp_p: dict token -> prob (support).  Bins with expected >= 5, rest pooled; out-of-support
    observations reported separately (must be 0)."""
    N = sum(obs.values()); oos = sum(v for k, v in obs.items() if exp_p.get(k, 0.0) == 0.0)
    big = [k for k, p in exp_p.items() if p * N >= 5]; small = [k for k in exp_p if k not in big]
    fo = [obs.get(k, 0) for k in big]; fe = [exp_p[k] * N for k in big]
    if small and sum(exp_p[k] for k in small) * N >= 5:
        fo.append(sum(obs.get(k, 0) for k in small)); fe.append(sum(exp_p[k] for k in small) * N)
    elif small and fo:
        fo[-1] += sum(obs.get(k, 0) for k in small); fe[-1] += sum(exp_p[k] for k in small) * N
    if fo:
        fo[-1] += oos
    tv = 0.5 * sum(abs(obs.get(k, 0) / N - exp_p.get(k, 0.0)) for k in set(obs) | set(exp_p))
    if len(fo) < 2:
        return {"N": N, "bins": len(fo), "chi2": 0.0, "p_value": 1.0, "tv": tv, "out_of_support": oos}
    c, pv = stats.chisquare(fo, [x * sum(fo) / sum(fe) for x in fe])
    return {"N": N, "bins": len(fo), "chi2": float(c), "p_value": float(pv), "tv": tv, "out_of_support": oos}


# ---------------------------------------------------------------------------------------------------------------------
def part_toy():
    torch.manual_seed(int(os.environ.get("TOY_SEED", 0)))
    dev = os.environ.get("TOY_DEV", "cpu"); V, n, LEN, NSEQ = 6, 3, 5, int(os.environ.get("TOY_N", 40000))
    P = torch.distributions.Dirichlet(torch.full((V,), 0.6)).sample((V,)).to(dev)
    Qd = torch.distributions.Dirichlet(torch.full((V,), 0.6)).sample((n, V)).to(dev)
    res = {}

    def verify_broken(p, d, q):                                         # NEGATIVE CONTROL: emits from p (not the residual) at a rejection
        n_ = d.shape[0]
        ratio = (p[:n_].gather(1, d[:, None])[:, 0] / q.gather(1, d[:, None])[:, 0].clamp(min=1e-20)).clamp(max=1.0)
        acc = (torch.rand(n_, device=p.device) < ratio).long().cumprod(0).sum()
        return acc, sample(p.index_select(0, acc[None]))[0]
    cfgs = [("sample", 1.0, 1.0, False), ("sample", 1.0, 0.7, True), ("sample", 0.6, 0.95, True), ("argmax", 1.0, 0.7, False),
            ("broken", 1.0, 1.0, False)]
    if os.environ.get("TOY_ONLY"):                                      # e.g. TOY_ONLY=1 -> cfgs[1] (+ the negative control)
        cfgs = [cfgs[int(i)] for i in os.environ["TOY_ONLY"].split(",")] + [cfgs[-1]]
    for mode, temp, top_p, q_topp in cfgs:
        logP = P.log()
        tgt = lambda prev: probs(logP[prev], temp, top_p)
        Texact = tgt(torch.arange(V, device=dev))
        seqs = torch.zeros(NSEQ, LEN, dtype=torch.long); accs = []; t0 = time.time()
        for s in range(NSEQ):
            y = torch.tensor(0, device=dev); out = []
            while len(out) < LEN:
                if mode in ("sample", "broken"):
                    q = probs(Qd[:, y].log(), temp, top_p if q_topp else 1.0); d = sample(q)
                else:
                    q = None; d = Qd[:, y].argmax(-1)
                p = tgt(torch.cat([y[None], d]))
                acc, e = (verify_broken if mode == "broken" else verify)(p, d, q)
                a = int(acc); out += d[:a].tolist() + [int(e)]; accs.append(a + 1); y = e
            seqs[s] = torch.tensor(out[:LEN])
        r = {"tau": sum(accs) / len(accs), "seconds": time.time() - t0}
        pi = torch.zeros(V, device=dev); pi[0] = 1.0; marg = []
        for t in range(LEN):
            pi = pi @ Texact; marg.append(pi.clone())
            r[f"pos{t + 1}"] = chi2({k: int((seqs[:, t] == k).sum()) for k in range(V)}, {k: float(pi[k]) for k in range(V)})
        for t in (0, 2):
            J = marg[t][:, None] * Texact; obs = {}
            for a_, b_ in zip(seqs[:, t].tolist(), seqs[:, t + 1].tolist()):
                obs[(a_, b_)] = obs.get((a_, b_), 0) + 1
            r[f"pair{t + 1}{t + 2}"] = chi2(obs, {(a_, b_): float(J[a_, b_]) for a_ in range(V) for b_ in range(V)})
        key = f"{mode}_T{temp}_topp{top_p}" + ("_qtopp" if q_topp else "")
        res[key] = r
        print(key, json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}) for k, v in r.items()}), flush=True)
    neg = res.pop("broken_T1.0_topp1.0")
    pv = [v["p_value"] for r in res.values() for v in r.values() if isinstance(v, dict)]
    res["negative_control_broken_residual"] = neg
    res["negative_control_min_p_value"] = min(v["p_value"] for v in neg.values() if isinstance(v, dict))
    res["summary"] = {"n_tests": len(pv), "min_p_value": min(pv), "n_p_below_0.01": sum(x < 0.01 for x in pv),
                      "out_of_support": sum(v["out_of_support"] for r in res.values() if isinstance(r, dict) for v in r.values() if isinstance(v, dict) and "out_of_support" in v),
                      "negative_control_min_p_value": res["negative_control_min_p_value"]}
    print("TOY SUMMARY", res["summary"], flush=True)
    return res


# ---------------------------------------------------------------------------------------------------------------------
def load(name):
    from floodgates.models.ouro_min import OuroMin
    from floodgates.drafters.drafter import LoopFlowDrafter
    from floodgates.tokenizer import load_tokenizer
    run = MODELS[name][0]
    a = json.load(open(os.path.join(run, "args.json")))
    ouro = OuroMin(model_dir(a["model"]))
    dr = LoopFlowDrafter(ouro, k=a["k"], rank=a["rank"], kappa=a["kappa"], use_state=not a["no_state"], use_noise=not a["no_noise"],
                         flowmap=a.get("flowmap", False), feedback=a.get("feedback", False))
    missing, unexpected = dr.load_state_dict(torch.load(os.path.join(run, "drafter.pt")), strict=False)
    assert not unexpected
    return ouro, dr.eval().requires_grad_(False), load_tokenizer(model_dir(a["model"]))


def part_greedy():
    from floodgates.engines.engine import Engine
    from floodgates.engines.engine_sample import OuroSampleEngine
    res = {}
    for name in SEL:
        ouro, dr, tok = load(name)
        for task, tr, NP, MAX_NEW, STOPS in MODELS[name][1]:
            rows = [json.loads(l) for l in open(tr)][:NP]
            LB = max(12, max(len(tok(x, add_special_tokens=False)["input_ids"]) for x in STOPS) + dr.k + 2)
            stop = lambda out: any(x in tok.decode(out[-LB:]) for x in STOPS) or tok.eos_token_id in out[-dr.k - 1:]
            max_len = max(len(r["input"]) for r in rows) + MAX_NEW + dr.k + 2; tmax = max(4096, 256 * (max_len // 256 + 1))
            outs = {}
            for kind in ("greedy", "sample_T0", "sample_T1_a", "sample_T1_b"):
                if kind == "greedy":
                    eng = Engine(ouro, dr, n_rounds=1, tmax=tmax)
                else:
                    eng = OuroSampleEngine(ouro, dr, n_rounds=1, temp=0.0 if kind == "sample_T0" else 1.0, top_p=0.7, q_topp=True, tmax=tmax)
                t0 = time.time(); eng.warmup(max_len); tc = time.time() - t0
                o = []
                for i, r in enumerate(rows):
                    torch.cuda.manual_seed(123 + i)
                    op, _ = eng.generate_plain(r["input"], MAX_NEW, stop)
                    torch.cuda.manual_seed(456 + i)
                    os_, taus, _ = eng.generate_spec(r["input"], MAX_NEW, stop)
                    o.append((op, os_, taus))
                outs[kind] = o
                print(f"  {name} {task} {kind}: captured in {tc:.0f}s; lens plain {[len(x[0]) for x in o]} spec {[len(x[1]) for x in o]} "
                      f"tau {[round(sum(x[2]) / len(x[2]), 2) for x in o]}", flush=True)
                del eng; torch.cuda.empty_cache()
            g, s, a1, b1 = outs["greedy"], outs["sample_T0"], outs["sample_T1_a"], outs["sample_T1_b"]
            r_ = {"plain_identical": sum(x[0] == y[0] for x, y in zip(g, s)), "spec_identical": sum(x[1] == y[1] for x, y in zip(g, s)),
                  "taus_identical": sum(x[2] == y[2] for x, y in zip(g, s)), "n": len(rows),
                  "tokens_plain": sum(len(x[0]) for x in g), "tokens_spec": sum(len(x[1]) for x in g),
                  "greedy_spec_eq_greedy_plain": sum(x[1][:len(x[0])] == x[0][:len(x[1])] for x in g),
                  "T1_same_seed_plain_identical": sum(x[0] == y[0] for x, y in zip(a1, b1)),
                  "T1_same_seed_spec_identical": sum(x[1] == y[1] for x, y in zip(a1, b1)),
                  "T1_spec_differs_from_greedy": sum(x[1] != y[1] for x, y in zip(a1, g))}
            res[f"{name}_{task}"] = r_
            print(f"GREEDY CHECK {name} {task}: {r_}", flush=True)
        del ouro, dr; torch.cuda.empty_cache()
    ok = all(v["plain_identical"] == v["n"] and v["spec_identical"] == v["n"] and v["taus_identical"] == v["n"]
             and v["T1_same_seed_plain_identical"] == v["n"] and v["T1_same_seed_spec_identical"] == v["n"] for v in res.values())
    res["summary"] = {"all_identical": ok}
    print("GREEDY SUMMARY", res["summary"], flush=True)
    return res


def part_dist():
    from floodgates.engines.engine_sample import OuroSampleEngine
    NS = int(os.environ.get("N_SAMPLES", 4000))
    res = {}
    for mi, name in enumerate(SEL):
        ouro, dr, tok = load(name)
        tr = MODELS[name][1][0][1]
        row = [json.loads(l) for l in open(tr)][0]
        prompt, ref = row["input"], row["output_ids"]; P = len(prompt)
        for si, (temp, top_p) in enumerate(SETTINGS):
            if os.environ.get("DIST_ONLY") and f"{temp}/{top_p}" not in os.environ["DIST_ONLY"].split(","):
                continue
            eng = OuroSampleEngine(ouro, dr, n_rounds=1, temp=temp, top_p=top_p, q_topp=True, keep_p=True, tmax=4096)
            eng.warmup(P + 400)
            torch.cuda.manual_seed(7); eng.prefill(prompt)                     # rows 0..P-1
            cut = None
            for t in range(len(ref) - 1):                                        # force the reference tokens; find a context whose first
                _, p = eng.plain_from(P + t, ref[t])                             # AND second token are random
                nz = int((p > 0).sum()); mx = float(p.max())
                if t >= 5 and nz >= 3 and mx <= 0.75:
                    p1d = {int(i): float(p[i]) for i in torch.nonzero(p > 0)[:, 0]}
                    p2 = {}
                    for a in p1d:
                        eng.plain_from(P + t, ref[t]); _, pa = eng.plain_from(P + t + 1, a)
                        p2[a] = {int(i): float(pa[i]) for i in torch.nonzero(pa > 0)[:, 0]}
                    a0 = max(p1d, key=p1d.get)
                    if len(p2[a0]) >= 2:
                        cut, ctx_t = P + t, t
                        break
                    eng.plain_from(P + t, ref[t])
            assert cut is not None, f"no suitable context for {name} T{temp}"
            y = ref[ctx_t]
            joint = {(a, b): p1d[a] * pb for a in p1d for b, pb in p2[a].items()}

            def draws(fn):
                c1, c2 = {}, {}
                for _ in range(NS):
                    t1, t2 = fn()
                    c1[t1] = c1.get(t1, 0) + 1; c2.setdefault(t1, {}); c2[t1][t2] = c2[t1].get(t2, 0) + 1
                return c1, c2

            def spec_draw():
                new = eng.cycle_from(cut, y)
                if len(new) < 2:
                    new = new + eng.cycle_from(cut + 1, new[0])
                return new[0], new[1]

            def plain_draw():
                t1, _ = eng.plain_from(cut, y); t2, _ = eng.plain_from(cut + 1, t1)
                return t1, t2

            ctx = {"context_tokens": P + ctx_t + 1, "cut": cut, "y": tok.decode([y]),
                   "nucleus_p1": {tok.decode([k]): round(v, 4) for k, v in p1d.items()},
                   "nucleus_p2_size": {tok.decode([a]): len(p2[a]) for a in p1d}}
            taus = []
            todo = (("plain sampling", plain_draw, 11 + 1000 * mi + 100 * si), ("LoopAhead spec sampling", spec_draw, 12 + 1000 * mi + 100 * si))
            if os.environ.get("DIST_SEED_OFFSET"):                             # recheck with fresh streams
                todo = tuple((n_, f_, s_ + int(os.environ["DIST_SEED_OFFSET"])) for n_, f_, s_ in todo)
            if os.environ.get("DIST_PLAIN_ONLY") == "1":
                todo = todo[:1]
            for nm, fn, seed in todo:
                torch.cuda.manual_seed(seed)
                t0 = time.time(); c1, c2 = draws(fn)
                cj = {(a, b): v for a in c2 for b, v in c2[a].items()}
                r = {"context": ctx, "seed": seed, "t1": chi2(c1, p1d), "joint_t1_t2": chi2(cj, joint), "seconds": time.time() - t0}
                for a in [k for k in sorted(c2, key=lambda k: -c1[k]) if k in p2][:2]:
                    r[f"t2|t1={tok.decode([a])!r}"] = chi2(c2[a], p2[a])
                key = f"{name} T{temp} top-p {top_p}: {nm}"
                res[key] = r
                print(f"DIST {key}: {json.dumps(r)}", flush=True)
            del eng; torch.cuda.empty_cache()
        del ouro, dr; torch.cuda.empty_cache()
    pv = [v["p_value"] for r in res.values() for k, v in r.items() if isinstance(v, dict) and "p_value" in v]
    res["summary"] = {"n_tests": len(pv), "min_p_value": min(pv), "n_p_below_0.01": sum(x < 0.01 for x in pv),
                      "n_p_below_0.05": sum(x < 0.05 for x in pv),
                      "out_of_support": sum(v["out_of_support"] for r in res.values() for v in r.values() if isinstance(v, dict) and "out_of_support" in v)}
    print("DIST SUMMARY", res["summary"], flush=True)
    return res


def part_dist2():
    """Two-sample test over whole continuations that cross draft-cycle boundaries: from the context prompt 0 + the first CTX_REF
    reference tokens, N_SEQ continuations of SEQ_LEN tokens (no stop string) by plain sampling and by LoopAhead speculative
    sampling (independent seeds, same engine); per position t, chi-square test of homogeneity of the token distributions
    (tokens with < 10 total occurrences pooled), and the same for the 3-token prefix (positions 1-3)."""
    from floodgates.engines.engine_sample import OuroSampleEngine
    NSEQ = int(os.environ.get("N_SEQ", 1500)); LEN = int(os.environ.get("SEQ_LEN", 20)); CTXR = int(os.environ.get("CTX_REF", 8))
    POS = [1, 2, 3, 4, 6, 8, 10, 12, 13, 14, 15, 16, 18, 20]
    settings = [tuple(float(v) for v in x.split("/")) for x in os.environ.get("D2_SETTINGS", "1.0/0.7,0.6/0.95").split(",")]
    res = {}

    def homog(a, b, min_count=10):
        keys = set(a) | set(b)
        big = [k for k in keys if a.get(k, 0) + b.get(k, 0) >= min_count]
        rest = [k for k in keys if k not in big]
        ta = [a.get(k, 0) for k in big]; tb = [b.get(k, 0) for k in big]
        ra, rb = sum(a.get(k, 0) for k in rest), sum(b.get(k, 0) for k in rest)
        if ra + rb >= min_count:
            ta.append(ra); tb.append(rb)
        elif ta:
            ta[-1] += ra; tb[-1] += rb
        if len(ta) < 2:
            return {"bins": len(ta), "p_value": 1.0, "tv": 0.0}
        c, pv, dof, _ = stats.chi2_contingency([ta, tb])
        na, nb = sum(a.values()), sum(b.values())
        tv = 0.5 * sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in keys)
        return {"bins": len(ta), "chi2": float(c), "p_value": float(pv), "tv": tv}
    for mi, name in enumerate(SEL):
        ouro, dr, tok = load(name)
        row = [json.loads(l) for l in open(MODELS[name][1][0][1])][0]
        ctx = row["input"] + row["output_ids"][:CTXR]
        for si, (temp, top_p) in enumerate(settings):
            eng = OuroSampleEngine(ouro, dr, n_rounds=1, temp=temp, top_p=top_p, q_topp=True, tmax=4096)
            eng.warmup(len(ctx) + LEN + 2 * eng.k1)
            never = lambda out: False
            seqs = {}
            t0 = time.time()
            for kind, seed in (("plain", 31 + 1000 * mi + 100 * si), ("spec", 32 + 1000 * mi + 100 * si)):
                torch.cuda.manual_seed(seed); out = []; taus = []
                for _ in range(NSEQ):
                    if kind == "plain":
                        o, _ = eng.generate_plain(ctx, LEN, never)
                    else:
                        o, tau, _ = eng.generate_spec(ctx, LEN, never); taus += tau
                    out.append(o[:LEN])
                seqs[kind] = out
                if taus:
                    res_tau = sum(taus) / len(taus)
            r = {"context_tokens": len(ctx), "n_seq": NSEQ, "len": LEN, "spec_tau": res_tau, "seconds": time.time() - t0,
                 "spec_cycles_per_seq": None}
            for t in POS:
                if t > LEN:
                    continue
                ca, cb = {}, {}
                for o in seqs["plain"]:
                    ca[o[t - 1]] = ca.get(o[t - 1], 0) + 1
                for o in seqs["spec"]:
                    cb[o[t - 1]] = cb.get(o[t - 1], 0) + 1
                r[f"pos{t}"] = homog(ca, cb)
            ca, cb = {}, {}
            for o in seqs["plain"]:
                ca[tuple(o[:3])] = ca.get(tuple(o[:3]), 0) + 1
            for o in seqs["spec"]:
                cb[tuple(o[:3])] = cb.get(tuple(o[:3]), 0) + 1
            r["prefix123"] = homog(ca, cb)
            key = f"{name} T{temp} top-p {top_p}"
            res[key] = r
            print(f"DIST2 {key}: " + json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}) for k, v in r.items()}), flush=True)
            del eng; torch.cuda.empty_cache()
        del ouro, dr; torch.cuda.empty_cache()
    pv = [v["p_value"] for r in res.values() for v in r.values() if isinstance(v, dict) and "p_value" in v]
    res["summary"] = {"n_tests": len(pv), "min_p_value": min(pv), "n_p_below_0.01": sum(x < 0.01 for x in pv),
                      "n_p_below_0.05": sum(x < 0.05 for x in pv)}
    print("DIST2 SUMMARY", res["summary"], flush=True)
    return res


def part_diag():
    """Where does LoopAhead's emitted-token law differ from the 1-row plain step?  At the dist contexts: p_plain (1-row plain step,
    fp32-P decode kernel) vs p_verify (row 0 of the (k+1)-row verification pass, the target the rejection sampler is exact for):
    TV, support, and the raw bf16 logits of the nucleus tokens; then N_DIAG speculative first tokens tested against BOTH."""
    from floodgates.engines.engine_sample import OuroSampleEngine
    NS = int(os.environ.get("N_DIAG", 2000)); res = {}
    for mi, name in enumerate(SEL):
        ouro, dr, tok = load(name)
        row = [json.loads(l) for l in open(MODELS[name][1][0][1])][0]
        prompt, ref = row["input"], row["output_ids"]; P = len(prompt)
        for si, (temp, top_p) in enumerate(SETTINGS):
            eng = OuroSampleEngine(ouro, dr, n_rounds=1, temp=temp, top_p=top_p, q_topp=True, keep_p=True, tmax=4096)
            eng.warmup(P + 400); torch.cuda.manual_seed(7); eng.prefill(prompt)
            cut = None
            for t in range(len(ref) - 1):                                    # same context rule as part_dist
                _, p = eng.plain_from(P + t, ref[t])
                if t >= 5 and int((p > 0).sum()) >= 3 and float(p.max()) <= 0.75:
                    cut, y = P + t, ref[t]; p_plain = p.clone(); lg_plain = eng._last_lm[0].float().clone(); break
            assert cut is not None
            eng.plain_from(cut, y)
            eng.cycle_from(cut, y)
            p_ver = eng.pbuf[0].clone(); lg_ver = eng._last_lm[0].float().clone()
            sup = sorted(set(torch.nonzero(p_plain > 0)[:, 0].tolist()) | set(torch.nonzero(p_ver > 0)[:, 0].tolist()))
            top = torch.topk(lg_plain, 8).indices.tolist()
            r = {"tv_plain_verify": 0.5 * float((p_plain - p_ver).abs().sum()),
                 "nucleus_plain_vs_verify": {tok.decode([i]): (round(float(p_plain[i]), 4), round(float(p_ver[i]), 4)) for i in sup},
                 "bf16_logits_plain_vs_verify_top8": {tok.decode([i]): (float(lg_plain[i]), float(lg_ver[i])) for i in top},
                 "n_top8_logits_differ": sum(float(lg_plain[i]) != float(lg_ver[i]) for i in top)}
            torch.cuda.manual_seed(900 + 100 * mi + 10 * si); c1 = {}
            for _ in range(NS):
                t1 = eng.cycle_from(cut, y)[0]; c1[t1] = c1.get(t1, 0) + 1
            r["spec_t1_vs_p_verify"] = chi2(c1, {int(i): float(p_ver[i]) for i in torch.nonzero(p_ver > 0)[:, 0]})
            r["spec_t1_vs_p_plain"] = chi2(c1, {int(i): float(p_plain[i]) for i in torch.nonzero(p_plain > 0)[:, 0]})
            key = f"{name} T{temp} top-p {top_p}"; res[key] = r
            print(f"DIAG {key}: {json.dumps(r)}", flush=True)
            del eng; torch.cuda.empty_cache()
        del ouro, dr; torch.cuda.empty_cache()
    pv = [r["spec_t1_vs_p_verify"]["p_value"] for r in res.values()]
    res["summary"] = {"n_tests_vs_verify": len(pv), "min_p_vs_verify": min(pv), "oos_vs_verify": sum(r["spec_t1_vs_p_verify"]["out_of_support"] for r in res.values() if "spec_t1_vs_p_verify" in r),
                      "max_tv_plain_verify": max(r["tv_plain_verify"] for r in res.values() if "tv_plain_verify" in r)}
    print("DIAG SUMMARY", res["summary"], flush=True)
    return res


if __name__ == "__main__":
    print(f"PART {PART} torch {torch.__version__} gpu {torch.cuda.get_device_name() if torch.cuda.is_available() else None}", flush=True)
    r = {"toy": part_toy, "greedy": part_greedy, "dist": part_dist, "dist2": part_dist2, "diag": part_diag}[PART]()
    json.dump(r, open(os.path.join(OUT, f"test_{PART}{os.environ.get('EVAL_TAG', '')}.json"), "w"), indent=1, default=str)
    print("ALL DONE", flush=True)
