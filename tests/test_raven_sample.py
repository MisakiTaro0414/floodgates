"""Correctness checks of raven_engine_sample.py (lossless speculative sampling in the CUDA-graph engine).
  PART=toy     verify() (the exact function the engine's graphs run) on a toy Markov target with a parallel drafter: many full
               generations; the empirical marginals / pair joints of the emitted sequence vs the exact chain (chi-square).
  PART=greedy  T=0 (exact greedy switch): RavenSampleEngine's plain and speculative outputs vs the GREEDY engine
               (raven_engine.RavenEngine) on the first N_PROBLEMS GSM8K prompts, for every method of the benchmark (identity).
  PART=dist    T=1, top-p 0.7 on the real model at one fixed context (GSM8K prompt 0 + reference prefix, first position whose
               nucleus has >= 3 tokens): N_SAMPLES draws of (t1, t2) per method (spec cycles from the same state) and for plain
               sampling; t1 vs p_1 and t2 | t1=a vs p_2(.|a) (both from the engine's own target pass) by chi-square.
Writes runs/raven/sampling_wc/test_<part>.json."""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import json, os, sys, time
import torch
import os
from floodgates.paths import LOOPSPEC, RUNS, trace
from scipy import stats
from floodgates.engines.raven_engine_sample import RavenSampleEngine, verify, probs, sample, probs_cand

PART = os.environ.get("PART", "toy")
OUT = os.environ.get("OUT", os.path.join(RUNS, "raven", "sampling_wc")); os.makedirs(OUT, exist_ok=True)
L = LOOPSPEC
MODEL = f"{L}/models/Recurrent-Llama-3.2-train-recurrence-32"
TRACE = trace(MODEL, "gsm8k")
DR = os.environ.get("RUN", os.path.join(RUNS, "floodgates-Recurrent-Llama-3.2-train-recurrence-32"))
METHODS = [  # (name, run_dir or None, n_rounds, env, das)
    ("floodgates n1", DR, 1, {}, None),
    ("daswave argmax J24 k1 g32", None, 1, {}, {"mode": "wave", "J": 24, "k": 1, "gamma": 32, "q": "argmax"}),
    ("daswave sample J24 k1 g32", None, 1, {}, {"mode": "wave", "J": 24, "k": 1, "gamma": 32, "q": "sample"}),
]
if os.environ.get("ONLY"):
    METHODS = [x for x in METHODS if any(s in x[0] for s in os.environ["ONLY"].split(","))]


def chi2(obs, exp_p):
    """obs: counts dict token -> n; exp_p: dict token -> prob (support).  Bins with expected >= 5, rest pooled; out-of-support
    observations reported separately (must be 0)."""
    N = sum(obs.values()); oos = sum(v for k, v in obs.items() if exp_p.get(k, 0.0) == 0.0)
    big = [k for k, p in exp_p.items() if p * N >= 5]; small = [k for k in exp_p if k not in big]
    fo = [obs.get(k, 0) for k in big]; fe = [exp_p[k] * N for k in big]
    if small and sum(exp_p[k] for k in small) * N >= 5:
        fo.append(sum(obs.get(k, 0) for k in small)); fe.append(sum(exp_p[k] for k in small) * N)
    elif small:                                                       # fold the small remainder into the last bin
        fo[-1] += sum(obs.get(k, 0) for k in small); fe[-1] += sum(exp_p[k] for k in small) * N
    fo[-1] += oos                                                     # (oos is reported; keep the totals equal)
    tv = 0.5 * sum(abs(obs.get(k, 0) / N - exp_p.get(k, 0.0)) for k in set(obs) | set(exp_p))
    if len(fo) < 2:
        return {"N": N, "bins": len(fo), "chi2": 0.0, "p_value": 1.0, "tv": tv, "out_of_support": oos}
    c, pv = stats.chisquare(fo, [x * sum(fo) / sum(fe) for x in fe])
    return {"N": N, "bins": len(fo), "chi2": float(c), "p_value": float(pv), "tv": tv, "out_of_support": oos}


# ---------------------------------------------------------------------------------------------------------------------
def part_toy():
    """Markov target p(x | prev) = nucleus(P[prev], top_p); parallel drafter q_j = Qd[j, y] (fixed before the draws, like ours and
    DAS-Wave sampled), d_j ~ q_j; or argmax drafts with one-hot q (DAS-Wave argmax).  Full generations of LEN tokens from y0."""
    torch.manual_seed(0)
    dev = os.environ.get("TOY_DEV", "cpu"); V, n, LEN, NSEQ = 6, 3, 5, int(os.environ.get("TOY_N", 40000))
    P = torch.distributions.Dirichlet(torch.full((V,), 0.6)).sample((V,)).to(dev)
    Qd = torch.distributions.Dirichlet(torch.full((V,), 0.6)).sample((n, V)).to(dev)
    res = {}

    def verify_broken(p, d, q):                                         # NEGATIVE CONTROL: emits from p (not the residual) at a rejection
        n_ = d.shape[0]
        ratio = (p[:n_].gather(1, d[:, None])[:, 0] / q.gather(1, d[:, None])[:, 0].clamp(min=1e-20)).clamp(max=1.0)
        acc = (torch.rand(n_, device=p.device) < ratio).long().cumprod(0).sum()
        return acc, sample(p.index_select(0, acc[None]))[0]
    for mode, top_p, q_topp in [("sample", 1.0, False), ("sample", 0.8, False), ("sample", 0.8, True), ("argmax", 0.8, False),
                                ("broken", 1.0, False)]:
        logP = P.log()
        tgt = lambda prev: probs(logP[prev], 1.0, top_p)                # rows (.., V)
        Texact = tgt(torch.arange(V, device=dev))                       # exact transition matrix
        y0 = 0
        seqs = torch.zeros(NSEQ, LEN, dtype=torch.long)
        accs = []
        t0 = time.time()
        for s in range(NSEQ):
            y = torch.tensor(y0, device=dev); out = []
            while len(out) < LEN:
                if mode in ("sample", "broken"):
                    q = probs(Qd[:, y].log(), 1.0, top_p if q_topp else 1.0)   # (n, V)
                    d = sample(q)
                else:
                    q = None; d = Qd[:, y].argmax(-1)
                p = tgt(torch.cat([y[None], d]))                        # (n+1, V): rows for inputs [y, d_1..d_n]
                acc, e = (verify_broken if mode == "broken" else verify)(p, d, q)
                a = int(acc); out += d[:a].tolist() + [int(e)]; accs.append(a + 1); y = e
            seqs[s] = torch.tensor(out[:LEN])
        r = {"tau": sum(accs) / len(accs), "seconds": time.time() - t0}
        pi = torch.zeros(V, device=dev); pi[y0] = 1.0
        marg = []
        for t in range(LEN):                                            # exact marginal of position t+1
            pi = pi @ Texact; marg.append(pi.clone())
            obs = {k: int((seqs[:, t] == k).sum()) for k in range(V)}
            r[f"pos{t + 1}"] = chi2(obs, {k: float(pi[k]) for k in range(V)})
        for t in (0, 2):                                                # pair joints (x_{t+1}, x_{t+2})
            J = marg[t][:, None] * Texact
            obs = {}
            for a_, b_ in zip(seqs[:, t].tolist(), seqs[:, t + 1].tolist()):
                obs[(a_, b_)] = obs.get((a_, b_), 0) + 1
            r[f"pair{t + 1}{t + 2}"] = chi2(obs, {(a_, b_): float(J[a_, b_]) for a_ in range(V) for b_ in range(V)})
        key = f"{mode}_topp{top_p}" + ("_qtopp" if q_topp else "")
        res[key] = r
        print(key, json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}) for k, v in r.items()}), flush=True)
    neg = res.pop("broken_topp1.0")
    pv = [v["p_value"] for r in res.values() for v in r.values() if isinstance(v, dict)]
    oos = sum(v["out_of_support"] for r in res.values() for v in r.values() if isinstance(v, dict))
    res["negative_control_broken_residual"] = neg
    res["negative_control_min_p_value"] = min(v["p_value"] for v in neg.values() if isinstance(v, dict))
    res["summary"] = {"n_tests": len(pv), "min_p_value": min(pv), "n_p_below_0.01": sum(x < 0.01 for x in pv), "out_of_support": oos}
    print("TOY SUMMARY", res["summary"], flush=True)
    return res


# ---------------------------------------------------------------------------------------------------------------------
def load_common():
    from floodgates.models.raven_min import RavenMin
    from transformers import PreTrainedTokenizerFast
    m = RavenMin(MODEL)
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(MODEL, "tokenizer.json"))
    return m, tok


def make_dr(m, run):
    from floodgates.drafters.drafter_raven import RavenSelfDrafter
    a = json.load(open(os.path.join(run, "args.json")))
    dr = RavenSelfDrafter(m, k=a["k"], rank=a["rank"], shift=a.get("shift", False), feedback=a.get("feedback", False), iters=a.get("iters", 1), carry=a.get("carry", False))
    missing, unexpected = dr.load_state_dict(torch.load(os.path.join(run, "drafter.pt")), strict=False)
    assert not unexpected
    return dr.eval().requires_grad_(False)


def with_env(env, fn):
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return fn()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def part_greedy():
    from floodgates.engines.raven_engine import RavenEngine
    m, tok = load_common()
    NP = int(os.environ.get("N_PROBLEMS", 3)); MAX_NEW = int(os.environ.get("MAX_NEW", 512))
    rows = [json.loads(l) for l in open(TRACE)][:NP]
    res = {}
    for name, run, n, env, das in METHODS:
        dr = make_dr(m, run) if run else None
        K = (dr.k + (1 if dr.shift else 0)) if dr is not None else das["gamma"]
        LB = 12 + K + 2
        stop = lambda out: "Q:" in tok.decode(out[-LB:])
        max_len = max(len(r["input"]) for r in rows) + MAX_NEW + K + 2; tmax = min(4096, 256 * (max_len // 256 + 1))
        outs = {}
        for kind in ("greedy", "sample_T0"):
            def build():
                if kind == "greedy":
                    d_ = None if das is None else {k: v for k, v in das.items() if k != "q"}
                    return RavenEngine(m, dr, n_rounds=n, tmax=tmax, das=d_)
                return RavenSampleEngine(m, dr, n_rounds=n, temp=0.0, top_p=0.7, q_topp=True, tmax=tmax, das=das)
            eng = with_env(env, build)
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
        r_ = {"plain_identical": sum(a[0] == b[0] for a, b in zip(g, s)), "spec_identical": sum(a[1] == b[1] for a, b in zip(g, s)),
              "taus_identical": sum(a[2] == b[2] for a, b in zip(g, s)), "n": len(rows),
              "greedy_spec_eq_greedy_plain": sum(a[1][:len(a[0])] == a[0][:len(a[1])] for a in g),
              "tokens": sum(len(a[1]) for a in g)}
        res[name] = r_
        print(f"GREEDY CHECK {name}: {r_}", flush=True)
    return res


def part_dist():
    m, tok = load_common()
    NS = int(os.environ.get("N_SAMPLES", 3000)); TOP_P = 0.7
    row = [json.loads(l) for l in open(TRACE)][0]
    prompt, ref = row["input"], row["output_ids"]
    P = len(prompt)
    res = {}
    plain_done = False
    for name, run, n, env, das in METHODS:
        dr = make_dr(m, run) if run else None
        eng = with_env(env, lambda: RavenSampleEngine(m, dr, n_rounds=n, temp=1.0, top_p=TOP_P, q_topp=True, keep_p=True,
                                                      tmax=1024, das=das))
        eng.warmup(P + 200)
        torch.cuda.manual_seed(7)
        eng.prefill(prompt)                                            # rows 0..P-1
        cut, ctx_t = None, None
        for t in range(len(ref) - 1):                                   # force the reference tokens; find a context whose first
            _, p = eng.plain_from(P + t, ref[t])                        # AND second token are random: p(. | prompt, ref[:t+1])
            nz = int((p > 0).sum()); mx = float(p.max())
            if t >= 5 and nz >= 2 and mx <= 0.8:
                p1d = {int(i): float(p[i]) for i in torch.nonzero(p > 0)[:, 0]}
                p2 = {}
                for a in p1d:                                           # p_2(. | a) from the engine's own target pass
                    eng.plain_from(P + t, ref[t]); _, pa = eng.plain_from(P + t + 1, a)
                    p2[a] = {int(i): float(pa[i]) for i in torch.nonzero(pa > 0)[:, 0]}
                a0 = max(p1d, key=p1d.get)
                if len(p2[a0]) >= 2:
                    cut, ctx_t = P + t, t
                    break
                eng.plain_from(P + t, ref[t])                           # (row P+t back to the reference token)
        assert cut is not None
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

        ctx = {"context_tokens": P + ctx_t + 1, "cut": cut, "y": tok.decode([y]), "nucleus_p1": {tok.decode([k]): round(v, 4) for k, v in p1d.items()},
               "nucleus_p2": {tok.decode([a]): {tok.decode([b]): round(v, 4) for b, v in p2[a].items()} for a in p1d}}
        todo = [("plain sampling", plain_draw, 11)] if not plain_done else []
        todo.append((name, spec_draw, 11 + 100 * (1 + [x[0] for x in METHODS].index(name))))     # independent streams per method
        for nm, fn, seed in todo:
            torch.cuda.manual_seed(seed)
            t0 = time.time(); c1, c2 = draws(fn)
            cj = {(a, b): v for a in c2 for b, v in c2[a].items()}
            r = {"context": ctx, "seed": seed, "t1": chi2(c1, p1d), "joint_t1_t2": chi2(cj, joint), "seconds": time.time() - t0}
            for a in [k for k in sorted(c2, key=lambda k: -c1[k]) if k in p2][:2]:   # the two most frequent first tokens
                r[f"t2|t1={tok.decode([a])!r}"] = chi2(c2[a], p2[a])
            res[nm] = r
            print(f"DIST {nm}: {json.dumps(r)}", flush=True)
        plain_done = True
        del eng; torch.cuda.empty_cache()
    pv = [v["p_value"] for r in res.values() for k, v in r.items() if isinstance(v, dict) and "p_value" in v]
    res["summary"] = {"n_tests": len(pv), "min_p_value": min(pv), "n_p_below_0.01": sum(x < 0.01 for x in pv),
                      "out_of_support": sum(v["out_of_support"] for r in res.values() for v in r.values() if isinstance(v, dict) and "out_of_support" in v)}
    print("DIST SUMMARY", res["summary"], flush=True)
    return res


def part_diag():
    """Second-token reference check at the dist context: p_2(. | y, a) from the 1-row plain step vs the same row computed inside a
    verification pass (Q = n+1 rows, read from pbuf when the cycle's first draft is a), repeated to check determinism; then
    DIAG_N draws of t2 | t1 = a per method tested against both references."""
    m, tok = load_common()
    NS = int(os.environ.get("DIAG_N", 10000)); TOP_P = 0.7
    row = [json.loads(l) for l in open(TRACE)][0]
    prompt, ref = row["input"], row["output_ids"]; P = len(prompt)
    res = {}
    for mi, (name, run, n, env, das) in enumerate(METHODS):
        dr = make_dr(m, run) if run else None
        eng = with_env(env, lambda: RavenSampleEngine(m, dr, n_rounds=n, temp=1.0, top_p=TOP_P, q_topp=True, keep_p=True, tmax=1024, das=das))
        eng.warmup(P + 200); torch.cuda.manual_seed(7); eng.prefill(prompt)
        cut = None
        for t in range(len(ref) - 1):                                   # same context search as part_dist
            _, p = eng.plain_from(P + t, ref[t])
            if t >= 5 and int((p > 0).sum()) >= 2 and float(p.max()) <= 0.8:
                a0 = int(p.argmax()); eng.plain_from(P + t, ref[t]); _, pa = eng.plain_from(P + t + 1, a0)
                if int((pa > 0).sum()) >= 2:
                    cut, y, a = P + t, ref[t], a0; p_plain = pa.clone(); break
                eng.plain_from(P + t, ref[t])
        assert cut is not None
        torch.cuda.manual_seed(5000 + 100 * mi)
        p_ver = []
        for _ in range(4000):                                           # verification rows for input [y, a] (first draft = a)
            eng.cycle_from(cut, y)
            if int(eng.ids[0, 1]) == a:
                p_ver.append(eng.pbuf[1].clone())
                if len(p_ver) == 3:
                    break
        det = all(torch.equal(p_ver[0], x) for x in p_ver[1:])
        pv = p_ver[0]
        sup = sorted(set(torch.nonzero(p_plain > 0)[:, 0].tolist()) | set(torch.nonzero(pv > 0)[:, 0].tolist()))
        cmp_ = {tok.decode([i]): (round(float(p_plain[i]), 5), round(float(pv[i]), 5)) for i in sup}
        r = {"context_cut": cut, "y": tok.decode([y]), "a": tok.decode([a]), "verify_rows": eng.nd + 1,
             "p2_plain_vs_verify": cmp_, "verify_row_deterministic": det, "tv_plain_verify": 0.5 * float((p_plain - pv).abs().sum())}
        if NS:
            def spec2():
                new = eng.cycle_from(cut, y)
                if len(new) < 2:
                    new = new + eng.cycle_from(cut + 1, new[0])
                return new[0], new[1]
            c2 = {}
            for _ in range(NS):
                t1, t2 = spec2()
                if t1 == a:
                    c2[t2] = c2.get(t2, 0) + 1
            d_pl = {int(i): float(p_plain[i]) for i in torch.nonzero(p_plain > 0)[:, 0]}
            d_vr = {int(i): float(pv[i]) for i in torch.nonzero(pv > 0)[:, 0]}
            r.update({"t2_counts": {tok.decode([k]): v for k, v in c2.items()}, "chi2_vs_plain_p": chi2(c2, d_pl), "chi2_vs_verify_p": chi2(c2, d_vr)})
        res[name] = r
        print(f"DIAG {name}: {json.dumps(r)}", flush=True)
        del eng; torch.cuda.empty_cache()
    return res


def part_overhead():
    """GPU time of the sampling / verification ops alone (CUDA graph, real vocab 128256) vs the greedy engine's argmax + accept, for
    the plain step (1 row) and the verification shapes (n drafts + 1 rows; sampled q needs the extra q rows)."""
    dev = "cuda"; V = 128256; res = {}
    for (n, qm), K in [(x, K) for K in (-1, 4096) for x in [(0, None), (12, "sample"), (25, "sample"), (32, "sample"), (32, "argmax")]]:
        P_ = (lambda l: probs(l, 1.0, 0.7)) if K < 0 else (lambda l, K=K: probs_cand(l, 1.0, 0.7, K)[0])
        lg = (torch.randn(n + 1, V, device=dev) * 3).to(torch.bfloat16)
        dl = torch.randn(max(n, 1), V, device=dev) * 3
        ids = torch.randint(0, V, (n + 1,), device=dev)

        def samp():
            if n == 0:
                return sample(P_(lg))
            q = None
            if qm == "sample":
                q = P_(dl); ids[1:].copy_(sample(q))
            return verify(P_(lg), ids[1:], q)[1]

        def greedy():
            P = lg.argmax(-1)
            return (ids[1:] == P[:n]).long().cumprod(0).sum() if n else P
        out = {}
        for nm, fn in (("sampling", samp), ("greedy", greedy)):
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            for _ in range(10):
                g.replay()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(); e0.record()
            for _ in range(200):
                g.replay()
            e1.record(); torch.cuda.synchronize()
            out[f"ms_{nm}"] = e0.elapsed_time(e1) / 200
        res[f"drafts{n}_{qm}_{'sortnucleus' if K < 0 else f'topk{K}'}"] = out
        print(f"OVERHEAD {'sort nucleus (v1)' if K < 0 else f'tie nucleus top-{K} (v2)'} drafts {n} q {qm}: {out}", flush=True)
    return res


if __name__ == "__main__":
    print(f"PART {PART} torch {torch.__version__} gpu {torch.cuda.get_device_name() if torch.cuda.is_available() else None}", flush=True)
    r = {"toy": part_toy, "greedy": part_greedy, "dist": part_dist, "overhead": part_overhead, "diag": part_diag}[PART]()
    tag = os.environ.get("EVAL_TAG", "")
    json.dump(r, open(os.path.join(OUT, f"test_{PART}{tag}.json"), "w"), indent=1, default=str)
    print("ALL DONE", flush=True)
