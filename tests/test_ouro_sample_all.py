"""Correctness checks of engine_baselines_sample.py / bench_ouro_sample_all.py (lossless speculative sampling for every method of the
Ouro table).  Methods (METHODS): DAS 1,8 / 2,8, DAS-Wave 24,1,32 / 8,1,32 (sampled q), Jacobi 4,4,32 (no growth, no carry), PLD 3,12 /
2,8, LAL + 1 own-readout refinement loop (hybrid 1,1,16, carried states, sampled q), LAL one pass (engine_sample, sampled q).
  PART=greedy  T = 0 (exact greedy switch): the sampling engine's plain and speculative outputs (and cycle taus) vs the GREEDY engine
               of the same method (engine_baselines / engine_baselines_own / engine) in the same process, token for token, on the
               first N_PROBLEMS (5) GSM8K prompts of Ouro-1.4B (scripts/bench_greedy.sh stops / limit); also vs the stored greedy outputs of the
               table runs (runs/ouro-baselines/*_gsm8k_pp*_outputs.jsonl, <RUN>/engine_n1_gsm8k_outputs.jsonl; prefix
               agreement); and T = 1 / top-p 0.7: two runs with the same seeds give identical outputs.
  PART=dist    exact law of the first emitted token at fixed contexts (prompt 0 + a reference prefix: positions with >= 3 nucleus
               tokens and max prob <= 0.75, N_CTX of them): N_SAMPLES speculative cycles from the same state -> chi-square of t1 vs
               p_ver = row 0 of the cycle's verification pass (the target the rejection sampler is exact for; = the plain law up to
               the bf16 numerics of a (gamma+1)-row vs a 1-row step, reported as TV(p_plain, p_ver) and chi-square vs p_plain).
               PLD: its natural lookup draft when there is one, plus injected drafts (d_1 = the most likely / the 3rd most likely
               nucleus token).  Negative control (NEG=1): DAS-Wave 8,1,32 verified with the proposal rows rolled by one (q of d_1 ->
               q of d_gamma): must FAIL.
  PART=repro   same seed -> identical outputs across processes: compares two bench output files (REPRO_A, REPRO_B; '{m}' expanded
               over REPRO_METHODS labels) on their common problems.
Env: MODEL (14 | 26), RUN (the drafter, default runs/floodgates-<model>), DIST_METHODS (comma list of METHODS keys), N_SAMPLES (3000), N_CTX (2), OUT,
EVAL_TAG.  Writes <OUT> (runs/sampling_full/tests_sbo)/test_<part><tag>.json."""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import json, os, sys, time
import torch
import os
from floodgates.paths import RUNS, ROOT, model_dir, trace

PART = os.environ.get("PART", "greedy")
HERE = ROOT   # repository root
OUT = os.environ.get("OUT", os.path.join(RUNS, "sampling_full", "tests_sbo")); os.makedirs(OUT, exist_ok=True)
MK = os.environ.get("MODEL", "14")
W = {"14": "Ouro-1.4B", "26": "Ouro-2.6B"}[MK]
DR = os.environ.get("RUN", os.path.join(RUNS, f"floodgates-{W}"))
MODEL_DIR = model_dir(W)
TRACE = trace(W, "gsm8k")
STOPS, MAX_NEW = ["Q:", "</s>"], 512                                   # scripts/bench_greedy.sh gsm8k
# name -> (kind, cfg, das extras, q_mode, stored greedy label under runs/ouro-baselines (None: other / none))
METHODS = {
    "das18": ("das", "1,8", {}, "argmax", f"das_1-8_{W}_gsm8k_pp"),
    "das28": ("das", "2,8", {}, "argmax", f"das_2-8_{W}_gsm8k_pp"),
    "wave24": ("wave", "24,1,32", {}, "sample", f"wave_24-1-32_{W}_gsm8k_pp"),
    "wave8": ("wave", "8,1,32", {}, "sample", None),
    "jacobi": ("wave", "4,4,32", {"grow": False, "carry_rounds": False}, "argmax", f"wave_4-4-32_{W}_gsm8k_pp_nogrow_nocarry"),
    "pld312": ("pld", "3,12", {}, "argmax", f"pld_3-12_{W}_gsm8k_pp"),
    "pld28": ("pld", "2,8", {}, "argmax", f"pld_2-8_{W}_gsm8k_pp"),
    "hybrid": ("hybrid", "1,1,16", {"carry": True}, "sample", f"hybrid_1-1-16_{W}_gsm8k_pp_own_carry"),
    "lal": ("lal", "1", {}, "sample", "LAL"),
}


def make_das(kind, cfg, extra):
    v = [int(x) for x in cfg.split(",")]
    if kind == "lal":
        return None
    das = ({"mode": "wave", "J": v[0], "k": v[1], "gamma": v[2]} if kind in ("wave", "hybrid") else
           {"mode": "pld", "n": v[0], "gamma": v[1]} if kind == "pld" else {"mode": "seq", "r": v[0], "gamma": v[1]})
    if kind == "hybrid":
        das["init"] = "loopahead"
    das.update(extra)
    return das


_cache = {}


def load():
    if "m" in _cache:
        return _cache["m"]
    from floodgates.models.ouro_min import OuroMin
    from floodgates.drafters.drafter import LoopFlowDrafter
    from floodgates.tokenizer import load_tokenizer
    a = json.load(open(os.path.join(DR, "args.json")))
    assert os.path.basename(a["model"].rstrip("/")) == W, (a["model"], W)
    ouro = OuroMin(MODEL_DIR)
    dr = LoopFlowDrafter(ouro, k=a["k"], rank=a["rank"], kappa=a["kappa"], use_state=not a["no_state"], use_noise=not a["no_noise"],
                         flowmap=a.get("flowmap", False), feedback=a.get("feedback", False))
    missing, unexpected = dr.load_state_dict(torch.load(os.path.join(DR, "drafter.pt")), strict=False)
    assert not unexpected
    _cache["m"] = (ouro, dr.eval().requires_grad_(False), load_tokenizer(MODEL_DIR))
    return _cache["m"]


def make_engine(name, sampling, temp=1.0, keep_p=False, tmax=4096, cls=None):
    ouro, dr, _ = load()
    kind, cfg, extra, q_mode, _ = METHODS[name]
    das = make_das(kind, cfg, extra)
    if kind == "lal":
        if not sampling:
            from floodgates.engines.engine import Engine
            return Engine(ouro, dr, n_rounds=1, tmax=tmax)
        from floodgates.engines.engine_sample import OuroSampleEngine
        return OuroSampleEngine(ouro, dr, n_rounds=1, temp=temp, top_p=0.7, q_topp=True, q_mode=q_mode, keep_p=keep_p, tmax=tmax)
    if not sampling:
        from floodgates.engines.engine_baselines import OuroBaselineEngine
        from floodgates.engines.engine_baselines_own import OuroOwnReadoutEngine
        E = OuroOwnReadoutEngine if kind == "hybrid" else OuroBaselineEngine
        return E(ouro, das, dr if kind == "hybrid" else None, tmax=tmax)
    from floodgates.engines.engine_baselines_sample import OuroBaselineSampleEngine, OuroOwnSampleEngine
    E = cls or (OuroOwnSampleEngine if kind == "hybrid" else OuroBaselineSampleEngine)
    return E(ouro, das, dr if kind == "hybrid" else None, temp=temp, top_p=0.7, q_topp=True, q_mode=q_mode, keep_p=keep_p, tmax=tmax)


def n_drafts(name):
    kind, cfg = METHODS[name][:2]
    return load()[1].k if kind == "lal" else int(cfg.split(",")[-1])


# ---------------------------------------------------------------------------------------------------------------------------
def part_greedy():
    NP = int(os.environ.get("N_PROBLEMS", 5))
    _, _, tok = load()
    rows = [json.loads(l) for l in open(TRACE)][:NP]
    res = {}
    sel = [m for m in os.environ.get("GREEDY_METHODS", ",".join(METHODS)).split(",") if m]
    for name in sel:
        K = n_drafts(name)
        LB = max(12, max(len(tok(x, add_special_tokens=False)["input_ids"]) for x in STOPS) + K + 2)
        stop = lambda out, LB=LB, K=K: any(x in tok.decode(out[-LB:]) for x in STOPS) or tok.eos_token_id in out[-K - 1:]
        max_len = max(len(r["input"]) for r in rows) + MAX_NEW + K + 2
        outs = {}
        for which in ("greedy", "sample_T0", "sample_T1"):
            eng = make_engine(name, which != "greedy", temp=0.0 if which == "sample_T0" else 1.0)
            t0 = time.time(); eng.warmup(max_len); tc = time.time() - t0
            runs = 2 if which == "sample_T1" else 1
            for rep in range(runs):
                o = []
                for i, r in enumerate(rows):
                    torch.cuda.manual_seed(1000 + i)
                    op, _ = eng.generate_plain(r["input"], MAX_NEW, stop)
                    torch.cuda.manual_seed(1000 + i)
                    os_, taus, _ = eng.generate_spec(r["input"], MAX_NEW, stop)
                    o.append((op, os_, taus))
                outs[which if runs == 1 else f"{which}_{'ab'[rep]}"] = o
                print(f"  {name} {which}{'_' + 'ab'[rep] if runs > 1 else ''}: captured in {tc:.0f}s; lens plain {[len(x[0]) for x in o]} "
                      f"spec {[len(x[1]) for x in o]} tau {[round(sum(x[2]) / max(len(x[2]), 1), 2) for x in o]}", flush=True)
            del eng; torch.cuda.empty_cache()
        g, s, a1, b1 = outs["greedy"], outs["sample_T0"], outs["sample_T1_a"], outs["sample_T1_b"]
        r_ = {"n": NP, "plain_identical": sum(x[0] == y[0] for x, y in zip(g, s)),
              "spec_identical": sum(x[1] == y[1] for x, y in zip(g, s)), "taus_identical": sum(x[2] == y[2] for x, y in zip(g, s)),
              "tokens_spec": sum(len(x[1]) for x in g), "greedy_tau": sum(t for x in g for t in x[2]) / sum(len(x[2]) for x in g),
              "greedy_spec_eq_greedy_plain": sum(x[1][:len(x[0])] == x[0][:len(x[1])] for x in g),
              "T1_same_seed_plain_identical": sum(x[0] == y[0] for x, y in zip(a1, b1)),
              "T1_same_seed_spec_identical": sum(x[1] == y[1] for x, y in zip(a1, b1)),
              "T1_same_seed_taus_identical": sum(x[2] == y[2] for x, y in zip(a1, b1)),
              "T1_spec_differs_from_greedy": sum(x[1] != y[1] for x, y in zip(a1, g)),
              "T1_tau": sum(t for x in a1 for t in x[2]) / sum(len(x[2]) for x in a1)}
        stored = METHODS[name][4]
        f = (os.path.join(DR, "engine_n1_gsm8k_outputs.jsonl") if stored == "LAL"
             else os.path.join(RUNS, "ouro-baselines", f"{stored}_outputs.jsonl") if stored else None)
        if f and os.path.exists(f):
            ref = [json.loads(l) for l in open(f)][:NP]
            pre = lambda a, b: a[:len(b)] == b[:len(a)]
            r_.update({"stored": os.path.relpath(f, HERE),
                       "T0_spec_identical_stored": sum(x[1] == y["spec"] for x, y in zip(s, ref)),
                       "T0_spec_prefix_eq_stored": sum(pre(x[1], y["spec"]) for x, y in zip(s, ref)),
                       "T0_plain_identical_stored": sum(x[0] == y["plain"] for x, y in zip(s, ref)),
                       "T0_plain_prefix_eq_stored": sum(pre(x[0], y["plain"]) for x, y in zip(s, ref)),
                       "T0_taus_identical_stored": sum(x[2] == y["tau"] for x, y in zip(s, ref))})
        res[name] = r_
        print(f"GREEDY CHECK {name}: {r_}", flush=True)
    ok = all(v["plain_identical"] == v["n"] and v["spec_identical"] == v["n"] and v["taus_identical"] == v["n"]
             and v["T1_same_seed_plain_identical"] == v["n"] and v["T1_same_seed_spec_identical"] == v["n"] for v in res.values())
    ok_stored = all(v.get("T0_spec_prefix_eq_stored", v["n"]) == v["n"] for v in res.values())
    res["summary"] = {"all_identical_in_process": ok, "all_prefix_eq_stored": ok_stored}
    print("GREEDY SUMMARY", res["summary"], flush=True)
    return res


# ---------------------------------------------------------------------------------------------------------------------------
def part_dist():
    from test_engine_sample import chi2                                 # chi-square helper of the LAL sampling tests
    from floodgates.engines.engine_baselines import pld_draft
    from floodgates.engines.engine_baselines_sample import OuroBaselineSampleEngine
    NS = int(os.environ.get("N_SAMPLES", 3000)); NCTX = int(os.environ.get("N_CTX", 2))
    _, _, tok = load()
    row = [json.loads(l) for l in open(TRACE)][0]
    prompt, ref = row["input"], row["output_ids"]; P = len(prompt)

    class RolledQ(OuroBaselineSampleEngine):                            # NEGATIVE CONTROL: proposal rows misaligned by one
        def _verify_q(self):
            return self._q.roll(1, 0)

    todo = [m for m in os.environ.get("DIST_METHODS", "wave24,wave8,pld312,das18,jacobi,hybrid").split(",") if m]
    if os.environ.get("NEG", "1") == "1":
        todo.append("NEG_wave8_rolledq")
    res = {}
    ckpt = os.path.join(OUT, f"test_dist{os.environ.get('EVAL_TAG', '')}.partial.jsonl")   # per-method checkpoint (preemptions)
    done = set()
    if os.path.exists(ckpt):
        for l in open(ckpt):
            try:
                x = json.loads(l)
            except Exception:                                       # a line cut by a preemption
                break
            if "__done__" in x:
                done.add(x["__done__"])
            else:
                res[x["key"]] = x["r"]
        res = {k: v for k, v in res.items() if v["method"] in done}
        print(f"resuming: methods done {sorted(done)}", flush=True)
    for mi, name in enumerate(todo):
        if name in done:
            continue
        base = name.replace("NEG_", "").replace("_rolledq", "")
        eng = make_engine(base, True, temp=1.0, keep_p=True, cls=RolledQ if name.startswith("NEG_") else None)
        k1 = eng.k1
        eng.warmup(P + len(ref) + 2 * k1 + 8)

        def force(t_end):
            """prefill the prompt and write the reference tokens ref[:t_end] at rows P .. P+t_end-1 (plain steps)."""
            torch.cuda.manual_seed(7); eng.prefill(prompt); ps = []
            for t in range(t_end):
                _, p = eng.plain_from(P + t, ref[t]); ps.append(p)
            return ps
        ctxs = []
        ps = force(len(ref) - 1)
        for t in range(5, len(ref) - 1):                                 # p of the token after ref[t] (position P+t+1)
            p = ps[t]
            if int((p > 0).sum()) >= 3 and float(p.max()) <= 0.75:
                ctxs.append(t)
            if len(ctxs) == NCTX:
                break
        assert ctxs, "no suitable context"
        for ci, t in enumerate(ctxs):
            cut, y = P + t, ref[t]
            p_plain = force(t + 1)[t]                                     # rows < cut = context; plain step of y at cut
            p1 = {int(i): float(p_plain[i]) for i in torch.nonzero(p_plain > 0)[:, 0]}
            variants = [("natural", None)]
            if getattr(eng, "das", None) is not None and eng.das["mode"] == "pld":
                g = eng.das["gamma"]; nat = pld_draft(prompt + ref[:t + 1], eng.das["n"], g)
                srt = sorted(p1, key=lambda x: -p1[x])
                variants = ([("natural_lookup", nat)] if nat else []) + [("inject_top1", [srt[0]] * g), ("inject_top3", [srt[2]] + [srt[0]] * (g - 1))]
            for vname, drafts in variants:
                torch.cuda.manual_seed(100 + 1000 * mi + 10 * ci)
                eng.cycle_from(cut, y, drafts) if getattr(eng, "das", None) is not None else eng.cycle_from(cut, y)
                p_ver = eng.pbuf[0].clone()
                c1 = {}; n_acc1 = 0; maxdiff = 0.0; t0 = time.time()
                torch.cuda.manual_seed(200 + 1000 * mi + 10 * ci + len(vname))
                for s in range(NS):
                    new = eng.cycle_from(cut, y, drafts) if getattr(eng, "das", None) is not None else eng.cycle_from(cut, y)
                    c1[new[0]] = c1.get(new[0], 0) + 1; n_acc1 += int(len(new) >= 2)
                    if s < 20:
                        maxdiff = max(maxdiff, float((eng.pbuf[0] - p_ver).abs().max()))
                pv = {int(i): float(p_ver[i]) for i in torch.nonzero(p_ver > 0)[:, 0]}
                r = {"method": name, "variant": vname, "context_tokens": cut + 1, "y": tok.decode([y]), "drafts_d1": tok.decode([drafts[0]]) if drafts else None,
                     "nucleus_p_ver": {tok.decode([k]): round(v_, 4) for k, v_ in sorted(pv.items(), key=lambda kv: -kv[1])},
                     "tv_plain_ver": 0.5 * float((p_plain - p_ver).abs().sum()), "p_ver_row0_max_abs_change": maxdiff,
                     "d1_accept_rate": n_acc1 / NS, "t1_vs_p_ver": chi2(c1, pv), "t1_vs_p_plain": chi2(c1, p1),
                     "seconds": time.time() - t0}
                key = f"{name} ctx{ci} {vname}"
                res[key] = r
                print(f"DIST {key}: {json.dumps(r)}", flush=True)
        with open(ckpt, "a") as fck:
            for k_, r_ in res.items():
                if r_["method"] == name:
                    fck.write(json.dumps({"key": k_, "r": r_}) + "\n")
            fck.write(json.dumps({"__done__": name}) + "\n")
        del eng; torch.cuda.empty_cache()
    pos = {k: v for k, v in res.items() if not k.startswith("NEG_")}
    neg = {k: v for k, v in res.items() if k.startswith("NEG_")}
    pvals = [v["t1_vs_p_ver"]["p_value"] for v in pos.values()]
    res["summary"] = {"n_tests": len(pvals), "min_p_value_vs_p_ver": min(pvals) if pvals else None,
                      "n_p_below_0.01": sum(x < 0.01 for x in pvals),
                      "out_of_support_vs_p_ver": sum(v["t1_vs_p_ver"]["out_of_support"] for v in pos.values()),
                      "max_tv_plain_ver": max(v["tv_plain_ver"] for v in pos.values()) if pos else None,
                      "max_p_ver_row0_change": max(v["p_ver_row0_max_abs_change"] for v in pos.values()) if pos else None,
                      "negative_control_p_values": [v["t1_vs_p_ver"]["p_value"] for v in neg.values()]}
    print("DIST SUMMARY", res["summary"], flush=True)
    return res


# ---------------------------------------------------------------------------------------------------------------------------
def part_repro():
    A, B = os.environ["REPRO_A"], os.environ["REPRO_B"]
    labels = [x for x in os.environ.get("REPRO_METHODS", "").split(",") if x] or [""]
    res = {}
    for m in labels:
        fa, fb = A.replace("{m}", m), B.replace("{m}", m)
        ra = {x["gidx"]: x for x in map(json.loads, open(fa))}; rb = {x["gidx"]: x for x in map(json.loads, open(fb))}
        common = sorted(set(ra) & set(rb))
        r = {"a": os.path.basename(fa), "b": os.path.basename(fb), "n_common": len(common),
             "plain_identical": sum(ra[i]["plain"] == rb[i]["plain"] for i in common),
             "spec_identical": sum(ra[i]["spec"] == rb[i]["spec"] for i in common),
             "taus_identical": sum(ra[i]["tau"] == rb[i]["tau"] for i in common),
             "nodes": sorted({ra[i].get("node") for i in common} | {rb[i].get("node") for i in common})}
        res[m or "file"] = r
        print(f"REPRO {m}: {r}", flush=True)
    res["summary"] = {"all_identical": all(v["plain_identical"] == v["spec_identical"] == v["taus_identical"] == v["n_common"] > 0
                                           for v in res.values())}
    print("REPRO SUMMARY", res["summary"], flush=True)
    return res


if __name__ == "__main__":
    print(f"PART {PART} torch {torch.__version__} gpu {torch.cuda.get_device_name() if torch.cuda.is_available() else None}", flush=True)
    r = {"greedy": part_greedy, "dist": part_dist, "repro": part_repro}[PART]()
    json.dump(r, open(os.path.join(OUT, f"test_{PART}{os.environ.get('EVAL_TAG', '')}.json"), "w"), indent=1, default=str)
    print("ALL DONE", flush=True)
