"""Offline accepted length of greedy speculative decoding with the Raven self-drafter (drafter_raven.py).
Reference = LoopSpec's deterministic greedy trace of the same model (100 GSM8K test problems; rows "input", "output_ids").
Per problem: one plain teacher-forced context pass over prompt+output (32 iterations; K/V of every block, last core
iteration), then cycles: at emitted token o[m], draft k tokens with context < P+m; accept the longest prefix equal to the
reference; the verifier then emits one more reference token -> tokens per verification tau = accepted + 1.
Cost in layers: a draft pass = prelude + core * n_rounds + coda; AR per token = prelude + core * R + coda (= 200 at R=32)
-> ideal speedup = tau * AR / (AR + draft).  Also reports the context pass's teacher-forced agreement with the trace.
  python eval_accept_raven.py <run_dir> [n_rounds ...]      env: N_PROBLEMS (100), TRACE, EVAL_TAG (suffix of accept.json)"""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import json, os, sys
import torch
import os
from floodgates.paths import model_dir, trace
from floodgates.models.raven_min import RavenMin
from floodgates.drafters.drafter_raven import RavenSelfDrafter, context_kv
run = sys.argv[1]
TR = os.environ.get("TRACE", trace("Recurrent-Llama-3.2-train-recurrence-32", "gsm8k"))
TAG = os.environ.get("EVAL_TAG", "")   # e.g. "_math500" -> output files carry the benchmark
a = json.load(open(os.path.join(run, "args.json")))
rounds_list = [int(x) for x in sys.argv[2:]] or list(range(1, a.get("rounds_max", 1) + 1))
PLAIN = [int(x) for x in os.environ.get("PLAIN", "0").split(",")]      # rounds of the model's own recurrence after the adapter pass(es)
PLAIN_CARRY = os.environ.get("PLAIN_CARRY", "1") == "1"
PLAIN_ITERS = int(os.environ.get("PLAIN_ITERS", "1"))                        # core iterations per plain round
K_WIN = int(os.environ.get("K_WIN", "0")) or None                            # window width for the plain rounds (>= k)
m = RavenMin(model_dir(a["model"]))
dr = RavenSelfDrafter(m, k=a["k"], rank=a["rank"], shift=a.get("shift", False), feedback=a.get("feedback", False), iters=a.get("iters", 1), carry=a.get("carry", False), causal=a.get("causal", False))
K = (K_WIN or a["k"]) + (1 if a.get("shift") else 0)
sd = torch.load(os.path.join(run, "drafter.pt"))
missing, unexpected = dr.load_state_dict(sd, strict=False)
assert not unexpected and all(n.startswith("m.") or n == "draft_emb" for n in missing), (unexpected, [n for n in missing if not n.startswith("m.")])
dr.eval()
c = m.c
ar_layers = c["n_layers_in_prelude"] + c["n_layers_in_recurrent_block"] * m.R + c["n_layers_in_coda"]
rows = [json.loads(l) for l in open(TR)][: int(os.environ.get("N_PROBLEMS", 100))]
res = {}
import itertools
BRANCH = [x for x in os.environ.get("BRANCH", "").split(",") if x]   # J:split -> two-branch window, tree verification with recomputed continuations
if BRANCH:
    from floodgates.drafters.drafter_raven import draft_branch
    def common(x, y):
        n = 0
        while n < min(len(x), len(y)) and x[n] == y[n]:
            n += 1
        return n
    rl = c["n_layers_in_prelude"] + c["n_layers_in_recurrent_block"] * PLAIN_ITERS + c["n_layers_in_coda"]
    for spec in BRANCH:
        J, split = [int(v) for v in spec.split(":")]
        key = f"B{J}s{split}x{PLAIN_ITERS}"
        taus, taus1, taus2, where, wins = [], [], [], [], 0
        cost = (c["n_layers_in_prelude"] + c["n_layers_in_recurrent_block"] * a.get("iters", 1) + c["n_layers_in_coda"]) + J * rl
        for r in rows:
            p, o = r["input"], r["output_ids"]
            ids = torch.tensor([p + o], device="cuda")
            _, kv = context_kv(m, ids, logits=False)
            mm = 0
            while mm < len(o) - 1:
                d1, d2, jd = draft_branch(dr, kv, len(p) + mm, o[mm], J, split, iters=PLAIN_ITERS)
                ref = o[mm + 1: mm + 1 + K]
                a1, a2 = common(d1, ref), common(d2, ref)
                taus1.append(a1 + 1); taus2.append(a2 + 1); taus.append(max(a1, a2) + 1); where.append(jd); wins += int(a2 > a1)
                mm += max(a1, a2) + 1
        tau = sum(taus) / len(taus)
        res[key] = {"tau": tau, "tau_branch1": sum(taus1) / len(taus1), "tau_branch2": sum(taus2) / len(taus2), "branch2_wins": wins / len(taus),
                    "mean_branch_pos": sum(where) / len(where), "draft_layers": cost, "ar_layers": ar_layers,
                    "ideal_speedup": tau * ar_layers / (ar_layers + cost), "cycles": len(taus), "n_problems": len(rows)}
        print(f"rounds={key}: tokens per verification tau={tau:.3f} (branch 1 alone {res[key]['tau_branch1']:.3f}, branch 2 alone "
              f"{res[key]['tau_branch2']:.3f}, branch 2 wins {100 * res[key]['branch2_wins']:.1f}%, mean branch position "
              f"{res[key]['mean_branch_pos']:.1f}), draft={cost} layers vs AR {ar_layers} -> ideal speedup {res[key]['ideal_speedup']:.3f}, "
              f"cycles={len(taus)}", flush=True)
    json.dump(res, open(os.path.join(run, f"accept{TAG}_branch.json"), "w"), indent=1)
    sys.exit(0)
CHEAP = [x for x in os.environ.get("CHEAP", "").split(",") if x]   # J:topk:full_every -> restricted-vocabulary readouts in the rounds
if CHEAP:
    from floodgates.drafters.drafter_raven import draft_cheap
    for spec in CHEAP:
        J, topk, fe = [int(v) for v in spec.split(":")]
        key = f"C{J}k{topk}f{fe}"
        taus, pos_acc, nfulls = [], [[0, 0] for _ in range(K)], []
        for r in rows:
            p, o = r["input"], r["output_ids"]
            ids = torch.tensor([p + o], device="cuda")
            _, kv = context_kv(m, ids, logits=False)
            mm = 0
            while mm < len(o) - 1:
                d, nf = draft_cheap(dr, kv, len(p) + mm, o[mm], J, topk=topk, full_every=fe, iters=PLAIN_ITERS)
                nfulls.append(nf)
                ref = o[mm + 1: mm + 1 + K]
                acc = 0
                for j, (x, y) in enumerate(zip(d, ref)):
                    pos_acc[j][1] += 1; pos_acc[j][0] += int(x == y)
                while acc < len(ref) and d[acc] == ref[acc]:
                    acc += 1
                taus.append(acc + 1); mm += acc + 1
        tau = sum(taus) / len(taus)
        res[key] = {"tau": tau, "J": J, "topk": topk, "full_every": fe, "full_readouts": sum(nfulls) / len(nfulls),
                    "cycles": len(taus), "pos_acc": [round(cnt / t, 3) if t else None for cnt, t in pos_acc], "n_problems": len(rows)}
        print(f"rounds={key}: tokens per verification tau={tau:.3f}, full readouts per cycle={res[key]['full_readouts']:.2f} of {J}, cycles={len(taus)}, per-slot acc={res[key]['pos_acc']}", flush=True)
    json.dump(res, open(os.path.join(run, f"accept{TAG}_cheap.json"), "w"), indent=1)
    sys.exit(0)
SCHED = [x for x in os.environ.get("SCHED", "").split(",") if x]   # e.g. L8L8,L4L4L4L4: alternate adapter passes and plain rounds
if SCHED:
    import re
    from floodgates.drafters.drafter_raven import draft_schedule
    rl = c["n_layers_in_prelude"] + c["n_layers_in_recurrent_block"] * PLAIN_ITERS + c["n_layers_in_coda"]
    TREE = [int(x) for x in os.environ.get("TREE", "0").split(",")]   # branch to the 2nd choice at the m least confident positions
    def tree_accept(d, d2, marg, ref, m):
        """Longest accepted prefix over the 2^m paths that flip the m lowest-margin positions to their second choice."""
        import itertools as it
        if m == 0:
            acc = 0
            while acc < len(ref) and d[acc] == ref[acc]: acc += 1
            return acc
        flips = sorted(range(len(ref)), key=lambda j: marg[j])[:m]
        best = 0
        for bits in it.product([0, 1], repeat=len(flips)):
            cand = list(d)
            for b, j in zip(bits, flips):
                if b: cand[j] = d2[j]
            acc = 0
            while acc < len(ref) and cand[acc] == ref[acc]: acc += 1
            best = max(best, acc)
        return best
    for sch in SCHED:
      for mtree in TREE:
          key = f"S{sch}x{PLAIN_ITERS}" + (f"t{mtree}" if mtree else "")
          taus, pos_acc = [], [[0, 0] for _ in range(K)]
          cost = sum((c["n_layers_in_prelude"] + c["n_layers_in_recurrent_block"] * a.get("iters", 1) + c["n_layers_in_coda"]) if t == "L" else int(t) * rl
                     for t in re.findall(r"L|\d+", sch))
          for r in rows:
              p, o = r["input"], r["output_ids"]
              ids = torch.tensor([p + o], device="cuda")
              _, kv = context_kv(m, ids, logits=False)
              mm = 0
              while mm < len(o) - 1:
                  d, d2, marg = draft_schedule(dr, kv, len(p) + mm, o[mm], sch, plain_iters=PLAIN_ITERS, top2=True)
                  ref = o[mm + 1: mm + 1 + K]
                  for j, (x, y) in enumerate(zip(d, ref)):
                      pos_acc[j][1] += 1; pos_acc[j][0] += int(x == y)
                  acc = tree_accept(d, d2, marg, ref, mtree)
                  taus.append(acc + 1); mm += acc + 1
          tau = sum(taus) / len(taus)
          res[key] = {"tau": tau, "draft_layers": cost, "ar_layers": ar_layers, "ideal_speedup": tau * ar_layers / (ar_layers + cost),
                      "cycles": len(taus), "pos_acc": [round(cnt / t, 3) if t else None for cnt, t in pos_acc], "n_problems": len(rows)}
          print(f"rounds={key}: tokens per verification tau={tau:.3f}, draft={cost} layers vs AR {ar_layers} -> ideal speedup "
                f"{res[key]['ideal_speedup']:.3f}, cycles={len(taus)}, per-slot acc={res[key]['pos_acc']}", flush=True)
    json.dump(res, open(os.path.join(run, f"accept{TAG}_sched.json"), "w"), indent=1)
    sys.exit(0)
PERSIST = int(os.environ.get("PERSIST", "0"))        # > 0: persistent window, this many plain rounds per cycle after seeding
PERSIST_J0 = int(os.environ.get("PERSIST_J0", "16")) # rounds at seeding (first cycle of each problem)
if PERSIST:
    from floodgates.drafters.drafter_raven import PersistentWindow
    for n in rounds_list:
        key = f"{n}+p{PERSIST}j{PERSIST_J0}x{PLAIN_ITERS}w{K_WIN or a['k']}"
        taus, costs, pos_acc = [], [], [[0, 0] for _ in range(K)]
        rl = c["n_layers_in_prelude"] + c["n_layers_in_recurrent_block"] * PLAIN_ITERS + c["n_layers_in_coda"]   # layers per plain round
        dl = c["n_layers_in_prelude"] + c["n_layers_in_recurrent_block"] * n * a.get("iters", 1) + c["n_layers_in_coda"]
        for r in rows:
            p, o = r["input"], r["output_ids"]
            ids = torch.tensor([p + o], device="cuda")
            _, kv = context_kv(m, ids, logits=False)
            win = PersistentWindow(dr, kv, k_win=K_WIN, iters=PLAIN_ITERS)
            mm = 0; first = True
            while mm < len(o) - 1:
                if first:
                    win.seed(len(p) + mm, o[mm], PERSIST_J0, n_lora=n, carry=PLAIN_CARRY); costs.append(dl + PERSIST_J0 * rl); first = False
                else:
                    win.rounds(PERSIST); costs.append(PERSIST * rl)
                d = win.drafts(); ref = o[mm + 1: mm + 1 + K]
                acc = 0
                for j, (x, y) in enumerate(zip(d, ref)):
                    pos_acc[j][1] += 1; pos_acc[j][0] += int(x == y)
                while acc < len(ref) and d[acc] == ref[acc]:
                    acc += 1
                taus.append(acc + 1); mm += acc + 1
                if mm < len(o) - 1:
                    win.commit(acc, o[mm])
        tau = sum(taus) / len(taus); cost = sum(costs) / len(costs)
        res[key] = {"tau": tau, "draft_layers": cost, "ar_layers": ar_layers, "ideal_speedup": tau * ar_layers / (ar_layers + cost),
                    "cycles": len(taus), "pos_acc": [round(cnt / t, 3) if t else None for cnt, t in pos_acc], "n_problems": len(rows)}
        print(f"rounds={key}: tokens per verification tau={tau:.3f}, mean draft cost={cost:.1f} layers vs AR {ar_layers} -> ideal speedup "
              f"{res[key]['ideal_speedup']:.3f}, cycles={len(taus)}, per-slot acc={res[key]['pos_acc']}", flush=True)
    json.dump(res, open(os.path.join(run, f"accept{TAG}_persist.json"), "w"), indent=1)
    sys.exit(0)
for n, J in itertools.product(rounds_list, PLAIN):
    key = n if not J else f"{n}+{J}x{PLAIN_ITERS}w{K_WIN or a['k']}{'c' if PLAIN_CARRY else ''}"
    taus, pos_acc = [], [[0, 0] for _ in range(K)]
    tf_hit = tf_tot = 0
    for r in rows:
        p, o = r["input"], r["output_ids"]
        ids = torch.tensor([p + o], device="cuda")
        lg, kv = context_kv(m, ids, logits=True)
        pred = lg[0, len(p) - 1: len(p) + len(o) - 1].argmax(-1)          # teacher-forced predictions of the output tokens
        tf_hit += int((pred == torch.tensor(o, device="cuda")).sum()); tf_tot += len(o)
        del lg
        mm = 0
        while mm < len(o) - 1:
            d = dr.draft(kv, len(p) + mm, o[mm], n_rounds=n, plain_rounds=J, plain_carry=PLAIN_CARRY, plain_iters=PLAIN_ITERS, k_win=K_WIN)
            ref = o[mm + 1: mm + 1 + K]
            acc = 0
            for j, (x, y) in enumerate(zip(d, ref)):
                pos_acc[j][1] += 1; pos_acc[j][0] += int(x == y)
            while acc < len(ref) and d[acc] == ref[acc]:
                acc += 1
            taus.append(acc + 1); mm += acc + 1
    tau = sum(taus) / len(taus)
    draft_layers = c["n_layers_in_prelude"] + c["n_layers_in_recurrent_block"] * n * a.get("iters", 1) + c["n_layers_in_coda"] + J * (c["n_layers_in_prelude"] + c["n_layers_in_recurrent_block"] * PLAIN_ITERS + c["n_layers_in_coda"])
    res[key] = {"tau": tau, "draft_layers": draft_layers, "ar_layers": ar_layers,
              "ideal_speedup": tau * ar_layers / (ar_layers + draft_layers), "cycles": len(taus),
              "pos_acc": [round(cnt / t, 3) if t else None for cnt, t in pos_acc], "tf_agree": tf_hit / tf_tot,
              "n_problems": len(rows)}
    print(f"rounds={key}: tokens per verification tau={tau:.3f}, draft={draft_layers} layers vs AR {ar_layers} -> ideal speedup "
          f"{res[key]['ideal_speedup']:.3f}, cycles={len(taus)}, per-slot acc={res[key]['pos_acc']}, "
          f"teacher-forced agreement with trace={tf_hit / tf_tot:.4f}", flush=True)
json.dump(res, open(os.path.join(run, f"accept{TAG}{'_plain' if PLAIN != [0] else ''}.json"), "w"), indent=1)
