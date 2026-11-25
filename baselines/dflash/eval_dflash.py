"""Offline tokens per verification (tau, incl. the bonus token) of a DFlash drafter on Ouro-1.4B's own greedy traces,
exactly the eval_accept.py protocol (same 100 problems, teacher-forced context = the reference, accept the longest
matching prefix, +1).  The drafts of every anchor are computed once; tau is then simulated for each draft length K
(K = 15 is DFlash's block 16; K = 8 / 12 match our k).  Loop-equivalent per 4-loop step: tau * 4 / (4 + 5/24).
  python baselines/dflash/eval_dflash.py <run_dir> [gsm8k math500]   (DRAFTER_PT=drafter_step4000.pt for a checkpoint)"""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # repository root
import json, os, sys, time
import torch
from floodgates.paths import TRACES, model_dir
from floodgates.models.ouro_min import OuroMin
from dflash_ouro import DFlashOuro, draft_all, simulate

run = sys.argv[1]
tasks = sys.argv[2:] or ["gsm8k", "math500"]
TR = os.path.join(TRACES, 'Ouro-1.4B-{}.jsonl')
a = json.load(open(os.path.join(run, "args.json")))
ouro = OuroMin(model_dir(a["model"]))
dr = DFlashOuro(ouro, layers=a["layers"], n_layers=a["n_layers"], block=a["block"], mask_id=a["mask_id"], gamma=a["gamma"])
pt = os.environ.get("DRAFTER_PT", "drafter.pt")
missing, unexpected = dr.load_state_dict(torch.load(os.path.join(run, pt)), strict=False)
assert not unexpected and all(m.startswith("ouro.") for m in missing), (unexpected, missing)
dr.eval()
n_tr = sum(p.numel() for _, p in dr.trainable())
for task in tasks:
    rows = [json.loads(l) for l in open(TR.format(task))][: int(os.environ.get("N_PROBLEMS", 100))]
    t0 = time.time()
    drafts = [draft_all(dr, r["input"], r["output_ids"]) for r in rows]
    res = {"trainable_params": n_tr, "drafter_pt": pt, "n_problems": len(rows)}
    for K in (8, 12, a["block"] - 1):
        taus, pos = [], [[0, 0] for _ in range(K)]
        for r, d in zip(rows, drafts):
            t, p = simulate(d, r["output_ids"], K)
            taus += t
            for j in range(K):
                pos[j][0] += p[j][0]; pos[j][1] += p[j][1]
        tau = sum(taus) / len(taus)
        cost = a["n_layers"] / ouro.L
        res[f"K{K}"] = {"tau": tau, "gamma_loop_equiv": tau * 4 / (4 + cost), "cycles": len(taus),
                        "pos_acc": [round(c / t, 3) if t else None for c, t in pos]}
        print(f"{task} K={K}: tau={tau:.3f} loop-equiv={tau * 4 / (4 + cost):.3f} cycles={len(taus)} "
              f"per-slot acc={res[f'K{K}']['pos_acc']}", flush=True)
    print(f"{task}: {time.time() - t0:.0f}s", flush=True)
    tag = "" if pt == "drafter.pt" else "_" + pt.replace("drafter_", "").replace(".pt", "")
    json.dump(res, open(os.path.join(run, f"accept_{task}{tag}.json"), "w"), indent=1)
