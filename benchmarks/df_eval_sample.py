"""[TASK=gsm8k|math500|humaneval|mbpp|bbh STRIDE=1] SAMPLING version of df_eval2.py: our reimplementation of the diffusion-forcing /
parallel sampler of Geiping et al. (training-free, not lossless) vs plain decoding in the same eager code (raven_min_sample.py:
RavenMin.generate / df_generate with every argmax replaced by a draw), BOTH drawing y ~ top-p(softmax(logits / T)) with the
tie-symmetric nucleus of sampling_ops / raven_engine_sample (T = 1, top-p 0.7 as in our other sampling runs).  How the sampler
samples (fresh draws for every wavefront position at every pass, Geiping et al. Alg. 1 l.12 / Alg. 2 l.12-17) and its unchanged
latent freeze rule: raven_min_sample.py.  Only the ratio is comparable with the CUDA-graph engine.
  OUT=<json> python df_eval_sample.py <model_dir> <trace.jsonl> <N> [inner=4] [eps=0.03] [eta=0.1]
Args, stops, MAX_NEW and the output keys (ar / df: tok_s, acc, tokens, sec; df: passes_per_token, iters_per_token, inner, eps,
eta; ratio) are those of df_eval2.py; added: "sampling" (setup), "per_problem" (seed, lengths, seconds, outputs), "df_stats".
Env: TEMP (1.0; 0 = greedy = df_eval2.py), TOP_P (0.7), TOPP_CAND (4096), SEED (1000): problem g (index in trace[::STRIDE][:N])
runs torch.manual_seed(SEED + g); torch.cuda.manual_seed(SEED + g) before the plain loop and again before the sampler;
DELTA (iter = df_eval2.py's freeze statistic | step = the paper's, raven_min_sample.py); SHARD / NSHARD (0 / 1): this run does
problems g = SHARD, SHARD + NSHARD, ... with the same seeds as an unsharded run, so that
  python df_eval_sample.py merge <out.json> <shard_0.json> ... <shard_{n-1}.json>
writes exactly the file of the unsharded run (ratio = (sum df tokens / sum df sec) / (sum ar tokens / sum ar sec)).
Cost model (batch-1 memory-bound layer reads): AR = pre + core*R + post per token; a pass = pre + core*inner + post."""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import json, os, re, sys, time
import os
from floodgates.paths import LOOPSPEC


def summarize(per, n_rows, setup, stats):
    res = {}
    for name in ("ar", "df"):
        ntok = sum(p[f"{name}_len"] for p in per); secs = sum(p[f"{name}_sec"] for p in per)
        res[name] = {"tok_s": ntok / secs, "acc": sum(p[f"{name}_correct"] for p in per) / n_rows, "tokens": ntok, "sec": secs}
    npass = sum(p["df_passes"] for p in per); nit = sum(p["df_iters"] for p in per); ntok = res["df"]["tokens"]
    res["df"].update({"passes_per_token": npass / ntok, "iters_per_token": nit / ntok, "inner": setup["inner"], "eps": setup["eps"],
                      "eta": setup["eta"]})
    res["ratio"] = res["df"]["tok_s"] / res["ar"]["tok_s"]
    res["sampling"] = setup
    res["df_stats"] = dict(stats, forced_frac=stats["forced"] / max(stats["committed"], 1),
                           mismatch_frac=stats["mismatch"] / max(stats["committed"], 1))
    res["per_problem"] = per
    return res


if len(sys.argv) > 1 and sys.argv[1] == "merge":                     # merge shards -> the unsharded run's file
    out, parts = sys.argv[2], [json.load(open(f)) for f in sys.argv[3:]]
    per = sorted((p for d in parts for p in d["per_problem"]), key=lambda p: p["g"])
    n = parts[0]["sampling"]["n_problems"]
    assert [p["g"] for p in per] == list(range(n)), f"incomplete or overlapping shards: {len(per)} of {n} problems"
    setup = dict(parts[0]["sampling"], shard=0, nshard=1)
    stats = {k: sum(d["df_stats"][k] for d in parts) for k in ("committed", "forced", "mismatch", "uncovered")}
    res = summarize(per, n, setup, stats)
    json.dump(res, open(out, "w"), indent=1)
    print(f"{out}: ratio {res['ratio']:.3f} over {n} problems (ar {res['ar']['tok_s']:.2f} tok/s, df {res['df']['tok_s']:.2f} tok/s)")
    sys.exit(0)

import glob
import torch
from floodgates.models.raven_min_sample import RavenMinSample
from transformers import PreTrainedTokenizerFast
M, TR, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
inner = int(sys.argv[4]) if len(sys.argv) > 4 else 4; eps = float(sys.argv[5]) if len(sys.argv) > 5 else 0.03
eta = float(sys.argv[6]) if len(sys.argv) > 6 else 0.1
TEMP = float(os.environ.get("TEMP", 1.0)); TOP_P = float(os.environ.get("TOP_P", 0.7)); KC = int(os.environ.get("TOPP_CAND", 4096))
SEED = int(os.environ.get("SEED", 1000)); DELTA = os.environ.get("DELTA", "iter")
SHARD, NSHARD = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(M, "tokenizer.json"))
m = RavenMinSample(M, temp=TEMP, top_p=TOP_P, k_cand=KC, delta=DELTA); c = m.c
pre, core, post = c["n_layers_in_prelude"], c["n_layers_in_recurrent_block"], c["n_layers_in_coda"]
gold = {}                                                                   # offline gold answers (question text -> answer)
for f in glob.glob(LOOPSPEC + '/runs/Ouro-1.4B-gsm8k-n100/baseline/samples_*.jsonl'):
    for l in open(f):
        d = json.loads(l)["doc"]; gold[d["question"].strip()] = d["answer"].split("####")[-1].strip().replace(",", "")
num = lambda s: (re.findall(r"answer is\s*\$?(-?[\d,]*\.?\d+)", s) or [None])[0]
TASK = os.environ.get("TASK", "gsm8k"); STRIDE = int(os.environ.get("STRIDE", 1))
CODE_STOPS = ["<|endoftext|>", "<|endofmask|>", "</s>", "\nif __name__", "\ndef main(", "\nprint(", "\n# Test", "\nassert ", "\n```"]
STOPS, MAXN = {"gsm8k": (["Q:", "</s>"], 512), "math500": (["Problem:"], 2048), "humaneval": (CODE_STOPS, 1024), "mbpp": (CODE_STOPS, 1024), "bbh": (["</s>", "Q", "\n\n"], 1024)}[TASK]
MAXN = int(os.environ.get("MAX_NEW", MAXN))
_tc = json.load(open(os.path.join(M, "tokenizer_config.json"))); _e = _tc.get("eos_token"); _e = _e["content"] if isinstance(_e, dict) else _e
eos = tok.convert_tokens_to_ids(_e) if _e else c.get("eos_token_id")
LB = max(12, max(len(tok(x, add_special_tokens=False)["input_ids"]) for x in STOPS) + 2)
stop = lambda out: any(x in tok.decode(out[-LB:]) for x in STOPS) or out[-1] == eos
rows_all = [json.loads(l) for l in open(TR)][::STRIDE][:N]
idx = list(range(len(rows_all)))[SHARD::NSHARD]                            # g = index in trace[::STRIDE][:N] (seed SEED + g)
print(f"task {TASK}: {len(idx)} of {len(rows_all)} prompts (stride {STRIDE}, shard {SHARD}/{NSHARD}), max_new {MAXN}, eos {_e} {eos}, "
      f"T {TEMP} top-p {TOP_P} (nucleus_tie K {KC}), seeds {SEED}+g, delta {DELTA}", flush=True)
def correct(o, q):
    a = num(tok.decode(o)); g = gold.get(q)
    return int(a is not None and g is not None and a.replace(",", "").rstrip(".") == g)
def seed_all(s):
    torch.manual_seed(s); torch.cuda.manual_seed(s)
t0 = time.time(); per = {g: {"g": g, "i": rows_all[g].get("i", g), "seed": SEED + g} for g in idx}
for name in ("ar", "df"):
    acc = ntok = 0; secs = 0.0
    for n_done, g in enumerate(idx, 1):
        r = rows_all[g]; p = per[g]
        q = re.findall(r"Q: (.*?)\nA:", tok.decode(r["input"]), re.S)[-1].strip() if TASK == "gsm8k" else None
        seed_all(SEED + g)
        torch.cuda.synchronize(); t = time.time()
        if name == "df":
            o, passes, its = m.df_generate(r["input"], MAXN, stop, inner=inner, eps=eps, eta=eta); p.update(df_passes=passes, df_iters=its)
        else:
            o = m.generate(r["input"], MAXN, stop); o = o[0] if isinstance(o, tuple) else o
        torch.cuda.synchronize(); dt = time.time() - t
        ok = correct(o, q) if TASK == "gsm8k" else 0
        p.update({f"{name}_len": len(o), f"{name}_sec": dt, f"{name}_correct": ok, f"{name}_out": o})
        secs += dt; acc += ok; ntok += len(o)
        if n_done % 20 == 0: print(f"{name}: {n_done}/{len(idx)} {ntok / secs:.1f} tok/s acc {acc / n_done:.2f}", flush=True)
setup = dict(temp=TEMP, top_p=TOP_P, nucleus="tie-symmetric (sampling_ops.probs_cand)", topp_cand=KC, seed=SEED,
             seed_rule="torch.manual_seed + torch.cuda.manual_seed(SEED + g) before the plain loop and before the sampler",
             delta=DELTA, inner=inner, eps=eps, eta=eta, max_new=MAXN, task=TASK, stride=STRIDE, n_problems=len(rows_all),
             shard=SHARD, nshard=NSHARD, model=M, trace=TR, gpu=torch.cuda.get_device_name(0))
stats = dict(m.stats, uncovered=m.uncovered)
res = summarize([per[g] for g in idx], len(idx), setup, stats)
print(json.dumps({k: v for k, v in res.items() if k not in ("per_problem", "sampling")}), f"({time.time() - t0:.0f}s)", flush=True)
json.dump(res, open(os.environ.get("OUT", "df_eval_sample_out.json"), "w"), indent=1)
