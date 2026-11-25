"""Wall-clock speculative decoding with LAL's drafter on a Raven looped model (Recurrent-Llama-3.2) in the SAME eager setting as
dense_spec_hf.py (plain PyTorch, batch size 1, greedy, same prompts, stops and length limits, time per problem incl. prefill), so that
the loop and the first layers of the dense base model are compared in one implementation.  Plain decoding = RavenMin.generate.
One step: the drafter's pass (prelude -> adapter -> one core iteration -> coda with LoRA on the slot rows) over [y, mask x k] at
positions c .. c+k, reading the context's prelude / last-core-iteration / coda K/V from the plain decoding cache; the full model
(R core iterations) verifies [y, d_0..d_{k-1}] with its cache, keeps the leading drafts that equal its greedy tokens plus its own next
token, and the cache is cropped to the kept positions.  Checks that the outputs equal plain decoding.
  python baselines/dense/raven_spec_hf.py <run dir> <prompts jsonl> <gsm8k|math500> <out.json>"""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # repository root
import json, os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from floodgates.paths import model_dir
from floodgates.models.raven_min import RavenMin
from floodgates.drafters.drafter_raven import RavenSelfDrafter
from transformers import PreTrainedTokenizerFast
RUN, PR, TASK, OUT = sys.argv[1:5]
STOPS = {"gsm8k": ["Q:"], "math500": ["Problem:"]}[TASK]; MAXNEW = {"gsm8k": 512, "math500": 1024}[TASK]
a = json.load(open(f"{RUN}/args.json")); a["model"] = model_dir(a["model"]); m = RavenMin(a["model"])
dr = RavenSelfDrafter(m, k=a["k"], rank=a["rank"], shift=a.get("shift", False), feedback=a.get("feedback", False), iters=a.get("iters", 1),
                      carry=a.get("carry", False), causal=a.get("causal", False))
dr.load_state_dict(torch.load(f"{RUN}/drafter.pt"), strict=False); dr.eval()
assert dr.shift, "next-token-aligned slots expected"
k = dr.k; k1 = k + 1; R = m.R; dev = m.device
tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(a["model"], "tokenizer.json"))   # the retrofit dirs have no loadable config
rows = [json.loads(l) for l in open(PR)][:int(os.environ.get("N_PROBLEMS", 100))]


def stopped(ids): return any(s in tok.decode(ids[-12:]) for s in STOPS)


def cut(ids):
    for n in range(1, len(ids) + 1):
        if any(s in tok.decode(ids[max(0, n - 8):n]) for s in STOPS): return ids[:n]
    return ids


@torch.no_grad()
def plain(ids):
    return m.generate(ids, MAXNEW, stop=stopped)


@torch.no_grad()
def spec(ids):
    cache = {}

    def attn_for(key):
        def attn(q, kk, v):
            if key in cache:
                pk, pv = cache[key]; kk, v = torch.cat([pk, kk], 2), torch.cat([pv, v], 2)
            cache[key] = (kk, v)
            T, S = q.shape[2], kk.shape[2]
            if T == S: return F.scaled_dot_product_attention(q, kk, v, is_causal=True, enable_gqa=True)
            return F.scaled_dot_product_attention(q, kk, v, attn_mask=torch.ones(T, S, dtype=torch.bool, device=q.device).tril(S - T), enable_gqa=True)
        return attn

    def run_block(x_ids, start):
        """The full model over x_ids (1, T) at positions start.., as RavenMin.generate: greedy tokens of every position."""
        T = x_ids.shape[1]; cos, sin = m.rope(torch.arange(start, start + T, device=dev)[None])
        e = m.wte(x_ids) * m.emb_scale
        for li, b in enumerate(m.prelude): e = b(e, cos, sin, attn_for(("p", li)))
        s = m.init_state_at(range(start, start + T), 0)
        for it in range(R):
            s = m.adapter(torch.cat([s, e], -1))
            for li, b in enumerate(m.core): s = b(s, cos, sin, attn_for(("c", it, li)))
        s = m.pre(s)
        for li, b in enumerate(m.coda): s = b(s, cos, sin, attn_for(("o", li)))
        return m.lm_head(m.ln_f(s[0])).argmax(-1)

    def crop(n):
        for key in cache: cache[key] = (cache[key][0][:, :, :n], cache[key][1][:, :, :n])

    P = len(ids); y = int(run_block(torch.tensor([ids], device=dev), 0)[-1]); gen = [y]; c = P; steps = 0
    while len(gen) < MAXNEW and not stopped(gen):
        ctx = ([cache[("p", li)] for li in range(len(m.prelude))] + [cache[("c", R - 1, li)] for li in range(len(m.core))]
               + [cache[("o", li)] for li in range(len(m.coda))])                 # draft-pass order, positions < c
        d = dr.draft(ctx, c, y)                                                   # k+1 drafts: d_j for position c+j+1
        blk = torch.tensor([[y] + d[:k]], device=dev)
        t = run_block(blk, c)                                                     # greedy token after each of the k+1 inputs
        a_ = 0
        while a_ < k1 and d[a_] == int(t[a_]): a_ += 1
        new = d[:a_] + ([int(t[a_])] if a_ < k1 else [])
        crop(c + len(new) if a_ < k1 else c + k1)
        gen += new; c += len(new); y = gen[-1]; steps += 1
    return gen[:MAXNEW], steps


res = {"run": RUN, "model": a["model"], "R": R, "task": TASK, "n": len(rows), "gpu": torch.cuda.get_device_name()}
for name, fn in (("plain", plain), ("lal", spec)):
    outs, secs, cyc = [], [], 0
    for i, r in enumerate([rows[0], rows[1]] + rows):                            # two warm-up problems, not timed
        ids = r["input"] if "input" in r else tok(r["context"])["input_ids"]
        torch.cuda.synchronize(); t0 = time.perf_counter(); o = fn(ids); torch.cuda.synchronize(); dt = time.perf_counter() - t0
        if name == "lal": o, s = o
        else: s = 0
        if i >= 2: outs.append(cut(o)); secs.append(dt); cyc += s
    res[name] = {"tok_s": sum(map(len, outs)) / sum(secs), "tokens": sum(map(len, outs)), "seconds": sum(secs)}
    if name == "plain": ref = outs
    else: res[name].update(speedup=res[name]["tok_s"] / res["plain"]["tok_s"], tau=res[name]["tokens"] / max(cyc, 1), identical_to_plain=sum(x == z for x, z in zip(outs, ref)))
    print(json.dumps({name: res[name]}), flush=True)
json.dump(res, open(OUT, "w"), indent=1); print("wrote", OUT)
