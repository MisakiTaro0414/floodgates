"""Wall-clock speculative decoding with LAL's drafter on the first E layers of a dense model (a run of floodgates/drafters/dense_drafter.py train),
in the same setting as LayerSkip's own self-speculative decoding (layerskip_selfspec.py): Hugging Face eager, batch size 1, greedy,
same prompts, stops and length limits, time per problem including prefill, against plain greedy generate of the same model.
One step: the slots [y, mask x k] at positions cut .. cut+k run through the first E layers with the drafter's LoRA, reading the
model's own cache of those layers for the context (positions < cut), and the output head drafts d_0..d_k (slot j drafts the token at
cut+j+1); the full model then verifies [y, d_0..d_{k-1}] with its cache, keeps the leading drafts that equal its greedy tokens plus
its own next token, and the cache is cropped to the kept positions.  Checks that the outputs equal plain decoding.
LayerSkip's own self-speculative decoding runs in the same loop (mode ls:E:d): the first E layers and the output head draft d tokens
one at a time, reading the model's cache of those layers plus the drafts' own keys and values, which are discarded before the full
model verifies [y, d_0..d_{d-1}] exactly as above (LayerSkip additionally reuses the drafts' first-E-layer keys and values in the
verification, which saves E/L of its cost).
  python baselines/dense/dense_spec_hf.py <run dir> <prompts jsonl> <gsm8k|math500> <out.json> [modes: lal,ls:E:d,...]"""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # repository root
import json, os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from transformers import DynamicCache, StopStringCriteria, StoppingCriteriaList
import torch.nn.functional as F
from floodgates.drafters.dense_drafter import DenseDrafter, load, layer_fwd
RUN, PR, TASK, OUT = sys.argv[1:5]
MODES = (sys.argv[5] if len(sys.argv) > 5 else "lal").split(",")
STOPS = {"gsm8k": ["Q:"], "math500": ["Problem:"]}[TASK]; MAXNEW = {"gsm8k": 512, "math500": 1024}[TASK]
ar = json.load(open(f"{RUN}/args.json")); m, tok = load(ar["model"])
dr = DenseDrafter(m, ar["E"], ar["k"], ar["rank"]); dr.load_state_dict(torch.load(f"{RUN}/drafter.pt"), strict=False); dr.eval()
k1 = dr.k + 1
rows = [json.loads(l) for l in open(PR)][:int(os.environ.get("N_PROBLEMS", 100))]


def stopped(ids): return any(s in tok.decode(ids[-12:]) for s in STOPS)


def cut(ids):
    for n in range(1, len(ids) + 1):
        if any(s in tok.decode(ids[max(0, n - 8):n]) for s in STOPS): return ids[:n]
    return ids


def layer_kv(cache, l):
    L = cache.layers[l] if hasattr(cache, "layers") else None
    return (L.keys, L.values) if L is not None else (cache.key_cache[l], cache.value_cache[l])


@torch.no_grad()
def plain(ids):
    sc = StoppingCriteriaList([StopStringCriteria(tok, STOPS)])
    return m.generate(ids, max_new_tokens=MAXNEW, do_sample=False, stopping_criteria=sc)[0, ids.shape[1]:].tolist()


@torch.no_grad()
def spec(ids):
    dev = ids.device; P = ids.shape[1]; cache = DynamicCache()
    out = m(input_ids=ids, past_key_values=cache, use_cache=True)
    y = int(out.logits[0, -1].argmax()); gen = [y]; c = P; steps = 0
    is_clean = torch.zeros(1, k1, dtype=torch.bool, device=dev); is_clean[0, 0] = True
    while len(gen) < MAXNEW and not stopped(gen):
        ctx = [layer_kv(cache, l) for l in range(dr.E)]                         # the model's own cache, positions < c
        sid = torch.zeros(1, k1, dtype=torch.long, device=dev); sid[0, 0] = y
        pos = torch.arange(c, c + k1, device=dev)[None]
        d = dr(sid, is_clean, pos, ctx, None)[0].argmax(-1)                     # d_j drafts the token at c+j+1
        blk = torch.cat([torch.tensor([y], device=dev), d[:-1]])[None]
        lg = m(input_ids=blk, past_key_values=cache, use_cache=True).logits[0]               # positions follow the cache length c
        t = lg.argmax(-1); match = (d == t).long(); a = int(match.cumprod(0).sum())
        if a < k1:
            new = d[:a].tolist() + [int(t[a])]; cache.crop(c + a + 1)
        else:
            new = d.tolist()                                                      # every draft kept; the last one is the next input
        gen += new; c += len(new); y = gen[-1]; steps += 1
    return gen[:MAXNEW], steps


@torch.no_grad()
def layerskip(ids, E, nd):
    dev = ids.device; P = ids.shape[1]; cache = DynamicCache()
    out = m(input_ids=ids, past_key_values=cache, use_cache=True)
    y = int(out.logits[0, -1].argmax()); gen = [y]; c = P; steps = 0
    layers = m.model.layers[:E]
    while len(gen) < MAXNEW and not stopped(gen):
        ctx = [layer_kv(cache, l) for l in range(E)]; dk = [[] for _ in range(E)]; dv = [[] for _ in range(E)]
        d, x_in = [], y
        for j in range(nd):                                                       # draft nd tokens, one at a time
            x = m.model.embed_tokens(torch.tensor([[x_in]], device=dev)); cos, sin = m.model.rotary_emb(x, torch.tensor([[c + j]], device=dev))
            for i, layer in enumerate(layers):
                def attn(q, k, v, i=i):
                    dk[i].append(k); dv[i].append(v)
                    K = torch.cat([ctx[i][0]] + dk[i], 2); V = torch.cat([ctx[i][1]] + dv[i], 2)
                    return F.scaled_dot_product_attention(q, K, V, enable_gqa=True)
                x = layer_fwd(layer, x, cos, sin, attn)
            x_in = int(m.lm_head(m.model.norm(x))[0, -1].argmax()); d.append(x_in)
        d = torch.tensor(d, device=dev)
        blk = torch.cat([torch.tensor([y], device=dev), d])[None]
        t = m(input_ids=blk, past_key_values=cache, use_cache=True).logits[0].argmax(-1)    # nd + 1 greedy tokens
        a = int((d == t[:nd]).long().cumprod(0).sum())
        new = d[:a].tolist() + [int(t[a])]; cache.crop(c + a + 1)
        gen += new; c += len(new); y = gen[-1]; steps += 1
    return gen[:MAXNEW], steps


res = {"run": RUN, "E": ar["E"], "task": TASK, "n": len(rows), "gpu": torch.cuda.get_device_name()}
METHODS = [("plain", plain)] + [(md, spec if md == "lal" else (lambda ids, E=int(md.split(":")[1]), nd=int(md.split(":")[2]): layerskip(ids, E, nd)))
                                for md in MODES]
for name, fn in METHODS:
    outs, secs, cyc = [], [], 0
    for i, r in enumerate([rows[0], rows[1]] + rows):                            # two warm-up problems, not timed
        ids = tok(r["context"], return_tensors="pt").input_ids.cuda()
        torch.cuda.synchronize(); t0 = time.perf_counter(); o = fn(ids); torch.cuda.synchronize(); dt = time.perf_counter() - t0
        if name != "plain": o, s = o
        else: s = 0
        if i >= 2: outs.append(cut(o)); secs.append(dt); cyc += s
    res[name] = {"tok_s": sum(map(len, outs)) / sum(secs), "tokens": sum(map(len, outs)), "seconds": sum(secs)}
    if name == "plain": ref = outs
    else:
        res[name].update(speedup=res[name]["tok_s"] / res["plain"]["tok_s"], tau=res[name]["tokens"] / max(cyc, 1),
                         identical_to_plain=sum(a == b for a, b in zip(outs, ref)))
    print(json.dumps({name: res[name]}), flush=True)
json.dump(res, open(OUT, "w"), indent=1); print("wrote", OUT)
