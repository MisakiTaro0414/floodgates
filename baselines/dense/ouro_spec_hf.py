"""Wall-clock speculative decoding on Ouro-1.4B with every drafter of the drafter comparison in ONE eager implementation (plain PyTorch, batch size 1,
greedy, same prompts, stops and length limits; time per problem incl. prefill), against plain greedy decoding of the same model:
  loop:<run>          LAL's drafting pass (drafter.py, one pass of the block with LoRA on the slots, final-loop context K/V)
  loopm:<run>         the same with the LoRA deltas merged into a copy of the block's weights (as the engine does), no adapter matmuls
  ext:<run>           a separate drafter of 1 or 4 transformer layers (drafter_ext.py), fed Ouro's final-loop hidden states
  dflash:<run>:<K>    the DFlash-recipe drafter (baselines/dflash/dflash_ouro.py), K of its 15 drafts verified
Every step: the drafter proposes d_0..d_{n-1} for positions c+1..c+n, the full model (R loops) verifies [y, d_0..d_{n-1}] with its
per-(loop, layer) cache, keeps the leading drafts that equal its greedy tokens plus its own next token, and the caches are cropped.
The separate drafters' own context caches (ext: their causal K/V over Ouro's hidden states; DFlash: K/V of the projected target
features) are extended with the accepted positions after every step.  Checks that the outputs equal plain decoding.
  python baselines/dense/ouro_spec_hf.py <prompts jsonl> <gsm8k|math500> <out.json> loop:<run> [ext:<run> dflash:<run>:<K> ...]"""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # repository root
import json, os, sys, time
import torch
import torch.nn.functional as F
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, D); sys.path.insert(0, os.path.join(D, "runs", "dflash"))
from floodgates.models.ouro_min import OuroMin
from floodgates.drafters.drafter import LoopFlowDrafter
from floodgates.drafters.drafter_ext import ExtDrafter
from dflash_ouro import DFlashOuro
from floodgates.tokenizer import load_tokenizer
PR, TASK, OUT = sys.argv[1:4]; MODES = sys.argv[4:]
STOPS = {"gsm8k": ["Q:", "</s>", "<|im_end|>"], "math500": ["Problem:"]}[TASK]; MAXNEW = {"gsm8k": 512, "math500": 1024}[TASK]
args = {md: json.load(open(f"{md.split(':')[1]}/args.json")) for md in MODES}
MODEL = next(iter(args.values()))["model"]
m = OuroMin(MODEL); R, L, dev = m.R, m.L, m.device
tok = load_tokenizer(MODEL)
rows = [json.loads(l) for l in open(PR)][:int(os.environ.get("N_PROBLEMS", 100))]
gen = torch.Generator(device=dev).manual_seed(0)


def stopped(ids): return any(s in tok.decode(ids[-12:]) for s in STOPS)


def cut(ids):
    for n in range(1, len(ids) + 1):
        if any(s in tok.decode(ids[max(0, n - 8):n]) for s in STOPS): return ids[:n]
    return ids


def sdpa_cached(q, k, v, pk, pv):
    """Attention of T new queries over cached (pk, pv) plus their own k, v (causal among the new positions)."""
    if pk is not None: k, v = torch.cat([pk, k], 2), torch.cat([pv, v], 2)
    T, S = q.shape[2], k.shape[2]
    if T == S: return F.scaled_dot_product_attention(q, k, v, is_causal=True), k, v
    return F.scaled_dot_product_attention(q, k, v, attn_mask=torch.ones(T, S, dtype=torch.bool, device=q.device).tril(S - T)), k, v


class Target:
    """Ouro with a per-(loop, layer) cache: prefill / verification blocks, final-loop context K/V, cropping."""
    def __init__(self, feat_ids=()):
        self.cache = [[None] * L for _ in range(R)]; self.feat_ids = set(feat_ids)

    @torch.no_grad()
    def run(self, x_ids, start):
        """x_ids (1, T) at positions start..: greedy token after every position, the normalized final-loop hidden states
        (1, T, d), and the residual stream at the effective layers feat_ids."""
        T = x_ids.shape[1]; cos, sin = m.rope(torch.arange(start, start + T, device=dev)[None])
        x = m.embed_tokens(x_ids); feats = {}
        for ut in range(R):
            for li, layer in enumerate(m.layers):
                pk, pv = self.cache[ut][li] if self.cache[ut][li] is not None else (None, None)
                def attn(q, k, v, ut=ut, li=li, pk=pk, pv=pv):
                    a, k2, v2 = sdpa_cached(q, k, v, pk, pv); self.cache[ut][li] = (k2, v2); return a
                x, _, _ = layer(x, cos, sin, attn)
                if ut * L + li in self.feat_ids: feats[ut * L + li] = x
            x = m.norm(x)
        return m.lm_head(x[0]).float().argmax(-1), x, feats

    def ctx(self): return [self.cache[R - 1][li] for li in range(L)]

    def crop(self, n):
        for ut in range(R):
            for li in range(L): self.cache[ut][li] = (self.cache[ut][li][0][:, :, :n], self.cache[ut][li][1][:, :, :n])


@torch.no_grad()
def plain(ids):
    tg = Target(); t = tg.run(torch.tensor([ids], device=dev), 0); y = int(t[0][-1]); gen_ = [y]; c = len(ids)
    while len(gen_) < MAXNEW and not stopped(gen_):
        y = int(tg.run(torch.tensor([[y]], device=dev), c)[0][-1]); gen_.append(y); c += 1
    return gen_


def make_drafter(md):
    kind, run = md.split(":")[0], md.split(":")[1]; a = args[md]
    if kind in ("loop", "loopm"):
        dr = LoopFlowDrafter(m, k=a["k"], rank=a["rank"], kappa=a["kappa"], use_state=not a["no_state"], use_noise=not a["no_noise"],
                             flowmap=a.get("flowmap", False), shift=a.get("shift", False), ctx_loop=a.get("ctx_loop"), feedback=a.get("feedback", False),
                             markov_rank=a.get("markov_rank", 0))
    elif kind == "ext":
        dr = ExtDrafter(m, k=a["k"], n_layers=a["ext_layers"], init=a.get("ext_init", "random"))
    else:
        dr = DFlashOuro(m, layers=a["layers"], n_layers=a["n_layers"], block=a["block"], mask_id=a["mask_id"], gamma=a["gamma"])
    dr.load_state_dict(torch.load(f"{run}/drafter.pt"), strict=False); dr.eval()
    if kind == "loopm":                                                          # merge W + B A into a copy of the layers
        import copy
        layers = copy.deepcopy(m.layers)
        with torch.no_grad():
            for li, layer in enumerate(layers):
                for name, lo in dr.lora[li].items():
                    w = getattr(layer, name).weight; w.add_((lo.B.float() @ lo.A.float()).to(w.dtype))
        om = copy.copy(m); om._modules = dict(m._modules); om.layers = layers      # own module table: m keeps its plain layers
        dr2 = copy.copy(dr); dr2._modules = dict(dr._modules); dr2._modules.pop("lora")     # own module table: no adapters
        dr2.ouro = om; dr2.lora = [None] * L; dr = dr2
    return kind, dr


@torch.no_grad()
def spec(ids, kind, dr, K=None):
    tg = Target(feat_ids=dr.target_layer_ids if kind == "dflash" else ())
    ext_cache = [None] * (len(dr.layers) if kind in ("ext", "dflash") else 0)

    def ext_update(new_ids, hidden, feats, start):
        """Extend the separate drafter's context caches with the accepted positions start.. (ids, Ouro states)."""
        n = new_ids.shape[1]; cos, sin = m.rope(torch.arange(start, start + n, device=dev)[None])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if kind == "ext":
                x = dr.in_proj(m.embed_tokens(new_ids)) + dr.feat_proj(hidden)
                for i, layer in enumerate(dr.layers):
                    pk, pv = ext_cache[i] if ext_cache[i] is not None else (None, None)
                    def attn(q, k, v, i=i, pk=pk, pv=pv):
                        a, k2, v2 = sdpa_cached(q, k, v, pk, pv); ext_cache[i] = (k2, v2); return a
                    x, _, _ = layer(x, cos, sin, attn)
            else:
                ctx = dr.hidden_norm(dr.fc(torch.cat([feats[i] for i in dr.target_layer_ids], -1)))
                for i, layer in enumerate(dr.layers):
                    k, v = layer.ctx_kv(ctx, cos, sin)
                    ext_cache[i] = (k, v) if ext_cache[i] is None else (torch.cat([ext_cache[i][0], k], 2), torch.cat([ext_cache[i][1], v], 2))

    def draft(c, y):
        if kind in ("loop", "loopm"):
            return dr.draft(tg.ctx(), c, y, n_rounds=1, gen=gen)[:dr.k]           # k+1 slot drafts, the block verifies k of them
        if kind == "ext":
            return dr.draft(ext_cache, c, y)                                        # k drafts
        bs = dr.block; noise = torch.full((1, bs), dr.mask_id, dtype=torch.long, device=dev); noise[0, 0] = y
        cos_b, sin_b = m.rope((c + torch.arange(bs, device=dev))[None])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = m.embed_tokens(noise).float()
            for i, layer in enumerate(dr.layers):
                x = layer(x, cos_b, sin_b, ext_cache[i][0], ext_cache[i][1], lambda q, k, v: F.scaled_dot_product_attention(q, k, v))
            return m.lm_head(dr.norm(x)[:, 1:].to(torch.bfloat16)).float().argmax(-1)[0].tolist()[:K]

    P = len(ids); x_ids = torch.tensor([ids], device=dev); t, hidden, feats = tg.run(x_ids, 0)
    if kind not in ("loop", "loopm"): ext_update(x_ids, hidden, feats, 0)
    y = int(t[-1]); gen_ = [y]; c = P; steps = 0
    while len(gen_) < MAXNEW and not stopped(gen_):
        d = draft(c, y)
        blk = torch.tensor([[y] + d], device=dev)
        t, hidden, feats = tg.run(blk, c)                                           # greedy token after each of the n+1 inputs
        a_ = 0
        while a_ < len(d) and d[a_] == int(t[a_]): a_ += 1
        new = d[:a_] + [int(t[a_])]
        tg.crop(c + a_ + 1)
        if kind not in ("loop", "loopm"): ext_update(blk[:, :a_ + 1], hidden[:, :a_ + 1], {e: f[:, :a_ + 1] for e, f in feats.items()}, c)
        gen_ += new; c += len(new); y = gen_[-1]; steps += 1
    return gen_[:MAXNEW], steps


res = {"model": MODEL, "task": TASK, "n": len(rows), "gpu": torch.cuda.get_device_name()}
fns = [("plain", lambda ids: plain(ids))]
for md in MODES:
    kind, dr = make_drafter(md); K = int(md.split(":")[2]) if kind == "dflash" else None
    fns.append((md, lambda ids, kind=kind, dr=dr, K=K: spec(ids, kind, dr, K)))
for name, fn in fns:
    outs, secs, cyc = [], [], 0
    for i, r in enumerate([rows[0], rows[1]] + rows):                              # two warm-up problems, not timed
        ids = r["input"] if "input" in r else tok(r["context"])["input_ids"]
        torch.cuda.synchronize(); t0 = time.perf_counter(); o = fn(ids); torch.cuda.synchronize(); dt = time.perf_counter() - t0
        if name != "plain": o, s = o
        else: s = 0
        if i >= 2: outs.append(cut(o)); secs.append(dt); cyc += s
    res[name] = {"tok_s": sum(map(len, outs)) / sum(secs), "tokens": sum(map(len, outs)), "seconds": sum(secs)}
    if name == "plain": ref = outs
    else: res[name].update(speedup=res[name]["tok_s"] / res["plain"]["tok_s"], tau=res[name]["tokens"] / max(cyc, 1), identical_to_plain=sum(x == z for x, z in zip(outs, ref)))
    print(json.dumps({name: res[name]}), flush=True)
json.dump(res, open(OUT, "w"), indent=1); print("wrote", OUT)
