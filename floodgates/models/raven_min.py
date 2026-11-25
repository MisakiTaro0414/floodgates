"""Minimal Raven-family recurrent-depth LM (Huginn-0125, smcleish Recurrent-Llama/TinyLlama/OLMo retrofits) in plain
PyTorch, matching raven_modeling_minimal.py: prelude blocks -> e (injected every iteration); state s_0 ~ truncated
normal(std) (like-init); each iteration s <- core(adapter([s, e])); coda blocks + ln_f + lm_head on the final state.
Pre-norm blocks (norm_1 -> attn -> +x, norm_2 -> gated MLP -> +x); GQA attention with a fused Wqkv; llama3 RoPE.
Exposes per-iteration states and a coda readout of ANY state (early-exit draft)."""
import json, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def llama3_inv_freq(c):
    hd = c["n_embd"] // c["n_heads"]
    base = c.get("rope_theta", c.get("rope_base"))
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    rs = c.get("rope_scaling")
    if not rs or rs.get("rope_type") != "llama3":
        return inv
    factor, lo, hi, old = rs["factor"], rs["low_freq_factor"], rs["high_freq_factor"], rs["original_max_position_embeddings"]
    lo_wl, hi_wl = old / lo, old / hi
    wl = 2 * math.pi / inv
    inv_l = torch.where(wl > lo_wl, inv / factor, inv)
    smooth = (old / wl - lo) / (hi - lo)
    smoothed = (1 - smooth) * inv_l / factor + smooth * inv_l
    medium = ~(wl < hi_wl) & ~(wl > lo_wl)
    return torch.where(medium, smoothed, inv_l)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def rope_interleaved(x, cos, sin):
    """Huginn's complex-like RoPE on adjacent pairs; x (B, H, T, D), cos/sin (B, T, D/2)."""
    xf = x.float().unflatten(-1, (-1, 2))
    c, s_ = cos.float()[:, None], sin.float()[:, None]
    out = torch.stack([xf[..., 0] * c - xf[..., 1] * s_, xf[..., 1] * c + xf[..., 0] * s_], -1).flatten(-2)
    return out.type_as(x)


class Block(nn.Module):
    def __init__(self, c, sandwich=False, interleaved=False):
        super().__init__()
        self.sandwich, self.interleaved = sandwich, interleaved
        d = c["n_embd"]
        self.H, self.KV = c["n_heads"], c["num_key_value_heads"]
        self.D = d // self.H
        self.norm_1 = RMSNorm(d, c["norm_eps"]); self.norm_2 = RMSNorm(d, c["norm_eps"])
        self.Wqkv = nn.Linear(d, (self.H + 2 * self.KV) * self.D, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc = nn.Linear(d, 2 * c["intermediate_size"], bias=False)
        self.mlp_proj = nn.Linear(c["intermediate_size"], d, bias=False)
        self.qk_bias = nn.Parameter(torch.zeros(2, 1, self.H, self.D)) if c.get("qk_bias") else None
        if sandwich:
            self.norm_3 = RMSNorm(d, c["norm_eps"]); self.norm_4 = RMSNorm(d, c["norm_eps"])

    def forward(self, x, cos, sin, attn_fn):
        """attn_fn(q, k, v) -> (B, H, T, D); q (B, H, T, D), k/v (B, KV, T, D) after RoPE."""
        B, T, d = x.shape
        h = self.norm_1(x)
        q, k, v = self.Wqkv(h).split([self.H * self.D, self.KV * self.D, self.KV * self.D], dim=-1)
        q = q.view(B, T, self.H, self.D); k = k.view(B, T, self.KV, self.D); v = v.view(B, T, self.KV, self.D)
        if self.qk_bias is not None:
            qb, kb = self.qk_bias.split(1, dim=0)
            q, k = (q + qb).to(q.dtype), (k + kb).to(q.dtype)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if self.interleaved:
            q, k = rope_interleaved(q, cos, sin), rope_interleaved(k, cos, sin)
        else:
            q = q * cos[:, None] + rotate_half(q) * sin[:, None]
            k = k * cos[:, None] + rotate_half(k) * sin[:, None]
        a = attn_fn(q, k, v).transpose(1, 2).reshape(B, T, d)
        if self.sandwich:                                   # Huginn: x = n2(attn(n1 x) + x); x = n4(mlp(n3 x) + x)
            x = self.norm_2(self.proj(a) + x)
            g, u = self.fc(self.norm_3(x)).chunk(2, dim=-1)
            return self.norm_4(self.mlp_proj(F.silu(g) * u) + x)
        x = x + self.proj(a)
        g, u = self.fc(self.norm_2(x)).chunk(2, dim=-1)
        return x + self.mlp_proj(F.silu(g) * u)


def _diag(T, S, dev):
    """(T, S) mask allowing each query its own key (keys S-T+i) -- keeps padded query rows finite."""
    m = torch.zeros(T, S, dtype=torch.bool, device=dev)
    m[torch.arange(T, device=dev), torch.arange(S - T, S, device=dev)] = True
    return m


def causal_attn(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)


class RavenMin(nn.Module):
    def __init__(self, model_dir, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        c = json.load(open(os.path.join(model_dir, "config.json")))
        self.c, self.R = c, int(os.environ.get("RAVEN_R", c["mean_recurrence"]))   # RAVEN_R: test-time number of core iterations
        d, V = c["n_embd"], c["padded_vocab_size"]
        idx = os.path.join(model_dir, "model.safetensors.index.json")
        names = json.load(open(idx))["weight_map"] if os.path.exists(idx) else {}
        sandwich = any(".norm_3." in k for k in names)
        self.interleaved = not (c.get("rope_scaling") or {}).get("rope_type") == "llama3" and sandwich
        Blk = lambda: Block(c, sandwich, self.interleaved)
        self.wte = nn.Embedding(V, d)
        self.prelude = nn.ModuleList([Blk() for _ in range(c["n_layers_in_prelude"])])
        self.adapter = nn.Linear(2 * d, d, bias=c.get("bias", False))
        self.core = nn.ModuleList([Blk() for _ in range(c["n_layers_in_recurrent_block"])])
        self.coda = nn.ModuleList([Blk() for _ in range(c["n_layers_in_coda"])])
        self.ln_f = RMSNorm(d, c["norm_eps"])
        mf = os.path.join(model_dir, "raven_modeling_minimal.py")   # Huginn applies ln_f to the recurrent state BEFORE the coda ("used twice"); the retrofits do not
        pcn = os.environ.get("RAVEN_PRE_CODA_NORM", "auto")
        self.pre_coda_norm = (os.path.exists(mf) and "return self.transformer.ln_f(x), num_steps_no_grad" in open(mf).read()) if pcn == "auto" else pcn == "1"
        self.pre = (lambda s: self.ln_f(s)) if self.pre_coda_norm else (lambda s: s)      # applied to a copy: the carried state stays unnormalized
        self.lm_head = nn.Linear(d, V, bias=False)
        sd = {}
        files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else ["model.safetensors"]
        for f in files:
            sd.update(load_file(os.path.join(model_dir, f)))
        own = {}
        for k, v in sd.items():
            if k == "freqs_cis":                            # Huginn stores its RoPE table; recomputed in rope()
                continue
            k = k.replace("transformer.core_block.", "core.").replace("transformer.", "")
            k = k.replace(".attn.Wqkv.", ".Wqkv.").replace(".attn.proj.", ".proj.").replace(".attn.qk_bias", ".qk_bias")
            k = k.replace(".mlp.fc.", ".fc.").replace(".mlp.proj.", ".mlp_proj.")
            own[k] = v
        if c.get("tie_embeddings") and "lm_head.weight" not in own:
            own["lm_head.weight"] = own["wte.weight"]
        missing, unexpected = self.load_state_dict(own, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        self.to(device=device, dtype=dtype).eval().requires_grad_(False)
        self.device, self.dtype = device, dtype
        self.inv_freq = llama3_inv_freq(c).to(device)          # (interleaved: plain theta, no scaling)
        self.emb_scale = c["init_values"]["embed_scale"]; self.init_std = c["init_values"]["std"]

    def rope(self, pos):
        f = pos.float()[..., None] * self.inv_freq
        if self.interleaved:
            return f.cos(), f.sin()
        emb = torch.cat([f, f], dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    def init_state(self, shape, gen=None):
        """like-init: truncated normal(std) * embed_scale (deterministic given gen)."""
        x = torch.randn(*shape, device=self.device, generator=gen) * self.init_std
        x = x.clamp(-3 * self.init_std, 3 * self.init_std)          # truncation (resampling-free approximation)
        return (x * self.emb_scale).to(self.dtype)

    def init_table(self, n, seed=0):
        """Rows j = like-init noise from a generator seeded by (seed, j); cached and grown on demand."""
        tab = getattr(self, "_init_tab", None)
        if tab is None or tab[0] != seed or tab[1].shape[0] < n:
            d = self.c["n_embd"]; n = max(n, 4096)
            rows = [self.init_state((1, d), torch.Generator(device=self.device).manual_seed(seed * 1_000_003 + j))
                    for j in range(n)]
            self._init_tab = (seed, torch.cat(rows, 0))
        return self._init_tab[1]

    def init_state_at(self, positions, seed=0):
        """Per-position deterministic init: position j gets row j of init_table -> (1, len, d)."""
        positions = list(positions)
        tab = self.init_table(max(positions) + 1, seed)
        return tab[torch.tensor(positions, device=self.device)][None]

    def embed(self, ids, cos, sin, attn=causal_attn):
        e = self.wte(ids) * self.emb_scale
        for b in self.prelude:
            e = b(e, cos, sin, attn)
        return e

    def core_step(self, s, e, cos, sin, attn=causal_attn):
        x = self.adapter(torch.cat([s, e], dim=-1))
        for b in self.core:
            x = b(x, cos, sin, attn)
        return x

    def readout(self, s, cos, sin, attn=causal_attn):
        s = self.pre(s)
        for b in self.coda:
            s = b(s, cos, sin, attn)
        return self.lm_head(self.ln_f(s))

    @torch.no_grad()
    def forward_full(self, ids, n_iter=None, s0=None, record=(), gen=None):
        """Teacher-forced causal pass.  Returns (final logits (B, T, V), {i: state after iteration i} for i in record)."""
        n_iter = n_iter or self.R
        B, T = ids.shape
        cos, sin = self.rope(torch.arange(T, device=ids.device)[None].expand(B, T))
        e = self.embed(ids, cos, sin)
        s = self.init_state(e.shape, gen) if s0 is None else s0
        rec = {}
        for i in range(1, n_iter + 1):
            s = self.core_step(s, e, cos, sin)
            if i in record:
                rec[i] = s
        return self.readout(s, cos, sin), rec

    @torch.no_grad()
    def generate(self, prompt, max_new, stop=lambda out: False, n_iter=None, seed=0):
        """Greedy decoding with per-(stage, iteration, layer) KV caches, batch 1.  The initial state of position j is
        drawn from a generator seeded by (seed, j), so the output is deterministic.  Returns the generated ids."""
        n_iter = n_iter or self.R
        cache = {}

        def attn_for(key):
            def attn(q, k, v):
                if key in cache:
                    pk, pv = cache[key]
                    k, v = torch.cat([pk, k], 2), torch.cat([pv, v], 2)
                cache[key] = (k, v)
                T, S = q.shape[2], k.shape[2]
                if T == S:
                    return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
                m = torch.ones(T, S, dtype=torch.bool, device=q.device).tril(S - T)
                return F.scaled_dot_product_attention(q, k, v, attn_mask=m, enable_gqa=True)
            return attn

        out, ids, start = [], list(prompt), 0
        x_ids = torch.tensor([ids], device=self.device)
        while len(out) < max_new:
            T = x_ids.shape[1]
            cos, sin = self.rope(torch.arange(start, start + T, device=self.device)[None])
            e = self.wte(x_ids) * self.emb_scale
            for li, b in enumerate(self.prelude):
                e = b(e, cos, sin, attn_for(("p", li)))
            s = self.init_state_at(range(start, start + T), seed)
            for it in range(n_iter):
                s = self.adapter(torch.cat([s, e], -1))
                for li, b in enumerate(self.core):
                    s = b(s, cos, sin, attn_for(("c", it, li)))
            s = self.pre(s)
            for li, b in enumerate(self.coda):
                s = b(s, cos, sin, attn_for(("o", li)))
            nxt = int(self.lm_head(self.ln_f(s[:, -1])).argmax(-1))
            out.append(nxt)
            if stop(out):
                break
            x_ids, start = torch.tensor([[nxt]], device=self.device), start + T
        return out

    # ---- staggered (wavefront) decoding, reference implementation --------------------------------------------------
    @torch.no_grad()
    def staggered_generate(self, prompt, max_new, R=None, lag=1, stop=lambda out: False, input_mode="hard", seed=0):
        """Every step runs ONE recurrence iteration for all in-flight positions (each at its own depth); a new position
        enters when its predecessor has done `lag` iterations; a position is committed after R iterations.  In-flight
        positions attend to committed positions' cached prelude/core/coda K/V (taken from their FINAL iteration) and to
        each other's current K/V (causal).  The input of an in-flight position is its predecessor's current readout:
        'hard' = argmax token, 'soft' = softmax-weighted embedding.  lag=R reproduces autoregressive decoding with
        final-iteration ("latest") KV sharing.  Returns (generated ids, number of steps)."""
        R = R or self.R
        d = self.c["n_embd"]
        W = self.wte.weight
        ck = {}                                                       # committed caches: key -> (k, v) (1, KV, T, D)

        def attn_with(key, commit):
            def attn(q, k, v):
                if key in ck:
                    pk, pv = ck[key]
                    K, V = torch.cat([pk, k], 2), torch.cat([pv, v], 2)
                else:
                    K, V = k, v
                if commit is not None:
                    commit[key] = (k, v)
                T, S = q.shape[2], K.shape[2]
                m = torch.ones(T, S, dtype=torch.bool, device=q.device).tril(S - T)
                return F.scaled_dot_product_attention(q, K, V, attn_mask=m, enable_gqa=True)
            return attn

        # prefill: synchronous iterations over the prompt; cache final-iteration K/V of every stage
        P = len(prompt)
        cos, sin = self.rope(torch.arange(P, device=self.device)[None])
        new = {}
        e = self.wte(torch.tensor([prompt], device=self.device)) * self.emb_scale
        for li, b in enumerate(self.prelude):
            e = b(e, cos, sin, attn_with(("p", li), new))
        s = self.init_state_at(range(P), seed)
        for it in range(R):
            s = self.adapter(torch.cat([s, e], -1))
            for li, b in enumerate(self.core):
                s = b(s, cos, sin, attn_with(("c", li), new if it == R - 1 else None))
        s = self.pre(s)
        for li, b in enumerate(self.coda):
            s = b(s, cos, sin, attn_with(("o", li), new))
        ck.update(new)
        first = int(self.lm_head(self.ln_f(s[:, -1])).argmax(-1))
        out = [first]
        # in-flight positions: dict(pos, s, n, logits); position P holds input `first` (committed) and predicts out[1]
        flight = [{"pos": P, "s": self.init_state_at([P], seed), "n": 0, "lg": None}]
        steps = 0
        while len(out) < max_new and not stop(out) and flight:
            pos = torch.tensor([f["pos"] for f in flight], device=self.device)
            # inputs: first in-flight position's input is the last committed token; others: predecessor's current readout
            embs = [W[out[-1]][None]]
            for f_prev in flight[:-1]:
                lg = f_prev["lg"]
                if input_mode == "soft":
                    embs.append((lg.float().softmax(-1).to(W.dtype) @ W)[None])
                else:
                    embs.append(W[int(lg.argmax())][None])
            x_in = torch.cat(embs, 0)[None] * self.emb_scale             # (1, F, d)
            cos, sin = self.rope(pos[None])
            commit = {} if flight[0]["n"] == R - 1 else None              # oldest commits after this step
            e = x_in
            for li, b in enumerate(self.prelude):
                e = b(e, cos, sin, attn_with(("p", li), commit))
            s = torch.cat([f["s"] for f in flight], 1)
            s = self.adapter(torch.cat([s, e], -1))
            for li, b in enumerate(self.core):
                s = b(s, cos, sin, attn_with(("c", li), commit))
            h = self.pre(s)
            for li, b in enumerate(self.coda):
                h = b(h, cos, sin, attn_with(("o", li), commit))
            lg = self.lm_head(self.ln_f(h))[0]                            # (F, V)
            for j, f in enumerate(flight):
                f["s"], f["n"], f["lg"] = s[:, j:j + 1], f["n"] + 1, lg[j]
            steps += 1
            if commit is not None:                                        # commit the oldest position
                for key, (k, v) in commit.items():
                    k1, v1 = k[:, :, :1], v[:, :, :1]
                    pk, pv = ck[key]
                    ck[key] = (torch.cat([pk, k1], 2), torch.cat([pv, v1], 2))
                tok = int(flight[0]["lg"].argmax())
                out.append(tok)
                flight.pop(0)
            # admit a new position when the newest has done `lag` iterations
            if flight == [] or flight[-1]["n"] >= lag:
                nxt = (flight[-1]["pos"] + 1) if flight else P + len(out) - 1
                flight.append({"pos": nxt, "s": self.init_state_at([nxt], seed), "n": 0, "lg": None})
        return out, steps

    @torch.no_grad()
    def generate_batch(self, prompts, max_new, stop=lambda out: False, seed=0):
        """Greedy decoding for a batch of prompts (left padding, per-row positions, key-padding masks, preallocated
        per-(stage, iteration, layer) KV buffers).  Same init noise as generate() (per absolute position)."""
        B = len(prompts); Pm = max(len(p) for p in prompts); dev = self.device
        Smax = Pm + max_new
        pad = torch.tensor([Pm - len(p) for p in prompts], device=dev)
        ids = torch.zeros(B, Pm, dtype=torch.long, device=dev)
        for b, p in enumerate(prompts):
            ids[b, Pm - len(p):] = torch.tensor(p, device=dev)
        pos = (torch.arange(Pm, device=dev)[None] - pad[:, None]).clamp(min=0)
        valid = torch.zeros(B, Smax, dtype=torch.bool, device=dev)
        valid[:, :Pm] = torch.arange(Pm, device=dev)[None] >= pad[:, None]
        tab = self.init_table(Smax + 1, seed)
        bufs = {}; L = [0]; MASK = [None]                                       # filled length, this step's mask

        def attn_for(key, T):
            def attn(q, k, v):
                if key not in bufs:
                    bufs[key] = (k.new_empty(B, k.shape[1], Smax, k.shape[3]), v.new_empty(B, v.shape[1], Smax, v.shape[3]))
                kb, vb = bufs[key]
                kb[:, :, L[0]:L[0] + T] = k; vb[:, :, L[0]:L[0] + T] = v
                S = L[0] + T
                return F.scaled_dot_product_attention(q, kb[:, :, :S], vb[:, :, :S], attn_mask=MASK[0], enable_gqa=True)
            return attn

        outs = [[] for _ in range(B)]; done = [False] * B
        x_ids, cur_pos = ids, pos
        for step in range(max_new):
            T = x_ids.shape[1]
            S = L[0] + T
            mk = torch.ones(T, S, dtype=torch.bool, device=dev).tril(S - T)[None] & valid[:, None, :S]
            MASK[0] = (mk | _diag(T, S, dev)[None])[:, None]
            cos, sin = self.rope(cur_pos)
            e = self.wte(x_ids) * self.emb_scale
            for li, b in enumerate(self.prelude):
                e = b(e, cos, sin, attn_for(("p", li), T))
            s = tab[cur_pos]
            for it in range(self.R):
                s = self.adapter(torch.cat([s, e], -1))
                for li, b in enumerate(self.core):
                    s = b(s, cos, sin, attn_for(("c", it, li), T))
            s = self.pre(s)
            for li, b in enumerate(self.coda):
                s = b(s, cos, sin, attn_for(("o", li), T))
            nxt = self.lm_head(self.ln_f(s[:, -1])).argmax(-1).tolist()
            L[0] += T
            for b in range(B):
                if not done[b]:
                    outs[b].append(nxt[b])
                    done[b] = stop(outs[b])
            if all(done):
                break
            valid[:, L[0]] = True
            x_ids = torch.tensor(nxt, device=dev)[:, None]; cur_pos = cur_pos[:, -1:] + 1
        return outs

    # ---- baseline: diffusion-forcing sampler of Geiping et al. (arXiv 2510.14961, Alg. 1), training-free -------------
    @torch.no_grad()
    def df_generate(self, prompt, max_new, stop=lambda out: False, inner=4, eta=0.1, eps=0.03, max_wave=128,
                    max_iters=None, seed=0):
        """Each forward pass: one new position joins the wavefront (input = predecessor's current argmax prediction);
        input embeddings of in-flight positions are refreshed from the predecessors' current predictions with momentum
        eta (e = eta e_prev + (1-eta) P(y)); `inner` recurrence steps run for all in-flight positions; the coda reads out
        every position; the longest prefix of the wavefront whose relative latent change ||z - z_prev|| / ||z|| < eps
        (or that reached max_iters) is frozen and committed.  KV: committed positions keep only their latest (final) K/V.
        Returns (generated ids, number of forward passes, total core iterations)."""
        max_iters = max_iters or self.R
        W = self.wte.weight
        ck = {}

        def attn_with(key, commit):
            def attn(q, k, v):
                if key in ck:
                    pk, pv = ck[key]
                    K, V = torch.cat([pk, k], 2), torch.cat([pv, v], 2)
                else:
                    K, V = k, v
                if commit is not None:
                    commit[key] = (k, v)
                T, S = q.shape[2], K.shape[2]
                m = torch.ones(T, S, dtype=torch.bool, device=q.device).tril(S - T)
                return F.scaled_dot_product_attention(q, K, V, attn_mask=m, enable_gqa=True)
            return attn

        P = len(prompt)
        cos, sin = self.rope(torch.arange(P, device=self.device)[None])
        new = {}
        e = self.wte(torch.tensor([prompt], device=self.device)) * self.emb_scale
        for li, b in enumerate(self.prelude):
            e = b(e, cos, sin, attn_with(("p", li), new))
        s = self.init_state_at(range(P), seed)
        for it in range(self.R):
            s = self.adapter(torch.cat([s, e], -1))
            for li, b in enumerate(self.core):
                s = b(s, cos, sin, attn_with(("c", li), new if it == self.R - 1 else None))
        s = self.pre(s)
        for li, b in enumerate(self.coda):
            s = b(s, cos, sin, attn_with(("o", li), new))
        ck.update(new)
        out = [int(self.lm_head(self.ln_f(s[:, -1])).argmax(-1))]
        wave = []                                      # dict(pos, z, e, pred, iters)
        passes = iters_total = 0
        while len(out) < max_new and not stop(out):
            nxt_pos = (wave[-1]["pos"] + 1) if wave else P + len(out) - 1
            if len(wave) < max_wave:
                wave.append({"pos": nxt_pos, "z": self.init_state_at([nxt_pos], seed), "e": None, "pred": None, "iters": 0})
            pos = torch.tensor([[w["pos"] for w in wave]], device=self.device)
            cos, sin = self.rope(pos)
            toks = [out[-1]] + [w["pred"] for w in wave[:-1]]       # input of each position = predecessor's prediction
            toks = [t if t is not None else out[-1] for t in toks]
            x_in = W[torch.tensor(toks, device=self.device)][None] * self.emb_scale
            rec = {}                                   # this pass's K/V of all wave positions (prelude, last inner iter, coda)
            e = x_in
            for li, b in enumerate(self.prelude):
                e = b(e, cos, sin, attn_with(("p", li), rec))
            e_mix = torch.cat([(eta * w["e"] + (1 - eta) * e[:, k:k + 1]) if w["e"] is not None else e[:, k:k + 1]
                               for k, w in enumerate(wave)], 1)
            z = torch.cat([w["z"] for w in wave], 1); z_prev = z
            for r_ in range(inner):
                z_prev = z
                z = self.adapter(torch.cat([z, e_mix], -1))
                for li, b in enumerate(self.core):
                    z = b(z, cos, sin, attn_with(("c", li), rec if r_ == inner - 1 else None))
            h = self.pre(z)
            for li, b in enumerate(self.coda):
                h = b(h, cos, sin, attn_with(("o", li), rec))
            preds = self.lm_head(self.ln_f(h))[0].argmax(-1).tolist()
            delta = ((z - z_prev).float().norm(dim=-1) / z.float().norm(dim=-1).clamp(min=1e-6))[0].tolist()
            passes += 1; iters_total += inner * len(wave)
            for k, w in enumerate(wave):
                w["z"], w["e"], w["pred"], w["iters"] = z[:, k:k + 1], e_mix[:, k:k + 1], preds[k], w["iters"] + inner
            # freeze the longest converged prefix: recompute its K/V once with its final latent and commit
            nfz = 0
            for w, dl in zip(wave, delta):
                if dl < eps or w["iters"] >= max_iters:
                    nfz += 1
                else:
                    break
            if nfz:
                fz = wave[:nfz]
                for key, (k, v) in rec.items():                           # freeze: keep their latest K/V
                    pk, pv = ck[key]
                    ck[key] = (torch.cat([pk, k[:, :, :nfz]], 2), torch.cat([pv, v[:, :, :nfz]], 2))
                for w in fz:
                    out.append(w["pred"])
                    if stop(out) or len(out) >= max_new:
                        break
                wave = wave[nfz:]
        return out, passes, iters_total
