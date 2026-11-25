"""DFlash drafter (Chen et al. 2026, arXiv 2602.06036) for Ouro-1.4B, reimplemented from the official sources:
  model   = z-lab/Qwen3-4B-DFlash-b16 dflash.py (Qwen3DFlashAttention / DecoderLayer / DFlashDraftModel), with Ouro's dims
  recipe  = SpecForge OnlineDFlashModel (sgl-project/SpecForge, specforge/algorithms/common/dflash_family_model.py):
            anchors sampled WITHOUT replacement among response positions p with p and p+1 supervised (default 512),
            block = [clean anchor token, mask x 15], block sees target features at positions < anchor (strict) and its
            own block bidirectionally, slot k predicts token anchor+k, CE weights exp(-(k-1)/gamma), gamma 7 for block 16.
Architecture: 5 Qwen3-style layers (q_norm/k_norm, pre-norm, SwiGLU) of Ouro's width (d 2048, 16 heads x 128, MHA,
ffn 5632, rope theta 1e6), randomly initialised N(0, 0.02) (HF post_init); target features = residual-stream outputs of
5 target layers, concatenated -> fc (5d -> d) -> RMSNorm -> injected as extra K/V (each layer's own k_proj/v_proj +
k_norm + RoPE at the context positions) into EVERY draft layer.  Frozen + shared with Ouro: token embedding, lm_head.
Mask token: SpecForge's resolve_mask_token_id on Ouro's tokenizer (no mask/pad token, len(tok) == vocab) -> eos id 0.
Target layers ("rule": uniformly between the second and the third-to-last layer, build_target_layer_ids):
  unroll = over Ouro's unrolled 96-layer stack (4 loops x 24): effective layers [1, 24, 47, 70, 93]
  last   = over the final loop's 24 layers: [1, 6, 11, 16, 21] of loop 4 = effective [73, 78, 83, 88, 93]"""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # repository root
import math, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from floodgates.models.ouro_min import RMSNorm, apply_rope


def build_target_layer_ids(num_target_layers, num_draft_layers):     # verbatim from the official utils.py
    if num_draft_layers == 1:
        return [(num_target_layers // 2)]
    start, end = 1, num_target_layers - 3
    span = end - start
    return [int(round(start + (i * span) / (num_draft_layers - 1))) for i in range(num_draft_layers)]


def layer_set(name, L=24, R=4, n=5):
    if name == "unroll":
        return build_target_layer_ids(L * R, n)
    if name == "last":
        return [(R - 1) * L + i for i in build_target_layer_ids(L, n)]
    raise ValueError(name)


@torch.no_grad()
def ouro_features(ouro, ids, eff_ids):
    """Teacher-forced Ouro pass over ids (B, T); returns cat of the residual-stream outputs of the effective layers
    eff_ids (loop * L + layer, 0-indexed; = HF hidden_states[layer_id + 1]) -> (B, T, n * d) bf16.  Stops early."""
    B, T = ids.shape
    pos = torch.arange(T, device=ids.device)[None].expand(B, T)
    cos, sin = ouro.rope(pos)
    causal = lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)
    x = ouro.embed_tokens(ids)
    feats, last = {}, max(eff_ids)
    for ut in range(ouro.R):
        for li, layer in enumerate(ouro.layers):
            x, _, _ = layer(x, cos, sin, causal)
            e = ut * ouro.L + li
            if e in eff_ids:
                feats[e] = x
            if e == last:
                return torch.cat([feats[i] for i in eff_ids], dim=-1)
        x = ouro.norm(x)


class DFlashLayer(nn.Module):
    def __init__(self, d, H, Hkv, D, f, eps):
        super().__init__()
        self.H, self.Hkv, self.D = H, Hkv, D
        self.q_proj = nn.Linear(d, H * D, bias=False)
        self.k_proj = nn.Linear(d, Hkv * D, bias=False)
        self.v_proj = nn.Linear(d, Hkv * D, bias=False)
        self.o_proj = nn.Linear(H * D, d, bias=False)
        self.q_norm = RMSNorm(D, eps)
        self.k_norm = RMSNorm(D, eps)
        self.gate_proj = nn.Linear(d, f, bias=False)
        self.up_proj = nn.Linear(d, f, bias=False)
        self.down_proj = nn.Linear(f, d, bias=False)
        self.input_layernorm = RMSNorm(d, eps)
        self.post_attention_layernorm = RMSNorm(d, eps)

    def _kv(self, h, cos, sin):
        B, T, _ = h.shape
        k = self.k_norm(self.k_proj(h).view(B, T, self.Hkv, self.D)).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.Hkv, self.D).transpose(1, 2)
        return apply_rope(k, cos, sin), v

    def ctx_kv(self, ctx, cos, sin):
        """ctx (B, S, d) = projected target features (NOT passed through input_layernorm, as in the official code)."""
        return self._kv(ctx, cos, sin)

    def forward(self, x, cos, sin, ck, cv, attn):
        B, Q, _ = x.shape
        h = self.input_layernorm(x)
        q = apply_rope(self.q_norm(self.q_proj(h).view(B, Q, self.H, self.D)).transpose(1, 2), cos, sin)
        k, v = self._kv(h, cos, sin)
        bf = torch.bfloat16                                                # fp32 norm weights promote q/k
        a = attn(q.to(bf), torch.cat([ck, k], 2).to(bf), torch.cat([cv, v], 2).to(bf))   # (B, H, Q, D)
        x = x + self.o_proj(a.transpose(1, 2).reshape(B, Q, -1))
        h = self.post_attention_layernorm(x)
        return x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


class DFlashOuro(nn.Module):
    def __init__(self, ouro, layers="unroll", n_layers=5, block=16, mask_id=0, gamma=7.0):
        super().__init__()
        c = ouro.c
        d = c["hidden_size"]
        self.ouro, self.block, self.mask_id, self.gamma = ouro, block, mask_id, gamma
        self.target_layer_ids = layer_set(layers, ouro.L, ouro.R, n_layers)
        self.fc = nn.Linear(len(self.target_layer_ids) * d, d, bias=False)
        self.hidden_norm = RMSNorm(d, c["rms_norm_eps"])
        self.layers = nn.ModuleList([DFlashLayer(d, c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"],
                                                 c["intermediate_size"], c["rms_norm_eps"]) for _ in range(n_layers)])
        self.norm = RMSNorm(d, c["rms_norm_eps"])
        std = c.get("initializer_range", 0.02)
        for name, m in self.named_modules():
            if name.startswith("ouro"):
                continue
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=std)
        self.to(ouro.device)                              # fp32 master weights; forward under bf16 autocast

    def trainable(self):
        return [(n, p) for n, p in self.named_parameters() if not n.startswith("ouro.")]

    def state(self):
        return {n: p.detach().cpu() for n, p in self.trainable()}

    def features(self, ids):
        return ouro_features(self.ouro, ids, self.target_layer_ids)

    def trunk(self, feats, ctx_pos, noise_ids, block_pos, attn):
        """feats (B, S, 5d) target features; noise_ids (B, Q) block tokens (anchor / mask); attn(q, K, V) handles the
        mask.  Returns the normed draft hidden states (B, Q, d)."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ctx = self.hidden_norm(self.fc(feats))
            cos_c, sin_c = self.ouro.rope(ctx_pos)
            cos_b, sin_b = self.ouro.rope(block_pos)
            x = self.ouro.embed_tokens(noise_ids).float()
            for layer in self.layers:
                ck, cv = layer.ctx_kv(ctx, cos_c, sin_c)
                x = layer(x, cos_b, sin_b, ck, cv, attn)
            return self.norm(x)


# ---------------------------------------------------------------------------------------------------------------------
# training (SpecForge OnlineDFlashModel semantics) with FlexAttention block masks
# ---------------------------------------------------------------------------------------------------------------------
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
torch._dynamo.config.cache_size_limit = 256
_flex = torch.compile(flex_attention, dynamic=False)


def sample_anchors(lens, plens, num_anchors, gen):
    """Per sequence: candidates p with p, p+1 in the response (p >= P, p + 1 < len); a random subset of size
    min(num_anchors, #candidates) without replacement (sorted).  Returns list of lists."""
    out = []
    for n, P in zip(lens, plens):
        cand = torch.arange(P, n - 1)
        if len(cand) > num_anchors:
            cand = cand[torch.randperm(len(cand), generator=gen)[:num_anchors]].sort().values
        out.append(cand.tolist())
    return out


def bucket(x, m):
    return max(m, ((x + m - 1) // m) * m)


def blocks_hidden(dr, ids, lens, anc, keep, t_bucket=1024):
    """Training forward of all blocks at once (FlexAttention).  ids (B, T); anc/keep (B, N).  Returns the draft hidden
    states (B, N, bs, d), the slot targets tgt (B, N, bs) (slot k -> token anchor + k), their validity, and offsets."""
    dev, bs = dr.ouro.device, dr.block
    B, T = ids.shape; N = anc.shape[1]
    feats = dr.features(ids)                                         # (B, T, 5d), no grad
    S = bucket(T, t_bucket)
    if S > T:
        feats = F.pad(feats, (0, 0, 0, S - T))
    ctx_pos = torch.arange(S, device=dev)[None].expand(B, S)
    offs = torch.arange(bs, device=dev)
    lab_idx = anc[:, :, None] + offs
    lens_t = torch.tensor(lens, device=dev)
    valid = keep[:, :, None] & (lab_idx < lens_t[:, None, None])
    tgt = torch.gather(ids[:, None, :].expand(B, N, T), 2, lab_idx.clamp(max=T - 1))
    noise = torch.full((B, N, bs), dr.mask_id, dtype=torch.long, device=dev)
    noise[:, :, 0] = torch.where(keep, tgt[:, :, 0], torch.full_like(tgt[:, :, 0], dr.mask_id))
    block_pos = lab_idx.view(B, N * bs)

    def mask_mod(b, h, q_idx, kv_idx):
        qb = q_idx // bs
        a = anc[b, qb]
        m_ctx = (kv_idx < S) & (kv_idx < a)
        m_drf = (kv_idx >= S) & (((kv_idx - S) // bs) == qb)
        return (m_ctx | m_drf) & keep[b, qb]
    bm = create_block_mask(mask_mod, B, None, N * bs, S + N * bs, device=dev, _compile=True)
    attn = lambda q, k, v: _flex(q, k, v, block_mask=bm)
    hid = dr.trunk(feats, ctx_pos, noise.view(B, N * bs), block_pos, attn).view(B, N, bs, -1)
    return hid, tgt, valid, offs


def dflash_loss(dr, seqs, plens, num_anchors, gen, t_bucket=1024, n_bucket=None, ce_chunk=8192):
    dev, bs = dr.ouro.device, dr.block
    B = len(seqs)
    lens = [len(s) for s in seqs]
    T = max(lens)
    ids = torch.zeros(B, T, dtype=torch.long)
    for b, s in enumerate(seqs):
        ids[b, : len(s)] = torch.tensor(s)
    anchors = sample_anchors(lens, plens, num_anchors, gen)
    N = max(len(a) for a in anchors)
    N = bucket(N, n_bucket) if n_bucket else num_anchors
    N = max(N, 8)
    anc = torch.zeros(B, N, dtype=torch.long); keep = torch.zeros(B, N, dtype=torch.bool)
    for b, a in enumerate(anchors):
        anc[b, : len(a)] = torch.tensor(a, dtype=torch.long); keep[b, : len(a)] = True
    ids, anc, keep = ids.to(dev), anc.to(dev), keep.to(dev)
    hid, tgt, valid, offs = blocks_hidden(dr, ids, lens, anc, keep, t_bucket)

    w = torch.exp(-(offs.float() - 1).clamp(min=0) / dr.gamma)[None, None].expand(B, N, bs)
    w = w * valid.float() * (offs > 0).float()[None, None]           # slot 0 = clean anchor; response-only labels
    sel = w.view(-1) > 0
    h = hid.view(-1, hid.shape[-1])[sel]; y = tgt.view(-1)[sel]; ww = w.view(-1)[sel]
    lm = dr.ouro.lm_head

    def chunk_ce(hc, yc):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = lm(hc.to(torch.bfloat16)).float()
        return F.cross_entropy(lg, yc, reduction="none"), (lg.argmax(-1) == yc).float()
    ce, cor = [], []
    for i in range(0, h.shape[0], ce_chunk):
        c, a = checkpoint(chunk_ce, h[i: i + ce_chunk], y[i: i + ce_chunk], use_reentrant=False)
        ce.append(c); cor.append(a)
    ce, cor = torch.cat(ce), torch.cat(cor)
    loss = (ce * ww).sum() / ww.sum()
    with torch.no_grad():
        slot = offs[None, None].expand(B, N, bs).reshape(-1)[sel]
        acc1 = cor[slot == 1].mean() if (slot == 1).any() else torch.tensor(0.)
        acc_all = cor.mean()
    return loss, {"acc_s1": float(acc1), "acc": float(acc_all), "anchors": int(keep.sum()) / B, "N": N, "T": T}


# ---------------------------------------------------------------------------------------------------------------------
# offline drafts for every anchor of a reference trace (exactly the eval_accept.py protocol)
# ---------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def draft_all(dr, prompt, out, chunk=128):
    """For every m in 0..n-2: the 15 drafts of the block [out[m], mask x 15] at positions P+m.., reading the target
    features of the teacher-forced reference at positions < P+m.  Returns (n-1, block-1) long tensor."""
    dev, bs = dr.ouro.device, dr.block
    ids = torch.tensor([prompt + out], device=dev)
    P, n = len(prompt), len(out)
    feats = dr.features(ids)
    T = ids.shape[1]
    ctx_pos = torch.arange(T, device=dev)[None]
    res = []
    for s in range(0, n - 1, chunk):
        ms = torch.arange(s, min(n - 1, s + chunk), device=dev)
        A = len(ms)
        anc = P + ms
        noise = torch.full((1, A, bs), dr.mask_id, dtype=torch.long, device=dev)
        noise[0, :, 0] = ids[0, anc]
        block_pos = (anc[:, None] + torch.arange(bs, device=dev)).view(1, A * bs)
        ctx_m = torch.arange(T, device=dev)[None, :] < anc[:, None]               # (A, T)
        ctx_m = ctx_m[:, None, :].expand(A, bs, T).reshape(A * bs, T)
        own = torch.block_diag(*[torch.ones(bs, bs, dtype=torch.bool, device=dev)] * A)
        mask = torch.cat([ctx_m, own], dim=1)[None, None]
        attn = lambda q, k, v: F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        hid = dr.trunk(feats, ctx_pos, noise.view(1, A * bs), block_pos, attn).view(A, bs, -1)[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            res.append(dr.ouro.lm_head(hid.to(torch.bfloat16)).float().argmax(-1))
    return torch.cat(res).cpu()


def simulate(drafts, out, K):
    """eval_accept.py loop: cycle at m drafts K tokens, accepts the longest prefix equal to the reference, +1 bonus."""
    taus, pos = [], [[0, 0] for _ in range(K)]
    m = 0
    while m < len(out) - 1:
        d = drafts[m][:K].tolist()
        ref = out[m + 1: m + 1 + K]
        for j, (x, y) in enumerate(zip(d, ref)):
            pos[j][1] += 1; pos[j][0] += int(x == y)
        acc = 0
        while acc < len(ref) and d[acc] == ref[acc]:
            acc += 1
        taus.append(acc + 1); m += acc + 1
    return taus, pos
