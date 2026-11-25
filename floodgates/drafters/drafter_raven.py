"""Self-drafter for a Raven-family recurrent-depth LM: the looped target drafts its own next block (port of drafter.py's
LoopFlowDrafter, mask variant, to RavenMin: prelude -> core x R with the prelude output injected every iteration -> coda).

A cycle starts after the verifier emitted y_m (known, not yet processed).  The canvas is k+1 slots at positions cut .. cut+k
(cut = P+m): slot 0 = y_m (clean token embedding), slots 1..k = a learned mask embedding; slot j predicts y_{m+j}.
One draft ROUND = prelude -> adapter([s0, e]) -> ONE core iteration -> coda -> ln_f -> lm_head over the slot rows only, where
s0 is the model's like-init state of the slot positions (init_table, as in generate()), LoRA (zero-init B) acts on the slot
rows of every block (+ the adapter), and in every block the slot queries attend to the CONTEXT's K/V of that block from the
plain teacher-forced pass (prelude blocks: prelude K/V; core blocks: K/V of the LAST iteration; coda blocks: coda K/V;
positions < cut) and to the slots of the same anchor (bidirectional).  The base model is never modified: the LoRA deltas
are added inside block_fwd (so the context pass stays the plain model after training).
Rounds (optional, rounds_max > 1): round r > 1 re-runs adapter -> core -> coda with the previous round's core output added to
the prelude output through a zero-init Linear (state_proj); s0 stays the like-init state; the prelude output is reused, so
an extra round costs core + coda layers (the eval counts prelude + core * n_rounds + coda).
Training: random anchors per sequence in the output region (slot targets = the next k tokens), prefix-weighted CE
exp(-(j-1)/4) on slots 1..k, deep supervision over rounds with the state detached.  Inference: argmax of slots 1..k."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from floodgates.models.raven_min import RavenMin, rope_interleaved, rotate_half

# RAVEN_CTX_ITER (default -1 = the last iteration): the core iteration whose context K/V the draft slots read; 0 = the first
# iteration, the shallow-prefix-cache control (training and decoding must use the same value).
import os
from floodgates.paths import LOOPSPEC
CTX_ITER = int(os.environ.get("RAVEN_CTX_ITER", "-1"))
MODEL_DEFAULT = LOOPSPEC + '/models/Recurrent-Llama-3.2-train-recurrence-32'
LORA_TARGETS = ("Wqkv", "proj", "fc", "mlp_proj")


class LoRA(nn.Module):
    """Additive low-rank delta x -> x A^T B^T (A random, B zero => zero at init; scale 1, as drafter.py)."""
    def __init__(self, din, dout, r):
        super().__init__()
        self.A = nn.Parameter(torch.randn(r, din) / math.sqrt(din))
        self.B = nn.Parameter(torch.zeros(dout, r))

    def forward(self, x):
        return (x.float() @ self.A.t() @ self.B.t()).to(x.dtype)


def block_fwd(blk, x, cos, sin, attn_fn, lora):
    """raven_min.Block.forward with the LoRA deltas of `lora` (ModuleDict Wqkv/proj/fc/mlp_proj) added to the slot rows."""
    B, T, d = x.shape
    h = blk.norm_1(x)
    q, k, v = (blk.Wqkv(h) + lora["Wqkv"](h)).split([blk.H * blk.D, blk.KV * blk.D, blk.KV * blk.D], dim=-1)
    q = q.view(B, T, blk.H, blk.D); k = k.view(B, T, blk.KV, blk.D); v = v.view(B, T, blk.KV, blk.D)
    if blk.qk_bias is not None:
        qb, kb = blk.qk_bias.split(1, dim=0)
        q, k = (q + qb).to(q.dtype), (k + kb).to(q.dtype)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if blk.interleaved:
        q, k = rope_interleaved(q, cos, sin), rope_interleaved(k, cos, sin)
    else:
        q = q * cos[:, None] + rotate_half(q) * sin[:, None]
        k = k * cos[:, None] + rotate_half(k) * sin[:, None]
    a = attn_fn(q, k, v).transpose(1, 2).reshape(B, T, d)
    pa = blk.proj(a) + lora["proj"](a)
    if blk.sandwich:                                        # Huginn: x = n2(attn(n1 x) + x); x = n4(mlp(n3 x) + x)
        x = blk.norm_2(pa + x)
        h3 = blk.norm_3(x)
        g, u = (blk.fc(h3) + lora["fc"](h3)).chunk(2, dim=-1)
        mm = F.silu(g) * u
        return blk.norm_4(blk.mlp_proj(mm) + lora["mlp_proj"](mm) + x)
    x = x + pa
    h2 = blk.norm_2(x)
    g, u = (blk.fc(h2) + lora["fc"](h2)).chunk(2, dim=-1)
    mm = F.silu(g) * u
    return x + blk.mlp_proj(mm) + lora["mlp_proj"](mm)


@torch.no_grad()
def context_kv(m, ids, n_iter=None, seed=0, logits=False):
    """Plain teacher-forced causal pass of RavenMin m over ids (B, T) with the per-position like-init state (init_table,
    as generate()/generate_batch()).  Returns (logits (B, T, V) float or None, kv): kv = list over the blocks in draft-pass
    order [prelude..., core... (K/V of the LAST iteration), coda...] of (k, v) (B, KV, T, D) after RoPE."""
    n_iter = n_iter or m.R
    B, T = ids.shape
    pos = torch.arange(T, device=ids.device)[None].expand(B, T)
    cos, sin = m.rope(pos)

    def rec(store):
        def attn(q, k, v):
            if store is not None:
                store.append((k, v))
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return attn

    pre, core, coda = [], [], []
    ci = CTX_ITER if CTX_ITER >= 0 else n_iter - 1                      # the core iteration whose K/V the slots read
    e = m.wte(ids) * m.emb_scale
    for b in m.prelude:
        e = b(e, cos, sin, rec(pre))
    s = m.init_table(T + 1, seed)[pos]
    for it in range(n_iter):
        s = m.adapter(torch.cat([s, e], dim=-1))
        for b in m.core:
            s = b(s, cos, sin, rec(core if it == ci else None))
    h = m.ln_f(s) if getattr(m, "pre_coda_norm", False) else s
    for b in m.coda:
        h = b(h, cos, sin, rec(coda))
    lg = m.lm_head(m.ln_f(h)).float() if logits else None
    return lg, pre + core + coda


@torch.no_grad()
def context_kv_all(m, ids, seed=0):
    """Like context_kv, but keeps the core K/V of EVERY iteration: returns (pre, core_all, coda) with pre/coda lists over
    blocks of (k, v) and core_all[it][j] the (k, v) of core block j at iteration it (B, KV, T, D), after RoPE."""
    B, T = ids.shape
    pos = torch.arange(T, device=ids.device)[None].expand(B, T)
    cos, sin = m.rope(pos)

    def rec(store):
        def attn(q, k, v):
            store.append((k, v))
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return attn

    pre, core_all, coda = [], [], []
    e = m.wte(ids) * m.emb_scale
    for b in m.prelude:
        e = b(e, cos, sin, rec(pre))
    s = m.init_table(T + 1, seed)[pos]
    for it in range(m.R):
        s = m.adapter(torch.cat([s, e], dim=-1))
        core_all.append([])
        for b in m.core:
            s = b(s, cos, sin, rec(core_all[-1]))
    h = m.ln_f(s) if getattr(m, "pre_coda_norm", False) else s
    for b in m.coda:
        h = b(h, cos, sin, rec(coda))
    return pre, core_all, coda


class RavenSelfDrafter(nn.Module):
    def __init__(self, m: RavenMin, k=8, rank=32, shift=False, feedback=False, iters=1, carry=False, causal=False):
        super().__init__()
        self.causal = causal    # slots attend causally to earlier slots (as the DAS-Wave window) instead of bidirectionally
        self.carry = carry        # rounds continue the core recurrence from the previous round's latent (depth accumulates)
        self.iters = iters        # core iterations inside one draft pass (LoRA shared across them, like the model's loop)
        self.shift = shift      # next-token-aligned slots: slot j (incl. clean slot 0) predicts the token at cut+j+1
        self.feedback = feedback  # rounds > 1 take the previous round's drafted TOKENS as slot inputs (trained refinement)
        self.m, self.k = m, k
        c = m.c
        d, f, V = c["n_embd"], c["intermediate_size"], c["padded_vocab_size"]
        H, KV = c["n_heads"], c["num_key_value_heads"]; D = d // H
        self.d, self.V = d, V
        self.nP, self.nC, self.nO = len(m.prelude), len(m.core), len(m.coda)
        dims = {"Wqkv": (d, (H + 2 * KV) * D), "proj": (d, d), "fc": (d, 2 * f), "mlp_proj": (f, d)}
        self.lora = nn.ModuleList([nn.ModuleDict({n: LoRA(*dims[n], rank) for n in LORA_TARGETS})
                                   for _ in range(self.nP + self.nC + self.nO)])
        self.lora_adapter = LoRA(2 * d, d, rank)
        self.mask_emb = nn.Parameter(torch.zeros(d))        # input of slots 1..k
        self.state_proj = nn.Linear(d, d, bias=False)       # carried state (rounds > 1), zero-init: round 1 = no state
        nn.init.zeros_(self.state_proj.weight)
        self.draft_emb = nn.Parameter(torch.zeros(d))       # added to fed-back draft tokens (marks them as provisional)
        self.to(m.device)

    # ---- inputs -------------------------------------------------------------------------------------------------
    def slot_inputs(self, sid, is_clean):
        """sid (N, S) long (slot 0 = y_m), is_clean (N, S) bool -> (N, S, d) token embedding (clean) or mask embedding."""
        clean = self.m.wte(sid) * self.m.emb_scale
        return torch.where(is_clean[..., None], clean, self.mask_emb.to(clean.dtype).expand_as(clean))

    def feedback_inputs(self, y_in, pred, k1):
        """Slot inputs of the next round from the previous round's argmax pred (N, k1): slot 0 keeps the clean token y_in (N,);
        shift: slot j's input is the token drafted for position cut+j, i.e. pred[j-1] (each readout seeds the next position);
        no shift: slot j's input is its own draft pred[j].  Draft tokens get the draft embedding added."""
        toks = torch.cat([y_in[:, None], pred[:, :k1 - 1]], 1) if self.shift else torch.cat([y_in[:, None], pred[:, 1:]], 1)
        e = self.m.wte(toks) * self.m.emb_scale
        prov = torch.ones_like(toks, dtype=torch.bool); prov[:, 0] = False
        return e + prov[..., None].to(e.dtype) * self.draft_emb.to(e.dtype)

    def init_states(self, slot_pos, seed=0):
        """(B, Q) positions -> (B, Q, d) like-init states (same table as generate())."""
        return self.m.init_table(int(slot_pos.max()) + 1, seed)[slot_pos]

    # ---- one round: slots through prelude -> adapter -> one core iteration -> coda --------------------------------
    def round(self, e_in, s0, slot_pos, ctx_kv, mask, state=None):
        """e_in (B, Q, d) slot inputs; s0 (B, Q, d) like-init states; slot_pos (B, Q); ctx_kv per block (k, v) (B, KV, T, D);
        mask (B, 1, Q, T+Q) bool (True = may attend); state (B, Q, d) previous round's core output or None.
        Returns (logits (B, Q, V) float, core output (B, Q, d))."""
        m = self.m
        cos, sin = m.rope(slot_pos)

        def attn_for(i):
            ck, cv = ctx_kv[i]
            def attn(q, k, v):
                return F.scaled_dot_product_attention(q, torch.cat([ck, k], 2), torch.cat([cv, v], 2), attn_mask=mask,
                                                      enable_gqa=True)
            return attn

        x = e_in
        for i, blk in enumerate(m.prelude):
            x = block_fwd(blk, x, cos, sin, attn_for(i), self.lora[i])
        if state is not None:
            x = x + self.state_proj(m.ln_f(state).float()).to(x.dtype)
        s = state if (self.carry and state is not None) else s0
        for _ in range(self.iters):
            ae = torch.cat([s, x], dim=-1)
            s = m.adapter(ae) + self.lora_adapter(ae)
            for j, blk in enumerate(m.core):
                s = block_fwd(blk, s, cos, sin, attn_for(self.nP + j), self.lora[self.nP + j])
        h = m.ln_f(s) if getattr(m, "pre_coda_norm", False) else s
        for j, blk in enumerate(m.coda):
            h = block_fwd(blk, h, cos, sin, attn_for(self.nP + self.nC + j), self.lora[self.nP + self.nC + j])
        return m.lm_head(m.ln_f(h)).float(), s

    # ---- the model's own round over the window (continuation of the recurrence, no adapters) ---------------------
    def plain_round(self, tok, state, slot_pos, ctx_kv, mask, iters=1, vocab=None):
        """tok (B, Q) window tokens (slot 0 = y_m, slot j = the token drafted for position cut+j); state (B, Q, d) carried
        core latent.  prelude -> adapter -> ONE core iteration from `state` -> coda -> lm_head, with the PLAIN blocks (no LoRA,
        no state_proj, no draft marker) reading the context at its final iteration: exactly the looped model's computation
        on these tokens (the DAS-Wave round of Li et al. 2026).  Returns (logits (B, Q, V) float, core state (B, Q, d))."""
        m = self.m
        cos, sin = m.rope(slot_pos)

        def attn_for(i):
            ck, cv = ctx_kv[i]
            def attn(q, k, v):
                return F.scaled_dot_product_attention(q, torch.cat([ck, k], 2), torch.cat([cv, v], 2), attn_mask=mask,
                                                      enable_gqa=True)
            return attn

        x = m.wte(tok) * m.emb_scale
        for i, blk in enumerate(m.prelude):
            x = blk(x, cos, sin, attn_for(i))
        s = state
        for _ in range(iters):
            s = m.adapter(torch.cat([s, x], dim=-1))
            for j, blk in enumerate(m.core):
                s = blk(s, cos, sin, attn_for(self.nP + j))
        h = m.ln_f(s) if getattr(m, "pre_coda_norm", False) else s
        for j, blk in enumerate(m.coda):
            h = blk(h, cos, sin, attn_for(self.nP + self.nC + j))
        h = m.ln_f(h)
        if vocab is not None:                                  # restricted readout: exact logits on the candidate tokens only
            lg = torch.full((*h.shape[:-1], self.V), float("-inf"), device=h.device)
            lg[..., vocab] = (h @ m.lm_head.weight[vocab].T).float()
            return lg, s
        return m.lm_head(h).float(), s

    @torch.no_grad()
    def teacher_targets(self, tok, slot_pos, kv_all, mask_c):
        """Full-depth pass of the PLAIN model over the window tokens tok (B, Q) (slot 0 = y_m, slot j = the provisional token
        at position cut+j), reading the prefix at every iteration and causal within the window: returns the model's own
        next-token prediction at every slot (B, Q), i.e. the target that verification will compare the drafts against."""
        m = self.m; pre, core_all, coda = kv_all
        cos, sin = m.rope(slot_pos)

        def attn_for(ck, cv):
            def attn(q, k, v):
                return F.scaled_dot_product_attention(q, torch.cat([ck, k], 2), torch.cat([cv, v], 2), attn_mask=mask_c,
                                                      enable_gqa=True)
            return attn

        x = m.wte(tok) * m.emb_scale
        for i, blk in enumerate(m.prelude):
            x = blk(x, cos, sin, attn_for(*pre[i]))
        s = self.init_states(slot_pos)
        for it in range(m.R):
            s = m.adapter(torch.cat([s, x], dim=-1))
            for j, blk in enumerate(m.core):
                s = blk(s, cos, sin, attn_for(*core_all[it][j]))
        h = m.ln_f(s) if getattr(m, "pre_coda_norm", False) else s
        for j, blk in enumerate(m.coda):
            h = blk(h, cos, sin, attn_for(*coda[j]))
        return m.lm_head(m.ln_f(h)).argmax(-1)

    @staticmethod
    def wave_feedback(tok, logits, B, A, k1):
        """DAS-Wave token feedback: the readout of slot j becomes the input of slot j+1; slot 0 keeps y_m.  tok (B, Q) -> (B, Q)."""
        z = logits.detach().argmax(-1).view(B, A, k1)
        t = tok.view(B, A, k1)
        return torch.cat([t[..., :1], z[..., :-1]], -1).reshape(B, A * k1)

    # ---- training ---------------------------------------------------------------------------------------------
    def build_batch(self, seqs, prompt_lens, anchors_per_seq, gen):
        """seqs: list of id lists (prompt+output); anchors m sampled in the output region (m < n-1 so slot 1 has a target).
        Returns ids (B, T), slot_ids (B, A, k+1), slot_valid (B, A, k+1), cut (B, A)."""
        B = len(seqs); T = max(len(s) for s in seqs); k1 = self.k + 1
        ids = torch.zeros(B, T, dtype=torch.long)
        for b, s in enumerate(seqs):
            ids[b, : len(s)] = torch.tensor(s)
        A = anchors_per_seq
        slot_ids = torch.zeros(B, A, k1, dtype=torch.long); slot_valid = torch.zeros(B, A, k1, dtype=torch.bool)
        cut = torch.zeros(B, A, dtype=torch.long)
        off = 1 if self.shift else 0
        for b, s in enumerate(seqs):
            P, n = prompt_lens[b], len(s) - prompt_lens[b]
            ms = torch.randint(0, max(n - 1, 1), (A,), generator=gen)
            for a, m in enumerate(ms.tolist()):
                cut[b, a] = P + m                                  # context = positions < P+m
                for j in range(k1):
                    pos = P + m + j + off                          # target position (shift: next token)
                    if pos < len(s):
                        slot_ids[b, a, j] = s[pos]; slot_valid[b, a, j] = True
        if self.shift:                                             # slot 0's INPUT is y_m = s[cut]
            self._slot0_in = torch.stack([torch.tensor([seqs[b][int(cut[b, a])] for a in range(A)]) for b in range(B)])
        return ids, slot_ids, slot_valid, cut

    def masks(self, cut, T, A, k1, device, causal=None):
        """(B, 1, A*k1, T + A*k1): slots of anchor a see context < cut[b,a] and the slots of anchor a (bidirectional, or
        causal within the anchor's slots if causal / self.causal)."""
        B = cut.shape[0]; Q = A * k1
        causal = self.causal if causal is None else causal
        ctx = torch.arange(T, device=device)[None, None, :] < cut[:, :, None]          # (B, A, T)
        ctx = ctx[:, :, None, :].expand(B, A, k1, T).reshape(B, Q, T)
        blk = torch.tril(torch.ones(k1, k1, dtype=torch.bool, device=device)) if causal else torch.ones(k1, k1, dtype=torch.bool, device=device)
        own = torch.block_diag(*[blk] * A)[None].expand(B, Q, Q)
        return torch.cat([ctx, own], dim=-1)[:, None]

    def loss(self, seqs, prompt_lens, anchors_per_seq=16, rounds_max=1, gen=None, prefix_decay=4.0,
             plain_rounds=0, plain_carry=True, plain_weight=1.0, local_target=False):
        """plain_rounds > 0: after the adapter pass(es), run that many rounds of the model's OWN recurrence over the drafted
        window (plain_round, causal window, token feedback), starting from the adapter pass's latent (plain_carry) or from
        the like-init state, with the loss after every round; the gradient reaches the adapters through the carried latent
        (the tokens fed back are discrete), so the adapter pass is trained to initialize the continuation well."""
        dev = self.m.device; k1 = self.k + 1; A = anchors_per_seq
        ids, slot_ids, slot_valid, cut = self.build_batch(seqs, prompt_lens, A, gen)
        ids, slot_ids, slot_valid, cut = ids.to(dev), slot_ids.to(dev), slot_valid.to(dev), cut.to(dev)
        B, T = ids.shape
        if local_target:                                                      # full-depth teacher needs every iteration's K/V
            kv_all = context_kv_all(self.m, ids)
            ctx_kv = kv_all[0] + kv_all[1][-1] + kv_all[2]
        else:
            _, ctx_kv = context_kv(self.m, ids)
        mask = self.masks(cut, T, A, k1, dev)
        Q = A * k1
        sid = slot_ids.view(B, Q); valid = slot_valid.view(B, Q)
        slot_pos = (cut[:, :, None] + torch.arange(k1, device=dev)[None, None]).view(B, Q)
        is_clean = torch.zeros(B, A, k1, dtype=torch.bool, device=dev); is_clean[:, :, 0] = True
        is_clean = is_clean.view(B, Q)
        in_ids = sid
        if self.shift:
            in_ids = sid.clone().view(B, A, k1); in_ids[:, :, 0] = self._slot0_in.to(dev); in_ids = in_ids.view(B, Q)
        e_in = self.slot_inputs(in_ids, is_clean)
        s0 = self.init_states(slot_pos)
        w = torch.exp(-(torch.arange(k1, device=dev).float() - (0 if self.shift else 1)).clamp(min=0) / prefix_decay)
        w = w[None, None].expand(B, A, k1).reshape(B, Q) * valid.float() * (torch.ones_like(is_clean) if self.shift else ~is_clean).float()
        r = rounds_max if self.feedback else int(torch.randint(1, rounds_max + 1, (1,), generator=gen))
        state = None; total = 0.0; logs = {}
        y_in = in_ids.view(B, A, k1)[:, :, 0].reshape(B * A)
        s_last = None; tgt = sid
        for i in range(r):
            logits, s = self.round(e_in, s0, slot_pos, ctx_kv, mask, state)
            ce = F.cross_entropy(logits.view(-1, self.V), tgt.view(-1), reduction="none").view(B, Q)
            total = total + (ce * w).sum() / w.sum().clamp(min=1)
            state = s.detach(); s_last = s
            if self.feedback:                                       # next round: own drafted tokens as inputs (on-policy)
                pred = logits.detach().argmax(-1).view(B * A, k1)
                e_in = self.feedback_inputs(y_in, pred, k1).view(B, Q, -1).to(e_in.dtype)
                if local_target:                                    # consistent target: the model's prediction given the drafts
                    assert self.shift
                    toks = torch.cat([y_in[:, None], pred[:, :k1 - 1]], 1).view(B, Q)
                    tgt = self.teacher_targets(toks, slot_pos, kv_all, self.masks(cut, T, A, k1, dev, causal=True))
                    with torch.no_grad():
                        logs[f"tgt_eq_x_r{i + 1}"] = float((((tgt == sid) & (w > 0)).sum()) / (w > 0).sum().clamp(min=1))
            with torch.no_grad():
                j0 = 0 if self.shift else 1
                hit = ((logits.argmax(-1) == sid) & (w > 0)).view(B, A, k1)[..., j0:]
                cnt = (w > 0).view(B, A, k1)[..., j0:]
                logs[f"acc_r{i}"] = float(hit.sum() / cnt.sum().clamp(min=1))
                if i == 0:
                    logs["slot_acc_r0"] = [round(float(x), 4) for x in (hit.sum((0, 1)) / cnt.sum((0, 1)).clamp(min=1))]
        logs["rounds"] = r
        if plain_rounds:
            assert self.shift, "plain rounds assume next-token-aligned slots"
            from torch.utils.checkpoint import checkpoint
            mask_c = self.masks(cut, T, A, k1, dev, causal=True)
            tok = self.wave_feedback(in_ids, logits, B, A, k1)                   # window seeded with the adapter pass's drafts
            ps = s_last if plain_carry else s0                                   # carried latent (differentiable) or like-init
            for j in range(plain_rounds):
                lg, ps = checkpoint(self.plain_round, tok, ps, slot_pos, ctx_kv, mask_c, use_reentrant=False)
                ce = F.cross_entropy(lg.view(-1, self.V), sid.view(-1), reduction="none").view(B, Q)
                total = total + plain_weight * (ce * w).sum() / w.sum().clamp(min=1)
                tok = self.wave_feedback(tok, lg, B, A, k1)
                with torch.no_grad():
                    hit = ((lg.argmax(-1) == sid) & (w > 0)).view(B, A, k1)
                    cnt = (w > 0).view(B, A, k1)
                    logs[f"acc_p{j + 1}"] = float(hit.sum() / cnt.sum().clamp(min=1))
                    if j + 1 == plain_rounds:
                        logs["slot_acc_plast"] = [round(float(x), 4) for x in (hit.sum((0, 1)) / cnt.sum((0, 1)).clamp(min=1))]
            return total / (r + plain_weight * plain_rounds), logs
        return total / r, logs

    # ---- inference --------------------------------------------------------------------------------------------
    @torch.no_grad()
    def draft(self, ctx_kv, cut, y_m, n_rounds=1, plain_rounds=0, plain_carry=True, plain_iters=1, k_win=None):
        """Single sequence: ctx_kv from context_kv (positions < cut used), y_m the just-emitted token at position cut.
        Returns the k drafted ids (list) = argmax of slots 1..k after n_rounds adapter passes and plain_rounds rounds of the
        model's own recurrence over the drafted window (continuation; plain_carry: from the adapter pass's latent)."""
        dev = self.m.device; k1 = self.k + 1
        T = ctx_kv[0][0].shape[2]
        cut_t = torch.tensor([[cut]], device=dev)
        mask = self.masks(cut_t, T, 1, k1, dev)
        sid = torch.zeros(1, k1, dtype=torch.long, device=dev); sid[0, 0] = y_m
        is_clean = torch.zeros(1, k1, dtype=torch.bool, device=dev); is_clean[0, 0] = True
        slot_pos = (cut + torch.arange(k1, device=dev))[None]
        e_in = self.slot_inputs(sid, is_clean)
        s0 = self.init_states(slot_pos)
        state = None
        for _ in range(n_rounds):
            logits, state = self.round(e_in, s0, slot_pos, ctx_kv, mask, state)
            if self.feedback:
                e_in = self.feedback_inputs(sid[:, 0], logits.argmax(-1), k1).to(e_in.dtype)
        if plain_rounds:
            assert self.shift
            kw = k_win + 1 if k_win else k1                                      # window may exceed the drafter's slots
            tok = self.wave_feedback(sid, logits, 1, 1, k1)
            ps = state if plain_carry else s0
            if kw > k1:                                                          # extend: repeat the last draft, like-init states
                tok = torch.cat([tok, tok[:, -1:].expand(1, kw - k1)], 1)
                pos_w = (cut + torch.arange(kw, device=dev))[None]
                ps = torch.cat([ps, self.init_states(pos_w)[:, k1:]], 1)
            else:
                pos_w = slot_pos
            mask_c = self.masks(cut_t, T, 1, kw, dev, causal=True)
            for _ in range(plain_rounds):
                logits, ps = self.plain_round(tok, ps, pos_w, ctx_kv, mask_c, iters=plain_iters)
                tok = self.wave_feedback(tok, logits, 1, 1, kw)
        return logits[0, (0 if self.shift else 1):].argmax(-1).tolist()


@torch.no_grad()
def draft_schedule(dr, ctx_kv, cut, y_m, schedule, plain_iters=1, top2=False):
    """Single sequence: alternate trained adapter passes ('L') and rounds of the model's own recurrence (digits) over the
    same window, sharing the window tokens and the carried latent, e.g. 'L8L8' = adapter pass, 8 plain rounds, adapter pass,
    8 plain rounds.  Returns the k drafts.  Cost in rounds: each 'L' = one adapter pass, each digit group = that many rounds."""
    import re
    dev = dr.m.device; k1 = dr.k + 1
    T = ctx_kv[0][0].shape[2]
    cut_t = torch.tensor([[cut]], device=dev)
    mask_b = dr.masks(cut_t, T, 1, k1, dev)                      # adapter pass: as trained (bidirectional unless causal)
    mask_c = dr.masks(cut_t, T, 1, k1, dev, causal=True)         # plain rounds: the model's causal attention
    sid = torch.zeros(1, k1, dtype=torch.long, device=dev); sid[0, 0] = y_m
    is_clean = torch.zeros(1, k1, dtype=torch.bool, device=dev); is_clean[0, 0] = True
    slot_pos = (cut + torch.arange(k1, device=dev))[None]
    s0 = dr.init_states(slot_pos)
    state = None; logits = None; tok = None
    for step in re.findall(r"L|\d+", schedule):
        if step == "L":
            if logits is None:
                e_in = dr.slot_inputs(sid, is_clean)
            else:                                                 # re-draft from the current window tokens and latent
                e_in = dr.feedback_inputs(sid[:, 0], logits.argmax(-1), k1).to(dr.m.dtype)
            logits, state = dr.round(e_in, s0, slot_pos, ctx_kv, mask_b, state)
            tok = dr.wave_feedback(sid, logits, 1, 1, k1)
        else:
            if state is None:
                state = s0
            for _ in range(int(step)):
                logits, state = dr.plain_round(tok, state, slot_pos, ctx_kv, mask_c, iters=plain_iters)
                tok = dr.wave_feedback(tok, logits, 1, 1, k1)
    lg = logits[0, (0 if dr.shift else 1):]
    if top2:                                                       # (drafts, second choices, confidence margins)
        v, i = lg.float().softmax(-1).topk(2, dim=-1)
        return i[:, 0].tolist(), i[:, 1].tolist(), (v[:, 0] - v[:, 1]).tolist()
    return lg.argmax(-1).tolist()


@torch.no_grad()
def draft_cheap(dr, ctx_kv, cut, y_m, J, topk=8, full_every=8, iters=1):
    """Trained pass, then J rounds of the model's own recurrence (carried latent) in which the output head is read only over
    a candidate vocabulary: the union over positions of the top-`topk` tokens of the last FULL readout and the current window
    tokens; a full readout every `full_every` rounds refreshes the candidates.  Returns (drafts, number of full readouts)."""
    dev = dr.m.device; k1 = dr.k + 1
    T = ctx_kv[0][0].shape[2]
    cut_t = torch.tensor([[cut]], device=dev)
    mask = dr.masks(cut_t, T, 1, k1, dev); mask_c = dr.masks(cut_t, T, 1, k1, dev, causal=True)
    sid = torch.zeros(1, k1, dtype=torch.long, device=dev); sid[0, 0] = y_m
    is_clean = torch.zeros(1, k1, dtype=torch.bool, device=dev); is_clean[0, 0] = True
    slot_pos = (cut + torch.arange(k1, device=dev))[None]
    s0 = dr.init_states(slot_pos)
    logits, state = dr.round(dr.slot_inputs(sid, is_clean), s0, slot_pos, ctx_kv, mask, None)
    tok = dr.wave_feedback(sid, logits, 1, 1, k1)
    cand = torch.unique(torch.cat([logits[0].topk(topk, -1).indices.flatten(), tok[0]]))
    nfull = 0
    for r in range(J):
        full = (r + 1) % full_every == 0
        logits, state = dr.plain_round(tok, state, slot_pos, ctx_kv, mask_c, iters=iters, vocab=None if full else cand)
        tok = dr.wave_feedback(tok, logits, 1, 1, k1)
        if full:
            nfull += 1
            cand = torch.unique(torch.cat([logits[0].topk(topk, -1).indices.flatten(), tok[0], cand]))
    return logits[0, (0 if dr.shift else 1):].argmax(-1).tolist(), nfull


@torch.no_grad()
def draft_branch(dr, ctx_kv, cut, y_m, J, split, iters=1):
    """Trained pass and `split` rounds of the model's own recurrence on one window; then the window is duplicated and, in the
    copy, the input of the slot after the least confident readout is replaced by that readout's second choice; the remaining
    J - split rounds run on both windows in one batch (the same weight reads as one window).  Returns the drafts of both
    windows and the index of the branched draft, for tree verification with recomputed continuations."""
    dev = dr.m.device; k1 = dr.k + 1
    T = ctx_kv[0][0].shape[2]
    cut_t = torch.tensor([[cut]], device=dev)
    mask_b = dr.masks(cut_t, T, 1, k1, dev); mask_c = dr.masks(cut_t, T, 1, k1, dev, causal=True)
    sid = torch.zeros(1, k1, dtype=torch.long, device=dev); sid[0, 0] = y_m
    is_clean = torch.zeros(1, k1, dtype=torch.bool, device=dev); is_clean[0, 0] = True
    slot_pos = (cut + torch.arange(k1, device=dev))[None]
    s0 = dr.init_states(slot_pos)
    logits, state = dr.round(dr.slot_inputs(sid, is_clean), s0, slot_pos, ctx_kv, mask_b, None)
    tok = dr.wave_feedback(sid, logits, 1, 1, k1)
    for _ in range(split):
        logits, state = dr.plain_round(tok, state, slot_pos, ctx_kv, mask_c, iters=iters)
        tok = dr.wave_feedback(tok, logits, 1, 1, k1)
    off = 0 if dr.shift else 1
    v, i = logits[0, off:].float().softmax(-1).topk(2, dim=-1)        # the readouts that are drafts (readout of slot j feeds slot j+1)
    jd = int((v[:-1, 0] - v[:-1, 1]).argmin())                         # least confident draft among those that feed a later slot
    j = jd + off
    tok2 = tok.clone(); tok2[0, j + 1] = i[jd, 1]
    tok = torch.cat([tok, tok2]); state = torch.cat([state, state])
    kv2 = [(ck.expand(2, -1, -1, -1), cv.expand(2, -1, -1, -1)) for ck, cv in ctx_kv]
    for _ in range(J - split):
        logits, state = dr.plain_round(tok, state, slot_pos.expand(2, -1), kv2, mask_c, iters=iters)
        tok = dr.wave_feedback(tok, logits, 2, 1, k1)
    d = logits[:, off:].argmax(-1)
    return d[0].tolist(), d[1].tolist(), jd


class PersistentWindow:
    """Sliding window of the model's own recurrence over provisional tokens that persists ACROSS verification cycles
    (single sequence).  seed(): trained draft pass + J0 plain rounds; drafts(): current predictions; commit(a, y): after
    verification accepted a drafts and the model emitted y, shift the window by a+1 (keep the latents of the surviving
    positions, refill the tail), then rounds(J) continues the recurrence with the context now covering the committed tokens.
    Cost per cycle in rounds: J0 (+ the draft pass) at seeding, J afterwards."""
    def __init__(self, dr, ctx_kv, k_win=None, iters=1):
        self.dr, self.ctx_kv, self.iters = dr, ctx_kv, iters
        self.kw = (k_win or dr.k) + 1
        self.T = ctx_kv[0][0].shape[2]
        self.dev = dr.m.device

    def _mask_pos(self, cut):
        cut_t = torch.tensor([[cut]], device=self.dev)
        return self.dr.masks(cut_t, self.T, 1, self.kw, self.dev, causal=True), (cut + torch.arange(self.kw, device=self.dev))[None]

    @torch.no_grad()
    def seed(self, cut, y_m, J0, n_lora=1, carry=True):
        dr, k1, kw = self.dr, self.dr.k + 1, self.kw
        self.cut = cut
        cut_t = torch.tensor([[cut]], device=self.dev)
        mask = dr.masks(cut_t, self.T, 1, k1, self.dev)
        sid = torch.zeros(1, k1, dtype=torch.long, device=self.dev); sid[0, 0] = y_m
        is_clean = torch.zeros(1, k1, dtype=torch.bool, device=self.dev); is_clean[0, 0] = True
        slot_pos = (cut + torch.arange(k1, device=self.dev))[None]
        e_in = dr.slot_inputs(sid, is_clean); s0 = dr.init_states(slot_pos); state = None
        for _ in range(n_lora):
            logits, state = dr.round(e_in, s0, slot_pos, self.ctx_kv, mask, state)
            if dr.feedback:
                e_in = dr.feedback_inputs(sid[:, 0], logits.argmax(-1), k1).to(e_in.dtype)
        tok = dr.wave_feedback(sid, logits, 1, 1, k1)
        st = state if carry else s0
        if kw > k1:
            tok = torch.cat([tok, tok[:, -1:].expand(1, kw - k1)], 1)
            st = torch.cat([st, dr.init_states((cut + torch.arange(kw, device=self.dev))[None])[:, k1:]], 1)
        self.tok, self.state, self.pred = tok, st, logits.argmax(-1)[:, :1].expand(1, kw)
        self.rounds(J0)

    @torch.no_grad()
    def rounds(self, J):
        mask_c, pos = self._mask_pos(self.cut)
        for _ in range(J):
            logits, self.state = self.dr.plain_round(self.tok, self.state, pos, self.ctx_kv, mask_c, iters=self.iters)
            self.pred = logits.argmax(-1)                              # pred[j] = token for position cut+j+1
            self.tok = self.dr.wave_feedback(self.tok, logits, 1, 1, self.kw)

    def drafts(self):
        return self.pred[0].tolist()                                   # kw drafts for positions cut+1 .. cut+kw

    @torch.no_grad()
    def commit(self, a, y_next):
        """Verification accepted drafts for positions cut+1..cut+a and the model emitted y_next at cut+a+1."""
        sh = a + 1; kw = self.kw
        self.cut += sh
        tok = torch.cat([torch.tensor([[y_next]], device=self.dev), self.tok[:, sh + 1:]], 1)   # slot 0 = the committed token
        st = self.state[:, sh:]
        pred = self.pred[:, sh:]
        fill = kw - tok.shape[1]
        if fill > 0:
            last = pred[:, -1:] if pred.shape[1] else torch.tensor([[y_next]], device=self.dev)
            tok = torch.cat([tok, last.expand(1, fill)], 1)
            pos_new = (self.cut + torch.arange(kw, device=self.dev))[None]
            st = torch.cat([st, self.dr.init_states(pos_new)[:, kw - fill:]], 1)
            pred = torch.cat([pred, last.expand(1, fill + 1)], 1)[:, :kw]
        self.tok, self.state, self.pred = tok[:, :kw], st[:, :kw], pred[:, :kw]


# ---- unit check: at init (LoRA B = 0, mask embedding 0) slot 0 must reproduce the model's own next-token logits -------
# With a 1-iteration context pass (forward_full(n_iter=1) with the same per-position init) the slot-0 pass is the same
# computation as the full pass at position cut, provided the context K/V wiring (block order, LAST-iteration core K/V,
# RoPE positions) is right.  k=0: exact (slot 0 sees context + itself); k=8: slot 0 also sees the 8 zero-input mask slots
# (bidirectional), a small dilution of its attention.  Also prints the 32-iteration numbers (slot-0 one-iteration
# readout vs the model's final logits; the drafter's starting point).
#   python drafter_raven.py <data.jsonl> [n_seqs=4] [model_dir]
if __name__ == "__main__":
    import json, sys
    data = sys.argv[1]; n_seq = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    model = sys.argv[3] if len(sys.argv) > 3 else MODEL_DEFAULT
    m = RavenMin(model)
    rows = [json.loads(l) for l in open(data) if l.strip()]
    rows = [r for r in rows if len(r["output_ids"]) >= 16][:n_seq]
    seqs = [r["prompt_ids"] + r["output_ids"] for r in rows]; plens = [len(r["prompt_ids"]) for r in rows]
    dev = m.device; A = 16
    for n_iter in (1, m.R):
        for k in (0, 8):
            gen = torch.Generator().manual_seed(0)
            dr = RavenSelfDrafter(m, k=k, rank=32).eval()
            k1 = k + 1
            ids, slot_ids, slot_valid, cut = dr.build_batch(seqs, plens, A, gen)
            ids, slot_ids, cut = ids.to(dev), slot_ids.to(dev), cut.to(dev)
            B, T = ids.shape
            pos = torch.arange(T, device=dev)[None].expand(B, T)
            full, _ = m.forward_full(ids, n_iter=n_iter, s0=m.init_table(T + 1)[pos])          # (B, T, V) bf16
            lg_ctx, kv = context_kv(m, ids, n_iter=n_iter, logits=True)
            mask = dr.masks(cut, T, A, k1, dev); Q = A * k1
            sid = slot_ids.view(B, Q)
            slot_pos = (cut[:, :, None] + torch.arange(k1, device=dev)[None, None]).view(B, Q)
            is_clean = torch.zeros(B, A, k1, dtype=torch.bool, device=dev); is_clean[:, :, 0] = True
            with torch.no_grad():
                logits, _ = dr.round(dr.slot_inputs(sid, is_clean.view(B, Q)), dr.init_states(slot_pos), slot_pos, kv, mask)
            s0_logits = logits.view(B, A, k1, -1)[:, :, 0]                                       # (B, A, V)
            ref = torch.gather(full.float(), 1, cut[..., None].expand(B, A, full.shape[-1]))      # full-pass logits at cut
            ref_ctx = torch.gather(lg_ctx, 1, cut[..., None].expand(B, A, full.shape[-1]))
            agree = float((s0_logits.argmax(-1) == ref.argmax(-1)).float().mean())
            agree_ctx = float((ref_ctx.argmax(-1) == ref.argmax(-1)).float().mean())
            maxdiff = float((s0_logits - ref).abs().max()); refscale = float(ref.abs().max())
            print(f"n_iter={n_iter} k={k}: slot-0 argmax == forward_full argmax at cut on {agree * 100:.1f}% of {B * A} anchors "
                  f"(context-pass logits vs forward_full: {agree_ctx * 100:.1f}%), max|logit diff|={maxdiff:.3f} "
                  f"(max|logit|={refscale:.1f})", flush=True)
            if n_iter == m.R:
                tgt = torch.gather(ids, 1, (cut + 1).clamp(max=T - 1))                          # y_(m+1) (B, A)
                print(f"   slot-0 one-iteration readout == trace token y_(m+1): {float((s0_logits.argmax(-1) == tgt).float().mean()) * 100:.1f}%; "
                      f"model final argmax == y_(m+1): {float((ref.argmax(-1) == tgt).float().mean()) * 100:.1f}%", flush=True)
            del full, lg_ctx, kv, logits, s0_logits, ref, ref_ctx
            torch.cuda.empty_cache()
