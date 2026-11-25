"""Looped-flow drafter for Ouro (Thinking with Looped Flows, arXiv 2609.11801, applied to a block of future tokens).

A cycle starts after the verifier emitted y_m (known, not yet processed).  The canvas is k+1 slots at positions
P+m .. P+m+k: slot 0 = y_m (clean, t=1), slots 1..k = drafts of y_{m+1..m+k}.  One draft ROUND = one pass of Ouro's own
24-layer stack (one loop) over the canvas with LoRA active only on slot tokens; slots attend to the verified context's
final-loop K/V (positions < P+m) and to each other (bidirectional).  Readout = Ouro's step norm + lm_head (mean
parameterization: p_hat = softmax).  Slot input = x_t @ E (x_t in vocab space: training interpolant (1-t) sigma eps +
t onehot, inference Euler state) + time embedding(t) + W_s(state of the previous round) (zero-init: round 1 = no state).
Training (paper recipe): sorted times per anchor, SHARED noise across the rounds of a rollout, state carried and detached
between rounds, loss on every round (deep supervision), prefix-weighted CE.  Inference: Euler on a fixed grid t_i = i/n."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRA(nn.Module):
    def __init__(self, din, dout, r):
        super().__init__()
        self.A = nn.Parameter(torch.randn(r, din) / math.sqrt(din))
        self.B = nn.Parameter(torch.zeros(dout, r))

    def forward(self, x):
        return (x.float() @ self.A.t() @ self.B.t()).to(x.dtype)


def sinusoidal(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float()[..., None] * f * 1000.0
    return torch.cat([a.sin(), a.cos()], dim=-1)


class LoopFlowDrafter(nn.Module):
    def __init__(self, ouro, k=8, rank=32, kappa=4.0, use_state=True, use_noise=True, flowmap=False, shift=False, ctx_loop=None,
                 feedback=False, markov_rank=0):
        super().__init__()
        self.markov_rank = markov_rank  # > 0: low-rank bias on slot j's logits from the token drafted at slot j-1 (DSpark-style)
        self.feedback = feedback  # rounds > 1 take the previous round's drafted TOKENS as slot inputs (trained refinement)
        self.ouro, self.k, self.use_state, self.use_noise, self.flowmap = ouro, k, use_state, use_noise, flowmap
        self.ctx_loop = ctx_loop  # which loop's K/V the slots read for the context (None = final loop; 0 = first loop, ablation)
        self.shift = shift      # next-token-aligned slots: slot j (incl. the clean slot 0) predicts the token at cut+j+1
        c = ouro.c
        d, f, V = c["hidden_size"], c["intermediate_size"], c["vocab_size"]
        self.d, self.V = d, V
        dims = {"q_proj": (d, d), "k_proj": (d, d), "v_proj": (d, d), "o_proj": (d, d),
                "gate_proj": (d, f), "up_proj": (d, f), "down_proj": (f, d)}
        self.lora = nn.ModuleList([nn.ModuleDict({n: LoRA(*dims[n], rank) for n in LORA_TARGETS}) for _ in range(ouro.L)])
        self.time_mlp = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        nn.init.zeros_(self.time_mlp[2].weight); nn.init.zeros_(self.time_mlp[2].bias)
        if flowmap:   # flow map D_{s,t}: second time embedding of the jump t - s (zero-init: D_{s,t} = D_{s,s} at init)
            self.time_end_mlp = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
            nn.init.zeros_(self.time_end_mlp[2].weight); nn.init.zeros_(self.time_end_mlp[2].bias)
        self.state_proj = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.state_proj.weight)
        self.mask_emb = nn.Parameter(torch.zeros(d))          # "no flow" ablation: a learned mask input instead of noise
        self.draft_emb = nn.Parameter(torch.zeros(d))         # added to fed-back draft tokens (marks them as provisional)
        # noise scale in vocab space: RMS(eps @ E) = kappa * RMS(token embedding)
        # eps ~ N(0, sigma^2 I_V) => RMS(eps @ E) = sigma * sqrt(V * mean(E^2)) = kappa * RMS(E) for sigma = kappa / sqrt(V)
        self.sigma = kappa / math.sqrt(V)
        if markov_rank:
            self.mk_a = nn.Embedding(V, markov_rank); nn.init.normal_(self.mk_a.weight, std=0.02)
            self.mk_b = nn.Linear(markov_rank, V, bias=False); nn.init.zeros_(self.mk_b.weight)   # zero-init: no effect at start
        self.to(ouro.device)

    def markov_bias(self, prev):
        """prev (...,) long: token drafted (or, in training, the true token) one position earlier -> (..., V) float bias."""
        return self.mk_b(self.mk_a(prev).float())

    def markov_decode(self, logits, y):
        """Greedy left-to-right readout with the Markov bias.  logits (S, k1, V) of the slots, y (S,) the clean slot-0 token.
        Returns the drafts (S, k): no shift = argmax of slots 1..k, shift = argmax of slots 0..k-1 (each conditioned on the
        previous draft, the first on y)."""
        j0 = 0 if self.shift else 1
        prev, out = y, []
        for j in range(j0, j0 + self.k):
            d = (logits[:, j] + self.markov_bias(prev)).argmax(-1); out.append(d); prev = d
        return torch.stack(out, 1)

    # ---- inputs -------------------------------------------------------------------------------------------------
    def slot_inputs(self, x_vocab, clean_ids, is_clean, t, state, t_end=None):
        """x_vocab (N, S, V) float (noisy slots); clean_ids (N, S) long; is_clean (N, S) bool; t (N, S); state (N, S, d);
        t_end (N, S) end time of the flow-map jump (flowmap only; None = diagonal t_end = t)."""
        E = self.ouro.embed_tokens.weight
        if self.use_noise:
            noisy = (x_vocab.to(E.dtype) @ E)                  # (N, S, d)
        else:
            noisy = self.mask_emb.to(E.dtype).expand(*clean_ids.shape, self.d)
        e = torch.where(is_clean[..., None], self.ouro.embed_tokens(clean_ids), noisy)
        e = e + self.time_mlp(sinusoidal(t, 256)).to(e.dtype)
        if self.flowmap:
            gap = torch.zeros_like(t) if t_end is None else t_end - t
            e = e + self.time_end_mlp(sinusoidal(gap, 256)).to(e.dtype)
        if self.use_state and state is not None:
            e = e + self.state_proj(self.ouro.norm(state).float()).to(e.dtype)
        return e

    def feedback_inputs(self, y_in, pred, k1, state):
        """Next round's slot inputs from the previous round's argmax pred (N, k1): slot 0 keeps y_in (N,); shift: slot j gets
        pred[j-1] (each readout seeds the next position), no shift: slot j gets its own pred[j]; provisional slots get the
        draft embedding; time embedding 1 (clean slot) / 0 (provisional), plus the carried state as in slot_inputs."""
        toks = torch.cat([y_in[:, None], pred[:, :k1 - 1]], 1) if self.shift else torch.cat([y_in[:, None], pred[:, 1:]], 1)
        prov = torch.ones_like(toks, dtype=torch.bool); prov[:, 0] = False
        t = (~prov).float()
        e = self.slot_inputs(None, toks, torch.ones_like(prov), t, state)
        return e + prov[..., None].to(e.dtype) * self.draft_emb.to(e.dtype)

    # ---- one round: slots through Ouro's stack ----------------------------------------------------------------
    def round(self, e, slot_pos, ctx_kv, mask):
        """e (B, Q, d) slot inputs; slot_pos (B, Q) positions; ctx_kv per layer (k, v) (B, H, T, D) of the context;
        mask (B, 1, Q, T+Q) bool (True = may attend).  Returns (logits (B, Q, V) float, hidden (B, Q, d) before norm)."""
        cos, sin = self.ouro.rope(slot_pos)
        x = e
        for li, layer in enumerate(self.ouro.layers):
            ck, cv = ctx_kv[li]
            def attn(q, k, v, ck=ck, cv=cv):
                return F.scaled_dot_product_attention(q, torch.cat([ck, k], 2), torch.cat([cv, v], 2), attn_mask=mask)
            x, _, _ = layer(x, cos, sin, attn, lora=self.lora[li])
        h = self.ouro.norm(x)
        return self.ouro.lm_head(h).float(), x

    # ---- training ---------------------------------------------------------------------------------------------
    def build_batch(self, seqs, prompt_lens, anchors_per_seq, gen):
        """seqs: list of id lists (prompt+output); anchors m sampled in the output region.  Returns tensors for a batch."""
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
            ms = torch.randint(0, max(n, 1), (A,), generator=gen)
            for a, m in enumerate(ms.tolist()):
                cut[b, a] = P + m                                  # context = positions < P+m
                for j in range(k1):
                    pos = P + m + j + off                          # target position (shift: next token)
                    if pos < len(s):
                        slot_ids[b, a, j] = s[pos]; slot_valid[b, a, j] = True
        if self.shift:                                             # slot_ids = targets; slot 0's INPUT is y_m = s[cut]
            self._slot0_in = torch.stack([torch.tensor([seqs[b][int(cut[b, a])] for a in range(A)]) for b in range(B)])
        return ids, slot_ids, slot_valid, cut

    def masks(self, cut, T, A, k1, device):
        """(B, 1, A*k1, T + A*k1): slots of anchor a see context < cut[b,a] and the slots of anchor a."""
        B = cut.shape[0]; Q = A * k1
        ctx = torch.arange(T, device=device)[None, None, :] < cut[:, :, None]          # (B, A, T)
        ctx = ctx[:, :, None, :].expand(B, A, k1, T).reshape(B, Q, T)
        own = torch.block_diag(*[torch.ones(k1, k1, dtype=torch.bool, device=device)] * A)[None].expand(B, Q, Q)
        return torch.cat([ctx, own], dim=-1)[:, None]

    def loss(self, seqs, prompt_lens, anchors_per_seq=16, rounds_max=3, gen=None, prefix_decay=4.0, acc_weight=False):
        dev = self.ouro.device; k1 = self.k + 1; A = anchors_per_seq
        ids, slot_ids, slot_valid, cut = self.build_batch(seqs, prompt_lens, A, gen)
        ids, slot_ids, slot_valid, cut = ids.to(dev), slot_ids.to(dev), slot_valid.to(dev), cut.to(dev)
        B, T = ids.shape
        _, ctx_kv = self.ouro.forward_context(ids, keep_kv_loop=self.ctx_loop, logits=False)
        mask = self.masks(cut, T, A, k1, dev)
        Q = A * k1
        sid = slot_ids.view(B, Q); valid = slot_valid.view(B, Q)
        slot_pos = (cut[:, :, None] + torch.arange(k1, device=dev)[None, None]).view(B, Q)
        is_clean = torch.zeros(B, A, k1, dtype=torch.bool, device=dev); is_clean[:, :, 0] = True
        is_clean = is_clean.view(B, Q)
        in_ids = sid
        if self.shift:                                             # clean slot 0 carries y_m, not its target
            in_ids = sid.clone().view(B, A, k1); in_ids[:, :, 0] = self._slot0_in.to(dev); in_ids = in_ids.view(B, Q)
        # rollout: r rounds with sorted times (first = 0 with prob 1/2), shared noise across rounds
        r = rounds_max if self.feedback else int(torch.randint(1, rounds_max + 1, (1,), generator=gen))
        ts = torch.sort(torch.rand(B, A, r, generator=gen), dim=-1).values
        if self.feedback:
            ts.zero_()                                               # feedback rounds: deterministic t = 0 (as at inference)
        zero_first = torch.rand(B, A, generator=gen) < 0.5
        ts[..., 0] = torch.where(zero_first, torch.zeros_like(ts[..., 0]), ts[..., 0])
        ts = ts.to(dev)
        eps = torch.randn(B, Q, self.V, device=dev) * self.sigma
        onehot = F.one_hot(sid, self.V).float()
        w = torch.exp(-(torch.arange(k1, device=dev).float() - (0 if self.shift else 1)).clamp(min=0) / prefix_decay)
        w = w[None, None].expand(B, A, k1).reshape(B, Q) * valid.float() * (torch.ones_like(is_clean) if self.shift else ~is_clean).float()
        state = None; total = 0.0; logs = {}
        for i in range(r):
            t = ts[..., i][..., None].expand(B, A, k1).reshape(B, Q)
            t_in = torch.where(is_clean, torch.ones_like(t), t)
            x_t = (1 - t)[..., None] * eps + t[..., None] * onehot
            if self.feedback and i > 0:
                e = self.feedback_inputs(in_ids.view(B, A, k1)[:, :, 0].reshape(B * A), pred.view(B * A, k1), k1,
                                         state.view(B * A, k1, -1)).view(B, Q, -1)
            else:
                e = self.slot_inputs(x_t, in_ids, is_clean, t_in, state)
            logits, h = self.round(e, slot_pos, ctx_kv, mask)
            if self.markov_rank:                                     # teacher-forced: slot j sees the true token at slot j-1
                prev = torch.cat([in_ids.view(B, A, k1)[:, :, :1], sid.view(B, A, k1)[:, :, :-1]], -1).view(B, Q)
                logits = logits + self.markov_bias(prev)
            pred = logits.detach().argmax(-1)
            ce = F.cross_entropy(logits.view(-1, self.V), sid.view(-1), reduction="none").view(B, Q)
            wr = w
            if acc_weight:                                            # expected-acceptance weighting: P(all earlier slots right)
                with torch.no_grad():
                    p_ok = torch.exp(-ce.detach()).view(B, A, k1)
                    j0 = 0 if self.shift else 1
                    reach = torch.ones_like(p_ok)
                    reach[..., j0 + 1:] = torch.cumprod(p_ok[..., j0:-1], -1)
                    wr = (valid.float() * (torch.ones_like(is_clean) if self.shift else ~is_clean).float()).view(B, A, k1)
                    wr = (wr * (0.25 + 0.75 * reach)).view(B, Q)
            total = total + (ce * wr).sum() / wr.sum().clamp(min=1)
            state = h.detach()
            with torch.no_grad():
                acc = ((logits.argmax(-1) == sid).float() * (w > 0)).sum() / (w > 0).sum().clamp(min=1)
            logs[f"acc_r{i}"] = float(acc)
        logs["rounds"] = r
        total = total / r
        if self.flowmap:
            psd, logs["acc_fm01"] = self.psd_loss(eps, onehot, sid, is_clean, slot_pos, ctx_kv, mask, w, B, A, k1, gen)
            logs["psd"] = float(psd)
            total = total + psd
        return total, logs

    def psd_loss(self, eps, onehot, sid, is_clean, slot_pos, ctx_kv, mask, w, B, A, k1, gen):
        """Flow-map progressive self-distillation (FMLM+ / Boffi et al.): per anchor a jump s -> t (s = 0 w.p. 1/2, t = 1
        w.p. 1/2, else uniform), midpoint u; teacher (no grad, current weights) = two half jumps,
        D_st* = lam D_su + (1 - lam) D_ut(X_{s->u}), lam = (1 - t) / (2 (1 - u)) (exact composition of the one-hot flow
        X_{s->t} = x_s + (t - s) / (1 - s) (D_st - x_s)); student soft-CE.  No carried state (the one-loop draft has none).
        Also returns the argmax accuracy of the one-jump draft D_01(eps) on the s=0, t=1 anchors."""
        dev = eps.device; Q = A * k1
        rnd = lambda: torch.rand(B, A, generator=gen).to(dev)
        s = torch.where(rnd() < 0.5, torch.zeros(B, A, device=dev), rnd())
        t = torch.where(rnd() < 0.5, torch.ones(B, A, device=dev), s + (1 - s) * rnd())
        u = (s + t) / 2
        ex = lambda z: z[..., None].expand(B, A, k1).reshape(B, Q)
        s, t, u = ex(s), ex(t), ex(u)
        one = torch.ones_like(s)
        cl = lambda z: torch.where(is_clean, one, z)
        x_s = (1 - s)[..., None] * eps + s[..., None] * onehot
        with torch.no_grad():
            lg, _ = self.round(self.slot_inputs(x_s, sid, is_clean, cl(s), None, cl(u)), slot_pos, ctx_kv, mask)
            D_su = lg.softmax(-1)
            x_u = x_s + ((u - s) / (1 - s))[..., None] * (D_su - x_s)
            lg, _ = self.round(self.slot_inputs(x_u, sid, is_clean, cl(u), None, cl(t)), slot_pos, ctx_kv, mask)
            lam = ((1 - t) / (2 * (1 - u)))[..., None]
            target = lam * D_su + (1 - lam) * lg.softmax(-1)
            del D_su, lg, x_u
        logits, _ = self.round(self.slot_inputs(x_s, sid, is_clean, cl(s), None, cl(t)), slot_pos, ctx_kv, mask)
        ce = -(target * logits.log_softmax(-1)).sum(-1)
        loss = (ce * w).sum() / w.sum().clamp(min=1)
        with torch.no_grad():
            m01 = (w > 0) & (s == 0) & (t == 1)
            acc = float(((logits.argmax(-1) == sid) & m01).sum() / m01.sum().clamp(min=1))
        return loss, acc

    # ---- inference --------------------------------------------------------------------------------------------
    @torch.no_grad()
    def draft(self, ctx_kv, cut, y_m, n_rounds=2, gen=None):
        """Single sequence: ctx_kv from forward_context (context positions < cut used), y_m the just-emitted token.
        Euler on t_i = i/n from x_0 = sigma*eps; returns the k drafted ids (list)."""
        dev = self.ouro.device; k1 = self.k + 1
        T = ctx_kv[0][0].shape[2]
        mask = self.masks(torch.tensor([[cut]], device=dev), T, 1, k1, dev)
        sid = torch.zeros(1, k1, dtype=torch.long, device=dev); sid[0, 0] = y_m
        is_clean = torch.zeros(1, k1, dtype=torch.bool, device=dev); is_clean[0, 0] = True
        slot_pos = (cut + torch.arange(k1, device=dev))[None]
        x = torch.randn(1, k1, self.V, device=dev, generator=gen) * self.sigma
        state = None
        if self.feedback:                                            # trained refinement: mask round, then token-feedback rounds
            e = self.slot_inputs(x, sid, is_clean, torch.where(is_clean, 1.0, 0.0), None)
            for i in range(n_rounds):
                logits, h = self.round(e, slot_pos, ctx_kv, mask)
                if i < n_rounds - 1:
                    e = self.feedback_inputs(sid[:, 0], logits.argmax(-1), k1, h)
            return logits[0, (0 if self.shift else 1):].argmax(-1).tolist()
        for i in range(n_rounds):
            t0, t1 = i / n_rounds, (i + 1) / n_rounds
            t = torch.full((1, k1), t0, device=dev); t_in = torch.where(is_clean, torch.ones_like(t), t)
            t_end = torch.where(is_clean, torch.ones_like(t), torch.full_like(t, t1))
            e = self.slot_inputs(x, sid, is_clean, t_in, state, t_end)
            logits, h = self.round(e, slot_pos, ctx_kv, mask)
            p = logits.softmax(-1)
            if i < n_rounds - 1:
                x = x + (t1 - t0) / (1 - t0) * (p - x)
            state = h
        if self.markov_rank:
            return self.markov_decode(logits, sid[:, 0])[0].tolist()
        return logits[0, (0 if self.shift else 1):].argmax(-1).tolist()


@torch.no_grad()
def draft_many(dr, ctx_kv, cut, y_m, n_rounds=2, S=4, sample_readout=False, temperature=1.0, gen=None):
    """S candidate drafts in one batched pass (draft-tree candidates): each candidate has its own noise sample x_0
    (noise drafter) and, with sample_readout, its tokens are sampled from the last round's slot distributions instead of
    argmax (gives the mask drafter diversity too).  Returns a list of S drafts (lists of k ids)."""
    dev = dr.ouro.device; k1 = dr.k + 1
    T = ctx_kv[0][0].shape[2]
    kv = [(k.expand(S, -1, -1, -1), v.expand(S, -1, -1, -1)) for k, v in ctx_kv]
    mask = dr.masks(torch.full((S, 1), cut, device=dev), T, 1, k1, dev)
    sid = torch.zeros(S, k1, dtype=torch.long, device=dev); sid[:, 0] = y_m
    is_clean = torch.zeros(S, k1, dtype=torch.bool, device=dev); is_clean[:, 0] = True
    slot_pos = (cut + torch.arange(k1, device=dev))[None].expand(S, k1)
    x = torch.randn(S, k1, dr.V, device=dev, generator=gen) * dr.sigma
    state = None
    for i in range(n_rounds):
        t0, t1 = i / n_rounds, (i + 1) / n_rounds
        t = torch.full((S, k1), t0, device=dev); t_in = torch.where(is_clean, torch.ones_like(t), t)
        t_end = torch.where(is_clean, torch.ones_like(t), torch.full_like(t, t1))
        e = dr.slot_inputs(x, sid, is_clean, t_in, state, t_end)
        logits, h = dr.round(e, slot_pos, kv, mask)
        p = logits.softmax(-1)
        if i < n_rounds - 1:
            x = x + (t1 - t0) / (1 - t0) * (p - x)
        state = h
    lg = logits[:, 1:]
    if sample_readout:
        pr = (lg / temperature).softmax(-1)
        toks = torch.multinomial(pr.reshape(-1, dr.V), 1, generator=gen).view(S, dr.k)
        toks[0] = lg[0].argmax(-1)             # candidate 0 = the plain argmax draft
    else:
        toks = lg.argmax(-1)
    return toks.tolist()


# ---- commit-and-re-noise refinement (FMLM+ posterior refinement on the draft block, looped: state carried) -------------
def loss_commit(dr, seqs, prompt_lens, anchors_per_seq=16, rounds_max=4, gen=None, prefix_decay=4.0):
    """Rounds i=1..r with nested committed sets: slot j (1..k) is committed (clean, its TRUE token) at round i if its
    priority pi_j < f_i, f_1 < ... < f_r sorted uniform (f_1 may be 0).  Uncommitted slots get FRESH pure noise (t=0) or the
    mask embedding (dr.use_noise False).  State carried (detached); CE on the uncommitted slots of every round."""
    dev = dr.ouro.device; k1 = dr.k + 1; A = anchors_per_seq
    ids, slot_ids, slot_valid, cut = dr.build_batch(seqs, prompt_lens, A, gen)
    ids, slot_ids, slot_valid, cut = ids.to(dev), slot_ids.to(dev), slot_valid.to(dev), cut.to(dev)
    B, T = ids.shape
    _, ctx_kv = dr.ouro.forward_context(ids, keep_kv_loop=dr.ctx_loop, logits=False)
    mask = dr.masks(cut, T, A, k1, dev)
    Q = A * k1
    sid = slot_ids.view(B, Q); valid = slot_valid.view(B, Q)
    slot_pos = (cut[:, :, None] + torch.arange(k1, device=dev)[None, None]).view(B, Q)
    r = int(torch.randint(1, rounds_max + 1, (1,), generator=gen))
    f = torch.sort(torch.rand(B, A, r, generator=gen), dim=-1).values
    f[..., 0] = torch.where(torch.rand(B, A, generator=gen) < 0.5, torch.zeros_like(f[..., 0]), f[..., 0])
    pri = torch.rand(B, A, k1, generator=gen); pri[..., 0] = -1.0          # slot 0 (y_m) always clean
    f, pri = f.to(dev), pri.to(dev)
    w0 = torch.exp(-(torch.arange(k1, device=dev).float() - 1).clamp(min=0) / prefix_decay)[None, None].expand(B, A, k1)
    state = None; total = 0.0; logs = {}
    for i in range(r):
        clean = (pri < f[..., i:i + 1]).view(B, Q)
        w = (w0.reshape(B, Q) * valid.float() * (~clean).float())
        x = torch.randn(B, Q, dr.V, device=dev) * dr.sigma
        t_in = clean.float()
        e = dr.slot_inputs(x, sid, clean, t_in, state)
        logits, h = dr.round(e, slot_pos, ctx_kv, mask)
        ce = torch.nn.functional.cross_entropy(logits.view(-1, dr.V), sid.view(-1), reduction="none").view(B, Q)
        total = total + (ce * w).sum() / w.sum().clamp(min=1)
        state = h.detach()
        with torch.no_grad():
            logs[f"acc_r{i}"] = float(((logits.argmax(-1) == sid).float() * (w > 0)).sum() / (w > 0).sum().clamp(min=1))
    logs["rounds"] = r
    return total / r, logs


@torch.no_grad()
def draft_commit(dr, ctx_kv, cut, y_m, n_rounds=3, thr=0.9, gen=None):
    """Each round: predict the open slots, commit the most confident ones (budget ceil(open/(rounds left)) plus any with
    p >= thr), re-noise the rest (fresh noise, or mask), carry the state.  Returns the k drafted ids."""
    dev = dr.ouro.device; k1 = dr.k + 1
    T = ctx_kv[0][0].shape[2]
    mask = dr.masks(torch.tensor([[cut]], device=dev), T, 1, k1, dev)
    sid = torch.zeros(1, k1, dtype=torch.long, device=dev); sid[0, 0] = y_m
    clean = torch.zeros(1, k1, dtype=torch.bool, device=dev); clean[0, 0] = True
    slot_pos = (cut + torch.arange(k1, device=dev))[None]
    state = None
    for i in range(n_rounds):
        x = torch.randn(1, k1, dr.V, device=dev, generator=gen) * dr.sigma
        e = dr.slot_inputs(x, sid, clean, clean.float(), state)
        logits, h = dr.round(e, slot_pos, ctx_kv, mask)
        conf, pred = logits.softmax(-1).max(-1)
        open_ = ~clean[0]
        n_open = int(open_.sum())
        if n_open == 0:
            break
        if i == n_rounds - 1:
            sid[0, open_] = pred[0, open_]
            clean[0, open_] = True
        else:
            budget = math.ceil(n_open / (n_rounds - i))
            c = conf[0].masked_fill(~open_, -1.0)
            top = torch.topk(c, budget).indices
            commit = torch.zeros_like(open_); commit[top] = True
            commit |= open_ & (conf[0] >= thr)
            sid[0, commit] = pred[0, commit]; clean[0, commit] = True
        state = h
    return sid[0, 1:].tolist()


# ---- shared-noise (Gumbel-max) coupled drafting for speculative SAMPLING -----------------------------------------------
# The target samples y_j = argmax(log p_j + g_j) with a Gumbel stream g shared with the drafter (lossless: exact sample of
# p_j whatever the drafts).  Round 1 drafts every slot (mask input); round 2 (coupled=True) additionally receives the flow
# state x = softmax(l^1 + g) -- the Gumbel-relaxed COUPLED sample of round 1, a point of the vocab simplex -- so all slots
# re-draft conditioned on one joint noise-selected sample.  Decision d_j = argmax(l_j + g_j); loss = CE(l + g, y) with g
# drawn from its posterior given the target's token (the drafter learns the map noise -> the target's sample).
def posterior_gumbel(a, y):
    """a (..., V) target log-probs, y (...) observed argmax of a + g.  Returns g ~ Gumbel(0)^V conditioned on
    argmax(a + g) = y (top-down construction: max ~ Gumbel(logsumexp a), others truncated below it)."""
    U = lambda *s: torch.rand(*s, device=a.device).clamp(1e-20, 1 - 1e-7)
    M = torch.logsumexp(a, -1, keepdim=True) - torch.log(-torch.log(U(*a.shape[:-1], 1)))
    G = a - torch.log(-torch.log(U(*a.shape)))
    Gt = -torch.logaddexp(-M, -G)
    Gt.scatter_(-1, y[..., None], M)
    return Gt - a


def loss_gumbel(dr, seqs, prompt_lens, anchors_per_seq=16, coupled=True, gen=None, prefix_decay=4.0, temp=1.0):
    dev = dr.ouro.device; k1 = dr.k + 1; A = anchors_per_seq
    ids, slot_ids, slot_valid, cut = dr.build_batch(seqs, prompt_lens, A, gen)
    ids, slot_ids, slot_valid, cut = ids.to(dev), slot_ids.to(dev), slot_valid.to(dev), cut.to(dev)
    B, T = ids.shape
    full_logits, ctx_kv = dr.ouro.forward_context(ids, keep_kv_loop=dr.ctx_loop, logits=True)
    mask = dr.masks(cut, T, A, k1, dev)
    Q = A * k1
    sid = slot_ids.view(B, Q); valid = slot_valid.view(B, Q)
    slot_pos = (cut[:, :, None] + torch.arange(k1, device=dev)[None, None]).view(B, Q)
    is_clean = torch.zeros(B, A, k1, dtype=torch.bool, device=dev); is_clean[:, :, 0] = True
    is_clean = is_clean.view(B, Q)
    with torch.no_grad():                                   # target distribution of slot j = logits at position cut+j-1
        src = (slot_pos - 1).clamp(min=0, max=T - 1)
        a = torch.gather(full_logits, 1, src[..., None].expand(B, Q, full_logits.shape[-1])).float() / temp
        del full_logits
        g = posterior_gumbel(a.log_softmax(-1), sid)
        del a
    w = torch.exp(-(torch.arange(k1, device=dev).float() - 1).clamp(min=0) / prefix_decay)
    w = w[None, None].expand(B, A, k1).reshape(B, Q) * valid.float() * (~is_clean).float()
    zero = torch.zeros(B, Q, device=dev)
    E = dr.ouro.embed_tokens.weight
    # round 1: mask input at t = 0
    e = dr.slot_inputs(None, sid, is_clean, torch.where(is_clean, zero + 1, zero), None)
    l1, h1 = dr.round(e, slot_pos, ctx_kv, mask)
    ce1 = F.cross_entropy((l1 + g).view(-1, dr.V), sid.view(-1), reduction="none").view(B, Q)
    # round 2: state + (coupled) relaxed joint sample
    e = dr.slot_inputs(None, sid, is_clean, torch.where(is_clean, zero + 1, zero + 0.5), h1.detach())
    if coupled:
        s = (l1.detach() + g).softmax(-1)
        e = e + torch.where(is_clean[..., None], torch.zeros_like(e), (s.to(E.dtype) @ E))
    l2, _ = dr.round(e, slot_pos, ctx_kv, mask)
    ce2 = F.cross_entropy((l2 + g).view(-1, dr.V), sid.view(-1), reduction="none").view(B, Q)
    nw = w.sum().clamp(min=1)
    loss = ((ce1 * w).sum() + (ce2 * w).sum()) / (2 * nw)
    with torch.no_grad():
        m = w > 0
        logs = {"acc_r0": float((((l1 + g).argmax(-1) == sid) & m).sum() / m.sum()),
                "acc_r1": float((((l2 + g).argmax(-1) == sid) & m).sum() / m.sum())}
    return loss, logs


@torch.no_grad()
def draft_gumbel(dr, ctx_kv, cut, y_m, g, n_rounds=2, coupled=True):
    """g (k, V): the shared Gumbel noise of the k draft positions.  Returns the last round's slot logits (k, V); the draft
    is argmax(logits + g)."""
    dev = dr.ouro.device; k1 = dr.k + 1
    T = ctx_kv[0][0].shape[2]
    mask = dr.masks(torch.tensor([[cut]], device=dev), T, 1, k1, dev)
    sid = torch.zeros(1, k1, dtype=torch.long, device=dev); sid[0, 0] = y_m
    is_clean = torch.zeros(1, k1, dtype=torch.bool, device=dev); is_clean[0, 0] = True
    slot_pos = (cut + torch.arange(k1, device=dev))[None]
    zero = torch.zeros(1, k1, device=dev)
    E = dr.ouro.embed_tokens.weight
    gg = torch.cat([torch.zeros_like(g[:1]), g])[None]       # slot 0 has no noise
    e = dr.slot_inputs(None, sid, is_clean, torch.where(is_clean, zero + 1, zero), None)
    l, h = dr.round(e, slot_pos, ctx_kv, mask)
    for i in range(1, n_rounds):
        e = dr.slot_inputs(None, sid, is_clean, torch.where(is_clean, zero + 1, zero + 0.5), h)
        if coupled:
            s = (l + gg).softmax(-1)
            e = e + torch.where(is_clean[..., None], torch.zeros_like(e), (s.to(E.dtype) @ E))
        l, h = dr.round(e, slot_pos, ctx_kv, mask)
    return l[0, 1:]
