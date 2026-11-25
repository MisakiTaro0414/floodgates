"""Lossless speculative SAMPLING in the batch-1 CUDA-graph engine (raven_engine.RavenEngine, unchanged): a subclass that replaces
the greedy pieces by sampling, with every sampling / acceptance op ON THE DEVICE inside the captured graphs (still one host sync
per cycle, raven_engine._result; the adaptive drafter keeps its own per-pass host check).
  target p = top-p(softmax(logits / T)) (eval_sample_raven.target_probs; T = 0: exact greedy = one-hot on the argmax, lowest
    index on ties as torch.argmax / the greedy engine).
  plain step: y ~ p (also the first token after the prefill).
  proposals (q_mode):
    "sample" (our drafter; DAS-Wave sampled): the draft graph(s) store the FINAL pass's slot logits (shift: logits[0:], else
      logits[1:]; DAS-Wave: the last round's readouts 0..gamma-1); the verification graph turns them into
      q_j = softmax(logits_j / T) (top-p truncated with the same threshold when q_topp, eval_sample_raven Q_TOPP=1), draws
      d_j ~ q_j independently, writes them as the verification inputs and runs the target.  Feedback passes of our drafter are
      the base engine's argmax feedback (as in training; eval_sample_raven FB=argmax), the adaptive rule (ADAPT_MAX, conf rule
      on the drafter's own argmax confidences) is unchanged; q is fixed before the d's are drawn, so the output is exact.
    "argmax" (DAS-Wave argmax): the base engine's argmax drafts with one-hot q (accept prob = p_j(d_j)).
  verification (verify()): Leviathan et al. / Chen et al.: accept d_j w.p. min(1, p_j(d_j) / q_j(d_j)) sequentially (cumprod of
    the accept flags), at the first rejection emit a draw from norm(max(p_j - q_j, 0)), if all drafts are accepted a bonus draw
    from p after the last draft; res = [acc+1, d_1..d_acc, e], next y = e.
Draws: the exponential race argmax_i p_i / E_i (E_i ~ Exp(1) from torch.rand, the default CUDA generator, which PyTorch's graph
capture handles: every replay advances the philox offset) -- no torch.multinomial / one_hot (validation syncs, device asserts),
so a draw never picks a zero-probability token and never syncs.  torch.cuda.manual_seed(s) before a generation makes it
reproducible (the replays read the generator's seed / offset).
top-p (nucleus_tie): keep every token whose probability is >= the smallest probability of the sorted-prefix nucleus (tie-
symmetric: a function of the row's values, identical in the plain step, the verification and the proposals), with the
threshold found among the TOPP_CAND (4096) most likely tokens (topk instead of a full-vocabulary sort); exact whenever the
top-4096 mass reaches top_p, self.uncov counts the rows where it did not.  TOPP_CAND=-1 = the first version of the runs
(eval_sample_raven's sort-based nucleus(), which keeps a kernel-dependent part of a tie group at the boundary).
Requires FUSED=1 (the logits are taken from RavenEngine._lm).  keep_p=True also stores the target p rows of every step in
self.pbuf (tests only)."""
import os, time
import torch
import torch.nn.functional as F
from floodgates.engines.raven_engine import RavenEngine


def nucleus(p, top_p):
    """p (..., V) restricted to its top-p nucleus (smallest prefix of the sorted probs reaching top_p), renormalised
    (= eval_sample_raven.nucleus)."""
    sp, si = p.sort(-1, descending=True)
    keep = (sp.cumsum(-1) - sp) < top_p
    p = torch.zeros_like(p).scatter_(-1, si, sp * keep)
    return p / p.sum(-1, keepdim=True)


def one_hot_argmax(logits):
    return torch.zeros(logits.shape, dtype=torch.float32, device=logits.device).scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)


def probs(logits, temp, top_p):
    """eval_sample_raven.target_probs / proposal_probs: (..., V) logits -> float32 probabilities."""
    if temp == 0:
        return one_hot_argmax(logits)
    p = (logits.float() / temp).softmax(-1)
    return nucleus(p, top_p) if top_p < 1.0 else p


def nucleus_tie(p, top_p, K=0):
    """Tie-symmetric top-p nucleus: keep x iff p_x >= t, t = the smallest probability in the sorted-prefix nucleus (equivalently:
    keep x iff the mass strictly above p_x is < top_p).  Unlike nucleus(), whose boundary keeps an arbitrary part of a group of
    equal probabilities (bf16 logits tie often) depending on the sort kernel's tie order, this depends on the row's values only,
    so the plain step (1 row) and the verification (n+1 rows) get the same target.  t is found among the K most likely tokens
    (topk, ~0.15 ms vs ~0.65 ms for a full sort on 26 x 128256, A100); exact whenever the top-K mass reaches top_p, else the
    row keeps everything >= the K-th probability, flagged in `uncovered` (K = 0: full sort, never uncovered).
    Returns (renormalised p, uncovered (N,) bool or None)."""
    sp = p.topk(min(K, p.shape[-1]), -1).values if K else p.sort(-1, descending=True).values
    cs = sp.cumsum(-1)
    m = ((cs - sp) < top_p).sum(-1, keepdim=True)                       # sorted-prefix nucleus size (>= 1)
    t = sp.gather(-1, m - 1)
    q = p * (p >= t)
    return q / q.sum(-1, keepdim=True), ((cs[..., -1] < top_p) if K else None)


def probs_cand(logits, temp, top_p, K):
    """The engine's target / proposal probabilities: probs() with the tie-symmetric nucleus searched among the top K tokens
    (nucleus_tie).  Returns (p (N, V) float32, uncovered (N,) bool or None)."""
    if temp == 0:
        return one_hot_argmax(logits), None
    p = (logits.float() / temp).softmax(-1)
    if top_p >= 1.0:
        return p, None
    return nucleus_tie(p, top_p, K)


def sample(p):
    """One draw per row of the non-negative (possibly unnormalised) rows p (N, V) -> (N,) long, P(i) = p_i / sum(p):
    argmax_i p_i / E_i with E_i = -log(U_i) ~ Exp(1).  Zero entries score 0 and are never chosen (E_i in [6e-8, 69])."""
    e = -torch.log(torch.rand(p.shape, device=p.device, dtype=torch.float32).clamp_(min=1e-30))
    return (p / e).argmax(-1)


def verify(p, d, q=None, u=None):
    """Rejection sampling for one cycle (vectorised, no host sync).  p (n+1, V) target probabilities at the inputs [y, d_1..d_n]
    (row j = distribution of the token after input j); d (n,) drafts; q (n, V) proposals or None = one-hot on d (argmax drafts);
    u (n,) uniforms (default: torch.rand).  Returns (acc, e): number of accepted drafts (0-d long, 0..n) and the emitted token
    (0-d long): the residual draw at the first rejection, or the bonus draw from p[n]."""
    n = d.shape[0]
    pd = p[:n].gather(1, d[:, None])[:, 0]
    qd = torch.ones_like(pd) if q is None else q.gather(1, d[:, None])[:, 0]
    ratio = (pd / qd.clamp(min=1e-20)).clamp(max=1.0)
    if u is None:
        u = torch.rand(n, device=p.device, dtype=torch.float32)
    acc = (u < ratio).long().cumprod(0).sum()                            # accepted prefix length
    pa = p.index_select(0, acc[None])[0]                                 # p at the first rejection / the bonus row
    ia = acc.clamp(max=n - 1)[None]
    qa = torch.zeros_like(pa).scatter_(0, d.index_select(0, ia), 1.0) if q is None else q.index_select(0, ia)[0]
    qa = qa * (acc < n)                                                  # all accepted: plain p[n] (bonus)
    resid = (pa - qa).clamp(min=0); rs = resid.sum()
    dist = torch.where(rs > 0, resid / rs.clamp(min=1e-30), pa)
    return acc, sample(dist[None])[0]


class RavenSampleEngine(RavenEngine):
    def __init__(self, m, dr=None, n_rounds=1, temp=1.0, top_p=0.7, q_topp=True, keep_p=False, k_cand=None, **kw):
        """temp / top_p: the target's sampling setting; q_topp: proposals restricted to their own top_p nucleus;
        k_cand (env TOPP_CAND, default 4096): tie-symmetric top-p nucleus (nucleus_tie) searched among the k_cand most likely
        tokens in every step (plain, verification, proposals, prefill); 0 = the same nucleus via a full sort; -1 = the first
        version (sort-based nucleus(), tie order up to the kernel).  self.uncov counts the rows whose top-k_cand mass was below
        top_p ([plain + prefill, verification target rows, proposal rows]; 0 = exact everywhere);
        das={"mode": "wave", "J", "k", "gamma", "q": "argmax" | "sample"} for DAS-Wave (hybrid / sequential modes unsupported)."""
        das = kw.get("das")
        assert das is None or das["mode"] == "wave", "only DAS-Wave and the drafter-seeded window (init=loopahead) are supported"
        super().__init__(m, dr, n_rounds, **kw)
        assert self.fused, "the sampling engine needs FUSED=1 (logits are taken from RavenEngine._lm)"
        self.temp, self.top_p, self.q_topp, self.keep_p = float(temp), float(top_p), bool(q_topp), keep_p
        self.q_mode = das.get("q", "argmax") if das is not None else "sample"
        V = m.lm_head.weight.shape[0]; self.vocab = V                       # (self.V is the base V-cache dict)
        self.off = (0 if dr.shift else 1) if (dr is not None and das is None) else 0   # first draft row of the final-pass logits
        self.spec = dr is not None or das is not None
        if self.spec and self.q_mode == "sample":
            self.dlogits = torch.zeros(self.nd, V, device=self.dev, dtype=torch.float32)   # final-pass draft logits
            self.state.append(self.dlogits)
        if keep_p:
            self.pbuf = torch.zeros(self.nd + 1, V, device=self.dev, dtype=torch.float32)
        self._tmode = "plain"
        self.k_cand = int(os.environ.get("TOPP_CAND", 4096)) if k_cand is None else int(k_cand)
        self.uncov = torch.zeros(3, dtype=torch.long, device=self.dev)
        self.state.append(self.uncov)

    def _probs(self, logits, top_p, which):
        if self.k_cand < 0:                                                   # TOPP_CAND=-1: the first version (sort-based nucleus())
            return probs(logits, self.temp, top_p)
        p, unc = probs_cand(logits, self.temp, top_p, self.k_cand)
        if unc is not None:
            self.uncov[which] += unc.sum()
        return p

    # ---- hooks into the base graph bodies ------------------------------------------------------------------------------
    def _lm(self, h):
        out = super()._lm(h)
        self._last_lm = out                                                    # the latest lm_head output of this body
        return out

    def _store_draft_logits(self):
        if self.q_mode == "sample":
            self.dlogits.copy_(self._last_lm[self.off:self.off + self.nd])

    def _draft(self, Lb):
        super()._draft(Lb)
        if self.das is None:                                                  # seeded window: q comes from the last round
            self._store_draft_logits()

    def _draft_first(self, Lb):
        super()._draft_first(Lb); self._store_draft_logits()

    def _draft_refine(self, Lb):
        super()._draft_refine(Lb); self._store_draft_logits()

    def _draft_wave(self, Lb, phase="full"):
        super()._draft_wave(Lb, phase); self._store_draft_logits()            # readouts 0..gamma-1 = drafts 1..gamma

    def _target(self, Q, Lb, mode):
        if mode == "spec" and self.q_mode == "sample":                      # draw the drafts from the final-pass proposals
            self._q = self._probs(self.dlogits, self.top_p if self.q_topp else 1.0, 2)
            self.ids[0, 1:Q].copy_(sample(self._q))
        self._tmode = mode
        super()._target(Q, Lb, "spec")                                       # base forward (+ argmax pred), then self._accept()

    def _accept(self):
        logits = self._last_lm
        if self._tmode == "plain":
            p = self._probs(logits[:1], self.top_p, 0)
            if self.keep_p:
                self.pbuf[:1].copy_(p)
            t = sample(p)
            self.res[:1].fill_(1); self.res[1:2].copy_(t); self.ids[0, :1].copy_(t); self.pos0 += 1
            return
        nd = self.nd; Q = nd + 1
        p = self._probs(logits[:Q], self.top_p, 1)
        if self.keep_p:
            self.pbuf[:Q].copy_(p)
        d = self.ids[0, 1:Q]
        acc, e = verify(p, d, self._q if self.q_mode == "sample" else None)
        out = torch.cat([d, d[:1]]).scatter_(0, acc[None], e[None])        # [d_1..d_acc, e, ...]
        self.res[:1].copy_(acc[None] + 1); self.res[1:Q + 1].copy_(out)
        e1 = e[None]
        self.ids[0, :1].copy_(e1); self.sid[0, :1].copy_(e1); self.sid_dr[0, :1].copy_(e1)
        self.pos0 += acc + 1

    # ---- prefill: as RavenEngine.prefill, the first token sampled from p ------------------------------------------------
    @torch.no_grad()
    def prefill(self, prompt):
        m = self.m; P = len(prompt)
        assert P + 1 < self.tmax
        ids = torch.tensor([prompt], device=self.dev)
        pos = torch.arange(P, device=self.dev)[None]
        cos, sin = m.rope(pos)

        def attn_for(key):
            def attn(q, k, v):
                self.K[key][:, :, :P] = k; self.V[key][:, :, :P] = v
                return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            return attn
        e = m.wte(ids) * m.emb_scale
        for li, b in enumerate(m.prelude):
            e = b(e, cos, sin, attn_for(("p", li)))
        s = m.init_table(self.tmax)[pos[0]][None]
        for it in range(m.R):
            s = m.adapter(torch.cat([s, e], -1))
            for li, b in enumerate(m.core):
                s = b(s, cos, sin, attn_for(("c", it, li)))
        s = m.ln_f(s) if getattr(m, "pre_coda_norm", False) else s
        for li, b in enumerate(m.coda):
            s = b(s, cos, sin, attn_for(("o", li)))
        return int(sample(self._probs(m.lm_head(m.ln_f(s[0, -1:])), self.top_p, 0))[0])

    # ---- tests: one cycle from a fixed state -------------------------------------------------------------------------
    @torch.no_grad()
    def cycle_from(self, cut, y):
        """One draft + verification cycle with y (not yet cached) at position cut over the cached context rows < cut (rows >= cut
        are overwritten as in decoding); returns the emitted tokens (positions cut+1 ..)."""
        self.ids[0, 0] = y; self.pos0.fill_(cut)
        if self.spec:
            self.sid[0, 0] = y; self.sid_dr[0, 0] = y
            need = cut + self.nd + 1
            if self.adapt_max:
                self._draft_adaptive(need)
            else:
                self._run("draft", need)
            self._run("spec", need)
        else:
            self._run("plain", cut + 1)
        return self._result()[1]

    @torch.no_grad()
    def plain_from(self, cut, y):
        """One plain sampling step at position cut (y not yet cached); returns (token, p row) (keep_p engines)."""
        self.ids[0, 0] = y; self.pos0.fill_(cut)
        self._run("plain", cut + 1)
        _, new = self._result()
        return new[0], self.pbuf[0].clone()
