"""Lossless speculative SAMPLING in the batch-1 Ouro CUDA-graph engine (engine.Engine, unchanged): a subclass that replaces the
greedy pieces by sampling, every sampling / acceptance op ON THE DEVICE inside the captured graphs (still one host read-back per
cycle, launched one cycle ahead with PIPE=1 exactly as the greedy engine).  Mirror of raven_engine_sample.py (Raven/Huginn).
  target p = top-p(softmax(logits / T)) of the final loop (T = 0: exact greedy = one-hot on the argmax, lowest index on ties as
    torch.argmax / the greedy engine).
  plain step: y ~ p (also the first token after the prefill).
  proposals (q_mode):
    "sample" (default): the draft graph stores the drafter's slot logits of draft slots 1..k (final pass); the verification graph
      turns them into q_j = softmax(logits_j / T) (restricted to their own top-p nucleus when q_topp), draws d_j ~ q_j
      independently (q_j is fixed before any d is drawn and does not depend on d_<j: a valid proposal, the output is exact),
      writes them as the verification inputs and runs the target.
    "argmax": the base engine's argmax drafts with one-hot q (accept prob = p_j(d_j)).
  verification (verify()): Leviathan et al. / Chen et al.: accept d_j w.p. min(1, p_j(d_j) / q_j(d_j)) sequentially (cumprod of
    the accept flags), at the first rejection emit a draw from norm(max(p_j - q_j, 0)), if all drafts are accepted a bonus draw
    from p after the last draft; res = [acc+1, d_1..d_acc, e], next y = e.  K/V rows of y and the accepted drafts stay in the
    cache (as in the greedy engine, rejected rows are overwritten later).
Draws: the exponential race argmax_i p_i / E_i (E_i ~ Exp(1) from torch.rand on the default CUDA generator; graph replays
advance its philox offset) -- no torch.multinomial (host syncs / device asserts); a draw never picks a zero-probability token.
torch.cuda.manual_seed(s) before a generation makes it reproducible.
top-p (nucleus_tie): keep every token whose probability is >= the smallest probability of the sorted-prefix nucleus (tie-
symmetric, a function of the row's values only: identical in the plain step, the verification rows and the proposals; this is
also the rule of LoopSpec's sampler, sglang_recurrent/loopspec/sampling/distributions.py), the threshold searched among the
TOPP_CAND (default 4096) most likely tokens; exact whenever the top-TOPP_CAND mass reaches top_p, self.uncov counts the rows where
it did not ([plain + prefill, verification target rows, proposal rows]); TOPP_CAND=0: full sort (never uncovered).
The sampling helpers (nucleus_tie, probs_cand, sample, verify) live in sampling_ops.py: the same code as raven_engine_sample.py's
(copied, not imported, so this file does not depend on the Raven stack).
How the logits are reached without touching engine.py: self.o is replaced by a thin proxy of the OuroMin model whose lm_head
records its latest output (bf16 logits) in self._last_lm; every graph body reads it right after the base forward.
Restrictions: S = 1 (no draft trees), no adaptive passes (ADAPT_MAX unset), non-shift drafters (all Ouro drafters).
keep_p=True also stores the target p rows of every step in self.pbuf (tests only)."""
import math, os, weakref
import torch
from floodgates.engines.engine import Engine
from floodgates.engines.sampling_ops import nucleus, one_hot_argmax, probs, nucleus_tie, probs_cand, sample, verify


class _LMTap:
    """Stands in for the OuroMin model inside the engine: every attribute is the model's, except lm_head, whose output (bf16
    logits) is also recorded in sink._last_lm (weak reference: no cycle engine <-> proxy)."""
    def __init__(self, model, sink):
        object.__setattr__(self, "_m", model)
        object.__setattr__(self, "_sink", weakref.ref(sink))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_m"), name)

    def lm_head(self, x):
        out = self._m.lm_head(x)
        self._sink()._last_lm = out
        return out


class OuroSampleEngine(Engine):
    def __init__(self, ouro, dr=None, n_rounds=1, temp=1.0, top_p=0.7, q_topp=True, q_mode="sample", keep_p=False, k_cand=None, **kw):
        """temp / top_p: the target's sampling setting (temp 0 = exact greedy); q_topp: proposals restricted to their own top-p
        nucleus; q_mode: "sample" (d_j ~ q_j = drafter slot softmax) | "argmax" (argmax drafts, one-hot q); k_cand (env TOPP_CAND,
        default 4096): nucleus threshold searched among the k_cand most likely tokens (0 = full sort)."""
        super().__init__(ouro, dr, n_rounds, **kw)
        assert self.S == 1, "draft trees (S > 1) are greedy-only"
        assert not self.adapt_max, "adaptive passes are not supported in the sampling engine (unset ADAPT_MAX)"
        assert dr is None or not getattr(dr, "shift", False), "the Ouro engine has no shift drafters"
        assert q_mode in ("sample", "argmax")
        self.o = _LMTap(ouro, self)
        self.temp, self.top_p, self.q_topp, self.q_mode, self.keep_p = float(temp), float(top_p), bool(q_topp), q_mode, keep_p
        V = ouro.lm_head.weight.shape[0]; self.vocab = V
        self.spec = dr is not None
        self.k_cand = int(os.environ.get("TOPP_CAND", 4096)) if k_cand is None else int(k_cand)
        self.uncov = torch.zeros(3, dtype=torch.long, device=self.dev)
        if self.spec and self.q_mode == "sample":
            self.dlogits = torch.zeros(self.k, V, device=self.dev, dtype=torch.float32)   # drafter logits of slots 1..k
        if keep_p:
            self.pbuf = torch.zeros(self.k1, V, device=self.dev, dtype=torch.float32)
        self._tmode = "plain"; self._q = None; self._last_lm = None

    def _probs(self, logits, top_p, which):
        p, unc = probs_cand(logits, self.temp, top_p, self.k_cand)
        if unc is not None:
            self.uncov[which] += unc.sum()
        return p

    def _state(self):
        return super()._state() + ([self.dlogits] if self.spec and self.q_mode == "sample" else []) + [self.uncov]

    # ---- graph bodies ----------------------------------------------------------------------------------------------------
    def _draft(self, Lb):
        super()._draft(Lb)                                                    # base pass(es); ids[0, 1:] = argmax drafts
        if self.q_mode == "sample":
            self.dlogits.copy_(self._last_lm.view(self.k1, -1)[1:])           # final pass, draft slots 1..k

    def _target(self, Q, Lb, mode="prefill"):
        if mode == "prefill":
            return super()._target(Q, Lb, mode)
        self._tmode = mode
        if mode == "spec" and self.q_mode == "sample":                      # draw the drafts from the final-pass proposals
            self._q = self._probs(self.dlogits, self.top_p if self.q_topp else 1.0, 2)
            self.ids[0, 1:Q].copy_(sample(self._q))
        super()._target(Q, Lb, "spec")                                       # base forward (+ argmax pred), then self._accept()

    def _accept(self, S):
        logits = self._last_lm                                                 # (Q, V) final-loop logits of this step
        if self._tmode == "plain":
            p = self._probs(logits[:1], self.top_p, 0)
            if self.keep_p:
                self.pbuf[:1].copy_(p)
            t = sample(p)
            self.res[:1].fill_(1); self.res[1:2].copy_(t); self.ids[0, :1].copy_(t); self.pos0 += 1
            return
        k1 = self.k1
        p = self._probs(logits[:k1], self.top_p, 1)
        if self.keep_p:
            self.pbuf.copy_(p)
        d = self.ids[0, 1:k1]
        acc, e = verify(p, d, self._q if self.q_mode == "sample" else None)
        out = torch.cat([d, d[:1]]).scatter_(0, acc[None], e[None])        # [d_1..d_acc, e, ...]
        self.res[:1].copy_(acc[None] + 1); self.res[1:k1 + 1].copy_(out)
        e1 = e[None]
        self.ids[0, :1].copy_(e1); self.sid[0, :1].copy_(e1)
        self.pos0 += acc + 1

    # ---- prefill: as Engine.prefill, the first token sampled from p --------------------------------------------------------
    @torch.no_grad()
    def prefill(self, prompt):
        P = len(prompt)
        ids_save, pred_save = self.ids, self.pred
        self.ids = torch.tensor([prompt], device=self.dev); self.pos0.fill_(0)
        self.pred = torch.zeros(P, dtype=torch.long, device=self.dev)
        Lb = min(self.tmax, self.bucket * math.ceil(P / self.bucket))
        super()._target(P, Lb, "prefill")
        y = int(sample(self._probs(self._last_lm[P - 1:P], self.top_p, 0))[0])
        self.ids, self.pred = ids_save, pred_save
        return y

    # ---- tests: one step / cycle from a fixed state ---------------------------------------------------------------------
    def _result(self):
        return self._wait(self._readback())[1]

    @torch.no_grad()
    def cycle_from(self, cut, y):
        """One draft + verification cycle with y (not yet cached) at position cut over the cached context rows < cut (rows >= cut
        are overwritten as in decoding); returns the emitted tokens (positions cut+1 ..)."""
        self.ids[0, 0] = y; self.pos0.fill_(cut)
        if self.spec:
            self.sid[0, 0] = y
            self._run("d", 0, cut + self.k1)
            self._run("t", self.k1, cut + self.k1, "spec")
        else:
            self._run("t", 1, cut + 1, "plain")
        return self._result()

    @torch.no_grad()
    def plain_from(self, cut, y):
        """One plain sampling step at position cut (y not yet cached; writes its K/V row); returns (token, p row) (keep_p)."""
        self.ids[0, 0] = y; self.pos0.fill_(cut)
        self._run("t", 1, cut + 1, "plain")
        new = self._result()
        return new[0], (self.pbuf[0].clone() if self.keep_p else None)
