"""Lossless speculative SAMPLING for the training-free Ouro baselines (engine_baselines.OuroBaselineEngine) and the own-readout
LAL + refinement (engine_baselines_own.OuroOwnReadoutEngine), in the same batch-1 CUDA-graph engine: subclasses that replace the
greedy pieces by sampling, every sampling / acceptance op ON THE DEVICE inside the captured graphs (one host read-back per cycle,
launched one cycle ahead with PIPE=1 exactly as the greedy engines; PLD stays serial).  Mirror of engine_sample.py (LAL one pass)
and raven_engine_sample.py (Raven/Huginn incl. DAS-Wave); the sampling ops are sampling_ops.py's (imported).
  target p = top-p(softmax(logits / T)) of the final loop, tie-symmetric nucleus searched among the TOPP_CAND (4096) most likely
    tokens (sampling_ops.probs_cand); T = 0: exact greedy (one-hot on the argmax, lowest index on ties as torch.argmax / the greedy
    engines), so a T = 0 run reproduces the greedy engine token for token.
  plain step: y ~ p (also the first token after the prefill).
  proposals (q_mode):
    "argmax": the greedy engine's drafts (sequential DAS, Jacobi decoding, PLD's host lookup, or any wave mode) with one-hot q, i.e.
      d_j is accepted w.p. p_j(d_j) and the residual at a rejection is p_j without d_j (renormalised).  The drafts are a
      deterministic function of the verified prefix, so this is exact.
    "sample" (DAS-Wave, hybrid refinement; = bench_raven_sample.py DAS_Q=sample): the wave graph stores the LAST ROUND's readout
      logits of the window rows that propose d_1..d_gamma (DAS-Wave: rows 0..gamma-1, the readout of row i is the token of row i+1;
      own-readout refinement: rows 1..gamma, every row keeps its own readout); the verification graph turns them into
      q_j = softmax(logits_j / T) (restricted to their own top-p nucleus when q_topp), draws d_j ~ q_j independently (q is fixed
      before any d is drawn and does not depend on d_<j), writes them as the verification inputs and runs the target.  The window
      itself (rounds, carried states) is the greedy engine's, unchanged.
  verification (sampling_ops.verify): Leviathan et al. / Chen et al. rejection sampling over [y, d_1..d_gamma] in ONE target step
    (Q = gamma + 1); res = [acc+1, d_1..d_acc, e], next y = e (also into the drafter's slot 0 for the hybrid seed).
torch.cuda.manual_seed(s) before a generation makes it reproducible (graph replays advance the philox offset).
The logits are taken from engine_sample._LMTap (a proxy of the OuroMin model whose lm_head records its latest output), so neither
engine.py nor engine_baselines*.py is touched.  keep_p=True also stores the target p rows of every step in self.pbuf (tests only)."""
import math, os
import torch
from floodgates.engines.engine_baselines import OuroBaselineEngine
from floodgates.engines.engine_baselines_own import OuroOwnReadoutEngine
from floodgates.engines.engine_sample import _LMTap
from floodgates.engines.sampling_ops import probs_cand, sample, verify


class _SampleMixin:
    q_off = 0                     # window row of the last round whose readout proposes d_1 (DAS-Wave 0, own readout 1)

    def __init__(self, ouro, das, dr=None, temp=1.0, top_p=0.7, q_topp=True, q_mode="argmax", keep_p=False, k_cand=None, **kw):
        """das: the spec of engine_baselines (seq / wave / pld, hybrid = wave with init loopahead); temp / top_p: the target's
        sampling setting (temp 0 = exact greedy); q_mode: "argmax" (one-hot q on the greedy drafts) | "sample" (wave modes only:
        d_j ~ q_j = softmax of the last round's readouts); q_topp: sampled proposals restricted to their own top-p nucleus;
        k_cand (env TOPP_CAND, default 4096): nucleus threshold searched among the k_cand most likely tokens (0 = full sort)."""
        super().__init__(ouro, das, dr, **kw)
        assert q_mode in ("sample", "argmax")
        if q_mode == "sample":
            assert das["mode"] == "wave" and das["J"] >= 1, "sampled proposals need a wave window with >= 1 round (its last readouts)"
        self.o = _LMTap(ouro, self)
        self.temp, self.top_p, self.q_topp, self.q_mode, self.keep_p = float(temp), float(top_p), bool(q_topp), q_mode, keep_p
        V = ouro.lm_head.weight.shape[0]; self.vocab = V
        self.k_cand = int(os.environ.get("TOPP_CAND", 4096)) if k_cand is None else int(k_cand)
        self.uncov = torch.zeros(3, dtype=torch.long, device=self.dev)     # rows whose top-k_cand mass < top_p: plain, verify, proposal
        if q_mode == "sample":
            self.dlogits = torch.zeros(das["gamma"], V, device=self.dev, dtype=torch.float32)
        if keep_p:
            self.pbuf = torch.zeros(self.k1, V, device=self.dev, dtype=torch.float32)
        self._tmode = "plain"; self._q = None; self._last_lm = None

    def _probs(self, logits, top_p, which):
        p, unc = probs_cand(logits, self.temp, top_p, self.k_cand)
        if unc is not None:
            self.uncov[which] += unc.sum()
        return p

    def _state(self):
        return super()._state() + ([self.dlogits] if self.q_mode == "sample" else []) + [self.uncov]

    def _verify_q(self):
        """The proposal rows passed to verify() (None = one-hot on the drafts); a hook for the tests' negative control."""
        return self._q if self.q_mode == "sample" else None

    # ---- graph bodies --------------------------------------------------------------------------------------------------------
    def _draft_wave(self, Lb):
        super()._draft_wave(Lb)                                               # greedy window; ids[0, 1:] = its argmax drafts
        if self.q_mode == "sample":                                           # last round's lm_head output (W, V)
            g = self.das["gamma"]
            self.dlogits.copy_(self._last_lm[self.q_off:self.q_off + g])

    def _target(self, Q, Lb, mode="prefill"):
        if mode == "prefill":
            return super()._target(Q, Lb, mode)
        self._tmode = mode
        if mode == "spec" and self.q_mode == "sample":                      # draw the drafts from the last round's proposals
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
        acc, e = verify(p, d, self._verify_q())
        out = torch.cat([d, d[:1]]).scatter_(0, acc[None], e[None])        # [d_1..d_acc, e, ...]
        self.res[:1].copy_(acc[None] + 1); self.res[1:k1 + 1].copy_(out)
        e1 = e[None]
        self.ids[0, :1].copy_(e1); self.sid[0, :1].copy_(e1)
        if self.sid_dr is not None:
            self.sid_dr[0, :1].copy_(e1)
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

    # ---- tests: one step / cycle from a fixed state -----------------------------------------------------------------------
    def _result(self):
        return self._wait(self._readback())[1]

    @torch.no_grad()
    def cycle_from(self, cut, y, drafts=None):
        """One draft + verification cycle with y (not yet cached) at position cut over the cached context rows < cut (rows >= cut
        are overwritten as in decoding); PLD: drafts = the host proposals (gamma ids; None = a plain step, as without a match).
        Returns the emitted tokens (positions cut+1 ..)."""
        k1 = self.k1
        self.ids[0, 0] = y; self.sid[0, 0] = y; self.pos0.fill_(cut)
        if self.sid_dr is not None:
            self.sid_dr[0, 0] = y
        if self.das["mode"] == "pld":
            if drafts is None:
                self._run("t", 1, cut + 1, "plain")
                return self._result()
            self.ids[0, 1:k1].copy_(torch.tensor(drafts, device=self.dev))
        else:
            self._run(self.das["mode"], 0, cut + k1)
        self._run("t", k1, cut + k1, "spec")
        return self._result()

    @torch.no_grad()
    def plain_from(self, cut, y):
        """One plain sampling step at position cut (y not yet cached; writes its K/V row); returns (token, p row) (keep_p)."""
        self.ids[0, 0] = y; self.pos0.fill_(cut)
        self._run("t", 1, cut + 1, "plain")
        new = self._result()
        return new[0], (self.pbuf[0].clone() if self.keep_p else None)


class OuroBaselineSampleEngine(_SampleMixin, OuroBaselineEngine):
    """Sequential DAS / DAS-Wave / Jacobi / PLD (engine_baselines semantics) under lossless speculative sampling."""
    q_off = 0                     # DAS-Wave: drafts d_1..d_gamma = the readouts of window rows 0..gamma-1


class OuroOwnSampleEngine(_SampleMixin, OuroOwnReadoutEngine):
    """LAL + refinement with the own-position readout (engine_baselines_own semantics) under lossless speculative sampling."""
    q_off = 1                     # drafts d_1..d_gamma = the own readouts of window rows 1..gamma
