"""raven_engine_sample.RavenSampleEngine (lossless speculative SAMPLING in the batch-1 CUDA-graph engine) extended to the
training-free baselines that the greedy engine (raven_engine.RavenEngine) runs but the sampling engine refuses:
  das={"mode": "seq", "r", "gamma"}   sequential DAS / Mature-Spec (RavenEngine._draft_seq): gamma argmax drafts of depth r,
  das={"mode": "pld", "n", "gamma"}   prompt lookup decoding (raven_engine.pld_draft on the host; no match -> a plain step),
  das={"mode": "wave", ..., "grow": False, "carry_rounds": False}   Jacobi decoding (already accepted by the parent).
These proposals are deterministic functions of the context (their greedy argmax drafts), so q is one-hot on the draft and the
verification accepts d_j with probability p_j(d_j), at a rejection it emits from norm(max(p_j - 1[d_j], 0)) (= p_j with d_j
removed), and after a fully accepted block a bonus draw from p (sampling_ops.verify with q=None).  Nothing else changes: the
target, the nucleus (tie-symmetric, TOPP_CAND), the device-side draws and the seeding are the parent's.  T = 0 is the exact
greedy switch (one-hot target) and reproduces RavenEngine token for token.
Tests only: cycle_from(cut, y, seq) also runs a PLD cycle (seq = the host sequence ending with y); BrokenVerifyEngine is a
NEGATIVE CONTROL (emits from p instead of the residual at a rejection) for the distribution test."""
import os, time
import torch
from floodgates.engines.raven_engine import RavenEngine, pld_draft
from floodgates.engines.raven_engine_sample import RavenSampleEngine, sample


class RavenSampleEngineAll(RavenSampleEngine):
    def __init__(self, m, dr=None, n_rounds=1, temp=1.0, top_p=0.7, q_topp=True, keep_p=False, k_cand=None, **kw):
        """As RavenSampleEngine.__init__ (same arguments), without its restriction to DAS-Wave / the seeded window."""
        das = kw.get("das")
        if das is not None and das["mode"] in ("seq", "pld"):
            assert das.get("q", "argmax") == "argmax", "sequential DAS and PLD propose argmax drafts (one-hot q)"
        RavenEngine.__init__(self, m, dr, n_rounds, **kw)
        # ---- from here on identical to RavenSampleEngine.__init__ ----
        assert self.fused, "the sampling engine needs FUSED=1 (logits are taken from RavenEngine._lm)"
        self.temp, self.top_p, self.q_topp, self.keep_p = float(temp), float(top_p), bool(q_topp), keep_p
        self.q_mode = das.get("q", "argmax") if das is not None else "sample"
        V = m.lm_head.weight.shape[0]; self.vocab = V
        self.off = (0 if dr.shift else 1) if (dr is not None and das is None) else 0
        self.spec = dr is not None or das is not None
        if self.spec and self.q_mode == "sample":
            self.dlogits = torch.zeros(self.nd, V, device=self.dev, dtype=torch.float32)
            self.state.append(self.dlogits)
        if keep_p:
            self.pbuf = torch.zeros(self.nd + 1, V, device=self.dev, dtype=torch.float32)
        self._tmode = "plain"
        self.k_cand = int(os.environ.get("TOPP_CAND", 4096)) if k_cand is None else int(k_cand)
        self.uncov = torch.zeros(3, dtype=torch.long, device=self.dev)
        self.state.append(self.uncov)
        self.pld_nomatch = 0                                                   # PLD cycles without an n-gram match (plain steps)

    @torch.no_grad()
    def generate_spec(self, prompt, max_new, stop=lambda out: False):
        """RavenEngine.generate_spec; the PLD branch is the same loop, also counting the cycles without a match."""
        if self.das is None or self.das["mode"] != "pld":
            return super().generate_spec(prompt, max_new, stop)
        y = self.prefill(prompt); out = [y]; cur = len(prompt); taus = []
        self.ids[0, 0] = y; self.sid[0, 0] = y; self.sid_dr[0, 0] = y; self.pos0.fill_(cur)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        while len(out) < max_new and not stop(out):
            need = cur + self.nd + 1
            d = pld_draft(prompt + out, self.das["n"], self.nd)                 # the latest n-gram match's continuation
            if d:
                self.ids[0, 1:1 + self.nd].copy_(torch.tensor(d, device=self.dev)); self._run("spec", need)
            else:
                self._run("plain", need); self.pld_nomatch += 1                  # no match: a plain sampling step
            n, new = self._result(); out += new; taus.append(n); cur += n
        torch.cuda.synchronize()
        return out, taus, time.perf_counter() - t0

    # ---- tests: one cycle from a fixed state (PLD needs the host sequence) --------------------------------------------
    @torch.no_grad()
    def cycle_from(self, cut, y, seq=None):
        if self.das is None or self.das["mode"] != "pld":
            return super().cycle_from(cut, y)
        self.ids[0, 0] = y; self.pos0.fill_(cut); self.sid[0, 0] = y; self.sid_dr[0, 0] = y
        need = cut + self.nd + 1
        d = pld_draft(list(seq), self.das["n"], self.nd)
        if d:
            self.ids[0, 1:1 + self.nd].copy_(torch.tensor(d, device=self.dev)); self._run("spec", need)
        else:
            self._run("plain", need)
        return self._result()[1]

    @torch.no_grad()
    def drafts_at(self, cut, y, seq=None):
        """The block the method proposes with y at position cut (rows >= cut are overwritten, as in decoding); [] = PLD no match."""
        if self.das is not None and self.das["mode"] == "pld":
            return pld_draft(list(seq), self.das["n"], self.nd)
        self.ids[0, 0] = y; self.pos0.fill_(cut); self.sid[0, 0] = y; self.sid_dr[0, 0] = y
        self._run("draft", cut + self.nd + 1)
        return self.ids[0, 1:1 + self.nd].tolist()


def verify_broken(p, d, q=None):
    """NEGATIVE CONTROL: accepts as sampling_ops.verify, but at a rejection emits a draw from p (not the residual)."""
    n = d.shape[0]
    pd = p[:n].gather(1, d[:, None])[:, 0]
    qd = torch.ones_like(pd) if q is None else q.gather(1, d[:, None])[:, 0]
    acc = (torch.rand(n, device=p.device) < (pd / qd.clamp(min=1e-20)).clamp(max=1.0)).long().cumprod(0).sum()
    return acc, sample(p.index_select(0, acc[None]))[0]


class BrokenVerifyEngine(RavenSampleEngineAll):
    """Distribution-test negative control only (RavenSampleEngine._accept with verify_broken)."""
    def _accept(self):
        if self._tmode == "plain":
            return super()._accept()
        logits = self._last_lm
        nd = self.nd; Q = nd + 1
        p = self._probs(logits[:Q], self.top_p, 1)
        if self.keep_p:
            self.pbuf[:Q].copy_(p)
        d = self.ids[0, 1:Q]
        acc, e = verify_broken(p, d, self._q if self.q_mode == "sample" else None)
        out = torch.cat([d, d[:1]]).scatter_(0, acc[None], e[None])
        self.res[:1].copy_(acc[None] + 1); self.res[1:Q + 1].copy_(out)
        e1 = e[None]
        self.ids[0, :1].copy_(e1); self.sid[0, :1].copy_(e1); self.sid_dr[0, :1].copy_(e1)
        self.pos0 += acc + 1
