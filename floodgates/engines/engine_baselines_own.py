"""LAL + refinement on Ouro with the OWN-POSITION readout rule (engine_baselines.OuroBaselineEngine subclass): the trained
one-pass drafter's slots are mask-filling (slot i predicts the token AT position cut+i), and runs/ouro-baselines/diag_carry.py
shows that an untrained loop of the block from the carried slot states keeps that convention (the readout of position i agrees
with the token at i, not at i+1), so the next-token rule of the Raven refinement (the readout of slot i becomes the token of
slot i+1) only repeats tokens on Ouro.  Here every slot takes its OWN readout as its refined token (the rule of the drafter's
trained feedback rounds, drafter.feedback_inputs without shift): window of gamma+1 positions cut .. cut+gamma (0 = the anchor
y, 1..gamma = the drafts, seeded with [y, d_1..d_k, d_k ...]), J loops without LoRA over the window with carried states and
the embedding-delta rule of engine_baselines, drafts = the window tokens 1..gamma after the last round.
Use through bench_own.py (same spec strings / envs as bench_ouro_baselines.py; tag the EVAL_TAG with _own)."""
import torch
from floodgates.engines.engine_baselines import OuroBaselineEngine


class OuroOwnReadoutEngine(OuroBaselineEngine):
    def _draft_wave(self, Lb):
        o, das, dev = self.o, self.das, self.dev
        assert das.get("init") == "loopahead", "own-readout refinement seeds the window with the trained drafter"
        J, kl, g = das["J"], das["k"], das["gamma"]
        W = g + 1; E = o.embed_tokens
        pos = self.pos0 + torch.arange(W, device=dev); cos, sin = o.rope(pos[None])
        reads = das.get("reads", "final"); carry = das.get("carry", False)
        d0 = 1 if carry else 0
        blk, st, e_in = self._seed(Lb); nd = self.k_dr
        tok = torch.cat([blk[:1 + nd], blk[nd:1 + nd].expand(W - 1 - nd)]) if W - 1 > nd else blk[:W]
        x = E(tok)
        if carry:                                                          # seeded positions continue from the drafter's states
            n_ = min(W, self.k1_dr)
            xs = st[:n_] + (E(tok[:n_]) - e_in[:n_] if das.get("seed_delta", True) else 0)
            x = torch.cat([xs, x[n_:]]) if n_ < W else xs
        state, prev = None, tok
        for j in range(J):
            if j > 0:
                e = E(tok); x = state + (e - E(prev))                      # carried state, embedding delta of the changed token
            dj = d0 + j * kl
            bufs = [min(dj + i, self.R - 1) for i in range(kl)] if reads == "depth" else [self.R - 1] * kl
            state = self._pass(x, cos, sin, self.pos0, W, bufs, False, False)
            z = o.lm_head(state).argmax(-1)                                # (W,)
            prev = tok
            tok = torch.cat([tok[:1], z[1:]])                              # own readout: slot i -> its refined token
        self.ids[0, 1:1 + g].copy_(tok[1:1 + g]); self.ids[0, :1].copy_(self.sid[0, :1])
