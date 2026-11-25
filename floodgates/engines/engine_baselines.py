"""Training-free baselines in the Ouro batch-1 engine (engine.Engine subclass; same CUDA graphs, static per-loop K/V
buffers, device-side acceptance and pipelined host readback as the trained one-pass drafter), mirroring the modes of
raven_engine.py ("das:", "wave:", "hybrid:", "pld:" of bench_raven_engine.py) with the Ouro semantics of
eval_das_ouro.py / eval_daswave_ouro.py (one L-layer block looped R times, the model norm after every loop):
  seq   {"mode": "seq", "r": r, "gamma": g}: sequential DAS / Mature-Spec (ALIGNED reads): gamma single-position drafts
        one after the other at positions cut .. cut+gamma-1, each r loops of the block whose loop u reads the verified
        prefix's K/V AT LOOP u (the K/V of the earlier drafts are written at loop u too); the readout seeds the next draft.
  wave  {"mode": "wave", "J", "k", "gamma"}: DAS-Wave: window of gamma positions cut .. cut+gamma-1 (position 0 = the
        anchor y) seeded with the anchor token; J rounds of k loops over the whole window, causal within the window,
        reading the prefix at its FINAL loop, with carried (normalized) residual states: a round's input is the previous
        round's output plus the embedding delta of the changed token (eval_daswave_ouro.py); the readout of position i
        becomes position i+1's token.  Window growth W_j = min(gamma, 2 + j) is emulated by letting a position restart
        from its embedding until it is active ("grow": False = full window from the first round); "carry_rounds": False
        = every round restarts from the embeddings.  "reads": "final" (DAS-Wave: every loop reads the prefix at its final
        loop) | "depth" (loop i of round j reads the prefix at loop min(depth + i, R-1), depth = the loops the carried
        window state has been through, i.e. the model's own recurrence continued over the window; the default for
        k = R without carry = Jacobi decoding = full-depth passes of the model, else "final").
        The drafts are the window tokens 1..gamma-1 after the last round plus the last position's readout.
  hybrid wave with "init": "loopahead": the window is seeded with the trained drafter's block [y, d_1..d_k] (Engine._draft,
        LoRA, one pass; the remaining positions take the last draft) and, with "carry", the seeded positions start the
        refinement loops (NO LoRA) from the drafter's slot states with the same embedding-delta rule (their previous
        input = the drafter's slot input; the drafting pass counts as one loop of depth for "reads": "depth");
        J = 0 reproduces the one-pass drafter exactly (drafts = its block).
  pld   {"mode": "pld", "n", "gamma"}: prompt lookup decoding on the host (latest earlier occurrence of the last n tokens
        of prompt + generation -> the gamma tokens that followed), verification only when there is a match, else a plain
        step; serial (one host sync per cycle), since the next draft needs the accepted tokens.
Verification = Engine._target over [y, d_1..d_gamma] (Q = gamma+1) with the device-side acceptance of Engine._accept,
so every mode is lossless (greedy outputs = plain decoding up to bf16 ties).  Needs ATTN=triton (the default)."""
import math, os, time
import torch
from floodgates.engines.engine import Engine
from floodgates.engines import fused_gemm as fg


def pld_draft(seq, n_max, k):
    """Prompt lookup decoding (Saxena 2023; = raven_engine.pld_draft): find the latest earlier occurrence of the last n
    tokens (n = n_max .. 1) and return the k tokens that followed it (padded by repeating its last token); [] if none."""
    L = len(seq)
    for n in range(min(n_max, L - 1), 0, -1):
        key = seq[L - n:]
        for i in range(L - n - 1, -1, -1):
            if seq[i:i + n] == key:
                d = seq[i + n:i + n + k]
                if not d:
                    continue
                return d + [d[-1]] * (k - len(d))
    return []


class OuroBaselineEngine(Engine):
    def __init__(self, ouro, das, dr=None, tmax=4096, graphs=True):
        super().__init__(ouro, dr, n_rounds=1, tmax=tmax, graphs=graphs, S=1)
        assert self.att is not None, "the baselines need the Triton decode attention (ATTN=triton)"
        assert das["mode"] in ("seq", "wave", "pld")
        self.das = das; g = das["gamma"]; dev = self.dev
        if das["mode"] == "wave":
            assert 1 <= das["k"] and 1 <= g and (das.get("init") != "loopahead" or dr is not None)
            assert das.get("init") == "loopahead" or das["J"] >= 1
        # the trained drafter keeps the buffers of Engine.__init__ (k1_dr = k+1 slots): swapped in around Engine._draft
        self.k_dr, self.k1_dr = self.k, self.k1
        self.ids_dr, self.sid_dr = (self.ids, self.sid) if dr is not None else (None, None)
        self.dr_state = torch.zeros(self.k1_dr, ouro.c["hidden_size"], device=dev, dtype=ouro.dtype) if dr is not None else None
        # verification buffers for gamma drafts: ids = [y, d_1 .. d_gamma], Q = gamma + 1 (Engine._target / _accept)
        self.k, self.k1 = g, g + 1; QV = g + 1; self.QM = QV
        self.ids = torch.zeros(1, QV, dtype=torch.long, device=dev)
        self.pred = torch.zeros(QV, dtype=torch.long, device=dev)
        self.res = torch.zeros(1 + QV, dtype=torch.long, device=dev)
        self.hres = torch.zeros(2, 1 + QV, dtype=torch.long).pin_memory()
        self.sid = torch.zeros(1, QV, dtype=torch.long, device=dev)        # slot 0 = y (next anchor), as Engine
        self.pos_d = torch.zeros(1, dtype=torch.long, device=dev)          # sequential DAS: position of the current draft
        if das["mode"] == "pld":
            self.pipe = False                                              # the host drafts from the accepted tokens
        self.n_nodraft = 0                                                 # pld: cycles without a match (plain steps)

    # ---- the block over Q positions -----------------------------------------------------------------------------
    def _pass(self, x, cos, sin, length, blk, bufs, full, write):
        """x (Q, d): the residual-stream input of the first loop (token embeddings, or a carried normalized state); runs
        len(bufs) loops of the block over the Q positions (RoPE cos/sin (1, Q, D)).  The attention of loop i reads the
        cache rows [0, length) of the K/V buffers of loop bufs[i], plus the step's own Q tokens (causal within blocks
        of blk, or full), and with write stores the own K/V at rows length .. length+Q-1 of the buffer it reads.
        Returns the normalized output (Q, d): the input of the next loop / of lm_head (Ouro norms after every loop).
        Same fused pieces and rounding points as Engine._stack_fast / Engine._layer."""
        o, Q = self.o, x.shape[0]
        if self.fused and self.gemm_kind == "triton" and Q <= fg.MAX_T:
            eps, ns = self.rms_eps, o.norm.weight
            res, dn, n_prev = x.contiguous(), None, None
            for i, b in enumerate(bufs):
                for li in range(self.L):
                    W = self.W[li]
                    xo, h = fg.prep(res, dn, n_prev, ns if (li == 0 and i > 0) else None, W["n1"], eps)
                    q, k, v = self.rope_split(fg.gemm(h, W["qkv"]), cos, sin, self.H)
                    a = self.att(q, k, v, self.kc[b, li], self.vc[b, li], length, blk, full, write)
                    x1, h2 = fg.prep(xo, fg.gemm(a.transpose(1, 2).reshape(Q, -1), W["o"]), W["n2"], None, W["n3"], eps)
                    dn = fg.gemm(fg.gemm(h2, W["gu"], silu=True), W["d"])
                    res, n_prev = x1, W["n4"]
            xf, _ = fg.prep(res, dn, n_prev, ns, ns, eps)                  # last residual add + model norm
            return xf
        x = x[None]
        for b in bufs:
            for li in range(self.L):
                kc, vc = self.kc[b, li], self.vc[b, li]

                def attn(q, k, v, kc=kc, vc=vc):
                    return self.att(q, k, v, kc, vc, length, blk, full, write)
                x = self._layer(li, x, cos, sin, attn)
            x = o.norm(x)
        return x[0]

    # ---- graph bodies ----------------------------------------------------------------------------------------------
    def _draft_seq(self, Lb):
        """Sequential DAS (aligned reads): gamma drafts, draft j = one new position cut+j (input: the previous draft, y for
        j = 0) through r loops; loop u reads the prefix and the earlier drafts at loop u and writes its own K/V there (the
        verification step overwrites these rows); ids[1..gamma] = the drafts."""
        o, r, g = self.o, self.das["r"], self.das["gamma"]
        tok = self.sid[0, :1]
        for j in range(g):
            self.pos_d.copy_(self.pos0 + j)
            cos, sin = o.rope(self.pos_d[None])
            xf = self._pass(o.embed_tokens(tok), cos, sin, self.pos_d, 1, list(range(r)), False, True)
            z = o.lm_head(xf).argmax(-1)                                   # (1,)
            self.ids[0, 1 + j:2 + j].copy_(z); tok = z
        self.ids[0, :1].copy_(self.sid[0, :1])

    def _seed(self, Lb):
        """The trained drafter's one pass (Engine._draft with its own k+1-slot buffers); returns its block [y, d_1..d_k]
        (k1_dr,), the slots' normalized states (k1_dr, d) and the slots' input embeddings (k1_dr, d)."""
        dr = self.dr
        sv = (self.k, self.k1, self.ids, self.sid)
        self.k, self.k1, self.ids, self.sid = self.k_dr, self.k1_dr, self.ids_dr, self.sid_dr
        try:
            self._draft(Lb)
        finally:
            self.k, self.k1, self.ids, self.sid = sv
        t = torch.zeros(1, self.k1_dr, device=self.dev); t_in = torch.where(self.is_clean, torch.ones_like(t), t)
        e_in = dr.slot_inputs(self.eps * dr.sigma, self.sid_dr, self.is_clean, t_in, None, torch.ones_like(t))[0]   # as Engine._draft, round 0
        return self.ids_dr[0], self.dr_state, e_in

    def _draft_wave(self, Lb):
        """DAS-Wave / Jacobi / LoopAhead-seeded refinement over the window of W = gamma positions (see the module doc)."""
        o, das, dev = self.o, self.das, self.dev
        J, kl, g = das["J"], das["k"], das["gamma"]
        W = g; E = o.embed_tokens
        pos = self.pos0 + torch.arange(W, device=dev); cos, sin = o.rope(pos[None])
        seeded = das.get("init") == "loopahead"
        grow = das.get("grow", not seeded)
        carry_rounds = das.get("carry_rounds", True)
        reads = das.get("reads", "depth" if (kl == self.R and not carry_rounds) else "final")
        d0 = 1 if (seeded and das.get("carry")) else 0                     # loops of depth of the window state before round 0
        seed_drafts = None
        if seeded:
            blk, st, e_in = self._seed(Lb); nd = self.k_dr
            tok = torch.cat([blk[:1 + nd], blk[nd:1 + nd].expand(W - 1 - nd)]) if W - 1 > nd else blk[:W]
            seed_drafts = torch.cat([blk[1:1 + nd], blk[nd:1 + nd].expand(g - nd)]) if g > nd else blk[1:1 + g]
            x = E(tok)
            if das.get("carry"):                                           # seeded positions continue from the drafter's states
                n_ = min(W, self.k1_dr)
                xs = st[:n_] + (E(tok[:n_]) - e_in[:n_] if das.get("seed_delta", True) else 0)
                x = torch.cat([xs, x[n_:]]) if n_ < W else xs
        else:
            tok = self.sid[0, :1].expand(W)
            x = E(tok)
        act = (torch.arange(W, device=dev) - 1) if grow else torch.zeros(W, dtype=torch.long, device=dev)
        state, prev, z = None, tok, None
        for j in range(J):
            if j > 0:
                e = E(tok)
                if carry_rounds:                                           # positions active in the previous round carry on
                    x = torch.where((act < j)[:, None], state + (e - E(prev)), e)
                else:
                    x = e
            dj = (d0 + j * kl) if carry_rounds else 0                        # depth of the state entering round j
            bufs = [min(dj + i, self.R - 1) for i in range(kl)] if reads == "depth" else [self.R - 1] * kl
            state = self._pass(x, cos, sin, self.pos0, W, bufs, False, False)
            z = o.lm_head(state).argmax(-1)                                # (W,)
            prev = tok
            tok = torch.cat([tok[:1], z[:-1]])
        drafts = torch.cat([tok[1:], z[-1:]]) if J > 0 else seed_drafts
        self.ids[0, 1:1 + g].copy_(drafts); self.ids[0, :1].copy_(self.sid[0, :1])

    def _accept(self, S):
        super()._accept(S)
        if self.sid_dr is not None:
            self.sid_dr[0, :1].copy_(self.sid[0, :1])

    # ---- graph management ------------------------------------------------------------------------------------------
    def _run(self, kind, Q, need_len, mode="plain"):
        Lb = self.tmax
        if kind == "t":
            body = lambda: self._target(Q, Lb, mode)
        elif kind == "seq":
            body = lambda: self._draft_seq(Lb)
        elif kind == "wave":
            body = lambda: self._draft_wave(Lb)
        else:
            raise ValueError(kind)
        if not self.use_graphs:
            return body()
        key = (kind, Q, Lb, mode)
        g = self.graphs.get(key)
        if g is None:
            saved = [t.clone() for t in self._state()]                     # the body advances the device state
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    body()                                                 # warm-up (allocations, autotune)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                body()
            self.pool = g.pool(); self.graphs[key] = g
            for t, v in zip(self._state(), saved):
                t.copy_(v)
        g.replay()

    def _state(self):
        st = [self.pos0, self.ids, self.res, self.sid, self.pos_d]
        if self.sid_dr is not None:
            st += [self.ids_dr, self.sid_dr, self.dr_state]
        return st

    def warmup(self, max_len):
        assert max_len <= self.tmax, (max_len, self.tmax)
        self.pos0.fill_(self.tmax // 2)
        self._run("t", 1, self.tmax, "plain")
        self._run("t", self.k1, self.tmax, "spec")
        if self.das["mode"] != "pld":
            self._run(self.das["mode"], 0, self.tmax)
        torch.cuda.synchronize()

    # ---- decoding ------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def generate_spec(self, prompt, max_new, stop, gen=None):
        k1 = self.k1
        y = self.prefill(prompt); out = [y]; cur = len(prompt); taus = []
        self.ids[0, 0] = y; self.sid[0, 0] = y; self.pos0.fill_(cur)
        if self.sid_dr is not None:
            self.sid_dr[0, 0] = y
        torch.cuda.synchronize(); t0 = time.perf_counter()
        go = lambda: len(out) < max_new and not stop(out)
        if not go():
            return out, taus, 0.0
        if self.das["mode"] == "pld":                                      # serial: the host drafts from prompt + out
            n_, g = self.das["n"], self.k
            while go():
                d = pld_draft(prompt + out, n_, g)
                if d:
                    self.ids[0, 1:1 + g].copy_(torch.tensor(d, device=self.dev)); self._run("t", k1, cur + k1, "spec")
                else:
                    self._run("t", 1, cur + 1, "plain"); self.n_nodraft += 1
                n, new = self._wait(self._readback()); out += new; taus.append(n); cur += n
            t1 = time.perf_counter(); torch.cuda.synchronize()
            return out, taus, t1 - t0
        kind = self.das["mode"]

        def launch(c):
            if self.dr is not None and self.dr.use_noise:
                self.eps.normal_(generator=gen)
            self._run(kind, 0, c + k1)
            self._run("t", k1, c + k1, "spec")
            return self._readback()
        s = launch(cur)
        while True:
            if self.pipe:
                s_next = launch(cur + k1)                                  # the in-flight cycle adds at most k+1 tokens
            n, new = self._wait(s); out += new; taus.append(n); cur += n
            if not go():
                break
            s = s_next if self.pipe else launch(cur)
        t1 = time.perf_counter(); torch.cuda.synchronize()
        return out, taus, t1 - t0
