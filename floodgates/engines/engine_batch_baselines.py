"""Training-free baselines in the Ouro batched engine (engine_batch.BatchEngine subclass; same static per-loop K/V buffers,
packed ragged verification, CUDA graphs, continuous batching, DET option) for the batch figure's Ouro panels:
  das<r>_<g>  sequential DAS / Mature-Spec with aligned reads (= engine_baselines.OuroBaselineEngine mode "seq"): g
              single-position drafts one after the other; draft j of every row = one token at position Lc + j through r
              loops of the block, loop u reading the row's prefix (and earlier drafts) at loop u and writing its own K/V
              there (overwritten by the verification); the readout seeds the next draft.  Every row verifies its g drafts.
  pld<n>_<g>  prompt lookup decoding (= engine_baselines mode "pld"): the host looks up each row's latest earlier occurrence
              of its last n tokens (raven_engine_batch.PLDIndex = pld_draft); rows with a match verify the g tokens that
              followed, rows without one take a plain step in the same packed verification (k_b = 0); one sync per cycle.
plain is BatchEngine's plain step.  Outputs: see bench_engine_batch_baselines.py."""
import os, time
from collections import deque
import torch
from floodgates.engines import fused_gemm as fg
from floodgates.engines.engine_batch import BatchEngine, mm, rope_qk
from floodgates.engines.raven_engine_batch import PLDIndex
import re


def parse_bmode(mode):
    if mode == "plain":
        return dict(cat="ahead", nd=0)
    m = re.fullmatch(r"das(\d+)_(\d+)", mode)
    if m:
        return dict(cat="ahead", r=int(m.group(1)), nd=int(m.group(2)))
    m = re.fullmatch(r"pld(\d+)_(\d+)", mode)
    if m:
        return dict(cat="pld", n=int(m.group(1)), nd=int(m.group(2)))
    raise ValueError(mode)


class BatchBaselineEngine(BatchEngine):
    def __init__(self, ouro, dr, B=8, tmax=1024, gmax=12):
        super().__init__(ouro, dr, B=B, tmax=tmax)
        assert self.k >= gmax, "the drafter's k bounds the verified block (d buffers)"
        self.kk = torch.zeros(B, dtype=torch.long, device=self.dev)
        self.h_d = torch.zeros(B, self.k, dtype=torch.long).pin_memory()
        self.h_kk = torch.zeros(B, dtype=torch.long).pin_memory()

    # ---- r loops of the block over Bs single tokens (BatchEngine._stack's fast path, first r loops) ------------------------
    def _stack_r(self, Bs, ids, pos, lens, r):
        o, eps, ns, H, D = self.o, self.eps, self.o.norm.weight, self.H, self.D
        mt = self._meta(Bs)
        N = Bs
        cos, sin = o.rope(pos[None])
        cos, sin = cos[0], sin[0]
        att = self._att(Bs)
        res, dn, n_prev = o.embed_tokens(ids), None, None
        for ut in range(r):
            for li in range(self.NL):
                W = self.W[li]
                xo, h = fg.prep(res, dn, n_prev, ns if (li == 0 and ut > 0) else None, W["n1"], eps)
                y = mm(h, W["qkv"])
                q, k = rope_qk(y, cos, sin, H, D)
                v = y.view(N, 3, H, D)[:, 2]
                a = att(q.view(1, N, H, D).transpose(1, 2), k.view(1, N, H, D).transpose(1, 2), v.unsqueeze(0).transpose(1, 2),
                        self.kc[ut, li, :Bs], self.vc[ut, li, :Bs], lens, mt["qs1"], mt["qc1"], full=False, write=True, one=True)
                x1, h2 = fg.prep(xo, mm(a.transpose(1, 2).reshape(N, H * D), W["o"]), W["n2"], None, W["n3"], eps)
                dn = mm(mm(h2, W["gu"], silu=True), W["d"])
                res, n_prev = x1, W["n4"]
        xf, _ = fg.prep(res, dn, n_prev, ns, ns, eps)
        return xf

    def _draft_seq(self, Bs, r, g):
        L = self.Lc[:Bs]
        tok = self.y[:Bs]
        for j in range(g):
            p = L + j
            z = self.o.lm_head(self._stack_r(Bs, tok, p, p, r)).argmax(-1)
            self.d[:Bs, j].copy_(z)
            tok = z

    def _verify_given(self, Bs, N):
        """BatchEngine._verify with host-given per-row block lengths kk (prompt lookup; sum(kk) = N - Bs)."""
        k, k1, dev = self.k, self.k1, self.dev
        act, L, y, d = self.act[:Bs], self.Lc[:Bs], self.y[:Bs], self.d[:Bs]
        j = self.ar[:k1]
        kk = self.kk[:Bs]
        cnt = kk + 1
        qs = torch.cumsum(cnt, 0) - cnt
        dest = torch.where(j[None, :] < cnt[:, None], qs[:, None] + j[None, :], N).reshape(-1)
        ids = torch.zeros(N + 1, dtype=torch.long, device=dev)
        ids.scatter_(0, dest, torch.cat([y[:, None], d], 1).reshape(-1))
        pp = torch.zeros(N + 1, dtype=torch.long, device=dev)
        pp.scatter_(0, dest, (L[:, None] + j[None, :]).reshape(-1))
        rid = None
        if self.det:
            rid = torch.zeros(N + 1, dtype=torch.long, device=dev)
            rid.scatter_(0, dest, self.ar[:Bs][:, None].expand(Bs, k1).reshape(-1))
            rid = rid[:N]
        xf = self._stack(Bs, N, ids[:N], pp[:N], L, qs, cnt, one=False, rowid=rid)
        pred = self._lm(xf).argmax(-1)
        P = pred[(qs[:, None] + j[None, :]).clamp(max=N - 1)]
        match = (d == P[:, :k]) & (j[None, :k] < kk[:, None])
        acc = match.long().cumprod(1).sum(1)
        n_new = acc + 1
        ynew = P.gather(1, acc[:, None])[:, 0]
        r = self.res[:Bs]
        r.zero_()
        r[:, 0].copy_(n_new); r[:, 1:1 + k1].copy_(P); r[:, 1 + k1].copy_(kk)
        self.y[:Bs].copy_(torch.where(act, ynew, y))
        self.Lc[:Bs] += n_new * act.long()

    def _body(self, key):
        if key[0] == "dseq":
            return lambda: self._draft_seq(key[1], key[2], key[3])
        if key[0] == "verg":
            return lambda: self._verify_given(key[1], key[2])
        return super()._body(key)

    def _state(self):
        return super()._state() + [self.kk]

    def keys_for(self, Bs, mode):
        if mode in ("plain", "fixed", "cost", "costu"):
            return super().keys_for(Bs, mode)
        mi = parse_bmode(mode); g = mi["nd"]
        if mi["cat"] == "pld":
            return [("verg", Bs, Bs + g * c) for c in range(Bs + 1)]
        return [("dseq", Bs, mi["r"], g), ("ver", Bs, Bs * (g + 1), "uni")]

    def prepare(self, Bs, modes, Lcal=384):
        self.Lc[:Bs].fill_(Lcal); self.y[:Bs].fill_(1); self.act[:Bs].fill_(True)
        for mode in modes:
            for key in self.keys_for(Bs, mode):
                self.capture(key)
        torch.cuda.synchronize()
        self.act[:Bs].fill_(False)

    # ---- continuous batching (BatchEngine.run_continuous + the host-drafted prompt-lookup cycle) ---------------------------
    @torch.no_grad()
    def run_continuous(self, prompts, mode, max_new, stop, compact=True):
        mi = parse_bmode(mode)
        B, k1, n = self.B, self.k1, len(prompts)
        ladder = [b for b in self.ladder() if all(key in self.graphs for key in self.keys_for(b, mode))] if compact else [B]
        assert B in ladder
        Bs = B
        outs, cut = [None] * n, [None] * n
        row_seq, row_from = [None] * B, [0] * B
        st = dict(qi=0, cid=0, t_pre=0.0, n_pre=0, ncomp=0)
        first_fill = [True]
        PF1 = os.environ.get("PF_GRAPH", "1") == "1"
        PREFIX = os.environ.get("PREFIX", "0") == "1"
        if PF1 or PREFIX:
            self.prepare_prefill1(min(len(p) for p in prompts), max(len(p) for p in prompts))
        if PREFIX:
            if getattr(self, "pref_src", None) is not prompts:
                self.build_prefix(prompts); self.pref_src = prompts
                self.prepare_suffix(prompts)
        recs, widths = [], []
        pidx = {}
        self.act.zero_(); self.act_h.zero_()

        def finish_check(i, p0):
            o_ = outs[i]
            if len(o_) >= max_new or stop(o_):
                n_stop = min(len(o_), max_new)
                for q in range(max(p0, 1), min(len(o_), max_new) + 1):
                    if stop(o_[:q]):
                        n_stop = q
                        break
                cut[i] = n_stop
                return True
            return False

        def refill(rows):
            rows = [b for b in rows if b < Bs]
            while rows and st["qi"] < n:
                take = []
                for b in rows:
                    if st["qi"] < n:
                        take.append((b, st["qi"])); st["qi"] += 1
                torch.cuda.synchronize(); ta = time.perf_counter()
                if PREFIX:
                    y0 = [self.prefill_prefixed(prompts[i], b) for b, i in take]
                elif first_fill[0] or not PF1:
                    y0 = self.prefill([prompts[i] for _, i in take], [b for b, _ in take]).tolist()
                else:
                    y0 = [self.prefill1(prompts[i], b) for b, i in take]
                first_fill[0] = False
                st["t_pre"] += time.perf_counter() - ta; st["n_pre"] += 1
                rows = []
                for (b, i), t in zip(take, y0):
                    outs[i] = [t]; row_seq[b] = i; row_from[b] = st["cid"]
                    if mi["cat"] == "pld":
                        pidx[i] = PLDIndex(list(prompts[i]) + [t], mi["n"], mi["nd"])
                    if finish_check(i, 1):
                        row_seq[b] = None; self.act_h[b] = False; rows.append(b)
            self._push_act()

        def process(c, r, Bs_c):
            freed, rec = [], []
            for b in range(Bs_c):
                i = row_seq[b]
                if i is None or c < row_from[b]:
                    continue
                q = r[b][0]; p0 = len(outs[i])
                new = r[b][1:1 + q]
                outs[i] += new
                if i in pidx:
                    pidx[i].extend(new)
                rec.append((b, q, r[b][1 + k1], 0.0))
                if finish_check(i, p0 + 1):
                    row_seq[b] = None; self.act_h[b] = False; freed.append(b)
                    pidx.pop(i, None)
            recs.append(rec); widths.append(Bs_c)
            return freed

        def maybe_compact(drain):
            nonlocal Bs
            if not compact or st["qi"] < n:
                return False
            act_rows = [b for b in range(Bs) if row_seq[b] is not None]
            if not act_rows or len(act_rows) > Bs // 2:
                return False
            drain()
            torch.cuda.synchronize()
            act_rows = [b for b in range(Bs) if row_seq[b] is not None]
            if not act_rows:
                return False
            target = min(b for b in ladder if b >= len(act_rows))
            if target >= Bs:
                return False
            mp = self._compact(act_rows, target)
            new_seq = [None] * B
            for old, new in mp.items():
                new_seq[new] = row_seq[old]
            row_seq[:] = new_seq
            for b in range(B):
                row_from[b] = st["cid"]
            self.act_h.zero_()
            for b in range(target):
                self.act_h[b] = row_seq[b] is not None
            self._push_act()
            Bs = target; st["ncomp"] += 1
            return True

        torch.cuda.synchronize(); T0 = time.perf_counter()
        refill(list(range(B)))
        if mi["cat"] == "ahead":
            inflight = deque()

            def launch():
                for key in self.keys_for(Bs, mode):
                    self.run(key)
                inflight.append((st["cid"], self._readback(Bs), Bs)); st["cid"] += 1

            def drain():
                while inflight:
                    c, s, Bs_c = inflight.popleft()
                    process(c, self._wait(s, Bs_c), Bs_c)
                self._push_act()
            while any(x is not None for x in row_seq) or st["qi"] < n:
                while len(inflight) < 2:
                    launch()
                c, s, Bs_c = inflight.popleft()
                freed = process(c, self._wait(s, Bs_c), Bs_c)
                if freed:
                    refill(freed) if st["qi"] < n else self._push_act()
                maybe_compact(drain)
        else:
            g = mi["nd"]
            while any(x is not None for x in row_seq) or st["qi"] < n:
                cnt = 0
                self.h_kk[:Bs].zero_()
                for b in range(Bs):
                    i = row_seq[b]
                    if i is None:
                        continue
                    dd = pidx[i].draft()
                    if dd:
                        self.h_d[b, :g] = torch.tensor(dd)
                        self.h_kk[b] = g; cnt += 1
                self.d[:Bs].copy_(self.h_d[:Bs], non_blocking=True)
                self.kk[:Bs].copy_(self.h_kk[:Bs], non_blocking=True)
                self.run(("verg", Bs, Bs + g * cnt))
                c = st["cid"]; s = self._readback(Bs); st["cid"] += 1
                freed = process(c, self._wait(s, Bs), Bs)
                if freed:
                    refill(freed) if st["qi"] < n else self._push_act()
                maybe_compact(lambda: None)
        T1 = time.perf_counter()
        torch.cuda.synchronize()
        self.act.zero_(); self.act_h.zero_()
        outs = [o_[:c_] for o_, c_ in zip(outs, cut)]
        return dict(outs=outs, cycles=recs, choices=[], widths=widths, t_total=T1 - T0, t_prefill=st["t_pre"],
                    t_decode=T1 - T0 - st["t_pre"], n_prefill_calls=st["n_pre"], n_compactions=st["ncomp"])
