"""Batched decoding engine for Ouro with LoopAhead self-drafting (batch B >= 1), used to measure REAL end-to-end
wall-clock throughput of plain batched decoding vs LoopAhead (fixed k / cost-aware block length) in the same engine.

State.  Static per-row K/V buffers per (loop, layer): kc/vc (R, L, B, H, tmax, D).  Device tensors per row: Lc (committed
context length = position of the next input token y), y, act (row still generating).  Graphs for a sub-batch Bs <= B use
the first Bs rows (views), so a final partial wave runs with Bs-row graphs.
Step kinds (one CUDA graph each, per Bs):
  plain           : every row feeds y at position Lc, all R loops, K/V written at Lc; y <- argmax; Lc += act.
  draft           : LoopAhead draft pass: per row the k+1 slots [y, mask x k] at positions Lc .. Lc+k through ONE loop of the
                    stack with the drafter's LoRA, slots attend to the row's final-loop K/V [0, Lc) and to each other
                    (bidirectional); d = argmax of slots 1..k; gains[b, j] = prod_{l<=j} max_v q_l(v) (drafter confidences).
  pol (cost-aware): per bucket N of verified tokens, expected tokens / measured cycle time; choice = argmax.
                    alloc 'rag': per-sequence block lengths = the top (N - Bs) gains over all rows (gains decrease along a row,
                    so the top set is a prefix per row); alloc 'uni': one k per batch (N = Bs (k + 1)).
  ver (N, alloc)  : verification of the PACKED ragged block: row b verifies [y, d_1..d_{k_b}] (k_b from the allocation;
                    'uni' with k_b = k = fixed-k LoopAhead) at positions Lc_b .. Lc_b + k_b, all R loops, causal within the row,
                    K/V written at the row's own cache rows; per-row greedy acceptance acc_b = longest prefix with
                    d_j == argmax at j-1; emits acc_b + 1 tokens; Lc_b += (acc_b + 1) act_b.  Rejected K/V rows are
                    overwritten later (no copies).
Attention: decode_attn_batch (ragged split-KV Triton kernels, per-row lengths from device tensors).  GEMMs: fused_gemm
small-M Triton kernels where they beat cuBLAS (GCFG table), cuBLAS above; sandwich-norm chains fused
(fused_gemm.prep); RoPE / SiLU*up small Triton kernels.  Draft pass: MERGE_LORA=1 (default) runs the target's fast path with
W + B A merged once (bf16); MERGE_LORA=0 = engine.py's compiled blocks with the fp32 LoRA side path.
DET=1: batch-invariant target path (det_kernels: M-invariant GEMMs, K/V written before attention, cache-only attention with
absolute tiling) -> plain and LoopAhead produce bitwise-identical greedy outputs for any batch composition.
Schedulers: run_wave (static waves: B prompts until all finish, finished rows idle) and run_continuous (freed rows refilled
from the queue with graph-captured single-prompt prefills; PREFIX=1 copies the cached K/V of the prompts' shared prefix
and prefills only the suffix; at the tail the active rows are compacted into B/2, B/4, ... rows).
Host <-> device: results [n_new, tokens..., k_b, 1000 E_b] per row are read back through pinned memory; plain / fixed-k
cycles are launched one cycle ahead (the GPU never waits for the host); cost-aware cycles read the policy's choice (one
small sync per cycle between draft and verification) because the verification graph depends on it."""
import math, os, time
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from floodgates.engines import fused_gemm as fg
from floodgates.engines.decode_attn_batch import BatchDecodeAttention
from floodgates.engines.det_kernels import det_mm, rope_kv, CacheAttention, tile_buckets
from floodgates.engines.engine import pre_block, post_block

import torch._dynamo
torch._dynamo.config.recompile_limit = 64
torch._dynamo.config.accumulated_recompile_limit = 4096

GEMM_TMAX = int(os.environ.get("GEMM_TMAX", 64))      # 0 = cuBLAS everywhere


@triton.jit
def _rope_kernel(Y, COS, SIN, QO, KO, syt, H: tl.constexpr, D: tl.constexpr):
    """Y (N, 3*H*D) fused q,k,v projection; COS/SIN (N, D); QO/KO (N, H, D): RoPE'd q, k computed in fp32 with one bf16
    rounding (= the inductor-compiled engine.rope_split)."""
    t = tl.program_id(0)
    h = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    rd = tl.arange(0, D)
    rr = (rd + HALF) % D
    sgn = tl.where(rd < HALF, -1.0, 1.0)
    c = tl.load(COS + t * D + rd).to(tl.float32)
    s = tl.load(SIN + t * D + rd).to(tl.float32)
    base = Y + t * syt + h * D
    q = tl.load(base + rd).to(tl.float32)
    qr = tl.load(base + rr).to(tl.float32)
    tl.store(QO + (t * H + h) * D + rd, (q * c + sgn * qr * s).to(tl.bfloat16))
    kb = base + H * D
    k = tl.load(kb + rd).to(tl.float32)
    kr = tl.load(kb + rr).to(tl.float32)
    tl.store(KO + (t * H + h) * D + rd, (k * c + sgn * kr * s).to(tl.bfloat16))


@triton.jit
def _silu_mul_kernel(GU, M, FF: tl.constexpr, BLOCK: tl.constexpr):
    """GU (T, 2 FF) = [gate | up] (bf16 GEMM output) -> M (T, FF) = silu(gate) * up with the rounding of the eager /
    fused_gemm epilogue (silu rounded to bf16, product rounded to bf16)."""
    t = tl.program_id(0)
    pb = tl.program_id(1)
    r = pb * BLOCK + tl.arange(0, BLOCK)
    m = r < FF
    g = tl.load(GU + t * 2 * FF + r, mask=m, other=0.0).to(tl.float32)
    u = tl.load(GU + t * 2 * FF + FF + r, mask=m, other=0.0).to(tl.float32)
    s = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(M + t * FF + r, (s * u).to(tl.bfloat16), mask=m)


def rope_qk(y, cos, sin, H, D):
    N = y.shape[0]
    q = torch.empty(N, H, D, device=y.device, dtype=y.dtype)
    k = torch.empty_like(q)
    _rope_kernel[(N, H)](y, cos, sin, q, k, y.stride(0), H=H, D=D, num_warps=1)
    return q, k


# fused_gemm configs (BN, BK, num_warps) per (padded rows TP, N, K) and the Triton/cuBLAS split, from
# runs/batch/dev/tune_gemm.py (A100-80GB, CUDA graphs, L2-defeating weight rotation): Triton wins every shape up to 32 rows
# and the K = 2048 shapes up to 64 rows; the down projection (K = 5632) goes to cuBLAS above 32 rows.
GCFG = {(16, 6144, 2048): (64, 128, 4), (32, 6144, 2048): (64, 128, 8), (64, 6144, 2048): (64, 128, 4),
        (16, 2048, 2048): (32, 512, 4), (32, 2048, 2048): (32, 512, 4), (64, 2048, 2048): (32, 256, 4),
        (16, 5632, 2048): (64, 128, 4), (32, 5632, 2048): (64, 128, 4), (64, 5632, 2048): (64, 128, 4),
        (16, 2048, 5632): (32, 512, 4), (32, 2048, 5632): (32, 512, 4)}


def mm(h, W, silu=False):
    """h (T, K) @ W^T: fused_gemm small-M Triton kernels where they beat cuBLAS (GCFG), cuBLAS otherwise (+ SiLU*up kernel)."""
    T, K = h.shape
    Nout = W.shape[0] // 2 if silu else W.shape[0]
    TP = max(16, triton.next_power_of_2(T))
    cfg = GCFG.get((TP, Nout, K)) if T <= GEMM_TMAX else None
    if cfg is not None:
        return fg.gemm(h, W, silu, cfg)
    y = h @ W.t()
    if not silu:
        return y
    FF = y.shape[1] // 2
    m = torch.empty(T, FF, device=y.device, dtype=y.dtype)
    _silu_mul_kernel[(T, triton.cdiv(FF, 1024))](y, m, FF=FF, BLOCK=1024, num_warps=4)
    return m


class BatchEngine:
    AVG = tuple(range(0, 17))                        # verified drafts per row of the cost-aware buckets (<= k)

    def __init__(self, ouro, dr=None, B=8, tmax=1024, host_overhead_ms=0.03):
        o = ouro; c = o.c
        self.o, self.dr, self.B, self.tmax = o, dr, B, tmax
        self.R, self.NL, self.H, self.D = o.R, o.L, c["num_attention_heads"], c["head_dim"]
        self.eps = c["rms_norm_eps"]
        dev, dt = o.device, o.dtype
        self.dev = dev
        self.kc = torch.zeros(self.R, self.NL, B, self.H, tmax, self.D, device=dev, dtype=dt)
        self.vc = torch.zeros_like(self.kc)
        self.k = dr.k if dr is not None else 0
        k, k1 = self.k, self.k + 1
        self.k1 = k1
        self.fp32p = os.environ.get("FP32P", "1") == "1"
        self.det = os.environ.get("DET", "0") == "1"     # batch-invariant target path (det_kernels): bitwise plain == LoopAhead
        self.catt = CacheAttention(self.H, self.D, tmax, dev, nqp=max(16, triton.next_power_of_2(k1))) if self.det else None
        self.atts = {}
        self.ar = torch.arange(max(B, k1) + 1, device=dev)
        self.ones = torch.ones(B, dtype=torch.long, device=dev)
        self.Lc = torch.zeros(B, dtype=torch.long, device=dev)
        self.y = torch.zeros(B, dtype=torch.long, device=dev)
        self.act = torch.zeros(B, dtype=torch.bool, device=dev)
        self.act_h = torch.zeros(B, dtype=torch.bool).pin_memory()
        self.res = torch.zeros(B, k1 + 3, dtype=torch.long, device=dev)      # [n_new, tokens (k1), k_b, 1000 E_b(k_b)]
        self.hres = torch.zeros(2, B, k1 + 3, dtype=torch.long).pin_memory()
        self.ev = [torch.cuda.Event() for _ in range(2)]
        self.slot = 0
        self.host_overhead = host_overhead_ms
        if dr is not None:
            assert not dr.use_noise and not dr.shift and not dr.flowmap and not getattr(dr, "feedback", False), \
                "engine_batch supports the mask drafter (no noise / shift / flow map / feedback), one pass"
            self.d = torch.zeros(B, k, dtype=torch.long, device=dev)
            self.gains = torch.zeros(B, k, device=dev)
            self.sid = torch.zeros(B, k1, dtype=torch.long, device=dev)
            self.is_clean = torch.zeros(B, k1, dtype=torch.bool, device=dev)
            self.is_clean[:, 0] = True
            self.t_in = self.is_clean.float()
            self.choice = torch.zeros(1, dtype=torch.long, device=dev)
            self.hchoice = torch.zeros(1, dtype=torch.long).pin_memory()
            self.calib = torch.ones(1, device=dev)          # scale of the confidences in the cost rule (CALIB=1: set online, see _calibrate)
            self.calib_on = os.environ.get("CALIB", "0") == "1"
            self.cal_n = [0.0, 0.0]                          # accepted and predicted drafts of the cost-aware cycles so far (host)
            self.evc = torch.cuda.Event()
            self.LQ, self.LO = [], []
            for lm in dr.lora:
                self.LQ.append(tuple(t for n in ("q_proj", "k_proj", "v_proj") for t in (lm[n].A, lm[n].B)))
                self.LO.append(tuple(t for n in ("o_proj", "gate_proj", "up_proj", "down_proj") for t in (lm[n].A, lm[n].B)))
            self.merge = os.environ.get("MERGE_LORA", "1") == "1"
            if not self.merge:
                self.pre = torch.compile(pre_block, dynamic=False)
                self.post = torch.compile(post_block, dynamic=False)
        self.W = []
        for ly in o.layers:
            self.W.append(dict(qkv=torch.cat([ly.q_proj.weight, ly.k_proj.weight, ly.v_proj.weight]).contiguous(),
                               o=ly.o_proj.weight, gu=torch.cat([ly.gate_proj.weight, ly.up_proj.weight]).contiguous(),
                               d=ly.down_proj.weight, n1=ly.input_layernorm.weight, n2=ly.input_layernorm_2.weight,
                               n3=ly.post_attention_layernorm.weight, n4=ly.post_attention_layernorm_2.weight))
        if dr is not None and self.merge:
            # MERGE_LORA=1 (default): the draft pass runs the target's fast path with W + B A (merged in fp32, rounded to
            # bf16 once); MERGE_LORA=0: engine.py's compiled blocks with the fp32 LoRA side path (same as the batch-1 engine)
            self.Wd = []
            for W, lm in zip(self.W, dr.lora):
                dw = lambda n: (lm[n].B.float() @ lm[n].A.float())
                qkv = W["qkv"].float() + torch.cat([dw("q_proj"), dw("k_proj"), dw("v_proj")])
                gu = W["gu"].float() + torch.cat([dw("gate_proj"), dw("up_proj")])
                self.Wd.append(dict(W, qkv=qkv.to(dt).contiguous(), o=(W["o"].float() + dw("o_proj")).to(dt).contiguous(),
                                    gu=gu.to(dt).contiguous(), d=(W["d"].float() + dw("down_proj")).to(dt).contiguous()))
        self.graphs, self.pool = {}, None
        self.buckets, self.Mb, self.Ab, self.Tt, self.cost = {}, {}, {}, {}, {}
        self.meta = {}

    # ---- per sub-batch helpers ---------------------------------------------------------------------------------------
    def _att(self, Bs):
        a = self.atts.get(Bs)
        if a is None:
            a = self.atts[Bs] = BatchDecodeAttention(Bs, self.H, self.D, self.tmax, self.dev, nqp=max(16, triton.next_power_of_2(self.k1)),
                                                     fp32p=self.fp32p)
        return a

    def _meta(self, Bs):
        m = self.meta.get(Bs)
        if m is None:
            k1 = self.k1
            m = self.meta[Bs] = dict(qs1=torch.arange(Bs, device=self.dev), qc1=torch.ones(Bs, dtype=torch.long, device=self.dev),
                                     qsd=torch.arange(Bs, device=self.dev) * k1, qcd=torch.full((Bs,), k1, dtype=torch.long, device=self.dev))
        return m

    def bucket_list(self, Bs):
        """Cost-aware buckets: (N, M = N - Bs verified drafts, a = drafts per row for 'uni') with a in AVG, a <= k."""
        bl = self.buckets.get(Bs)
        if bl is None:
            bl = [(Bs * (a + 1), Bs * a, a) for a in self.AVG if a <= self.k]
            bl = sorted(bl + tile_buckets(Bs, Bs * self.k1, {b[0] for b in bl}))
            self.buckets[Bs] = bl
            self.Mb[Bs] = torch.tensor([b[1] for b in bl], device=self.dev)
            self.Ab[Bs] = torch.tensor([b[2] for b in bl], device=self.dev)
            self.Tt[Bs] = torch.ones(len(bl), device=self.dev)
        return bl

    # ---- the target stack over N packed tokens -------------------------------------------------------------------------
    def _stack(self, Bs, N, ids, pos, lens, qs, qc, one, rowid=None):
        o, eps, ns, H, D = self.o, self.eps, self.o.norm.weight, self.H, self.D
        cos, sin = o.rope(pos[None])
        cos, sin = cos[0], sin[0]
        att = self._att(Bs)
        res, dn, n_prev = o.embed_tokens(ids), None, None
        if self.det:                                   # batch-invariant path: same per-token bits for any step composition
            for ut in range(self.R):
                for li in range(self.NL):
                    W = self.W[li]
                    kc, vc = self.kc[ut, li, :Bs], self.vc[ut, li, :Bs]
                    xo, h = fg.prep(res, dn, n_prev, ns if (li == 0 and ut > 0) else None, W["n1"], eps)
                    q = rope_kv(det_mm(h, W["qkv"]), cos, sin, kc, vc, rowid, pos, H, D)
                    a = self.catt(q, kc, vc, lens, qs, qc)
                    x1, h2 = fg.prep(xo, det_mm(a, W["o"]), W["n2"], None, W["n3"], eps)
                    dn = det_mm(det_mm(h2, W["gu"], silu=True), W["d"])
                    res, n_prev = x1, W["n4"]
            xf, _ = fg.prep(res, dn, n_prev, ns, ns, eps)
            return xf
        for ut in range(self.R):
            for li in range(self.NL):
                W = self.W[li]
                xo, h = fg.prep(res, dn, n_prev, ns if (li == 0 and ut > 0) else None, W["n1"], eps)
                y = mm(h, W["qkv"])
                q, k = rope_qk(y, cos, sin, H, D)
                v = y.view(N, 3, H, D)[:, 2]
                a = att(q.view(1, N, H, D).transpose(1, 2), k.view(1, N, H, D).transpose(1, 2), v.unsqueeze(0).transpose(1, 2),
                        self.kc[ut, li, :Bs], self.vc[ut, li, :Bs], lens, qs, qc, full=False, write=True, one=one)
                x1, h2 = fg.prep(xo, mm(a.transpose(1, 2).reshape(N, H * D), W["o"]), W["n2"], None, W["n3"], eps)
                dn = mm(mm(h2, W["gu"], silu=True), W["d"])
                res, n_prev = x1, W["n4"]
        xf, _ = fg.prep(res, dn, n_prev, ns, ns, eps)
        return xf

    def _lm(self, xf):
        return det_mm(xf, self.o.lm_head.weight) if self.det else self.o.lm_head(xf)

    # ---- graph bodies --------------------------------------------------------------------------------------------------
    def _plain(self, Bs):
        m = self._meta(Bs)
        act, L, y = self.act[:Bs], self.Lc[:Bs], self.y[:Bs]
        xf = self._stack(Bs, Bs, y, L, L, m["qs1"], m["qc1"], one=True, rowid=m["qs1"])
        pred = self._lm(xf).argmax(-1)
        r = self.res[:Bs]
        r[:, 0].fill_(1); r[:, 1].copy_(pred); r[:, 2:].zero_()
        self.y[:Bs].copy_(torch.where(act, pred, y))
        self.Lc[:Bs] += act.long()

    def _draft(self, Bs):
        o, dr, k, k1, eps, H, D = self.o, self.dr, self.k, self.k1, self.eps, self.H, self.D
        m = self._meta(Bs)
        Nd = Bs * k1
        L = self.Lc[:Bs]
        pos = (L[:, None] + self.ar[:k1][None, :]).reshape(1, Nd)
        cos, sin = o.rope(pos)
        self.sid[:Bs, 0].copy_(self.y[:Bs])
        h = dr.slot_inputs(None, self.sid[:Bs], self.is_clean[:Bs], self.t_in[:Bs], None).reshape(1, Nd, -1)
        att = self._att(Bs)
        if self.merge:
            res, dn, n_prev, ns = h[0].contiguous(), None, None, o.norm.weight
            for li in range(self.NL):
                W = self.Wd[li]
                xo, hh = fg.prep(res, dn, n_prev, None, W["n1"], eps)
                y = mm(hh, W["qkv"])
                q, kk = rope_qk(y, cos[0], sin[0], H, D)
                v = y.view(Nd, 3, H, D)[:, 2]
                a = att(q.view(1, Nd, H, D).transpose(1, 2), kk.view(1, Nd, H, D).transpose(1, 2), v.unsqueeze(0).transpose(1, 2),
                        self.kc[self.R - 1, li, :Bs], self.vc[self.R - 1, li, :Bs], L, m["qsd"], m["qcd"], full=True, write=False)
                x1, h2 = fg.prep(xo, mm(a.transpose(1, 2).reshape(Nd, H * D), W["o"]), W["n2"], None, W["n3"], eps)
                dn = mm(mm(h2, W["gu"], silu=True), W["d"])
                res, n_prev = x1, W["n4"]
            xf, _ = fg.prep(res, dn, n_prev, ns, ns, eps)
            lg = o.lm_head(xf).float().view(Bs, k1, -1)[:, 1:]
        else:
            for li in range(self.NL):
                W = self.W[li]
                q, kk, v = self.pre(h, cos, sin, W["n1"], W["qkv"], eps, H, self.LQ[li])
                a = att(q, kk, v, self.kc[self.R - 1, li, :Bs], self.vc[self.R - 1, li, :Bs], L, m["qsd"], m["qcd"], full=True, write=False)
                h = self.post(h, a, W["o"], W["n2"], W["n3"], W["gu"], W["d"], W["n4"], eps, self.LO[li])
            lg = o.lm_head(o.norm(h)).float().view(Bs, k1, -1)[:, 1:]
        self.d[:Bs].copy_(lg.argmax(-1))
        self.gains[:Bs].copy_(lg.softmax(-1).amax(-1).cumprod(1))

    def _policy(self, Bs, alloc):
        """Expected emitted tokens of each bucket / its measured cycle time -> choice (bucket index)."""
        self.bucket_list(Bs)
        act = self.act[:Bs].float()
        g = self.gains[:Bs] * act[:, None] * self.calib
        if alloc == "rag":
            S = torch.cat([torch.zeros(1, device=self.dev), g.reshape(-1).sort(descending=True).values.cumsum(0)])
            val = act.sum() + S[self.Mb[Bs]]
        else:
            Ec = torch.cat([torch.zeros(Bs, 1, device=self.dev), g.cumsum(1)], 1)
            val = act.sum() + Ec[:, self.Ab[Bs]].sum(0)
        self.choice.copy_((val / self.Tt[Bs]).argmax()[None])

    def _calibrate(self, mode, rec):
        """CALIB=1, after each cost-aware cycle: the drafter's confidences overestimate the accepted drafts (by ~20% on Ouro), so
        the cost rule verified too many drafts at large batch sizes.  Scale them by the accepted / predicted drafts of all cost-aware
        cycles of the run so far (a uniform scale changes how many drafts are verified, not which: the top set is unchanged)."""
        if not getattr(self, "calib_on", False) or mode != "cost":
            return
        for _, n, _, e in rec:
            self.cal_n[0] += n - 1; self.cal_n[1] += e
        if self.cal_n[1] >= 20.0:
            self.calib.fill_(min(1.5, max(0.2, self.cal_n[0] / self.cal_n[1])))

    def _calib_reset(self):
        if getattr(self, "calib_on", False):
            self.cal_n = [0.0, 0.0]; self.calib.fill_(1.0)

    def _verify(self, Bs, N, alloc):
        k, k1, dev = self.k, self.k1, self.dev
        act, L, y, d = self.act[:Bs], self.Lc[:Bs], self.y[:Bs], self.d[:Bs]
        M = N - Bs
        j = self.ar[:k1]
        if alloc == "uni" or M == 0:
            assert M % Bs == 0
            kk = torch.full((Bs,), M // Bs, dtype=torch.long, device=dev)
        else:
            g = self.gains[:Bs] * act[:, None]
            idx = torch.topk(g.reshape(-1), M, sorted=False).indices
            sel = torch.zeros(Bs * k, device=dev)
            sel.scatter_(0, idx, 1.0)
            kk = sel.view(Bs, k).sum(1).long()
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
        r[:, 0].copy_(n_new); r[:, 1:1 + k1].copy_(P); r[:, 1 + k1].copy_(kk)
        e = (self.gains[:Bs] * (j[None, :k] < kk[:, None])).sum(1)
        r[:, 2 + k1].copy_((e * 1000).round().long())
        self.y[:Bs].copy_(torch.where(act, ynew, y))
        self.Lc[:Bs] += n_new * act.long()

    # ---- graph management ----------------------------------------------------------------------------------------------
    def _body(self, key):
        kind, Bs = key[0], key[1]
        if kind == "plain":
            return lambda: self._plain(Bs)
        if kind == "draft":
            return lambda: self._draft(Bs)
        if kind == "pol":
            return lambda: self._policy(Bs, key[2])
        return lambda: self._verify(Bs, key[2], key[3])

    def _state(self):
        st = [self.Lc, self.y, self.act, self.res]
        if self.dr is not None:
            st += [self.d, self.gains, self.sid, self.choice]
        return st

    def capture(self, key):
        if key in self.graphs:
            return
        body = self._body(key)
        saved = [t.clone() for t in self._state()]
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                body()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            body()
        self.pool = g.pool(); self.graphs[key] = g
        for t, v in zip(self._state(), saved):
            t.copy_(v)

    def run(self, key):
        self.graphs[key].replay()

    def keys_for(self, Bs, mode):
        if mode == "plain":
            return [("plain", Bs)]
        if mode == "fixed":
            return [("draft", Bs), ("ver", Bs, Bs * self.k1, "uni")]
        alloc = "rag" if mode == "cost" else "uni"
        if alloc == "uni" and any(a < 0 for N, M, a in self.bucket_list(Bs)):
            raise ValueError("EXTRA_N_STEP buckets are not multiples of the batch size: use the ragged cost mode")
        return [("draft", Bs), ("pol", Bs, alloc)] + [("ver", Bs, N, alloc) for N, M, a in self.bucket_list(Bs)]

    @torch.no_grad()
    def prepare(self, Bs, modes, Lcal=384):
        """Capture every graph of the modes for Bs rows (state set to a dummy context of length Lcal; the K/V rows it writes
        are garbage that a later prefill / decode overwrites) and measure the cycle cost of each verification bucket."""
        self.Lc[:Bs].fill_(Lcal); self.y[:Bs].fill_(1); self.act[:Bs].fill_(True)
        for mode in modes:
            for key in self.keys_for(Bs, mode):
                self.capture(key)
        torch.cuda.synchronize()
        if any(m in ("cost", "costu") for m in modes):
            self.calibrate(Bs, Lcal)
        self.act[:Bs].fill_(False)

    def _time(self, key, reps=10):
        saved = [t.clone() for t in self._state()]
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(reps):
            for t, v in zip(self._state(), saved):
                t.copy_(v)
            e0.record(); self.run(key); e1.record(); e1.synchronize()
            ts.append(e0.elapsed_time(e1))
        for t, v in zip(self._state(), saved):
            t.copy_(v)
        ts.sort()
        return ts[len(ts) // 2]

    def calibrate(self, Bs, Lcal):
        """T(Bs, N) = draft + policy + verification(N) (+ host overhead), measured on this GPU by replaying the graphs."""
        self.Lc[:Bs].fill_(Lcal); self.act[:Bs].fill_(True)
        if ("draft", Bs) in self.graphs:
            self.run(("draft", Bs))
        td = self._time(("draft", Bs))
        tp = self._time(("pol", Bs, "rag")) if ("pol", Bs, "rag") in self.graphs else self._time(("pol", Bs, "uni"))
        tv = {}
        for N, M, a in self.bucket_list(Bs):
            key = ("ver", Bs, N, "rag") if ("ver", Bs, N, "rag") in self.graphs else ("ver", Bs, N, "uni")
            tv[N] = self._time(key)
        tpl = self._time(("plain", Bs)) if ("plain", Bs) in self.graphs else None
        T = [td + tp + tv[N] + self.host_overhead for N, M, a in self.bucket_list(Bs)]
        self.Tt[Bs].copy_(torch.tensor(T, device=self.dev))
        self.cost[Bs] = dict(draft_ms=td, policy_ms=tp, verify_ms={int(N): v for N, v in tv.items()}, plain_ms=tpl,
                             cycle_ms={int(N): t for (N, M, a), t in zip(self.bucket_list(Bs), T)}, Lcal=Lcal)
        return self.cost[Bs]

    # ---- host <-> device ---------------------------------------------------------------------------------------------
    def _readback(self, Bs):
        s = self.slot; self.slot ^= 1
        self.hres[s, :Bs].copy_(self.res[:Bs], non_blocking=True); self.ev[s].record()
        return s

    def _wait(self, s, Bs):
        self.ev[s].synchronize()
        return self.hres[s, :Bs].tolist()

    # ---- prefill ---------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, prompts, rows):
        """Causal pass over right-padded prompts of the given rows (pads come after every real token, so they never
        influence it; their K/V rows lie beyond the row's length and are overwritten by decoding).  PREFILL=fused (default):
        fused norm chains (fused_gemm.prep), cuBLAS GEMMs, flash SDPA; PREFILL=eager: ouro_min.Layer modules (= engine.py).
        Sets Lc / y / act of the rows (device) and returns the first generated tokens as a device tensor."""
        o, dev, H, D, eps = self.o, self.dev, self.H, self.D, self.eps
        n = len(prompts); P = [len(p) for p in prompts]; Pm = max(P)
        ids = torch.zeros(n, Pm, dtype=torch.long)
        for i, p in enumerate(prompts):
            ids[i, :len(p)] = torch.tensor(p)
        ids = ids.to(dev)
        pos = torch.arange(Pm, device=dev)[None].expand(n, Pm)
        cos, sin = o.rope(pos)
        ridx = torch.tensor(rows, device=dev)
        Pt = torch.tensor(P, device=dev)
        last = torch.arange(n, device=dev) * Pm + Pt - 1
        if os.environ.get("PREFILL", "fused") == "eager":
            x = o.embed_tokens(ids)
            for ut in range(self.R):
                for li, layer in enumerate(o.layers):
                    kc, vc = self.kc[ut, li], self.vc[ut, li]

                    def attn(q, k, v, kc=kc, vc=vc):
                        kc[ridx, :, :Pm] = k; vc[ridx, :, :Pm] = v
                        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
                    x = layer(x, cos, sin, attn)[0]
                x = o.norm(x)
            xl = x.reshape(n * Pm, -1)[last]
        else:
            T, ns = n * Pm, o.norm.weight
            cos, sin = cos.reshape(T, D), sin.reshape(T, D)
            res, dn, n_prev = o.embed_tokens(ids.reshape(T)), None, None
            for ut in range(self.R):
                for li in range(self.NL):
                    W = self.W[li]
                    xo, h = fg.prep(res, dn, n_prev, ns if (li == 0 and ut > 0) else None, W["n1"], eps)
                    y = h @ W["qkv"].t()
                    q, k = rope_qk(y, cos, sin, H, D)
                    q = q.view(n, Pm, H, D).transpose(1, 2); k = k.view(n, Pm, H, D).transpose(1, 2)
                    v = y.view(n, Pm, 3, H, D)[:, :, 2].transpose(1, 2)
                    self.kc[ut, li][ridx, :, :Pm] = k; self.vc[ut, li][ridx, :, :Pm] = v
                    a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                    x1, h2 = fg.prep(xo, a.transpose(1, 2).reshape(T, H * D) @ W["o"].t(), W["n2"], None, W["n3"], eps)
                    gu = h2 @ W["gu"].t()
                    FF = gu.shape[1] // 2
                    m = torch.empty(T, FF, device=dev, dtype=gu.dtype)
                    _silu_mul_kernel[(T, triton.cdiv(FF, 1024))](gu, m, FF=FF, BLOCK=1024, num_warps=4)
                    dn = m @ W["d"].t()
                    res, n_prev = x1, W["n4"]
            xl, _ = fg.prep(res[last].contiguous(), dn[last].contiguous(), n_prev, ns, ns, eps)
        y0 = o.lm_head(xl).argmax(-1)
        self.Lc[ridx] = Pt; self.y[ridx] = y0; self.act[ridx] = True
        if self.dr is not None:
            self.gains[ridx] = 0.0                       # a refilled row verifies no stale drafts in its first cycle
        for r in rows:
            self.act_h[r] = True
        return y0

    # ---- graph-captured single-prompt prefill (continuous-batching refills) -------------------------------------------
    PF_BUCKET = 64

    def _prefill1_body(self, Pm):
        """One prompt right-padded to Pm tokens (ids in self.pf_ids[:Pm], true length in self.pf_P, target row in
        self.pf_row): same math as the fused prefill; writes the row's K/V [0, Pm) of every (loop, layer), sets Lc / y / act
        (/ gains = 0) of the row and leaves the first generated token in self.pf_y."""
        o, H, D, eps, ns = self.o, self.H, self.D, self.eps, self.o.norm.weight
        pos = self.ar_pf[:Pm]
        cos, sin = o.rope(pos[None])
        cos, sin = cos[0], sin[0]
        row = self.pf_row
        res, dn, n_prev = o.embed_tokens(self.pf_ids[:Pm]), None, None
        for ut in range(self.R):
            for li in range(self.NL):
                W = self.W[li]
                xo, h = fg.prep(res, dn, n_prev, ns if (li == 0 and ut > 0) else None, W["n1"], eps)
                y = h @ W["qkv"].t()
                q, k = rope_qk(y, cos, sin, H, D)
                q = q.view(1, Pm, H, D).transpose(1, 2); k = k.view(1, Pm, H, D).transpose(1, 2)
                v = y.view(1, Pm, 3, H, D)[:, :, 2].transpose(1, 2)
                self.kc[ut, li][:, :, :Pm].index_copy_(0, row, k)
                self.vc[ut, li][:, :, :Pm].index_copy_(0, row, v)
                a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                x1, h2 = fg.prep(xo, a.transpose(1, 2).reshape(Pm, H * D) @ W["o"].t(), W["n2"], None, W["n3"], eps)
                gu = h2 @ W["gu"].t()
                FF = gu.shape[1] // 2
                m = torch.empty(Pm, FF, device=self.dev, dtype=gu.dtype)
                _silu_mul_kernel[(Pm, triton.cdiv(FF, 1024))](gu, m, FF=FF, BLOCK=1024, num_warps=4)
                dn = m @ W["d"].t()
                res, n_prev = x1, W["n4"]
        last = (self.pf_P - 1).view(1)
        xl, _ = fg.prep(res.index_select(0, last), dn.index_select(0, last), n_prev, ns, ns, eps)
        y0 = o.lm_head(xl).argmax(-1)
        self.pf_y.copy_(y0)
        self.y.index_copy_(0, row, y0); self.Lc.index_copy_(0, row, self.pf_P.view(1))
        self.act.index_fill_(0, row, True)
        if self.dr is not None:
            self.gains.index_fill_(0, row, 0.0)

    def prepare_prefill1(self, min_len, max_len):
        """Capture the single-prompt prefill graphs for the padded-length buckets covering [min_len, max_len]."""
        if not hasattr(self, "pf_ids"):
            self.pf_ids = torch.zeros(self.tmax, dtype=torch.long, device=self.dev)
            self.ar_pf = torch.arange(self.tmax, device=self.dev)
            self.pf_P = torch.ones(1, dtype=torch.long, device=self.dev)
            self.pf_row = torch.zeros(1, dtype=torch.long, device=self.dev)
            self.pf_y = torch.zeros(1, dtype=torch.long, device=self.dev)
            self.hpf = torch.zeros(1, dtype=torch.long).pin_memory()
        lo = self.PF_BUCKET * math.ceil(min_len / self.PF_BUCKET)
        for Pm in range(lo, max_len + self.PF_BUCKET, self.PF_BUCKET):
            key = ("pf1", Pm)
            if key in self.graphs:
                continue
            self.pf_P.fill_(Pm); self.pf_row.fill_(0)
            body = lambda Pm=Pm: self._prefill1_body(Pm)
            saved = [t.clone() for t in self._state()]
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    body()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                body()
            self.pool = g.pool(); self.graphs[key] = g
            for t, v in zip(self._state(), saved):
                t.copy_(v)
        torch.cuda.synchronize()

    def prefill1(self, prompt, row):
        """Graph-captured prefill of one prompt into one row (falls back to the eager fused prefill beyond the buckets).
        Returns the first generated token (int; one host sync)."""
        P = len(prompt)
        Pm = self.PF_BUCKET * math.ceil(P / self.PF_BUCKET)
        if ("pf1", Pm) not in self.graphs:
            return self.prefill([prompt], [row]).tolist()[0]
        self.pf_ids[:P].copy_(torch.tensor(prompt, dtype=torch.long), non_blocking=False)
        self.pf_ids[P:Pm].zero_()
        self.pf_P.fill_(P); self.pf_row.fill_(row)
        self.graphs[("pf1", Pm)].replay()
        self.act_h[row] = True
        self.hpf.copy_(self.pf_y, non_blocking=True); torch.cuda.current_stream().synchronize()
        return int(self.hpf[0])

    # ---- prefix cache (continuous batching, PREFIX=1): shared prompt prefix prefilled once, suffix-only prefills ----------
    SUF_BUCKET = 32

    @torch.no_grad()
    def build_prefix(self, prompts, row=0):
        """Reference = the longest prefix shared by all prompts (e.g. the few-shot block).  Its K/V (every loop and layer) is
        computed ONCE by a single-prompt prefill of prompts[0] into `row` and kept in (R, L, H, LCP, D) buffers."""
        ref = prompts[0]
        lcp = len(ref)
        for p in prompts[1:]:
            i = 0
            while i < min(lcp, len(p)) and p[i] == ref[i]:
                i += 1
            lcp = i
        lcp = min(lcp, min(len(p) for p in prompts) - 1)
        self.pref_len = lcp
        if lcp < self.SUF_BUCKET:
            self.pref_len = 0
            return 0
        self.prefill1(ref, row)
        self.pk = self.kc[:, :, row, :, :lcp].clone()
        self.pv = self.vc[:, :, row, :, :lcp].clone()
        return lcp

    def _suffix_body(self, S):
        """Suffix of one prompt (ids in self.sf_ids[:S], right-padded; true prompt length self.pf_P; row self.pf_row) at
        positions LCP .. LCP+S-1 over the row's cache that already holds the prefix K/V: writes the suffix K/V, attends
        lower-right causally to [0, LCP+S), sets Lc / y / act (/ gains) of the row; first token -> self.pf_y."""
        o, H, D, eps, ns, P0 = self.o, self.H, self.D, self.eps, self.o.norm.weight, self.pref_len
        from torch.nn.attention.bias import causal_lower_right
        pos = self.ar_pf[P0:P0 + S]
        cos, sin = o.rope(pos[None])
        cos, sin = cos[0], sin[0]
        row = self.pf_row
        mask = causal_lower_right(S, P0 + S)
        res, dn, n_prev = o.embed_tokens(self.sf_ids[:S]), None, None
        for ut in range(self.R):
            for li in range(self.NL):
                W = self.W[li]
                xo, h = fg.prep(res, dn, n_prev, ns if (li == 0 and ut > 0) else None, W["n1"], eps)
                y = h @ W["qkv"].t()
                q, k = rope_qk(y, cos, sin, H, D)
                q = q.view(1, S, H, D).transpose(1, 2); k = k.view(1, S, H, D).transpose(1, 2)
                v = y.view(1, S, 3, H, D)[:, :, 2].transpose(1, 2)
                kc, vc = self.kc[ut, li], self.vc[ut, li]
                kc[:, :, P0:P0 + S].index_copy_(0, row, k)
                vc[:, :, P0:P0 + S].index_copy_(0, row, v)
                ka = kc[:, :, :P0 + S].index_select(0, row)
                va = vc[:, :, :P0 + S].index_select(0, row)
                a = F.scaled_dot_product_attention(q, ka, va, attn_mask=mask)
                x1, h2 = fg.prep(xo, a.transpose(1, 2).reshape(S, H * D) @ W["o"].t(), W["n2"], None, W["n3"], eps)
                gu = h2 @ W["gu"].t()
                FF = gu.shape[1] // 2
                m = torch.empty(S, FF, device=self.dev, dtype=gu.dtype)
                _silu_mul_kernel[(S, triton.cdiv(FF, 1024))](gu, m, FF=FF, BLOCK=1024, num_warps=4)
                dn = m @ W["d"].t()
                res, n_prev = x1, W["n4"]
        last = (self.pf_P - 1 - P0).view(1)
        xl, _ = fg.prep(res.index_select(0, last), dn.index_select(0, last), n_prev, ns, ns, eps)
        y0 = o.lm_head(xl).argmax(-1)
        self.pf_y.copy_(y0)
        self.y.index_copy_(0, row, y0); self.Lc.index_copy_(0, row, self.pf_P.view(1))
        self.act.index_fill_(0, row, True)
        if self.dr is not None:
            self.gains.index_fill_(0, row, 0.0)

    def prepare_suffix(self, prompts):
        if not getattr(self, "pref_len", 0):
            return
        if not hasattr(self, "sf_ids"):
            self.sf_ids = torch.zeros(self.tmax, dtype=torch.long, device=self.dev)
        smax = max(len(p) for p in prompts) - self.pref_len
        for S in range(self.SUF_BUCKET, smax + self.SUF_BUCKET, self.SUF_BUCKET):
            key = ("sfx", self.pref_len, S)
            if key in self.graphs:
                continue
            self.pf_P.fill_(self.pref_len + 1); self.pf_row.fill_(0)
            body = lambda S=S: self._suffix_body(S)
            saved = [t.clone() for t in self._state()]
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(2):
                    body()
            torch.cuda.current_stream().wait_stream(st)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                body()
            self.pool = g.pool(); self.graphs[key] = g
            for t, v in zip(self._state(), saved):
                t.copy_(v)
        torch.cuda.synchronize()

    def prefill_prefixed(self, prompt, row):
        """Copy the cached prefix K/V into the row, then the graph-captured suffix prefill.  Returns the first token (int)."""
        P0 = getattr(self, "pref_len", 0)
        S = len(prompt) - P0
        Sp = self.SUF_BUCKET * math.ceil(max(S, 1) / self.SUF_BUCKET)
        key = ("sfx", P0, Sp)
        if not P0 or key not in self.graphs or S < 1 or Sp + P0 > self.tmax:
            return self.prefill1(prompt, row)
        self.kc[:, :, row, :, :P0].copy_(self.pk); self.vc[:, :, row, :, :P0].copy_(self.pv)
        self.sf_ids[:S].copy_(torch.tensor(prompt[P0:], dtype=torch.long), non_blocking=False)
        self.sf_ids[S:Sp].zero_()
        self.pf_P.fill_(len(prompt)); self.pf_row.fill_(row)
        self.graphs[key].replay()
        self.act_h[row] = True
        self.hpf.copy_(self.pf_y, non_blocking=True); torch.cuda.current_stream().synchronize()
        return int(self.hpf[0])

    # ---- static wave -----------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def run_wave(self, prompts, mode, max_new, stop):
        """Decode len(prompts) prompts together (rows 0..Bs-1) until all finish (finished rows idle).  Returns per-row
        outputs (first token from the prefill + decoded tokens, cut at the stop), per-row cycle records, and times."""
        Bs = len(prompts)
        k1 = self.k1
        self._calib_reset()
        torch.cuda.synchronize(); tp0 = time.perf_counter()
        self.act_h.zero_(); self.act.zero_()
        y0 = self.prefill(prompts, list(range(Bs))).tolist()
        torch.cuda.synchronize(); tp1 = time.perf_counter()
        outs = [[t] for t in y0]
        done = [False] * Bs
        cut = [None] * Bs
        cyc = []                                          # per cycle: list of (row, n_new, k_b) for rows active on the host
        nchoice = []

        def finish_check(b, p0):
            o_ = outs[b]
            if len(o_) >= max_new or stop(o_):
                n_stop = min(len(o_), max_new)
                for n in range(max(p0, 1), min(len(o_), max_new) + 1):
                    if stop(o_[:n]):
                        n_stop = n
                        break
                cut[b] = n_stop
                return True
            return False

        for b in range(Bs):                                # stop on the first token already
            if finish_check(b, 1):
                done[b] = True; self.act_h[b] = False
        self.act[:Bs].copy_(self.act_h[:Bs], non_blocking=True)
        t0 = time.perf_counter()

        def process(r):
            changed = False
            rec = []
            for b in range(Bs):
                if done[b]:
                    continue
                n = r[b][0]
                p0 = len(outs[b])
                outs[b] += r[b][1:1 + n]
                rec.append((b, n, r[b][1 + k1], r[b][2 + k1] / 1000.0))
                if finish_check(b, p0 + 1):
                    done[b] = True; self.act_h[b] = False; changed = True
            cyc.append(rec)
            self._calibrate(mode, rec)
            if changed:
                self.act[:Bs].copy_(self.act_h[:Bs], non_blocking=True)
            return all(done)

        if all(done):
            pass
        elif mode in ("plain", "fixed"):
            keys = self.keys_for(Bs, mode)

            def launch():
                for key in keys:
                    self.run(key)
                return self._readback(Bs)
            s = launch()
            while True:
                s_next = launch()                          # launch-ahead: inputs are produced on the device
                if process(self._wait(s, Bs)):
                    break
                s = s_next
        else:
            alloc = "rag" if mode == "cost" else "uni"
            bl = self.bucket_list(Bs)

            def launch_draft():
                self.run(("draft", Bs)); self.run(("pol", Bs, alloc))
                self.hchoice.copy_(self.choice, non_blocking=True); self.evc.record()
            launch_draft()
            while True:
                self.evc.synchronize()
                ci = int(self.hchoice[0]); nchoice.append(ci)
                self.run(("ver", Bs, bl[ci][0], alloc))
                s = self._readback(Bs)
                launch_draft()                             # next cycle's draft runs while the host processes this result
                if process(self._wait(s, Bs)):
                    break
        t1 = time.perf_counter()
        torch.cuda.synchronize()                           # drain the look-ahead work (not timed)
        self.act[:Bs].zero_(); self.act_h.zero_()
        outs = [o_[:c] for o_, c in zip(outs, cut)]
        return dict(outs=outs, cycles=cyc, choices=nchoice, t_prefill=tp1 - tp0, t_decode=t1 - t0,
                    calib=self.calib.item() if getattr(self, "calib_on", False) else None)

    # ---- continuous batching -------------------------------------------------------------------------------------------
    def ladder(self):
        """Widths used at the tail of continuous batching: B, B/2, ..., 1."""
        out, b = [], self.B
        while b >= 1:
            out.append(b); b //= 2
        return out

    def _push_act(self):
        """Host active mask -> device, through a ring of pinned buffers (an in-flight async copy never sees later edits)."""
        if not hasattr(self, "_ring"):
            self._ring = [torch.zeros(self.B, dtype=torch.bool).pin_memory() for _ in range(8)]; self._ri = 0
        buf = self._ring[self._ri]; self._ri = (self._ri + 1) % len(self._ring)
        buf.copy_(self.act_h)
        self.act.copy_(buf, non_blocking=True)

    @torch.no_grad()
    def _compact(self, rows, Bs_new):
        """Move the active rows (host list) into rows < Bs_new: K/V rows up to each row's length (+ slack) and the per-row
        device state.  Call after a full sync (no cycle in flight).  Returns {old row: new row}."""
        Lh = self.Lc.tolist()
        keep = [r for r in rows if r < Bs_new]
        free = [r for r in range(Bs_new) if r not in keep]
        mp = {r: r for r in keep}
        st = [self.Lc, self.y, self.act] + ([self.d, self.gains] if self.dr is not None else [])
        for r in rows:
            if r >= Bs_new:
                f = free.pop(0)
                Lm = min(self.tmax, Lh[r] + 2 * self.k1 + 2)
                self.kc[:, :, f, :, :Lm].copy_(self.kc[:, :, r, :, :Lm])
                self.vc[:, :, f, :, :Lm].copy_(self.vc[:, :, r, :, :Lm])
                for t in st:
                    t[f] = t[r]
                mp[r] = f
        return mp

    @torch.no_grad()
    def run_continuous(self, prompts, mode, max_new, stop, compact=True):
        """Continuous batching of the prompt queue on B rows.  A finished row is refilled with the next prompt (one batched
        prefill of all rows freed by a cycle; results of cycles launched before the refill are dropped for that row).  Once
        the queue is empty and the active rows fit half the current width, they are compacted into the first rows (K/V row
        copies) and the narrower graphs take over (as a serving engine's batch shrinks).  Prefills are timed between syncs;
        decode time = total wall time - prefill time."""
        from collections import deque
        B, k1, n = self.B, self.k1, len(prompts)
        ladder = [b for b in self.ladder() if all(key in self.graphs for key in self.keys_for(b, mode))] if compact else [B]
        assert B in ladder
        Bs = B
        outs, cut = [None] * n, [None] * n
        row_seq, row_from = [None] * B, [0] * B
        st = dict(qi=0, cid=0, t_pre=0.0, n_pre=0, ncomp=0)
        first_fill = [True]                                   # the initial fill of all B rows: one batched prefill
        PF1 = os.environ.get("PF_GRAPH", "1") == "1"
        PREFIX = os.environ.get("PREFIX", "0") == "1"
        if PF1 or PREFIX:
            self.prepare_prefill1(min(len(p) for p in prompts), max(len(p) for p in prompts))
        if PREFIX:
            if getattr(self, "pref_src", None) is not prompts:   # the shared prefix is computed once per prompt set (not timed)
                self.build_prefix(prompts); self.pref_src = prompts
                self.prepare_suffix(prompts)
        recs, choices, widths = [], [], []
        self.act.zero_(); self.act_h.zero_()
        self._calib_reset()

        def finish_check(i, p0):
            o_ = outs[i]
            if len(o_) >= max_new or stop(o_):
                n_stop = min(len(o_), max_new)
                for m in range(max(p0, 1), min(len(o_), max_new) + 1):
                    if stop(o_[:m]):
                        n_stop = m
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
                if PREFIX:                                   # prefix cache: copy shared-prefix K/V + suffix-only prefill
                    y0 = [self.prefill_prefixed(prompts[i], b) for b, i in take]
                elif first_fill[0] or not PF1:
                    y0 = self.prefill([prompts[i] for _, i in take], [b for b, _ in take]).tolist()
                else:                                        # refills: one graph-captured prefill per prompt
                    y0 = [self.prefill1(prompts[i], b) for b, i in take]
                first_fill[0] = False
                st["t_pre"] += time.perf_counter() - ta; st["n_pre"] += 1
                rows = []
                for (b, i), t in zip(take, y0):
                    outs[i] = [t]; row_seq[b] = i; row_from[b] = st["cid"]
                    if finish_check(i, 1):                      # stopped on the prefill token: free the row again
                        row_seq[b] = None; self.act_h[b] = False; rows.append(b)
            self._push_act()

        def process(c, r, Bs_c):
            freed, rec = [], []
            for b in range(Bs_c):
                i = row_seq[b]
                if i is None or c < row_from[b]:
                    continue
                m = r[b][0]; p0 = len(outs[i])
                outs[i] += r[b][1:1 + m]
                rec.append((b, m, r[b][1 + k1], r[b][2 + k1] / 1000.0))
                if finish_check(i, p0 + 1):
                    row_seq[b] = None; self.act_h[b] = False; freed.append(b)
            recs.append(rec); widths.append(Bs_c)
            self._calibrate(mode, rec)
            return freed

        def maybe_compact(drain):
            """Queue empty and the active rows fit half the width: drain, sync, compact.  Returns True if compacted."""
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
        if mode in ("plain", "fixed"):
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
                    launch()                                      # launch-ahead: inputs are produced on the device
                c, s, Bs_c = inflight.popleft()
                freed = process(c, self._wait(s, Bs_c), Bs_c)
                if freed:
                    refill(freed) if st["qi"] < n else self._push_act()
                maybe_compact(drain)
        else:
            alloc = "rag" if mode == "cost" else "uni"
            pend = {}

            def launch_draft():
                self.run(("draft", Bs)); self.run(("pol", Bs, alloc))
                self.hchoice.copy_(self.choice, non_blocking=True); self.evc.record(); pend["w"] = Bs
            launch_draft()
            while any(x is not None for x in row_seq) or st["qi"] < n:
                self.evc.synchronize()
                ci = int(self.hchoice[0]); bl = self.bucket_list(Bs); choices.append(bl[ci][0] / Bs)
                self.run(("ver", Bs, bl[ci][0], alloc))
                c = st["cid"]; s = self._readback(Bs); st["cid"] += 1
                launch_draft()                                    # next cycle's draft runs while the host processes
                freed = process(c, self._wait(s, Bs), Bs)
                if freed:
                    refill(freed) if st["qi"] < n else self._push_act()
                if maybe_compact(lambda: None):
                    launch_draft()                                # the in-flight draft used the old row layout
        T1 = time.perf_counter()
        torch.cuda.synchronize()                                  # drain the look-ahead work (not timed)
        self.act.zero_(); self.act_h.zero_()
        outs = [o_[:c_] for o_, c_ in zip(outs, cut)]
        return dict(outs=outs, cycles=recs, choices=choices, widths=widths, t_total=T1 - T0, t_prefill=st["t_pre"],
                    t_decode=T1 - T0 - st["t_pre"], n_prefill_calls=st["n_pre"], n_compactions=st["ncomp"],
                    calib=self.calib.item() if getattr(self, "calib_on", False) else None)
