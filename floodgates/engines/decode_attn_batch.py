"""Batched, RAGGED split-KV decode attention (Triton) for engine_batch.py; extends decode_attn.py to B sequences.

Layout.  The step's tokens of all B rows are PACKED along one token axis: row b owns packed tokens
qs[b] .. qs[b] + qc[b] - 1 (1 <= qc[b] <= NQP).  q / k_new / v_new are (1, H, N, D) views (any strides, last dim
contiguous), e.g. of the fused q,k,v projection output.  Each row has its own static cache (B, H, tmax, D) per
(loop, layer); its committed length L[b] is read from a DEVICE tensor, so one CUDA graph serves every length mix.
Row b's queries attend to its cache rows [0, L[b]) plus its own new tokens: causal (verification / plain step) or full
(draft slots, bidirectional within the row's block); optionally the new K/V are written at cache rows L[b] + j.
Stage 1: grid (H, NS, B); program (h, s, b) handles cache chunk s (CH rows) of row b; program (h, 0, b) also handles the
row's new block.  Stage 2 (decode_attn._combine_kernel): merges the NS partials per (head, packed token).
Numerics: bf16 QK^T / PV on tensor cores, fp32 softmax, P cast to bf16 before PV (as decode_attn._split_kernel); the NQ=1
kernel (plain steps, every row one token) keeps scores / probabilities in fp32 on CUDA cores (as decode_attn._split1_kernel,
flashinfer-style numerics)."""
import math
import torch
import triton
import triton.language as tl
from floodgates.engines.decode_attn import _combine_kernel

BN = 64          # cache rows per tile


@triton.jit
def _bsplit_kernel(Q, KN, VN, KC, VC, LEN, QS, QC, OP, MP, LP,
                   sqh, sqt, skh, skt, svh, svt, scb, sch, sct, svcb, svch, svct, soph, sops, sopt, smh, sms, sm_scale,
                   NQP: tl.constexpr, D: tl.constexpr, CH: tl.constexpr, BT: tl.constexpr,
                   FULL: tl.constexpr, WRITE: tl.constexpr):
    h = tl.program_id(0)
    s = tl.program_id(1)
    b = tl.program_id(2)
    L = tl.load(LEN + b).to(tl.int32)
    q0 = tl.load(QS + b).to(tl.int32)
    nq = tl.load(QC + b).to(tl.int32)
    rq = tl.arange(0, NQP)
    rd = tl.arange(0, D)
    qm = rq < nq
    q = tl.load(Q + h * sqh + (q0 + rq)[:, None] * sqt + rd[None, :], mask=qm[:, None], other=0.0)
    m_i = tl.full([NQP], float("-inf"), tl.float32)
    l_i = tl.zeros([NQP], tl.float32)
    acc = tl.zeros([NQP, D], tl.float32)
    kb = KC + b * scb + h * sch
    vb = VC + b * svcb + h * svch
    start = s * CH
    end = tl.minimum(start + CH, L)
    for n0 in range(start, end, BT):
        rn = n0 + tl.arange(0, BT)
        km = rn < end
        k = tl.load(kb + rn[:, None] * sct + rd[None, :], mask=km[:, None], other=0.0)
        v = tl.load(vb + rn[:, None] * svct + rd[None, :], mask=km[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk = tl.where(km[None, :], qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    if s == 0:
        rn = tl.arange(0, NQP)
        nm = rn < nq
        k = tl.load(KN + h * skh + (q0 + rn)[:, None] * skt + rd[None, :], mask=nm[:, None], other=0.0)
        v = tl.load(VN + h * svh + (q0 + rn)[:, None] * svt + rd[None, :], mask=nm[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        if FULL:
            allowed = nm[None, :] & (rq[:, None] >= 0)
        else:
            allowed = (rn[None, :] <= rq[:, None]) | (~qm[:, None])      # padded query rows: anything valid (discarded)
            allowed = allowed & nm[None, :]
        qk = tl.where(allowed, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
        if WRITE:
            tl.store(kb + (L + rn)[:, None] * sct + rd[None, :], k, mask=nm[:, None])
            tl.store(vb + (L + rn)[:, None] * svct + rd[None, :], v, mask=nm[:, None])
    tl.store(OP + h * soph + s * sops + (q0 + rq)[:, None] * sopt + rd[None, :], acc, mask=qm[:, None])
    tl.store(MP + h * smh + s * sms + q0 + rq, m_i, mask=qm)
    tl.store(LP + h * smh + s * sms + q0 + rq, l_i, mask=qm)


@triton.jit
def _bsplit1_kernel(Q, KN, VN, KC, VC, LEN, OP, MP, LP,
                    sqh, sqt, skh, skt, svh, svt, scb, sch, sct, svcb, svch, svct, soph, sops, sopt, smh, sms, sm_scale,
                    D: tl.constexpr, CH: tl.constexpr, BT: tl.constexpr, WRITE: tl.constexpr):
    """Every row exactly one new token (packed token b = row b): fp32 scores / probabilities on CUDA cores."""
    h = tl.program_id(0)
    s = tl.program_id(1)
    b = tl.program_id(2)
    L = tl.load(LEN + b).to(tl.int32)
    rd = tl.arange(0, D)
    q = tl.load(Q + h * sqh + b * sqt + rd).to(tl.float32)
    m_i = tl.full([1], float("-inf"), tl.float32)
    l_i = tl.zeros([1], tl.float32)
    acc = tl.zeros([1, D], tl.float32)
    kb = KC + b * scb + h * sch
    vb = VC + b * svcb + h * svch
    start = s * CH
    end = tl.minimum(start + CH, L)
    for n0 in range(start, end, BT):
        rn = n0 + tl.arange(0, BT)
        km = rn < end
        k = tl.load(kb + rn[:, None] * sct + rd[None, :], mask=km[:, None], other=0.0).to(tl.float32)
        v = tl.load(vb + rn[:, None] * svct + rd[None, :], mask=km[:, None], other=0.0).to(tl.float32)
        qk = tl.sum(q[None, :] * k, 1) * sm_scale
        qk = tl.where(km, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        l_i = l_i * alpha + tl.sum(p, 0)
        acc = acc * alpha[:, None] + tl.sum(p[:, None] * v, 0)[None, :]
        m_i = m_new
    if s == 0:
        k = tl.load(KN + h * skh + b * skt + rd)
        v = tl.load(VN + h * svh + b * svt + rd)
        qk = tl.sum(q * k.to(tl.float32), 0) * sm_scale
        m_new = tl.maximum(m_i, qk)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        l_i = l_i * alpha + p
        acc = acc * alpha[:, None] + (p[:, None] * v.to(tl.float32)[None, :])
        m_i = m_new
        if WRITE:
            tl.store(kb + L * sct + rd, k)
            tl.store(vb + L * svct + rd, v)
    tl.store(OP + h * soph + s * sops + b * sopt + rd[None, :], acc)
    tl.store(MP + h * smh + s * sms + b + tl.arange(0, 1), m_i)
    tl.store(LP + h * smh + s * sms + b + tl.arange(0, 1), l_i)


# (CH, BT, num_warps, num_stages) per batch size, from runs/batch/dev/tune_attn.py (A100-80GB, CUDA graphs, L = 500, H = 16,
# D = 128, tmax = 1024): NQ=1 fp32 kernel ~1.4 TB/s and tensor-core kernel ~1.2 TB/s at B = 32.
CFG1 = {1: (128, 128, 4, 2), 4: (128, 128, 4, 2), 8: (128, 128, 4, 2), 16: (128, 128, 4, 2), 32: (128, 128, 4, 2)}
CFGN = {1: (128, 128, 4, 3), 4: (512, 128, 4, 3), 8: (512, 64, 4, 3), 16: (512, 64, 4, 2), 32: (512, 64, 4, 2)}


def _cfg(table, B):
    return table[max([b for b in table if b <= B] or [min(table)])]


class BatchDecodeAttention:
    """Workspace holder for one (B, H, D, tmax); call
    (q, kn, vn, kc, vc, lens, qs, qc, full, write, one=False) -> (1, H, N, D).
    q/kn/vn: (1, H, N, D) packed views; kc/vc: (B, H, tmax, D); lens/qs/qc: (B,) int64 device tensors.
    one=True: every row has exactly one token (N = B, packed token b = row b) -> fp32 kernel when fp32p."""

    def __init__(self, B, H, D, tmax, device, nqp=16, fp32p=True, cfg1=None, cfgn=None):
        self.B, self.H, self.D, self.tmax, self.dev, self.NQP = B, H, D, tmax, device, nqp
        self.fp32p = fp32p
        self.cfg1 = cfg1 or _cfg(CFG1, B)          # (CH, BT, num_warps, num_stages) of the NQ=1 fp32 kernel
        self.cfgn = cfgn or _cfg(CFGN, B)          # ... of the tensor-core kernel
        self.ws = {}
        self.scale = 1.0 / math.sqrt(D)

    def _ws(self, N, NS):
        w = self.ws.get((N, NS))
        if w is None:
            H, D = self.H, self.D
            w = (torch.empty(H, NS, N, D, device=self.dev, dtype=torch.float32),
                 torch.empty(H, NS, N, device=self.dev, dtype=torch.float32),
                 torch.empty(H, NS, N, device=self.dev, dtype=torch.float32))
            self.ws[(N, NS)] = w
        return w

    def __call__(self, q, kn, vn, kc, vc, lens, qs, qc, full=False, write=True, one=False):
        _, H, N, D = q.shape
        B = kc.shape[0]
        assert D == self.D and H == self.H and kc.shape[2] == self.tmax
        assert all(t.stride(3) == 1 for t in (q, kn, vn, kc, vc))
        use1 = one and self.fp32p
        ch, bt, nw, ns = self.cfg1 if use1 else self.cfgn
        NS = math.ceil(self.tmax / ch)
        op, mp, lp = self._ws(N, NS)
        out = torch.empty(1, N, H, D, device=q.device, dtype=q.dtype).permute(0, 2, 1, 3)   # (1, H, N, D) view of (1, N, H, D)
        args = (q.stride(1), q.stride(2), kn.stride(1), kn.stride(2), vn.stride(1), vn.stride(2),
                kc.stride(0), kc.stride(1), kc.stride(2), vc.stride(0), vc.stride(1), vc.stride(2),
                op.stride(0), op.stride(1), op.stride(2), mp.stride(0), mp.stride(1), self.scale)
        if use1:
            assert N == B
            _bsplit1_kernel[(H, NS, B)](q, kn, vn, kc, vc, lens, op, mp, lp, *args,
                                        D=D, CH=ch, BT=bt, WRITE=write, num_warps=nw, num_stages=ns)
        else:
            _bsplit_kernel[(H, NS, B)](q, kn, vn, kc, vc, lens, qs, qc, op, mp, lp, *args,
                                       NQP=self.NQP, D=D, CH=ch, BT=bt, FULL=full, WRITE=write,
                                       num_warps=nw, num_stages=ns)
        _combine_kernel[(H, N)](op, mp, lp, out, op.stride(0), op.stride(1), op.stride(2), mp.stride(0), mp.stride(1),
                                out.stride(1), out.stride(2), NS=NS, NSP=triton.next_power_of_2(NS), D=D,
                                num_warps=4)
        return out


def reference(q, kn, vn, kc, vc, lens, qs, qc, full):
    """Per-row SDPA over [cache[b, :, :L_b], new tokens of row b] (bf16 inputs) -> (1, H, N, D)."""
    out = torch.empty_like(q)
    for b in range(kc.shape[0]):
        L, s, c = int(lens[b]), int(qs[b]), int(qc[b])
        qq = q[:, :, s:s + c]
        k = torch.cat([kc[b:b + 1, :, :L], kn[:, :, s:s + c]], 2)
        v = torch.cat([vc[b:b + 1, :, :L], vn[:, :, s:s + c]], 2)
        i = torch.arange(c, device=q.device)
        blk = torch.ones(c, c, dtype=torch.bool, device=q.device) if full else (i[None, :] <= i[:, None])
        mask = torch.cat([torch.ones(c, L, dtype=torch.bool, device=q.device), blk], 1)[None, None]
        out[:, :, s:s + c] = torch.nn.functional.scaled_dot_product_attention(qq, k, v, attn_mask=mask)
    return out


if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    H, D, tmax = 16, 128, 1024
    for B in (1, 3, 8, 32):
        att = BatchDecodeAttention(B, H, D, tmax, "cuda")
        for mode in ("one", "ragged_causal", "uniform_full", "ragged_full"):
            g = torch.Generator().manual_seed(B)
            if mode == "one":
                qc = torch.ones(B, dtype=torch.long)
            elif mode == "uniform_full":
                qc = torch.full((B,), 13, dtype=torch.long)
            else:
                qc = torch.randint(1, 14, (B,), generator=g)
            lens = torch.randint(0, tmax - 20, (B,), generator=g); lens[0] = min(int(lens[0]), 5)
            qs = torch.cumsum(qc, 0) - qc
            N = int(qc.sum())
            lens, qs, qc = lens.cuda(), qs.cuda(), qc.cuda()
            kc = torch.randn(B, H, tmax, D, device="cuda", dtype=torch.bfloat16)
            vc = torch.randn(B, H, tmax, D, device="cuda", dtype=torch.bfloat16)
            y = torch.randn(1, N, 3, H, D, device="cuda", dtype=torch.bfloat16)          # fused-qkv-like strides
            q, kn, vn = y[:, :, 0].permute(0, 2, 1, 3), y[:, :, 1].permute(0, 2, 1, 3), y[:, :, 2].permute(0, 2, 1, 3)
            full = "full" in mode
            write = not full
            ref = reference(q, kn, vn, kc, vc, lens, qs, qc, full)
            kc2, vc2 = kc.clone(), vc.clone()
            out = att(q, kn, vn, kc2, vc2, lens, qs, qc, full, write, one=(mode == "one"))
            err = (out.float() - ref.float()).abs().max().item()
            wok = True
            if write:
                for b in range(B):
                    L, s, c = int(lens[b]), int(qs[b]), int(qc[b])
                    wok &= torch.equal(kc2[b, :, L:L + c], kn[0, :, s:s + c]) and torch.equal(vc2[b, :, L:L + c], vn[0, :, s:s + c])
                    wok &= torch.equal(kc2[b, :, :L], kc[b, :, :L]) and torch.equal(kc2[b, :, L + c:], kc[b, :, L + c:])
            else:
                wok = torch.equal(kc2, kc) and torch.equal(vc2, vc)
            print(f"B={B:2d} cfg1={att.cfg1} cfgn={att.cfgn} {mode:13s} N={N:4d}: max|err|={err:.2e} nan={bool(out.isnan().any())} cache_ok={wok}", flush=True)
    # timing (eager launches incl.) at a typical GSM8K context
    for B in (1, 8, 32):
        att = BatchDecodeAttention(B, H, D, tmax, "cuda")
        kc = torch.randn(B, H, tmax, D, device="cuda", dtype=torch.bfloat16); vc = torch.randn_like(kc)
        for nq in (1, 13):
            N = B * nq
            q = torch.randn(1, H, N, D, device="cuda", dtype=torch.bfloat16); kn = torch.randn_like(q); vn = torch.randn_like(q)
            lens = torch.full((B,), 500, device="cuda"); qc = torch.full((B,), nq, device="cuda"); qs = torch.cumsum(qc, 0) - qc
            f = lambda: att(q, kn, vn, kc, vc, lens, qs, qc, False, False, one=(nq == 1))
            for _ in range(3): f()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(100): f()
            torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / 100 * 1e6
            print(f"timing B={B:2d} nq={nq:2d} L=500: {dt:7.1f} us", flush=True)
