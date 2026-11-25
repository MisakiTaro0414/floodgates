"""Split-KV decode attention (Triton) for engine.py: q (1, H, NQ, D) attends to the static cache rows [0, L) (L read from a
DEVICE tensor, so one CUDA graph serves every context length) plus the NQ new tokens of the step with a static intra-step
mask (causal / block-causal / block-full); optionally writes the new K/V rows into the cache at L .. L+NQ-1.
Stage 1: grid (H, NS); program (h, s) handles cache chunk s (CHUNK rows); program (h, 0) also handles the new block.
Stage 2: merges the NS partial (m, l, acc) per (head, query) -> (1, H, NQ, D) bf16.  bf16 QK^T / PV on tensor cores, fp32
softmax, P cast to bf16 before PV (flash-attention numerics)."""
import math
import torch
import triton
import triton.language as tl

CHUNK = 256      # cache rows per split program
BN = 64          # rows per tile


@triton.jit
def _split_kernel(Q, KN, VN, KC, VC, LEN, OP, MP, LP,
                  sqh, sqt, skh, skt, svh, svt, sch, sct, svch, svct, soph, sops, sopt, smh, sms, sm_scale,
                  NQ: tl.constexpr, NQP: tl.constexpr, D: tl.constexpr, CH: tl.constexpr, BT: tl.constexpr,
                  BLK: tl.constexpr, FULL: tl.constexpr, WRITE: tl.constexpr):
    h = tl.program_id(0)
    s = tl.program_id(1)
    L = tl.load(LEN).to(tl.int32)
    rq = tl.arange(0, NQP)
    rd = tl.arange(0, D)
    qm = rq < NQ
    q = tl.load(Q + h * sqh + rq[:, None] * sqt + rd[None, :], mask=qm[:, None], other=0.0)
    m_i = tl.full([NQP], float("-inf"), tl.float32)
    l_i = tl.zeros([NQP], tl.float32)
    acc = tl.zeros([NQP, D], tl.float32)
    start = s * CH
    end = tl.minimum(start + CH, L)
    for n0 in range(start, end, BT):
        rn = n0 + tl.arange(0, BT)
        km = rn < end
        k = tl.load(KC + h * sch + rn[:, None] * sct + rd[None, :], mask=km[:, None], other=0.0)
        v = tl.load(VC + h * svch + rn[:, None] * svct + rd[None, :], mask=km[:, None], other=0.0)
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
        nm = rn < NQ
        k = tl.load(KN + h * skh + rn[:, None] * skt + rd[None, :], mask=nm[:, None], other=0.0)
        v = tl.load(VN + h * svh + rn[:, None] * svt + rd[None, :], mask=nm[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        same = (rq[:, None] // BLK) == (rn[None, :] // BLK)
        if FULL:
            allowed = same
        else:
            allowed = same & ((rn[None, :] % BLK) <= (rq[:, None] % BLK))
        allowed = (allowed | (~qm[:, None])) & nm[None, :]        # padded query rows: anything valid (discarded)
        qk = tl.where(allowed, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
        if WRITE:
            tl.store(KC + h * sch + (L + rn)[:, None] * sct + rd[None, :], k, mask=nm[:, None])
            tl.store(VC + h * svch + (L + rn)[:, None] * svct + rd[None, :], v, mask=nm[:, None])
    tl.store(OP + h * soph + s * sops + rq[:, None] * sopt + rd[None, :], acc, mask=qm[:, None])
    tl.store(MP + h * smh + s * sms + rq, m_i, mask=qm)
    tl.store(LP + h * smh + s * sms + rq, l_i, mask=qm)


@triton.jit
def _split1_kernel(Q, KN, VN, KC, VC, LEN, OP, MP, LP,
                   sqh, skh, svh, sch, sct, svch, svct, soph, sops, smh, sms, sm_scale,
                   D: tl.constexpr, CH: tl.constexpr, BT: tl.constexpr, WRITE: tl.constexpr):
    """NQ = 1 (plain decoding): fp32 scores / probabilities on CUDA cores, P never rounded to bf16 (the numerics of
    flashinfer's decode kernel, i.e. of the SGLang server), same split/merge structure as _split_kernel."""
    h = tl.program_id(0)
    s = tl.program_id(1)
    L = tl.load(LEN).to(tl.int32)
    rd = tl.arange(0, D)
    q = tl.load(Q + h * sqh + rd).to(tl.float32)
    m_i = tl.full([1], float("-inf"), tl.float32)
    l_i = tl.zeros([1], tl.float32)
    acc = tl.zeros([1, D], tl.float32)
    start = s * CH
    end = tl.minimum(start + CH, L)
    for n0 in range(start, end, BT):
        rn = n0 + tl.arange(0, BT)
        km = rn < end
        k = tl.load(KC + h * sch + rn[:, None] * sct + rd[None, :], mask=km[:, None], other=0.0).to(tl.float32)
        v = tl.load(VC + h * svch + rn[:, None] * svct + rd[None, :], mask=km[:, None], other=0.0).to(tl.float32)
        qk = tl.sum(q[None, :] * k, 1) * sm_scale
        qk = tl.where(km, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        l_i = l_i * alpha + tl.sum(p, 0)
        acc = acc * alpha[:, None] + tl.sum(p[:, None] * v, 0)[None, :]
        m_i = m_new
    if s == 0:
        k = tl.load(KN + h * skh + rd)
        v = tl.load(VN + h * svh + rd)
        qk = tl.sum(q * k.to(tl.float32), 0) * sm_scale
        m_new = tl.maximum(m_i, qk)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        l_i = l_i * alpha + p
        acc = acc * alpha[:, None] + (p[:, None] * v.to(tl.float32)[None, :])
        m_i = m_new
        if WRITE:
            tl.store(KC + h * sch + L * sct + rd, k)
            tl.store(VC + h * svch + L * svct + rd, v)
    tl.store(OP + h * soph + s * sops + rd[None, :], acc)
    tl.store(MP + h * smh + s * sms + tl.arange(0, 1), m_i)
    tl.store(LP + h * smh + s * sms + tl.arange(0, 1), l_i)


@triton.jit
def _combine_kernel(OP, MP, LP, O, soph, sops, sopt, smh, sms, soh, sot,
                    NS: tl.constexpr, NSP: tl.constexpr, D: tl.constexpr):
    h = tl.program_id(0)
    t = tl.program_id(1)
    rs = tl.arange(0, NSP)
    sm = rs < NS
    m = tl.load(MP + h * smh + rs * sms + t, mask=sm, other=float("-inf"))
    l = tl.load(LP + h * smh + rs * sms + t, mask=sm, other=0.0)
    mx = tl.max(m, 0)
    w = tl.exp(m - mx)
    den = tl.sum(w * l, 0)
    rd = tl.arange(0, D)
    o = tl.load(OP + h * soph + rs[:, None] * sops + t * sopt + rd[None, :], mask=sm[:, None], other=0.0)
    out = tl.sum(o * w[:, None], 0) / den
    tl.store(O + h * soh + t * sot + rd, out.to(tl.bfloat16))


class DecodeAttention:
    """Workspace holder for one (H, D, tmax); call (q, kn, vn, kc, vc, length, blk, full, write) -> (1, H, NQ, D)."""

    def __init__(self, H, D, tmax, device, fp32p=True):
        self.H, self.D, self.tmax, self.dev = H, D, tmax, device
        self.fp32p = fp32p                      # NQ = 1: fp32-probability kernel (server-style numerics)
        self.NS = math.ceil(tmax / CHUNK)
        self.ws = {}
        self.scale = 1.0 / math.sqrt(D)

    def _ws(self, NQ):
        w = self.ws.get(NQ)
        if w is None:
            H, NS, D = self.H, self.NS, self.D
            w = (torch.empty(H, NS, NQ, D, device=self.dev, dtype=torch.float32),
                 torch.empty(H, NS, NQ, device=self.dev, dtype=torch.float32),
                 torch.empty(H, NS, NQ, device=self.dev, dtype=torch.float32))
            self.ws[NQ] = w
        return w

    def __call__(self, q, kn, vn, kc, vc, length, blk, full=False, write=True):
        H, NQ, D = q.shape[1], q.shape[2], q.shape[3]
        assert q.shape[0] == 1 and kc.shape[0] == 1 and D == self.D
        assert all(t.stride(3) == 1 for t in (q, kn, vn, kc, vc))
        op, mp, lp = self._ws(NQ)
        out = torch.empty(1, NQ, H, D, device=q.device, dtype=q.dtype).permute(0, 2, 1, 3)   # (1, H, NQ, D) view of (1, NQ, H, D)
        NQP = max(16, triton.next_power_of_2(NQ))
        if NQ == 1 and self.fp32p:
            _split1_kernel[(H, self.NS)](q, kn, vn, kc, vc, length, op, mp, lp,
                                         q.stride(1), kn.stride(1), vn.stride(1), kc.stride(1), kc.stride(2), vc.stride(1), vc.stride(2),
                                         op.stride(0), op.stride(1), mp.stride(0), mp.stride(1), self.scale,
                                         D=D, CH=CHUNK, BT=BN, WRITE=write, num_warps=4, num_stages=2)
        else:
            _split_kernel[(H, self.NS)](q, kn, vn, kc, vc, length, op, mp, lp,
                                        q.stride(1), q.stride(2), kn.stride(1), kn.stride(2), vn.stride(1), vn.stride(2),
                                        kc.stride(1), kc.stride(2), vc.stride(1), vc.stride(2),
                                        op.stride(0), op.stride(1), op.stride(2), mp.stride(0), mp.stride(1), self.scale,
                                        NQ=NQ, NQP=NQP, D=D, CH=CHUNK, BT=BN, BLK=blk, FULL=full, WRITE=write,
                                        num_warps=4 if NQP <= 32 else 8, num_stages=2)
        _combine_kernel[(H, NQ)](op, mp, lp, out, op.stride(0), op.stride(1), op.stride(2), mp.stride(0), mp.stride(1),
                                 out.stride(1), out.stride(2), NS=self.NS, NSP=triton.next_power_of_2(self.NS), D=D,
                                 num_warps=4)
        return out


def reference(q, kn, vn, kc, vc, L, blk, full):
    """SDPA over [cache[:L], new] with the same masks (bf16 inputs)."""
    NQ = q.shape[2]
    k = torch.cat([kc[:, :, :L], kn], 2); v = torch.cat([vc[:, :, :L], vn], 2)
    i = torch.arange(NQ, device=q.device)
    same = (i[:, None] // blk) == (i[None, :] // blk)
    blockm = same if full else same & ((i[None, :] % blk) <= (i[:, None] % blk))
    mask = torch.cat([torch.ones(NQ, L, dtype=torch.bool, device=q.device), blockm], 1)[None, None]
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)


if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    H, D, tmax = 16, 128, 4096
    att = DecodeAttention(H, D, tmax, "cuda")
    for NQ, blk, full, write in [(1, 1, False, True), (9, 9, False, True), (72, 9, False, True), (72, 9, True, False), (36, 9, True, False)]:
        att.fp32p = NQ == 1 and write
        for L in [1, 5, 255, 256, 300, 1000, 2500, tmax - 80]:
            kc = torch.randn(1, H, tmax, D, device="cuda", dtype=torch.bfloat16)
            vc = torch.randn(1, H, tmax, D, device="cuda", dtype=torch.bfloat16)
            q = torch.randn(1, H, NQ, D, device="cuda", dtype=torch.bfloat16)
            kn = torch.randn(1, H, NQ, D, device="cuda", dtype=torch.bfloat16)
            vn = torch.randn(1, NQ, 3, H, D, device="cuda", dtype=torch.bfloat16).permute(0, 3, 1, 2, 4)[:, :, :, 2]  # strided like the fused qkv's v
            ref = reference(q, kn, vn, kc, vc, L, blk, full)
            kc2, vc2 = kc.clone(), vc.clone()
            out = att(q, kn, vn, kc2, vc2, torch.tensor([L], device="cuda"), blk, full, write)
            err = (out.float() - ref.float()).abs().max().item()
            wok = (not write) or (torch.equal(kc2[:, :, L:L + NQ], kn) and torch.equal(vc2[:, :, L:L + NQ], vn)
                                  and torch.equal(kc2[:, :, :L], kc[:, :, :L]) and torch.equal(kc2[:, :, L + NQ:], kc[:, :, L + NQ:]))
            print(f"NQ={NQ:3d} blk={blk} full={int(full)} write={int(write)} L={L:5d}: max|err|={err:.2e} nan={bool(out.isnan().any())} write_ok={wok}")
    # timing vs SDPA (masked over a 4096 bucket) at a few lengths
    for NQ, blk, fp in [(1, 1, True), (1, 1, False), (9, 9, False), (72, 9, False)]:
        att.fp32p = fp
        q = torch.randn(1, H, NQ, D, device="cuda", dtype=torch.bfloat16); kn = torch.randn_like(q); vn = torch.randn_like(q)
        kc = torch.randn(1, H, tmax, D, device="cuda", dtype=torch.bfloat16); vc = torch.randn_like(kc)
        for L in [300, 1000, 2500, 4000]:
            Lt = torch.tensor([L], device="cuda")
            for _ in range(3): att(q, kn, vn, kc, vc, Lt, blk, False, True)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(200): att(q, kn, vn, kc, vc, Lt, blk, False, True)
            torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / 200 * 1e6
            pos = L + torch.arange(NQ, device="cuda")
            Lb = 256 * math.ceil((L + NQ) / 256)
            allowed = (torch.arange(Lb, device="cuda")[None, :] <= pos[:, None])[None, None]
            for _ in range(3): torch.nn.functional.scaled_dot_product_attention(q, kc[:, :, :Lb], vc[:, :, :Lb], attn_mask=allowed)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(200): torch.nn.functional.scaled_dot_product_attention(q, kc[:, :, :Lb], vc[:, :, :Lb], attn_mask=allowed)
            torch.cuda.synchronize(); ds = (time.perf_counter() - t0) / 200 * 1e6
            print(f"timing NQ={NQ:3d} fp32p={int(fp)} L={L:5d}: triton {dt:6.1f} us   sdpa(bucket {Lb}) {ds:6.1f} us  (eager launches incl.)")
