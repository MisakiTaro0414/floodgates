"""Batch-invariant kernels for engine_batch.py's deterministic mode (DET=1): every per-token result is bitwise independent of
how many tokens share the step (plain step: B tokens; LoopAhead verification: any packed ragged block), so greedy LoopAhead
reproduces plain batched decoding EXACTLY (not only up to bf16 near-ties).
  det_mm      : M-tiled GEMM with ONE (BN, BK) per weight shape for every M (the per-element K reduction order never
                depends on M; no split-K), optional SiLU*up epilogue (fused_gemm rounding).
  rope_kv     : RoPE of q, k from the fused q,k,v projection + writes k, v into each token's own cache row
                (row id, position per token) BEFORE attention.
  cache_attn  : split-KV attention over the cache only: a query at position p reads keys [0, p] in chunks / tiles aligned to
                ABSOLUTE positions, so its result does not depend on which tokens were in the same step (the decode_attn
                kernels instead treat the step's own tokens as a separate block -> step-dependent order); fixed config.
                Partials merged by decode_attn._combine_kernel (fixed NS)."""
import math
import torch
import triton
import triton.language as tl
from floodgates.engines.decode_attn import _combine_kernel


@triton.jit
def _det_gemm_kernel(H, W, W2, Y, M, N, K, swn, syt, TM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SILU: tl.constexpr,
                     I64: tl.constexpr = False):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rm = pid_m * TM + tl.arange(0, TM)
    if I64:
        rm = rm.to(tl.int64)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros([TM, BN], tl.float32)
    acc2 = tl.zeros([TM, BN], tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        h = tl.load(H + rm[:, None] * K + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        w = tl.load(W + rn[:, None] * swn + rk[None, :], mask=(rn[:, None] < N) & (rk[None, :] < K), other=0.0)
        acc += tl.dot(h, tl.trans(w))
        if SILU:
            w2 = tl.load(W2 + rn[:, None] * swn + rk[None, :], mask=(rn[:, None] < N) & (rk[None, :] < K), other=0.0)
            acc2 += tl.dot(h, tl.trans(w2))
    if SILU:
        g = acc.to(tl.bfloat16).to(tl.float32)
        u = acc2.to(tl.bfloat16).to(tl.float32)
        s = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
        out = (s * u).to(tl.bfloat16)
    else:
        out = acc.to(tl.bfloat16)
    tl.store(Y + rm[:, None] * syt + rn[None, :], out, mask=(rm[:, None] < M) & (rn[None, :] < N))


# BK per (N, K) is FIXED for every M (batch invariance); (BN, TM, num_warps, num_stages) per M bucket only choose which
# program computes an element.  DET_TUNED (runs/batch/dev/tune_det.json, written by runs/batch/dev/tune_det.py) overrides
# the defaults when present.
DET_CFG = {(6144, 2048): (64, 64), (2048, 2048): (64, 64), (5632, 2048): (64, 64), (2048, 5632): (64, 64), (49152, 2048): (128, 64)}
DET_TUNED = {}
try:
    import json as _json, os as _os
    _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "runs/batch/dev/tune_det.json")
    if _os.environ.get("DET_TUNED", "1") == "1" and _os.path.exists(_p):
        for _k, _v in _json.load(open(_p)).items():
            if _v.get("invariant"):
                _n, _kk = map(int, _k.split(","))
                DET_TUNED[(_n, _kk)] = (_v["BK"], sorted((int(m), tuple(c)) for m, c in _v["per_M"].items()))
except Exception:
    DET_TUNED = {}


def det_mm(h, W, silu=False):
    M, K = h.shape
    N = W.shape[0] // 2 if silu else W.shape[0]
    t = DET_TUNED.get((N, K))
    if t is not None:
        BK, table = t
        BN, TM, nw, ns = next((c for m, c in table if m >= M), table[-1][1])
    else:
        BN, BK = DET_CFG.get((N, K), (64, 64))
        TM = 16 if M <= 16 else (32 if M <= 32 else 64)
        nw, ns = (4 if TM <= 32 else 8), (3 if not silu else 2)
    y = torch.empty(M, N, device=h.device, dtype=h.dtype)
    _det_gemm_kernel[(triton.cdiv(N, BN), triton.cdiv(M, TM))](h, W, W[N:] if silu else W, y, M, N, K, W.stride(0), y.stride(0),
                                                              TM=TM, BN=BN, BK=BK, SILU=silu, I64=needs_i64(M, K, y.stride(0)),
                                                              num_warps=nw, num_stages=ns)
    return y


def needs_i64(M, K, syt):
    """Row offsets rm*K and rm*syt overflow int32 once M*max(K, syt) >= 2^31 (~16.7K rows for a 128K vocabulary)."""
    return M * max(K, syt) >= 2 ** 31


def tile_buckets(Bs, top, have):
    """EXTRA_N_STEP=s (off by default): extra cost-rule buckets (N, N - Bs, -1) at every multiple of s verified tokens, for the
    ragged allocation only.  The verification time rises in steps at multiples of the 64-row GEMM tiles, and the buckets
    B (a + 1) of a batch size that is not a multiple of the tile miss the cheap end of each step (B 40: 240 and 280 tokens around
    the jump above 256).  Capped at EXTRA_N_MAX tokens (default 1024): beyond it one tile is a small fraction of the step and
    every bucket is one more graph."""
    import os
    s = int(os.environ.get("EXTRA_N_STEP") or 0)
    if s <= 0:
        return []
    cap = min(top, int(os.environ.get("EXTRA_N_MAX") or 1024))
    return [(N, N - Bs, -1) for N in range(s, cap + 1, s) if Bs < N < top and N not in have]


@triton.jit
def _rope_kv_kernel(Y, COS, SIN, QO, KC, VC, ROW, POS, syt, scb, sch, sct, H: tl.constexpr, D: tl.constexpr):
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
    k = tl.load(base + H * D + rd).to(tl.float32)
    kr = tl.load(base + H * D + rr).to(tl.float32)
    v = tl.load(base + 2 * H * D + rd)
    b = tl.load(ROW + t)
    p = tl.load(POS + t)
    off = b * scb + h * sch + p * sct + rd
    tl.store(KC + off, (k * c + sgn * kr * s).to(tl.bfloat16))
    tl.store(VC + off, v)


def rope_kv(y, cos, sin, kc, vc, row, pos, H, D):
    """y (N, 3HD) -> q (N, H, D) roped; k (roped), v written to kc/vc[row[t], :, pos[t]] (kc/vc (B, H, tmax, D))."""
    N = y.shape[0]
    q = torch.empty(N, H, D, device=y.device, dtype=y.dtype)
    _rope_kv_kernel[(N, H)](y, cos, sin, q, kc, vc, row, pos, y.stride(0), kc.stride(0), kc.stride(1), kc.stride(2),
                            H=H, D=D, num_warps=1)
    return q


@triton.jit
def _cattn_kernel(Q, KC, VC, LEN, QS, QC, OP, MP, LP, sqh, sqt, scb, sch, sct, soph, sops, sopt, smh, sms, sm_scale,
                  NQP: tl.constexpr, D: tl.constexpr, CH: tl.constexpr, BT: tl.constexpr):
    h = tl.program_id(0)
    s = tl.program_id(1)
    b = tl.program_id(2)
    L = tl.load(LEN + b).to(tl.int32)
    q0 = tl.load(QS + b).to(tl.int32)
    nq = tl.load(QC + b).to(tl.int32)
    rq = tl.arange(0, NQP)
    rd = tl.arange(0, D)
    qm = rq < nq
    qpos = L + rq                                              # absolute position of each query
    q = tl.load(Q + h * sqh + (q0 + rq)[:, None] * sqt + rd[None, :], mask=qm[:, None], other=0.0)
    m_i = tl.full([NQP], float("-inf"), tl.float32)
    l_i = tl.zeros([NQP], tl.float32)
    acc = tl.zeros([NQP, D], tl.float32)
    kb = KC + b * scb + h * sch
    vb = VC + b * scb + h * sch
    start = s * CH
    end = tl.minimum(start + CH, L + nq)
    for n0 in range(start, end, BT):
        rn = n0 + tl.arange(0, BT)
        km = rn < end
        k = tl.load(kb + rn[:, None] * sct + rd[None, :], mask=km[:, None], other=0.0)
        v = tl.load(vb + rn[:, None] * sct + rd[None, :], mask=km[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        ok = km[None, :] & (rn[None, :] <= qpos[:, None])
        qk = tl.where(ok, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    tl.store(OP + h * soph + s * sops + (q0 + rq)[:, None] * sopt + rd[None, :], acc, mask=qm[:, None])
    tl.store(MP + h * smh + s * sms + q0 + rq, m_i, mask=qm)
    tl.store(LP + h * smh + s * sms + q0 + rq, l_i, mask=qm)


class CacheAttention:
    """Fixed (CH, BT) for every batch size and step kind (batch invariance)."""

    def __init__(self, H, D, tmax, device, nqp=16, ch=256, bt=64):
        self.H, self.D, self.tmax, self.dev, self.NQP, self.CH, self.BT = H, D, tmax, device, nqp, ch, bt
        self.NS = math.ceil(tmax / ch)
        self.scale = 1.0 / math.sqrt(D)
        self.ws = {}

    def __call__(self, q, kc, vc, lens, qs, qc):
        """q (N, H, D) packed; kc/vc (B, H, tmax, D) already holding the step's own K/V; -> (N, H*D) bf16."""
        N, H, D = q.shape
        B = kc.shape[0]
        w = self.ws.get(N)
        if w is None:
            w = self.ws[N] = (torch.empty(H, self.NS, N, D, device=self.dev, dtype=torch.float32),
                              torch.empty(H, self.NS, N, device=self.dev, dtype=torch.float32),
                              torch.empty(H, self.NS, N, device=self.dev, dtype=torch.float32))
        op, mp, lp = w
        out = torch.empty(N, H, D, device=q.device, dtype=q.dtype)
        _cattn_kernel[(H, self.NS, B)](q, kc, vc, lens, qs, qc, op, mp, lp, q.stride(1), q.stride(0),
                                       kc.stride(0), kc.stride(1), kc.stride(2), op.stride(0), op.stride(1), op.stride(2),
                                       mp.stride(0), mp.stride(1), self.scale, NQP=self.NQP, D=D, CH=self.CH, BT=self.BT,
                                       num_warps=4, num_stages=2)
        _combine_kernel[(H, N)](op, mp, lp, out, op.stride(0), op.stride(1), op.stride(2), mp.stride(0), mp.stride(1),
                                out.stride(1), out.stride(0), NS=self.NS, NSP=triton.next_power_of_2(self.NS), D=D, num_warps=4)
        return out.view(N, H * D)


if __name__ == "__main__":
    # batch-invariance checks: a row's GEMM / attention result must not depend on the other rows of the step
    torch.manual_seed(0)
    for (N, K, silu) in [(6144, 2048, False), (2048, 2048, False), (5632, 2048, True), (2048, 5632, False), (49152, 2048, False)]:
        W = torch.randn(N * (2 if silu else 1), K, device="cuda", dtype=torch.bfloat16) * 0.02
        X = torch.randn(416, K, device="cuda", dtype=torch.bfloat16)
        ref = det_mm(X, W, silu)
        ok = all(torch.equal(det_mm(X[i:i + m], W, silu), ref[i:i + m]) for m, i in [(1, 0), (1, 7), (13, 20), (32, 64), (52, 100), (104, 200), (256, 0)])
        cub = (X.float() @ W[:N].float().t()) if not silu else None
        err = (ref.float() - cub).abs().max().item() if cub is not None else float("nan")
        print(f"det_mm N={N} K={K} silu={silu}: rows bitwise invariant across M: {ok}; max|err| vs fp32 {err:.2e}", flush=True)
    H, D, tmax, B = 16, 128, 1024, 6
    att = CacheAttention(H, D, tmax, "cuda")
    kc = torch.randn(B, H, tmax, D, device="cuda", dtype=torch.bfloat16); vc = torch.randn_like(kc)
    L = torch.tensor([300, 5, 700, 255, 256, 900], device="cuda")
    # step A: every row verifies 13 tokens; step B: every row decodes 1 token at position L + 4 (its 5th token)
    qa = torch.randn(B * 13, H, D, device="cuda", dtype=torch.bfloat16)
    oa = att(qa, kc, vc, L, torch.arange(B, device="cuda") * 13, torch.full((B,), 13, device="cuda"))
    qb = qa.view(B, 13, H, D)[:, 4].contiguous()
    ob = att(qb, kc, vc, L + 4, torch.arange(B, device="cuda"), torch.ones(B, dtype=torch.long, device="cuda"))
    same = torch.equal(oa.view(B, 13, -1)[:, 4], ob)
    # reference SDPA for one row / query
    b, j = 2, 4
    p = int(L[b]) + j
    r = torch.nn.functional.scaled_dot_product_attention(qa.view(B, 13, H, D)[b, j][None, :, None], kc[b:b + 1, :, :p + 1], vc[b:b + 1, :, :p + 1])
    err = (r[0, :, 0].reshape(-1).float() - oa.view(B, 13, -1)[b, j].float()).abs().max().item()
    print(f"cache_attn: verify-step vs plain-step query results bitwise equal: {same}; max|err| vs SDPA {err:.2e}", flush=True)
