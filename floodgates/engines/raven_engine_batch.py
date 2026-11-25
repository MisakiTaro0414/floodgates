"""Batched decoding engine (continuous batching, CUDA graphs) for Raven-family recurrent-depth LMs (raven_min.RavenMin:
prelude -> core x R -> coda), the batched counterpart of raven_engine.RavenEngine, measuring plain batched decoding against
self-drafting (LoopAhead one pass / with refinement) and the training-free baselines (DAS-Wave, sequential DAS, prompt
lookup) in the same engine.

State.  Every (stage, iteration, layer) cache of the batch-1 engine becomes ONE flat buffer (1, KV, B * tmax, D): row b owns
token rows [b * tmax, (b + 1) * tmax), so raven_engine.rope_write writes a token's K/V at the flat position b * tmax + p
unchanged.  Device tensors per row: Lc (committed length = position of the next input token y), y, act, drafts d / gains,
per-row verified drafts kk, the drafter's slot latents (carry).  Graphs for a sub-batch Bs <= B use the first Bs rows, so
the tail of continuous batching runs on narrower graphs after compaction (as engine_batch.py).
Step kinds (one CUDA graph each per Bs):
  plain            every row feeds y at position Lc through prelude -> R core iterations -> coda; y <- argmax; Lc += act.
  draft lal        LoopAhead drafting pass (drafter_raven.RavenSelfDrafter, one round): per row the k+1 slots [y, mask x k] at
                   Lc .. Lc+k, LoRA deltas, slots read the row's prelude / LAST-iteration core / coda K/V and each other.
  draft ref J      lal, then J of the model's own loops (prelude -> adapter -> 1 core iteration -> coda) over the window of
                   gamma+1 = 33 positions per row, seeded with the drafts (last draft repeated), the first k+1 positions
                   starting from the drafting pass's slot latents (carry) = raven_engine hybrid:J,1,32 + WAVE_CARRY=1.
  draft wave J     DAS-Wave (Li et al. 2026): window seeded with the anchor, J rounds with carried latents and the paper's
                   window growth = raven_engine wave:J,1,32.  All rows' windows are refined together.
  draft seq r,g    sequential DAS: g single-position drafts of depth r one after the other = raven_engine das:r,g.
  pol              cost rule: expected accepted tokens of each bucket (N packed tokens) / measured cycle time -> choice.
  ver N alloc      verification of a PACKED ragged block: row b verifies [y, d_1 .. d_kb] at Lc_b .. Lc_b + kb, all R loops,
                   causal within the row; kb = uniform (fixed block), top-(N - Bs) confidences over all rows (cost rule,
                   'rag'), or host-given (prompt lookup: kb = gamma for rows with a match, 0 = a plain step otherwise).
Kernels.  Attention: ragged split-KV kernel = raven_engine._att_split per row (same CH / BT / query-tile size / numerics) with
per-row base positions and query offsets read from device tensors; partials merged by raven_engine._att_combine.
ELT=inductor: the elementwise chains are raven_engine's torch.compile'd pieces (emulate_precision_casts) and every GEMM up to
64 rows is fused_gemm with raven_engine.GEMM_CFG = the batch-1 engine's numerics exactly (used at B = 1, so that every mode
reproduces the batch-1 engine token for token); ELT=triton: one-program-per-row Triton norm / RoPE / SiLU kernels (fixed
reduction order, no recompiles per shape) and cuBLAS above GEMM_TMAX rows.
DET=1: batch-invariant target steps (plain / verification): det_kernels.det_mm for every GEMM, the row-wise Triton norms,
16-query attention tiles, so a token's logits never depend on the other tokens of the step -> every mode reproduces plain
batched decoding bitwise for any batch composition.  Drafts need no invariance (they only decide which tokens are verified)."""
import math, os, time
from collections import deque
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import torch._dynamo
torch._dynamo.config.recompile_limit = 1024
torch._dynamo.config.accumulated_recompile_limit = 65536
from floodgates.engines import fused_gemm as fg
from floodgates.engines.raven_engine import prep, finish, rope_write, silu_mul, post_mid, _rms, GEMM_CFG, gemm_lora, pld_draft, _att_combine, ATT_CH, ATT_BT
from floodgates.engines.det_kernels import det_mm, _det_gemm_kernel, needs_i64, tile_buckets

# DET GEMM configs for the Recurrent-Llama shapes (runs/batch/dev/tune_det_rl32.py: one K-tile BK per weight shape for every M,
# per M bucket the fastest (BN, TM, num_warps, num_stages); every row's result verified bitwise invariant across the buckets);
# shapes without an entry fall back to det_kernels.det_mm.
DET_RL = {}
try:
    import json as _json
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs/batch/dev/tune_det_rl32.json")
    if os.environ.get("DET_RL_TUNED", "1") == "1" and os.path.exists(_p):
        for _k, _v in _json.load(open(_p)).items():
            if _v.get("invariant"):
                _n, _kk, _s = map(int, _k.split(","))
                DET_RL[(_n, _kk, bool(_s))] = (_v["BK"], sorted((int(m), tuple(c)) for m, c in _v["per_M"].items()))
except Exception:
    DET_RL = {}


def det_mm_rl(h, W, silu=False):
    M, K = h.shape
    N = W.shape[0] // 2 if silu else W.shape[0]
    t = DET_RL.get((N, K, silu))
    if t is None:
        return det_mm(h, W, silu)
    BK, table = t
    BN, TM, nw, ns = next((c for m, c in table if m >= M), table[-1][1])
    y = torch.empty(M, N, device=h.device, dtype=h.dtype)
    _det_gemm_kernel[(triton.cdiv(N, BN), triton.cdiv(M, TM))](h, W, W[N:] if silu else W, y, M, N, K, W.stride(0), y.stride(0),
                                                              TM=TM, BN=BN, BK=BK, SILU=silu, I64=needs_i64(M, K, y.stride(0)),
                                                              num_warps=nw, num_stages=ns)
    return y


# ---- ragged batched split-KV attention (raven_engine._att_split per row and query tile) --------------------------------
@triton.jit
def _batt_split(Q, KC, VC, BASE, QS, QC, ROW, OP, MP, LP, sqh, sqt, skh, skt, svh, svt, soph, sops, sopt, smh, sms, sm_scale, TROW,
                NQT: tl.constexpr, NQP: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, G: tl.constexpr,
                CH: tl.constexpr, BT: tl.constexpr, FULL: tl.constexpr):
    """Program (row b * NQT + query tile, head, cache chunk).  Row b's queries are packed tokens QS[b] .. QS[b] + QC[b] - 1 at
    positions BASE[b] .. ; its K/V live at flat rows ROW[b] * TROW + j.  CAUSAL: key j allowed for query t iff j <= its position;
    FULL: all keys < BASE[b] + QC[b] (draft slots)."""
    pid = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    b = pid // NQT
    t0 = (pid % NQT) * NQP
    kvh = h // G
    base = tl.load(BASE + b).to(tl.int32)
    q0 = tl.load(QS + b).to(tl.int32) + t0
    nqr = tl.load(QC + b).to(tl.int32)
    nq = nqr - t0
    pos0 = base + t0
    if FULL:
        L = base + nqr
    else:
        L = pos0 + tl.minimum(nq, NQP)
    L = tl.where(nq > 0, L, 0)
    rq = tl.arange(0, NQP)
    rd = tl.arange(0, DP)
    qm = rq < nq
    dm = rd < D
    q = tl.load(Q + h * sqh + (q0 + rq)[:, None] * sqt + rd[None, :], mask=qm[:, None] & dm[None, :], other=0.0)
    m_i = tl.full([NQP], float("-inf"), tl.float32)
    l_i = tl.zeros([NQP], tl.float32)
    acc = tl.zeros([NQP, DP], tl.float32)
    crow = tl.load(ROW + b).to(tl.int32)
    kb = KC + kvh * skh + crow * TROW * skt
    vb = VC + kvh * svh + crow * TROW * svt
    start = s * CH
    end = tl.minimum(start + CH, L)
    for n0 in range(start, end, BT):
        rn = n0 + tl.arange(0, BT)
        km = rn < end
        k = tl.load(kb + rn[:, None] * skt + rd[None, :], mask=km[:, None] & dm[None, :], other=0.0)
        v = tl.load(vb + rn[:, None] * svt + rd[None, :], mask=km[:, None] & dm[None, :], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        if FULL:
            allowed = km[None, :] & (rq[:, None] < NQP)
        else:
            allowed = km[None, :] & (rn[None, :] <= pos0 + rq[:, None])
        qk = tl.where(allowed, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    tl.store(OP + h * soph + s * sops + (q0 + rq)[:, None] * sopt + rd[None, :], acc, mask=qm[:, None] & dm[None, :])
    tl.store(MP + h * smh + s * sms + q0 + rq, m_i, mask=qm)
    tl.store(LP + h * smh + s * sms + q0 + rq, l_i, mask=qm)


class BatchAttention:
    def __init__(self, H, KV, D, tmax, nmax, device):
        self.H, self.KV, self.D, self.G, self.tmax = H, KV, D, H // KV, tmax
        self.DP = triton.next_power_of_2(D); self.scale = 1.0 / math.sqrt(D)
        self.NS = math.ceil(tmax / ATT_CH)
        self.nsc = self.NS                     # cache chunks launched / merged by the next call (set per graph by the engine)
        self.op = torch.empty(H, self.NS, nmax, self.DP, device=device, dtype=torch.float32)
        self.mp = torch.empty(H, self.NS, nmax, device=device, dtype=torch.float32)
        self.lp = torch.empty(H, self.NS, nmax, device=device, dtype=torch.float32)

    def __call__(self, q, Kb, Vb, base, qs, qc, Bs, nqt, nqp, full, rows):
        """q (1, H, N, D) (stride(3) == 1) packed queries; Kb/Vb flat caches (1, KV, B*tmax, D); rows (Bs,) cache row of
        each packed row -> (N, H*D) bf16."""
        H, N, D = q.shape[1], q.shape[2], q.shape[3]
        op, mp, lp, ns = self.op, self.mp, self.lp, self.nsc
        _batt_split[(Bs * nqt, H, ns)](q, Kb, Vb, base, qs, qc, rows, op, mp, lp, q.stride(1), q.stride(2), Kb.stride(1),
                                            Kb.stride(2), Vb.stride(1), Vb.stride(2), op.stride(0), op.stride(1), op.stride(2),
                                            mp.stride(0), mp.stride(1), self.scale, self.tmax, NQT=nqt, NQP=nqp, D=D,
                                            DP=self.DP, G=self.G, CH=ATT_CH, BT=ATT_BT, FULL=full, num_warps=4, num_stages=2)
        out = torch.empty(N, H * D, device=q.device, dtype=q.dtype)
        _att_combine[(H, N)](op, mp, lp, out, op.stride(0), op.stride(1), op.stride(2), mp.stride(0), mp.stride(1),
                             out.stride(0), NS=ns, NSP=triton.next_power_of_2(ns), D=D, DP=self.DP, num_warps=4)
        return out


# ---- row-wise elementwise kernels (ELT=triton; batch-invariant: one program per row, fixed reduction order) -------------
@triton.jit
def _rnd(v):
    """Exact bf16 round-to-nearest-even of an fp32 value in integer arithmetic (never contracted into an FMA)."""
    b = v.to(tl.int32, bitcast=True)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & -65536
    return b.to(tl.float32, bitcast=True)


@triton.jit
def _rms_row(x, w, eps, K: tl.constexpr):
    """raven_min.RMSNorm: (x * rsqrt(mean(x^2) + eps)) rounded to bf16, times the bf16 weight, rounded."""
    inv = tl.rsqrt(tl.sum(x * x, 0) / K + eps)
    return _rnd(_rnd(x * inv) * w)


@triton.jit
def _chain_kernel(A, Bt, W1, W2, YO, HO, eps, K: tl.constexpr, ADD: tl.constexpr, NORM1: tl.constexpr, OUTY: tl.constexpr,
                  OUTH: tl.constexpr):
    """x = A [+ Bt (bf16 add)] [-> rms(x) * W1]; YO = x; HO = rms(x) * W2."""
    t = tl.program_id(0)
    rk = tl.arange(0, K)
    x = tl.load(A + t * K + rk).to(tl.float32)
    if ADD:
        x = _rnd(x + tl.load(Bt + t * K + rk).to(tl.float32))
    if NORM1:
        x = _rms_row(x, tl.load(W1 + rk).to(tl.float32), eps, K)
    if OUTY:
        tl.store(YO + t * K + rk, x.to(tl.bfloat16))
    if OUTH:
        h = _rms_row(x, tl.load(W2 + rk).to(tl.float32), eps, K)
        tl.store(HO + t * K + rk, h.to(tl.bfloat16))


def _chain(a, b, w1, w2, eps, add, norm1, outy, outh):
    T, K = a.shape
    yo = torch.empty_like(a) if outy else a
    ho = torch.empty_like(a) if outh else a
    _chain_kernel[(T,)](a, b if b is not None else a, w1 if w1 is not None else w2, w2 if w2 is not None else w1, yo, ho, eps,
                        K=K, ADD=add, NORM1=norm1, OUTY=outy, OUTH=outh, num_warps=8)
    return yo, ho


@triton.jit
def _rope_kv_kernel(Y, COS, SIN, QO, KC, VC, WPOS, syt, skh, skt, svh, svt, H: tl.constexpr, KV: tl.constexpr,
                    D: tl.constexpr):
    """Non-interleaved RoPE of q (all heads) and k (KV heads) from the fused q,k,v projection (eager bf16 rounding:
    q*cos and rotate_half(q)*sin rounded, then the sum); k (roped) and v written at flat cache row WPOS[t]."""
    t = tl.program_id(0)
    h = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    rd = tl.arange(0, D)
    rr = (rd + HALF) % D
    sgn = tl.where(rd < HALF, -1.0, 1.0)
    c = tl.load(COS + t * D + rd).to(tl.float32)
    s = tl.load(SIN + t * D + rd).to(tl.float32)
    base = Y + t * syt
    q = tl.load(base + h * D + rd).to(tl.float32)
    qr = tl.load(base + h * D + rr).to(tl.float32)
    tl.store(QO + (t * H + h) * D + rd, (_rnd(q * c) + _rnd(sgn * qr * s)).to(tl.bfloat16))
    if h < KV:
        kp = base + H * D + h * D
        k = tl.load(kp + rd).to(tl.float32)
        kr = tl.load(kp + rr).to(tl.float32)
        v = tl.load(base + (H + KV) * D + h * D + rd)
        p = tl.load(WPOS + t)
        tl.store(KC + h * skh + p * skt + rd, (_rnd(k * c) + _rnd(sgn * kr * s)).to(tl.bfloat16))
        tl.store(VC + h * svh + p * svt + rd, v)


@triton.jit
def _silu_mul_kernel(GU, M, FF: tl.constexpr, BLOCK: tl.constexpr):
    """GU (T, 2 FF) = [gate | up] -> silu(gate) (rounded to bf16) * up (rounded)."""
    t = tl.program_id(0)
    pb = tl.program_id(1)
    r = pb * BLOCK + tl.arange(0, BLOCK)
    m = r < FF
    g = tl.load(GU + t * 2 * FF + r, mask=m, other=0.0).to(tl.float32)
    u = tl.load(GU + t * 2 * FF + FF + r, mask=m, other=0.0).to(tl.float32)
    s = _rnd(g / (1.0 + tl.exp(-g)))
    tl.store(M + t * FF + r, (s * u).to(tl.bfloat16), mask=m)


def silu_mul_t(gu):
    T, F2 = gu.shape
    FF = F2 // 2
    m = torch.empty(T, FF, device=gu.device, dtype=gu.dtype)
    _silu_mul_kernel[(T, triton.cdiv(FF, 1024))](gu, m, FF=FF, BLOCK=1024, num_warps=4)
    return m


class PLDIndex:
    """Incremental prompt lookup (raven_engine.pld_draft semantics: the k tokens after the LATEST earlier occurrence of the
    last n tokens, n = n_max .. 1, padded by repeating the last one; [] without a match) over a growing sequence: per n a
    dict n-gram -> latest start among the starts <= len - n - 1, extended lazily."""
    def __init__(self, seq, n_max, k):
        self.seq, self.n_max, self.k = list(seq), n_max, k
        self.idx = [None] + [dict() for _ in range(n_max)]
        self.done = [0] * (n_max + 1)

    def extend(self, toks):
        self.seq += toks

    def draft(self):
        s, L, k = self.seq, len(self.seq), self.k
        for n in range(min(self.n_max, L - 1), 0, -1):
            d = self.idx[n]
            for i in range(self.done[n], L - n):
                d[tuple(s[i:i + n])] = i
            self.done[n] = max(self.done[n], L - n)
            i = d.get(tuple(s[L - n:]))
            if i is not None:
                dd = s[i + n:i + n + k]
                return dd + [dd[-1]] * (k - len(dd))
        return []


def parse_mode(mode, k_dr):
    """plain | fixed | cost | ref<J>[c] | wave<J> | das<r>_<g> | pld<n>_<g>  ->  dict(cat, draft, nd, ...)."""
    import re
    if mode == "plain":
        return dict(cat="ahead", draft=None, nd=0)
    if mode in ("fixed", "cost"):
        return dict(cat="ahead" if mode == "fixed" else "cost", draft=("lal", mode == "cost"), nd=k_dr)
    m = re.fullmatch(r"ref(\d+)(c?)", mode)
    if m:
        c = m.group(2) == "c"
        return dict(cat="cost" if c else "ahead", draft=("ref", int(m.group(1)), c), nd=32)
    m = re.fullmatch(r"wave(\d+)(c?)", mode)
    if m:
        c = m.group(2) == "c"
        return dict(cat="cost" if c else "ahead", draft=("wave", int(m.group(1)), c), nd=32)
    m = re.fullmatch(r"das(\d+)_(\d+)", mode)
    if m:
        return dict(cat="ahead", draft=("seq", int(m.group(1)), int(m.group(2))), nd=int(m.group(2)))
    m = re.fullmatch(r"pld(\d+)_(\d+)", mode)
    if m:
        return dict(cat="pld", draft=None, nd=int(m.group(2)), n=int(m.group(1)))
    raise ValueError(mode)


class RavenBatchEngine:
    WIN = 33                                             # gamma + 1 window positions (refinement / DAS-Wave)
    AVG13 = tuple(range(0, 14))                          # verified drafts per row of the cost-rule buckets (one pass)
    AVG32 = tuple(range(0, 17)) + (20, 24, 28, 32)        # ... (window drafts)

    def __init__(self, m, dr=None, B=8, tmax=1024, det=False, elt=None, host_overhead_ms=0.03):
        c = m.c; dev, dt = m.device, m.dtype
        self.m, self.dr, self.B, self.tmax, self.dev = m, dr, B, tmax, dev
        self.H, self.KV = c["n_heads"], c["num_key_value_heads"]; self.D = c["n_embd"] // self.H
        self.d_model, self.eps = c["n_embd"], c["norm_eps"]
        self.det = det
        self.elt = elt or os.environ.get("ELT", "inductor" if (B == 1 and not det) else "triton")
        self.gemm_tmax = int(os.environ.get("GEMM_TMAX", 64))
        self.pre_coda_norm = bool(getattr(m, "pre_coda_norm", False))
        self.sandwich, self.interleaved = m.core[0].sandwich, m.core[0].interleaved
        if self.elt == "triton":
            assert not self.interleaved and m.core[0].qk_bias is None, "ELT=triton: non-interleaved RoPE, no qk bias (use ELT=inductor)"
        assert m.adapter.bias is None
        self.keys_p = [("p", i) for i in range(len(m.prelude))]
        self.keys_c = [[("c", it, i) for i in range(len(m.core))] for it in range(m.R)]
        self.keys_o = [("o", i) for i in range(len(m.coda))]
        self.keys = self.keys_p + [k for ks in self.keys_c for k in ks] + self.keys_o
        self.share = os.environ.get("SHARE", "0") == "1"      # KV-sharing (Prairie et al., 2026): one cache per core layer, the final iteration's
        if self.share:
            self.keys = self.keys_p + self.keys_c[-1] + self.keys_o      # the buffers that exist; every earlier iteration aliases the final one
        self.K = {k: torch.zeros(1, self.KV, B * tmax, self.D, device=dev, dtype=dt) for k in self.keys}
        self.V = {k: torch.zeros(1, self.KV, B * tmax, self.D, device=dev, dtype=dt) for k in self.keys}
        if self.share:
            for ks in self.keys_c[:-1]:
                for key, kl in zip(ks, self.keys_c[-1]):
                    self.K[key], self.V[key] = self.K[kl], self.V[kl]
        self.k1_dr = (dr.k + 1) if dr is not None else 1
        self.nd_dr = (dr.k + (1 if dr.shift else 0)) if dr is not None else 0
        self.NDMAX = max(self.WIN - 1, self.nd_dr, 16)
        K1 = self.NDMAX + 1
        self.ar = torch.arange(max(B, K1, self.WIN) + 1, device=dev)
        self.rowar = torch.arange(B, device=dev)
        self.Lc = torch.zeros(B, dtype=torch.long, device=dev)
        self.y = torch.zeros(B, dtype=torch.long, device=dev)
        self.act = torch.zeros(B, dtype=torch.bool, device=dev)
        self.act_h = torch.zeros(B, dtype=torch.bool).pin_memory()
        self.d = torch.zeros(B, self.NDMAX, dtype=torch.long, device=dev)
        self.gains = torch.zeros(B, self.NDMAX, device=dev)
        self.kk = torch.zeros(B, dtype=torch.long, device=dev)
        self.res = torch.zeros(B, K1 + 3, dtype=torch.long, device=dev)            # [n_new, tokens (K1), k_b, 1000 E_b]
        self.hres = torch.zeros(4, B, K1 + 3, dtype=torch.long).pin_memory()
        self.ev = [torch.cuda.Event() for _ in range(4)]
        self.slot = 0
        self.choice = torch.zeros(1, dtype=torch.long, device=dev)
        self.hchoice = torch.zeros(1, dtype=torch.long).pin_memory()
        self.calib = torch.ones(1, device=dev)                   # scale of the confidences in the cost rule (CALIB=1: set online, see _calibrate)
        self.calib_on = os.environ.get("CALIB", "0") == "1"
        self.cal_n = [0.0, 0.0]                                   # accepted and predicted drafts of the cost-aware cycles so far (host)
        self.evc = torch.cuda.Event()
        self.h_d = torch.zeros(B, self.NDMAX, dtype=torch.long).pin_memory()      # host-side drafts (prompt lookup)
        self.h_kk = torch.zeros(B, dtype=torch.long).pin_memory()
        self.host_overhead = host_overhead_ms
        if dr is not None:
            self.sid_dr = torch.zeros(B, self.k1_dr, dtype=torch.long, device=dev)
            self.is_clean_dr = torch.zeros(B, self.k1_dr, dtype=torch.bool, device=dev); self.is_clean_dr[:, 0] = True
            self.dr_state = torch.zeros(B, self.k1_dr, self.d_model, device=dev, dtype=dt)
        nmax = max(B * max(self.WIN, K1, self.k1_dr), tmax) + 8          # packed tokens per step (prefill: up to tmax)
        self.att = BatchAttention(self.H, self.KV, self.D, tmax, nmax, dev)
        self.NSall = self.att.NS
        # EXACT (B = 1, ELT=inductor, no DET): the batch-1 engine's numerics also in the attention merge: it merges the split-KV
        # partials of ceil(Lb / 256) chunks, Lb = the 256-bucket of the step's `need` (cur + 1 for plain decoding, cur + nd + 1
        # for every step of a speculative cycle); the merge width changes the fp32 rounding (FMA contraction), so graphs are
        # keyed by the merge width class (2 chunks for need <= 512, else 4) and chosen per cycle from the host-known length
        # (one sync per cycle, no launch-ahead, as raven_engine.py).
        self.exact = os.environ.get("EXACT1", "1") == "1" and self.elt == "inductor" and B == 1 and not det
        m.init_table(tmax + self.WIN + 8)
        if self.elt == "inductor":
            import torch._inductor.config as icfg
            icfg.emulate_precision_casts = True
            self.c_prep = torch.compile(prep, dynamic=False)
            self.c_finish = torch.compile(finish, dynamic=False)
            self.c_rope = torch.compile(rope_write, dynamic=False)
            self.c_mid = torch.compile(post_mid, dynamic=False)
            self.c_rms = torch.compile(_rms, dynamic=False)
            self.c_silu = torch.compile(silu_mul, dynamic=False)
        self.graphs, self.pool = {}, None
        self.meta, self.buckets, self.Mb, self.Tt, self.cost = {}, {}, {}, {}, {}

    # ---- elementwise dispatch ------------------------------------------------------------------------------------------
    def e_prep(self, x, dn, n_res, n_h):
        if self.elt == "inductor":
            return self.c_prep(x, dn, n_res, n_h, self.eps, self.sandwich)
        if dn is None:
            return x, _chain(x, None, None, n_h, self.eps, False, False, False, True)[1]
        return _chain(x, dn, n_res, n_h, self.eps, True, self.sandwich, True, True)

    def e_mid(self, x, o, n2, n3):
        if self.elt == "inductor":
            return self.c_mid(x, o, n2, n3, self.eps, self.sandwich)
        return _chain(x, o, n2, n3 if self.sandwich else n2, self.eps, True, self.sandwich, True, True)

    def e_fin(self, blocks, x, dn):
        n4 = blocks[-1].norm_4.weight if self.sandwich else None
        if self.elt == "inductor":
            return self.c_finish(x, dn, n4, self.eps, self.sandwich)
        return _chain(x, dn, n4, None, self.eps, True, self.sandwich, True, False)[0]

    def e_rms(self, x, w):
        if self.elt == "inductor":
            return self.c_rms(x, w, self.eps)
        return _chain(x, None, w, None, self.eps, False, True, True, False)[0]

    def e_silu(self, gu):
        return self.c_silu(gu) if self.elt == "inductor" else silu_mul_t(gu)

    def e_rope(self, y, cos, sin, b, Kb, Vb, wpos):
        """-> q (1, H, T, D) view; k, v written at the flat cache rows wpos."""
        if self.elt == "inductor":
            return self.c_rope(y, cos, sin, b.qk_bias, Kb, Vb, wpos, b.H, b.KV, b.D, b.interleaved)
        T = y.shape[0]
        q = torch.empty(T, self.H, self.D, device=y.device, dtype=y.dtype)
        _rope_kv_kernel[(T, self.H)](y, cos[0], sin[0], q, Kb, Vb, wpos, y.stride(0), Kb.stride(1), Kb.stride(2), Vb.stride(1),
                                     Vb.stride(2), H=self.H, KV=self.KV, D=self.D, num_warps=1)
        return q.view(1, T, self.H, self.D).transpose(1, 2)

    def _pre(self, s):
        return self.e_rms(s, self.m.ln_f.weight) if self.pre_coda_norm else s

    # ---- GEMMs -----------------------------------------------------------------------------------------------------------
    def _mm(self, h, W, silu=False, det=False):
        T, K = h.shape
        N = W.shape[0] // 2 if silu else W.shape[0]
        if det:
            return det_mm_rl(h.contiguous(), W, silu)
        if T <= self.gemm_tmax:                          # = raven_engine._mm (GEMM=triton) for every T the batch-1 engine sees
            return fg.gemm(h, W, silu, GEMM_CFG.get((max(16, 1 << (T - 1).bit_length()), N, K)))
        y = h @ W.t()
        return silu_mul_t(y) if silu else y

    def _lm(self, h, det=False):
        return self._mm(h, self.m.lm_head.weight, det=det)

    @staticmethod
    def _lora(x, l):
        return (x.float() @ l.A.t() @ l.B.t()).to(x.dtype)

    def _mm_lora(self, h, W, l):
        T, K = h.shape
        if T <= 16 and not (l.A.shape[0] & (l.A.shape[0] - 1)):       # = raven_engine._mm_lora (batch-1 shapes)
            u = (h.float() @ l.A.t()).contiguous()
            return gemm_lora(h, W, u, l.B, GEMM_CFG.get((16, W.shape[0], K)))
        return self._mm(h, W) + self._lora(h, l)

    # ---- one stage of blocks over N packed tokens -------------------------------------------------------------------------
    def _stack(self, blocks, keys, x, cos, sin, wpos, at, lora=None, det=False):
        """at = (base, qs, qc, Bs, nqt, nqp, full, rows) for the attention.  Returns (x_last, dn_last)."""
        sw = self.sandwich; n4 = None; dn = None
        for i, (b, key) in enumerate(zip(blocks, keys)):
            lo = lora[i] if lora is not None else None
            Kb, Vb = self.K[key], self.V[key]
            x, h = self.e_prep(x, dn, n4, b.norm_1.weight)
            y = self._mm(h, b.Wqkv.weight, det=det) if lo is None else self._mm_lora(h, b.Wqkv.weight, lo["Wqkv"])
            q = self.e_rope(y, cos, sin, b, Kb, Vb, wpos)
            a = self.att(q, Kb, Vb, *at)
            o = self._mm(a, b.proj.weight, det=det) if lo is None else self._mm_lora(a, b.proj.weight, lo["proj"])
            x, h2 = self.e_mid(x, o, b.norm_2.weight, b.norm_3.weight if sw else None)
            if lo is None:
                dn = self._mm(self._mm(h2, b.fc.weight, silu=True, det=det), b.mlp_proj.weight, det=det)
            else:
                mm_ = self.e_silu(self._mm_lora(h2, b.fc.weight, lo["fc"]))
                dn = self._mm_lora(mm_, b.mlp_proj.weight, lo["mlp_proj"])
            n4 = b.norm_4.weight if sw else None
        return x, dn

    def _target(self, ids, pos, wpos, at, det, last=None):
        """Full model (prelude -> R core iterations -> coda) over packed tokens -> logits (N, V) (only rows `last` if given)."""
        m = self.m
        cos, sin = m.rope(pos[None])
        e = m.wte(ids) * m.emb_scale
        e = self.e_fin(m.prelude, *self._stack(m.prelude, self.keys_p, e, cos, sin, wpos, at, det=det))
        s = m.init_table(self.tmax)[pos]
        for it in range(m.R):
            s = self._mm(torch.cat([s, e], -1), m.adapter.weight, det=det)
            s = self.e_fin(m.core, *self._stack(m.core, self.keys_c[it], s, cos, sin, wpos, at, det=det))
        s = self.e_fin(m.coda, *self._stack(m.coda, self.keys_o, self._pre(s), cos, sin, wpos, at, det=det))
        if last is not None:
            s = s.index_select(0, last)
        return self._lm(self.e_rms(s, m.ln_f.weight), det)

    def _meta(self, Bs):
        mt = self.meta.get(Bs)
        if mt is None:
            ar = torch.arange(Bs, device=self.dev)
            mt = self.meta[Bs] = dict(qs1=ar, ones=torch.ones(Bs, dtype=torch.long, device=self.dev),
                                      qsd=ar * self.k1_dr, qcd=torch.full((Bs,), self.k1_dr, dtype=torch.long, device=self.dev),
                                      qsw=ar * self.WIN, qcw=torch.full((Bs,), self.WIN, dtype=torch.long, device=self.dev))
        return mt

    def _nsc(self, need):
        """Merge width class of a step whose batch-1 length bucket is that of `need` (see EXACT)."""
        if not self.exact:
            return self.NSall
        Lb = min(self.tmax, 256 * math.ceil(need / 256))
        nsp = triton.next_power_of_2(math.ceil(Lb / ATT_CH))
        return self.NSall if nsp >= triton.next_power_of_2(self.NSall) else max(2, nsp)

    def nsc_list(self):
        return sorted({self.NSall} | ({2} if self.exact and self.NSall > 2 else set()))

    def _nqp(self, n):
        return max(16, triton.next_power_of_2(n))

    # ---- graph bodies -------------------------------------------------------------------------------------------------------
    def _plain(self, Bs):
        mt = self._meta(Bs)
        L, y, act = self.Lc[:Bs], self.y[:Bs], self.act[:Bs]
        wpos = self.rowar[:Bs] * self.tmax + L
        pred = self._target(y, L, wpos, (L, mt["qs1"], mt["ones"], Bs, 1, 16, False, self.rowar[:Bs]), self.det).argmax(-1)
        r = self.res[:Bs]
        r[:, 0].fill_(1); r[:, 1].copy_(pred); r[:, 2:].zero_()
        self.y[:Bs].copy_(torch.where(act, pred, y))
        self.Lc[:Bs] += act.long()

    def _draft_lal(self, Bs, gains):
        """LoopAhead drafting pass (= RavenEngine._draft, n_rounds 1): d[:, :nd_dr], dr_state, gains (optional)."""
        dr, m, k1 = self.dr, self.m, self.k1_dr
        mt = self._meta(Bs)
        L = self.Lc[:Bs]
        pos = (L[:, None] + self.ar[:k1][None, :]).reshape(-1)
        wpos = (self.rowar[:Bs, None] * self.tmax + L[:, None] + self.ar[:k1][None, :]).reshape(-1)
        cos, sin = m.rope(pos[None])
        at = (L, mt["qsd"], mt["qcd"], Bs, 1, self._nqp(k1), True, self.rowar[:Bs])      # full: as RavenEngine._draft
        self.sid_dr[:Bs, 0].copy_(self.y[:Bs])
        e_in = dr.slot_inputs(self.sid_dr[:Bs], self.is_clean_dr[:Bs]).reshape(Bs * k1, -1)
        nP, nC = dr.nP, dr.nC
        x = self.e_fin(m.prelude, *self._stack(m.prelude, self.keys_p, e_in, cos, sin, wpos, at, dr.lora[:nP]))
        s = m.init_table(self.tmax)[pos]
        for _ in range(getattr(dr, "iters", 1)):
            ae = torch.cat([s, x], -1)
            s = self._mm(ae, m.adapter.weight) + dr.lora_adapter(ae)
            s = self.e_fin(m.core, *self._stack(m.core, self.keys_c[-1], s, cos, sin, wpos, at, dr.lora[nP:nP + nC]))
        h = self.e_fin(m.coda, *self._stack(m.coda, self.keys_o, self._pre(s), cos, sin, wpos, at, dr.lora[nP + nC:]))
        lg = self._lm(self.e_rms(h, m.ln_f.weight)).float().view(Bs, k1, -1)[:, (0 if dr.shift else 1):]
        self.d[:Bs, :self.nd_dr].copy_(lg.argmax(-1))
        self.dr_state[:Bs].copy_(s.view(Bs, k1, -1))
        if gains:
            self.gains[:Bs, :self.nd_dr].copy_(torch.exp(lg.amax(-1) - torch.logsumexp(lg, -1)).cumprod(1))

    def _draft_window(self, Bs, J, seeded, gains, kc=1):
        """Refinement over the window of WIN positions per row (= RavenEngine._draft_wave, phase 'full'): seeded = the
        hybrid (LoopAhead drafts + carried slot latents, no growth); else DAS-Wave (anchor seed, window growth)."""
        m, W1 = self.m, self.WIN
        mt = self._meta(Bs)
        L = self.Lc[:Bs]
        pos = (L[:, None] + self.ar[:W1][None, :]).reshape(-1)
        wpos = (self.rowar[:Bs, None] * self.tmax + L[:, None] + self.ar[:W1][None, :]).reshape(-1)
        cos, sin = m.rope(pos[None])
        at = (L, mt["qsw"], mt["qcw"], Bs, 1, self._nqp(W1), False, self.rowar[:Bs])
        s_init = m.init_table(self.tmax)[pos]
        if seeded:
            self._draft_lal(Bs, False)
            nd = self.nd_dr
            tok = torch.cat([self.y[:Bs, None], self.d[:Bs, :nd], self.d[:Bs, nd - 1:nd].expand(Bs, W1 - 1 - nd)], 1)
            n_ = min(W1, self.k1_dr)
            state = torch.cat([self.dr_state[:Bs, :n_], s_init.view(Bs, W1, -1)[:, n_:]], 1).reshape(Bs * W1, -1)
            actp = torch.zeros(Bs * W1, dtype=torch.long, device=self.dev)
        else:
            tok = self.y[:Bs, None].expand(Bs, W1).clone()
            state = s_init
            actp = (self.ar[:W1] - 1).repeat(Bs)
        for j in range(J):
            e = m.wte(tok.reshape(-1)) * m.emb_scale
            e = self.e_fin(m.prelude, *self._stack(m.prelude, self.keys_p, e, cos, sin, wpos, at))
            s = torch.where((actp <= j)[:, None], state, s_init)
            for _ in range(kc):
                s = self._mm(torch.cat([s, e], -1), m.adapter.weight)
                s = self.e_fin(m.core, *self._stack(m.core, self.keys_c[-1], s, cos, sin, wpos, at))
            state = s
            h = self.e_fin(m.coda, *self._stack(m.coda, self.keys_o, self._pre(s), cos, sin, wpos, at))
            lg = self._lm(self.e_rms(h, m.ln_f.weight))
            z = lg.argmax(-1).view(Bs, W1)
            tok = torch.cat([tok[:, :1], z[:, :-1]], 1)
        self.d[:Bs, :W1 - 1].copy_(tok[:, 1:])
        if gains:
            lf = lg.float()
            mp = torch.exp(lf.amax(-1) - torch.logsumexp(lf, -1)).view(Bs, W1)
            self.gains[:Bs, :W1 - 1].copy_(mp[:, :W1 - 1].cumprod(1))

    def _draft_seq(self, Bs, r, g):
        """Sequential DAS (= RavenEngine._draft_seq): g single-position drafts of depth r, one after the other."""
        m = self.m
        mt = self._meta(Bs)
        L = self.Lc[:Bs]
        tok = self.y[:Bs]
        for j in range(g):
            p = L + j
            wpos = self.rowar[:Bs] * self.tmax + p
            cos, sin = m.rope(p[None])
            at = (p, mt["qs1"], mt["ones"], Bs, 1, 16, False, self.rowar[:Bs])
            e = m.wte(tok) * m.emb_scale
            e = self.e_fin(m.prelude, *self._stack(m.prelude, self.keys_p, e, cos, sin, wpos, at))
            s = m.init_table(self.tmax)[p]
            for _ in range(r):
                s = self._mm(torch.cat([s, e], -1), m.adapter.weight)
                s = self.e_fin(m.core, *self._stack(m.core, self.keys_c[-1], s, cos, sin, wpos, at))
            h = self.e_fin(m.coda, *self._stack(m.coda, self.keys_o, self._pre(s), cos, sin, wpos, at))
            z = self._lm(self.e_rms(h, m.ln_f.weight)).argmax(-1)
            self.d[:Bs, j].copy_(z)
            tok = z

    def _draft(self, Bs, spec):
        if spec[0] == "lal":
            self._draft_lal(Bs, spec[1])
        elif spec[0] == "ref":
            self._draft_window(Bs, spec[1], True, spec[2])
        elif spec[0] == "wave":
            self._draft_window(Bs, spec[1], False, spec[2])
        else:
            self._draft_seq(Bs, spec[1], spec[2])

    def bucket_list(self, Bs, nd):
        key = (Bs, nd)
        bl = self.buckets.get(key)
        if bl is None:
            avg = self.AVG13 if nd <= 13 else self.AVG32
            bl = [(Bs * (a + 1), Bs * a, a) for a in avg if a <= nd]
            if bl[-1][2] != nd:
                bl.append((Bs * (nd + 1), Bs * nd, nd))
            bl = sorted(bl + tile_buckets(Bs, Bs * (nd + 1), {b[0] for b in bl}))
            self.buckets[key] = bl
            self.Mb[key] = torch.tensor([b[1] for b in bl], device=self.dev)
        return bl

    def _policy(self, Bs, mode, nd):
        """Cost rule: expected emitted tokens of each bucket (the top N - Bs confidences over all rows) / its measured time."""
        self.bucket_list(Bs, nd)
        act = self.act[:Bs].float()
        g = self.gains[:Bs, :nd] * act[:, None] * self.calib
        S = torch.cat([torch.zeros(1, device=self.dev), g.reshape(-1).sort(descending=True).values.cumsum(0)])
        val = act.sum() + S[self.Mb[(Bs, nd)]]
        self.choice.copy_((val / self.Tt[(Bs, mode)]).argmax()[None])

    def _calibrate(self, mode, rec):
        """CALIB=1, after each cycle of Floodgates' cost rule: the drafter's confidences overestimate the accepted drafts (by ~40% on
        Recurrent-Llama), so the rule verified too many drafts at large batch sizes.  Scale them by the accepted / predicted drafts of
        all such cycles of the run so far (a uniform scale changes how many drafts are verified, not which)."""
        if not getattr(self, "calib_on", False) or mode != "cost":
            return
        for _, n, _, e in rec:
            self.cal_n[0] += n - 1; self.cal_n[1] += e
        if self.cal_n[1] >= 20.0:
            self.calib.fill_(min(1.5, max(0.2, self.cal_n[0] / self.cal_n[1])))

    def _verify(self, Bs, N, alloc, nd):
        dev, det = self.dev, self.det
        act, L, y, d = self.act[:Bs], self.Lc[:Bs], self.y[:Bs], self.d[:Bs, :nd]
        M = N - Bs
        k1 = nd + 1
        j = self.ar[:k1]
        if alloc == "uni" or M == 0:
            assert M % Bs == 0
            kk = torch.full((Bs,), M // Bs, dtype=torch.long, device=dev)
        elif alloc == "rag":
            g = self.gains[:Bs, :nd] * act[:, None]
            idx = torch.topk(g.reshape(-1), M, sorted=False).indices
            sel = torch.zeros(Bs * nd, device=dev)
            sel.scatter_(0, idx, 1.0)
            kk = sel.view(Bs, nd).sum(1).long()
        else:                                                   # host-given (prompt lookup); sum(kk) == M
            kk = self.kk[:Bs]
        cnt = kk + 1
        qs = torch.cumsum(cnt, 0) - cnt
        dest = torch.where(j[None, :] < cnt[:, None], qs[:, None] + j[None, :], N).reshape(-1)
        ids = torch.zeros(N + 1, dtype=torch.long, device=dev)
        ids.scatter_(0, dest, torch.cat([y[:, None], d], 1).reshape(-1))
        pp = torch.zeros(N + 1, dtype=torch.long, device=dev)
        pp.scatter_(0, dest, (L[:, None] + j[None, :]).reshape(-1))
        wp = torch.zeros(N + 1, dtype=torch.long, device=dev)
        wp.scatter_(0, dest, (self.rowar[:Bs, None] * self.tmax + L[:, None] + j[None, :]).reshape(-1))
        nqp = 16 if det else self._nqp(k1)
        at = (L, qs, cnt, Bs, math.ceil(k1 / nqp), nqp, False, self.rowar[:Bs])
        pred = self._target(ids[:N], pp[:N], wp[:N], at, det).argmax(-1)
        P = pred[(qs[:, None] + j[None, :]).clamp(max=N - 1)]
        match = (d == P[:, :nd]) & (j[None, :nd] < kk[:, None])
        acc = match.long().cumprod(1).sum(1)
        n_new = acc + 1
        ynew = P.gather(1, acc[:, None])[:, 0]
        r = self.res[:Bs]
        r.zero_()
        r[:, 0].copy_(n_new); r[:, 1:1 + k1].copy_(P); r[:, -2].copy_(kk)
        e = (self.gains[:Bs, :nd] * (j[None, :nd] < kk[:, None])).sum(1) if alloc == "rag" else torch.zeros(Bs, device=dev)
        r[:, -1].copy_((e * 1000).round().long())
        self.y[:Bs].copy_(torch.where(act, ynew, y))
        self.Lc[:Bs] += n_new * act.long()

    # ---- graph management -----------------------------------------------------------------------------------------------
    def _state(self):
        st = [self.Lc, self.y, self.act, self.res, self.d, self.gains, self.kk, self.choice]
        if self.dr is not None:
            st += [self.sid_dr, self.dr_state]
        return st

    def _body(self, key):
        nsc, inner = key[-1], self._body0(key[:-1])

        def body():
            self.att.nsc = nsc
            inner()
        return body

    def _body0(self, key):
        kind, Bs = key[0], key[1]
        if kind == "plain":
            return lambda: self._plain(Bs)
        if kind == "draft":
            return lambda: self._draft(Bs, key[2])
        if kind == "pol":
            return lambda: self._policy(Bs, key[2], key[3])
        if kind == "pf":
            return lambda: self._pf_body(key[1], key[2])
        return lambda: self._verify(Bs, key[2], key[3], key[4])

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

    def keys_for(self, Bs, mode, nsc=None):
        nsc = nsc or self.NSall
        return [k + (nsc,) for k in self._keys0(Bs, mode)]

    def _keys0(self, Bs, mode):
        mi = parse_mode(mode, self.nd_dr)
        nd = mi["nd"]
        if mode == "plain":
            return [("plain", Bs)]
        if mi["cat"] == "pld":
            return [("ver", Bs, Bs + nd * c, "given", nd) for c in range(Bs + 1)]
        if mi["cat"] == "ahead":
            return [("draft", Bs, mi["draft"]), ("ver", Bs, Bs * (nd + 1), "uni", nd)]
        return [("draft", Bs, mi["draft"]), ("pol", Bs, mode, nd)] + [("ver", Bs, N, "rag", nd) for N, M, a in self.bucket_list(Bs, nd)]

    @torch.no_grad()
    def prepare(self, Bs, modes, Lcal=400):
        """Capture every graph of the modes for Bs rows (dummy context of length Lcal; the K/V it writes is garbage that a
        later prefill / decode overwrites) and measure the cycle time of every cost-rule bucket."""
        self.Lc[:Bs].fill_(Lcal); self.y[:Bs].fill_(1); self.act[:Bs].fill_(True)
        for mode in modes:
            mi = parse_mode(mode, self.nd_dr)
            if mi["cat"] == "cost":
                self.Tt[(Bs, mode)] = torch.ones(len(self.bucket_list(Bs, mi["nd"])), device=self.dev)
            for nsc in self.nsc_list():
                for key in self.keys_for(Bs, mode, nsc):
                    self.capture(key)
        torch.cuda.synchronize()
        for mode in modes:
            if parse_mode(mode, self.nd_dr)["cat"] == "cost":
                self.calibrate(Bs, mode, Lcal)
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

    def calibrate(self, Bs, mode, Lcal):
        mi = parse_mode(mode, self.nd_dr); nd = mi["nd"]
        self.Lc[:Bs].fill_(Lcal); self.act[:Bs].fill_(True)
        nsc = self._nsc(Lcal + nd + 1)
        dkey = ("draft", Bs, mi["draft"], nsc)
        self.run(dkey)
        td = self._time(dkey)
        tp = self._time(("pol", Bs, mode, nd, nsc))
        bl = self.bucket_list(Bs, nd)
        tv = {N: self._time(("ver", Bs, N, "rag", nd, nsc)) for N, M, a in bl}
        T = [td + tp + tv[N] + self.host_overhead for N, M, a in bl]
        self.Tt[(Bs, mode)].copy_(torch.tensor(T, device=self.dev))
        self.cost[(Bs, mode)] = dict(draft_ms=td, policy_ms=tp, verify_ms={int(N): v for N, v in tv.items()},
                                     cycle_ms={int(N): t for (N, M, a), t in zip(bl, T)}, Lcal=Lcal)
        return self.cost[(Bs, mode)]

    def time_steps(self, Bs, modes, Lcal=400):
        """GPU time per graph (ms) at a dummy context of Lcal (for the report)."""
        self.Lc[:Bs].fill_(Lcal); self.act[:Bs].fill_(True)
        out = {}
        for mode in modes:
            for key in self.keys_for(Bs, mode, self._nsc(Lcal + parse_mode(mode, self.nd_dr)["nd"] + 1))[:2]:
                if key in self.graphs:
                    out[str(key)] = self._time(key)
        self.act[:Bs].fill_(False)
        return out

    # ---- host <-> device ----------------------------------------------------------------------------------------------
    def _readback(self, Bs):
        s = self.slot; self.slot = (self.slot + 1) % len(self.ev)
        self.hres[s, :Bs].copy_(self.res[:Bs], non_blocking=True); self.ev[s].record()
        return s

    def _wait(self, s, Bs):
        self.ev[s].synchronize()
        return self.hres[s, :Bs].tolist()

    def _push_act(self):
        if not hasattr(self, "_ring"):
            self._ring = [torch.zeros(self.B, dtype=torch.bool).pin_memory() for _ in range(8)]; self._ri = 0
        buf = self._ring[self._ri]; self._ri = (self._ri + 1) % len(self._ring)
        buf.copy_(self.act_h)
        self.act.copy_(buf, non_blocking=True)

    # ---- prefill (eager, one prompt = raven_engine.RavenEngine.prefill into row `row`) -----------------------------------
    @torch.no_grad()
    def prefill1(self, prompt, row):
        m = self.m; P = len(prompt); o = row * self.tmax
        assert P + 1 < self.tmax
        ids = torch.tensor([prompt], device=self.dev)
        pos = torch.arange(P, device=self.dev)[None]
        cos, sin = m.rope(pos)

        def attn_for(key):
            def attn(q, k, v):
                self.K[key][:, :, o:o + P] = k; self.V[key][:, :, o:o + P] = v
                return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            return attn
        e = m.wte(ids) * m.emb_scale
        for li, b in enumerate(m.prelude):
            e = b(e, cos, sin, attn_for(("p", li)))
        s = m.init_table(self.tmax)[pos[0]][None]
        for it in range(m.R):
            s = m.adapter(torch.cat([s, e], -1))
            for li, b in enumerate(m.core):
                s = b(s, cos, sin, attn_for(("c", it, li)))
        s = m.ln_f(s) if self.pre_coda_norm else s
        for li, b in enumerate(m.coda):
            s = b(s, cos, sin, attn_for(("o", li)))
        y0 = m.lm_head(m.ln_f(s[0, -1:])).argmax(-1)
        self.y[row:row + 1].copy_(y0); self.Lc[row].fill_(P); self.act[row] = True
        self.gains[row].zero_(); self.kk[row].zero_()
        self.act_h[row] = True
        return int(y0)

    # ---- graph-captured single-prompt prefill (PREFILL=graph) and shared-prefix cache (PREFIX=1) -------------------------
    PF_BUCKET = 32

    def _pf_body(self, S, P0):
        """Tokens P0 .. P0+S-1 of one prompt (ids pf_ids[:S], right-padded; true length pf_P; cache row pf_row) through the
        model with causal ragged attention over the row's cache (rows < P0 = the copied shared prefix): writes the K/V of
        every cache, sets y / Lc / act / gains / kk of the row; first token -> pf_y.  The padded tokens' K/V land beyond the
        prompt and are overwritten by decoding before any query reads them."""
        pos = P0 + self.ar_pf[:S]
        wpos = self.pf_row * self.tmax + pos
        at = (torch.full((1,), P0, dtype=torch.long, device=self.dev), torch.zeros(1, dtype=torch.long, device=self.dev),
              torch.full((1,), S, dtype=torch.long, device=self.dev), 1, math.ceil(S / 16), 16, False, self.pf_row)
        y0 = self._target(self.pf_ids[:S], pos, wpos, at, self.det, last=(self.pf_P - 1 - P0).view(1)).argmax(-1)
        row = self.pf_row
        self.pf_y.copy_(y0)
        self.y.index_copy_(0, row, y0); self.Lc.index_copy_(0, row, self.pf_P.view(1))
        self.act.index_fill_(0, row, True); self.gains.index_fill_(0, row, 0.0); self.kk.index_fill_(0, row, 0)

    def _pf_init(self):
        if not hasattr(self, "pf_ids"):
            self.pf_ids = torch.zeros(self.tmax, dtype=torch.long, device=self.dev)
            self.ar_pf = torch.arange(self.tmax, device=self.dev)
            self.pf_P = torch.ones(1, dtype=torch.long, device=self.dev)
            self.pf_row = torch.zeros(1, dtype=torch.long, device=self.dev)
            self.pf_y = torch.zeros(1, dtype=torch.long, device=self.dev)
            self.hpf = torch.zeros(1, dtype=torch.long).pin_memory()
            self.pref_len = 0

    def _pf_capture(self, S, P0):
        key = ("pf", S, P0, self.NSall)
        if key not in self.graphs:
            self.pf_P.fill_(P0 + 1); self.pf_row.fill_(0)
            self.capture(key)
        return key

    @torch.no_grad()
    def prepare_prefill(self, prompts, prefix=False):
        """Capture the prefill graphs for the padded suffix lengths of the prompts; prefix: the longest prefix shared by
        all prompts (the few-shot block) is prefilled ONCE (prompts[0] into row 0) and its K/V copied into each new row."""
        self._pf_init()
        if prefix:
            ref = list(prompts[0])
            if getattr(self, "pref_ref", None) != ref:                # K/V of prompts[0] (all but its last token), once
                self.prefill_graph(ref, 0)
                self.pk = {k: self.K[k][:, :, :len(ref) - 1].clone() for k in self.keys}
                self.pv = {k: self.V[k][:, :, :len(ref) - 1].clone() for k in self.keys}
                self.pref_ref = ref
                self.act.zero_(); self.act_h.zero_()
            lcp = len(ref) - 1
            for p in prompts[1:]:
                i = 0
                while i < min(lcp, len(p) - 1) and p[i] == ref[i]:
                    i += 1
                lcp = i
            self.pref_len = lcp
        P0 = self.pref_len if prefix else 0
        for p in prompts:
            self._pf_capture(self.PF_BUCKET * math.ceil((len(p) - P0) / self.PF_BUCKET), P0)
        torch.cuda.synchronize()

    @torch.no_grad()
    def prefill_graph(self, prompt, row, prefix=False):
        """Graph-captured prefill of one prompt into one row (prefix: copy the shared-prefix K/V, prefill the suffix).
        Returns the first generated token (one host sync)."""
        self._pf_init()
        P0 = self.pref_len if prefix else 0
        P = len(prompt); S = self.PF_BUCKET * math.ceil((P - P0) / self.PF_BUCKET)
        key = self._pf_capture(S, P0)
        if P0:
            o = row * self.tmax
            dst = [self.K[k][:, :, o:o + P0] for k in self.keys] + [self.V[k][:, :, o:o + P0] for k in self.keys]
            src = [self.pk[k][:, :, :P0] for k in self.keys] + [self.pv[k][:, :, :P0] for k in self.keys]
            torch._foreach_copy_(dst, src)
        self.pf_ids[:P - P0].copy_(torch.tensor(prompt[P0:], dtype=torch.long))
        self.pf_ids[P - P0:S].zero_()
        self.pf_P.fill_(P); self.pf_row.fill_(row)
        self.run(key)
        self.act_h[row] = True
        self.hpf.copy_(self.pf_y, non_blocking=True); torch.cuda.current_stream().synchronize()
        return int(self.hpf[0])

    # ---- continuous batching --------------------------------------------------------------------------------------------
    def ladder(self):
        out, b = [], self.B
        while b >= 1:
            out.append(b); b //= 2
        return out

    @torch.no_grad()
    def _compact(self, rows, Bs_new):
        """Move the active rows into rows < Bs_new (K/V of every cache up to the row's length + slack, per-row state).
        Call after a full sync.  Returns {old row: new row}."""
        Lh = self.Lc.tolist()
        keep = [r for r in rows if r < Bs_new]
        free = [r for r in range(Bs_new) if r not in keep]
        mp = {r: r for r in keep}
        tm = self.tmax
        st = [self.Lc, self.y, self.act, self.d, self.gains, self.kk]
        for r in rows:
            if r >= Bs_new:
                f = free.pop(0)
                Lm = min(tm, Lh[r] + 2 * self.WIN + 2)
                for key in self.keys:
                    self.K[key][:, :, f * tm:f * tm + Lm].copy_(self.K[key][:, :, r * tm:r * tm + Lm])
                    self.V[key][:, :, f * tm:f * tm + Lm].copy_(self.V[key][:, :, r * tm:r * tm + Lm])
                for t in st:
                    t[f] = t[r]
                mp[r] = f
        return mp

    @torch.no_grad()
    def run_continuous(self, prompts, mode, max_new, stop, compact=True):
        """Continuous batching of the prompt queue on B rows (= engine_batch.BatchEngine.run_continuous): a finished row
        is refilled with the next prompt (eager single-prompt prefill; results of cycles launched before the refill are
        dropped for that row); once the queue is empty and the active rows fit half the width, they are compacted and the
        narrower graphs take over.  Decode time = wall time - prefill time."""
        mi = parse_mode(mode, self.nd_dr)
        B, n = self.B, len(prompts)
        K1 = self.NDMAX + 1
        ladder = [b for b in self.ladder() if all(key in self.graphs for key in self.keys_for(b, mode))] if compact else [B]
        assert B in ladder, (mode, ladder)
        Bs = B
        outs, cut = [None] * n, [None] * n
        row_seq, row_from = [None] * B, [0] * B
        st = dict(qi=0, cid=0, t_pre=0.0, n_pre=0, ncomp=0)
        self.cal_n = [0.0, 0.0]; self.calib.fill_(1.0)
        recs, choices, widths = [], [], []
        pidx = {}
        PF = os.environ.get("PREFILL", "eager")            # eager (= the batch-1 engine's prefill) | graph
        PREFIX = os.environ.get("PREFIX", "0") == "1"
        assert PF == "graph" or not PREFIX, "PREFIX=1 needs PREFILL=graph"
        if PF == "graph":
            self.prepare_prefill(prompts, PREFIX)
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
                if PF == "graph":
                    y0 = [self.prefill_graph(prompts[i], b, PREFIX) for b, i in take]
                else:
                    y0 = [self.prefill1(prompts[i], b) for b, i in take]
                torch.cuda.synchronize()
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
                rec.append((b, q, r[b][K1 + 1], r[b][K1 + 2] / 1000.0))
                if finish_check(i, p0 + 1):
                    row_seq[b] = None; self.act_h[b] = False; freed.append(b)
                    pidx.pop(i, None)
            recs.append(rec); widths.append(Bs_c)
            self._calibrate(mode, rec)
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
        if self.exact:                                            # B = 1, the batch-1 engine's cycle (see EXACT)
            nd = mi["nd"]
            c_need = 1 if mode == "plain" else nd + 1
            while any(x is not None for x in row_seq) or st["qi"] < n:
                i = row_seq[0]
                nsc = self._nsc(len(prompts[i]) + len(outs[i]) - 1 + c_need)
                if mode == "plain":
                    self.run(("plain", 1, nsc))
                elif mi["cat"] == "pld":
                    dd = pidx[i].draft()
                    self.h_kk[0] = nd if dd else 0
                    if dd:
                        self.h_d[0, :nd] = torch.tensor(dd)
                    self.d[:1, :nd].copy_(self.h_d[:1, :nd], non_blocking=True)
                    self.kk[:1].copy_(self.h_kk[:1], non_blocking=True)
                    self.run(("ver", 1, 1 + (nd if dd else 0), "given", nd, nsc))
                elif mi["cat"] == "cost":
                    self.run(("draft", 1, mi["draft"], nsc)); self.run(("pol", 1, mode, nd, nsc))
                    self.hchoice.copy_(self.choice, non_blocking=True); self.evc.record(); self.evc.synchronize()
                    ci = int(self.hchoice[0]); bl = self.bucket_list(1, nd); choices.append(bl[ci][0])
                    self.run(("ver", 1, bl[ci][0], "rag", nd, nsc))
                else:
                    self.run(("draft", 1, mi["draft"], nsc)); self.run(("ver", 1, nd + 1, "uni", nd, nsc))
                c = st["cid"]; s = self._readback(1); st["cid"] += 1
                freed = process(c, self._wait(s, 1), 1)
                if freed:
                    refill(freed) if st["qi"] < n else self._push_act()
        elif mi["cat"] == "ahead":
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
        elif mi["cat"] == "cost":
            nd = mi["nd"]
            dkey = lambda: ("draft", Bs, mi["draft"], self.NSall)

            def launch_draft():
                self.run(dkey()); self.run(("pol", Bs, mode, nd, self.NSall))
                self.hchoice.copy_(self.choice, non_blocking=True); self.evc.record()
            launch_draft()
            while any(x is not None for x in row_seq) or st["qi"] < n:
                self.evc.synchronize()
                ci = int(self.hchoice[0]); bl = self.bucket_list(Bs, nd); choices.append(bl[ci][0] / Bs)
                self.run(("ver", Bs, bl[ci][0], "rag", nd, self.NSall))
                c = st["cid"]; s = self._readback(Bs); st["cid"] += 1
                launch_draft()                                    # next cycle's draft runs while the host processes
                freed = process(c, self._wait(s, Bs), Bs)
                if freed:
                    refill(freed) if st["qi"] < n else self._push_act()
                if maybe_compact(lambda: None):
                    launch_draft()                                # the in-flight draft used the old row layout
        else:                                                     # prompt lookup: host drafts, one sync per cycle
            nd = mi["nd"]
            while any(x is not None for x in row_seq) or st["qi"] < n:
                cnt = 0
                self.h_kk[:Bs].zero_()
                for b in range(Bs):
                    i = row_seq[b]
                    if i is None:
                        continue
                    dd = pidx[i].draft()
                    if dd:
                        self.h_d[b, :nd] = torch.tensor(dd)
                        self.h_kk[b] = nd; cnt += 1
                self.d[:Bs, :nd].copy_(self.h_d[:Bs, :nd], non_blocking=True)
                self.kk[:Bs].copy_(self.h_kk[:Bs], non_blocking=True)
                self.run(("ver", Bs, Bs + nd * cnt, "given", nd, self.NSall))
                c = st["cid"]; s = self._readback(Bs); st["cid"] += 1
                freed = process(c, self._wait(s, Bs), Bs)
                if freed:
                    refill(freed) if st["qi"] < n else self._push_act()
                maybe_compact(lambda: None)
        T1 = time.perf_counter()
        torch.cuda.synchronize()
        self.act.zero_(); self.act_h.zero_()
        outs = [o_[:c_] for o_, c_ in zip(outs, cut)]
        return dict(outs=outs, cycles=recs, choices=choices, widths=widths, t_total=T1 - T0, t_prefill=st["t_pre"],
                    t_decode=T1 - T0 - st["t_pre"], n_prefill_calls=st["n_pre"], n_compactions=st["ncomp"],
                    calib=self.calib.item() if self.calib_on else None)
