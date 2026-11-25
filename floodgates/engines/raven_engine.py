"""Batch-1 WALL-CLOCK decoding engine for Raven-family recurrent-depth LMs (raven_min.RavenMin: prelude -> core x R with
adapter injection -> coda) with static per-(stage, iteration, layer) KV caches and CUDA graphs: plain greedy decoding and
lossless greedy speculative decoding with the trained self-drafter (drafter_raven.RavenSelfDrafter).
Semantics = RavenMin.generate(): EVERY core iteration keeps its own K/V (R x n_core caches) plus prelude and coda caches,
and the init state of position j is init_table[j].
  - target step: Q tokens at positions pos0 .. pos0+Q-1 through prelude -> R core iterations -> coda; their K/V rows are
    written in place (row = position); every query attends to the cache rows <= its position (mask over a length bucket
    Lb, as stagger_engine.ARDecoder); argmax per position.  Q = 1: plain decoding (advance by one).  Q = nd+1:
    verification of [y, d_1 .. d_nd] with the acceptance on the device (longest prefix d_j == argmax at position
    pos0+j-1; the argmax at the first rejection / the bonus is the next y); rejected rows are simply overwritten later.
  - draft: RavenSelfDrafter.round (prelude -> adapter -> ONE core iteration -> coda, LoRA deltas on the slot rows) over
    the k+1 slots at positions cut .. cut+k; the slots read the context's prelude / LAST-iteration core / coda K/V
    directly from the static buffers (views, no copies), masked to positions < cut; nd = k (k+1 with shift) draft ids.
  - one CUDA graph per (kind, Lb); the host reads ONE result per cycle (pinned copy + event).
Per-layer math (FUSED=1, default): the elementwise chains are torch.compile'd with inductor's emulate_precision_casts,
i.e. the same ops and rounding points as raven_min.Block.forward (bitwise on the tested blocks); the K/V cache write is
fused into the RoPE kernel; GEMM=triton (default: fused_gemm.gemm, small-M Triton GEMMs incl. the SiLU*up epilogue and
the lm_head; differs from cuBLAS only in the fp32 accumulation order, and is the same kernel for 1 and k+2 rows) |
cublas (F.linear).  FUSED=0: raven_min.Block.forward (reference path).
ATTN=triton (default): split-KV decode attention kernel (below) reading the cache rows up to the position; ATTN=gqa:
SDPA(enable_gqa, bool mask over the bucket) = the math backend = exactly RavenMin.generate's kernels (slow).
The draft pass (FUSED=1) runs the same fused pieces with the drafter's LoRA deltas (fp32 x A^T as drafter_raven.LoRA; the
second matmul and the bf16 roundings fused into the base GEMM's epilogue, _gemm_lora_kernel) and a full mask over the
slots; FUSED=0 calls RavenSelfDrafter.round on views of the buffers.  PROFILE=1: per-cycle draft / verify GPU times
(CUDA events).  tau_teacher_forced() reproduces eval_accept_raven.py's offline acceptance inside the engine."""
import math, os, time
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from floodgates.models.raven_min import rotate_half, rope_interleaved


# ---- compiled elementwise pieces (same ops / rounding as raven_min.Block.forward) ---------------------------------------
def _rms(x, w, eps: float):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).type_as(x) * w


def prep(res, dn, n_res, n_h, eps: float, sandwich: bool):
    """Previous layer's residual add (x = res + dn; sandwich: x = rms(x) * n_res) fused with this layer's norm_1.
    Returns (x, rms(x) * n_h); dn None: x = res."""
    if dn is None:
        x = res
    elif sandwich:
        x = _rms(res + dn, n_res, eps)
    else:
        x = res + dn
    return x, _rms(x, n_h, eps)


def finish(res, dn, n_res, eps: float, sandwich: bool):
    return _rms(res + dn, n_res, eps) if sandwich else res + dn


def rope_write(y, cos, sin, qk_bias, Kb, Vb, pos, H: int, KV: int, D: int, interleaved: bool):
    """y (T, (H+2KV)*D) = fused qkv projection -> (qk bias) -> RoPE; writes k, v into rows `pos` of the caches
    (1, KV, tmax, D) in place; returns q (1, H, T, D) (strided view of the RoPE'd projection)."""
    T = y.shape[0]
    q, k, v = y.split([H * D, KV * D, KV * D], dim=-1)
    q = q.view(1, T, H, D); k = k.view(1, T, KV, D); v = v.view(1, T, KV, D)
    if qk_bias is not None:
        qb, kb = qk_bias.split(1, dim=0)
        q, k = (q + qb).to(q.dtype), (k + kb).to(q.dtype)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if interleaved:
        q, k = rope_interleaved(q, cos, sin), rope_interleaved(k, cos, sin)
    else:
        q = q * cos[:, None] + rotate_half(q) * sin[:, None]
        k = k * cos[:, None] + rotate_half(k) * sin[:, None]
    Kb[:, :, pos] = k
    Vb[:, :, pos] = v
    return q                                                            # (1, H, T, D) view; the attention kernel takes strides


def silu_mul(gu):
    g, u = gu.chunk(2, dim=-1)
    return F.silu(g) * u


def post_mid(x, o, n2, n3, eps: float, sandwich: bool):
    """x (T, d) residual stream, o = proj(attn) -> (x1, rms(x1) * n_mlp)  (sandwich: x1 = rms(o + x) * n2, norm n3)."""
    if sandwich:
        x1 = _rms(o + x, n2, eps)
        return x1, _rms(x1, n3, eps)
    x1 = x + o
    return x1, _rms(x1, n2, eps)


# fused_gemm.gemm configs (BN, BK, num_warps) per (padded rows TP, N, K) for the Raven shapes (runs/raven/tune_gemm.py, A100)
GEMM_CFG = {(16, 3072, 2048): (64, 256, 4), (16, 2048, 2048): (32, 512, 4), (16, 8192, 2048): (32, 64, 4),      # RL32 (Llama-3.2-1B dims)
            (16, 2048, 8192): (32, 512, 4), (16, 2048, 4096): (32, 512, 4),
            (16, 15840, 5280): (32, 256, 4), (16, 5280, 5280): (64, 256, 4), (16, 17920, 5280): (64, 256, 4),  # Huginn-0125
            (16, 5280, 17920): (64, 256, 4), (16, 5280, 10560): (64, 256, 4),
            (16, 128256, 2048): (128, 128, 4), (16, 65536, 5280): (128, 256, 4)}                            # lm_heads


# ---- split-KV decode attention (Triton): q (1, H, NQ, D) over the cache rows [0, pos0 + NQ) of ONE buffer -------------
# GQA (kv head = h // G), head dim padded to a power of 2 (Huginn: 96 -> 128), bf16 QK^T / PV on tensor cores, fp32 online
# softmax, P cast to bf16 before PV (flash-attention numerics).  CAUSAL: key j allowed for query t iff j <= pos0 + t
# (the new rows were written at pos0 .. pos0+NQ-1 by rope_write); FULL: all keys < pos0 + NQ (the drafter's slots see the
# context < cut and each other).  Grid (H, chunks of CH rows) -> partial (m, l, acc) -> combine -> (NQ, H*D) bf16.
# NQ = 1 and NQ = k+2 run the SAME kernel configuration, so the rows' results are bitwise identical between the plain
# step and the verification step (masked keys contribute exact zeros).
ATT_CH, ATT_BT = 256, 64


@triton.jit
def _att_split(Q, KC, VC, POS0, OP, MP, LP, sqh, sqt, skh, skt, svh, svt, soph, sops, sopt, smh, sms, sm_scale,
               NQ: tl.constexpr, NQP: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, G: tl.constexpr,
               CH: tl.constexpr, BT: tl.constexpr, FULL: tl.constexpr):
    h = tl.program_id(0)
    s = tl.program_id(1)
    kvh = h // G
    pos0 = tl.load(POS0).to(tl.int32)
    L = pos0 + NQ
    rq = tl.arange(0, NQP)
    rd = tl.arange(0, DP)
    qm = rq < NQ
    dm = rd < D
    q = tl.load(Q + h * sqh + rq[:, None] * sqt + rd[None, :], mask=qm[:, None] & dm[None, :], other=0.0)
    m_i = tl.full([NQP], float("-inf"), tl.float32)
    l_i = tl.zeros([NQP], tl.float32)
    acc = tl.zeros([NQP, DP], tl.float32)
    start = s * CH
    end = tl.minimum(start + CH, L)
    for n0 in range(start, end, BT):
        rn = n0 + tl.arange(0, BT)
        km = rn < end
        k = tl.load(KC + kvh * skh + rn[:, None] * skt + rd[None, :], mask=km[:, None] & dm[None, :], other=0.0)
        v = tl.load(VC + kvh * svh + rn[:, None] * svt + rd[None, :], mask=km[:, None] & dm[None, :], other=0.0)
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
    tl.store(OP + h * soph + s * sops + rq[:, None] * sopt + rd[None, :], acc, mask=qm[:, None] & dm[None, :])
    tl.store(MP + h * smh + s * sms + rq, m_i, mask=qm)
    tl.store(LP + h * smh + s * sms + rq, l_i, mask=qm)


@triton.jit
def _att_combine(OP, MP, LP, O, soph, sops, sopt, smh, sms, sot, NS: tl.constexpr, NSP: tl.constexpr, D: tl.constexpr,
                 DP: tl.constexpr):
    h = tl.program_id(0)
    t = tl.program_id(1)
    rs = tl.arange(0, NSP)
    sm = rs < NS
    m = tl.load(MP + h * smh + rs * sms + t, mask=sm, other=float("-inf"))
    l = tl.load(LP + h * smh + rs * sms + t, mask=sm, other=0.0)
    mx = tl.max(m, 0)
    w = tl.exp(m - mx)
    den = tl.sum(w * l, 0)
    rd = tl.arange(0, DP)
    dm = rd < D
    o = tl.load(OP + h * soph + rs[:, None] * sops + t * sopt + rd[None, :], mask=sm[:, None] & dm[None, :], other=0.0)
    out = tl.sum(o * w[:, None], 0) / den
    tl.store(O + t * sot + h * D + rd, out.to(tl.bfloat16), mask=dm)


class DecodeAttention:
    def __init__(self, H, KV, D, tmax, QM, device):
        self.H, self.KV, self.D, self.G = H, KV, D, H // KV
        self.DP = triton.next_power_of_2(D); self.scale = 1.0 / math.sqrt(D)
        NS = math.ceil(tmax / ATT_CH)
        self.op = torch.empty(H, NS, QM, self.DP, device=device, dtype=torch.float32)
        self.mp = torch.empty(H, NS, QM, device=device, dtype=torch.float32)
        self.lp = torch.empty(H, NS, QM, device=device, dtype=torch.float32)

    def __call__(self, q, kc, vc, pos0, Lb, full):
        """q (1, H, NQ, D) (stride(3) == 1), kc/vc (1, KV, tmax, D), pos0 (1,) long device, Lb: bucket -> (NQ, H*D) bf16."""
        H, NQ, D = q.shape[1], q.shape[2], q.shape[3]
        NS = math.ceil(Lb / ATT_CH); NQP = max(16, triton.next_power_of_2(NQ))
        out = torch.empty(NQ, H * D, device=q.device, dtype=q.dtype)
        op, mp, lp = self.op, self.mp, self.lp
        _att_split[(H, NS)](q, kc, vc, pos0, op, mp, lp, q.stride(1), q.stride(2), kc.stride(1), kc.stride(2), vc.stride(1),
                            vc.stride(2), op.stride(0), op.stride(1), op.stride(2), mp.stride(0), mp.stride(1), self.scale,
                            NQ=NQ, NQP=NQP, D=D, DP=self.DP, G=self.G, CH=ATT_CH, BT=ATT_BT, FULL=full,
                            num_warps=4, num_stages=2)
        _att_combine[(H, NQ)](op, mp, lp, out, op.stride(0), op.stride(1), op.stride(2), mp.stride(0), mp.stride(1),
                              out.stride(0), NS=NS, NSP=triton.next_power_of_2(NS), D=D, DP=self.DP, num_warps=4)
        return out


# ---- small-M GEMM with the drafter's LoRA delta fused in (draft pass) --------------------------------------------------
# y = bf16(h @ W^T) + bf16(u @ B^T) with u = fp32(h) @ A^T computed outside (fp32, as drafter_raven.LoRA), i.e. the rounding
# sequence of block_fwd (base and delta rounded to bf16, then the bf16 add); the fp32 delta dot uses IEEE precision.
@triton.jit
def _gemm_lora_kernel(H, W, U, Bm, Y, N, K, T, swn, syt, sbn, TP: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                      RP: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rt = tl.arange(0, TP)
    rr = tl.arange(0, RP)
    acc = tl.zeros([TP, BN], tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        h = tl.load(H + rt[:, None] * K + rk[None, :], mask=(rt[:, None] < T) & (rk[None, :] < K), other=0.0)
        w = tl.load(W + rn[:, None] * swn + rk[None, :], mask=(rn[:, None] < N) & (rk[None, :] < K), other=0.0)
        acc += tl.dot(h, tl.trans(w))
    u = tl.load(U + rt[:, None] * RP + rr[None, :], mask=rt[:, None] < T, other=0.0)
    b = tl.load(Bm + rn[:, None] * sbn + rr[None, :], mask=rn[:, None] < N, other=0.0)
    delta = tl.dot(u, tl.trans(b), input_precision="ieee")
    out = acc.to(tl.bfloat16).to(tl.float32) + delta.to(tl.bfloat16).to(tl.float32)
    tl.store(Y + rt[:, None] * syt + rn[None, :], out.to(tl.bfloat16), mask=(rt[:, None] < T) & (rn[None, :] < N))


def gemm_lora(h, W, u, B, cfg):
    """h (T, K) bf16, W (N, K) bf16, u (T, R) fp32 contiguous (R a power of 2), B (N, R) fp32 -> (T, N) bf16."""
    T, K = h.shape; N = W.shape[0]; R = u.shape[1]
    TP = max(16, triton.next_power_of_2(T))
    BN, BK, nw = cfg or (32, 256, 4)
    y = torch.empty(T, N, device=h.device, dtype=h.dtype)
    _gemm_lora_kernel[(triton.cdiv(N, BN),)](h, W, u, B, y, N, K, T, W.stride(0), y.stride(0), B.stride(0),
                                             TP=TP, BN=BN, BK=BK, RP=R, num_warps=nw)
    return y


def pld_draft(seq, n_max, k):
    """Prompt lookup decoding (Saxena 2023): find the latest earlier occurrence of the last n tokens (n = n_max .. 1) and
    return the k tokens that followed it (padded by repeating its last token); [] if nothing matches."""
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


class _Graphs:
    """One CUDA graph per key, captured lazily (stagger_engine._Graphs): warm-up runs execute the body (side effects on
    the device state!), so `state` tensors are snapshotted before and restored after capture; then the graph is replayed."""
    def __init__(self):
        self.g, self.pool = {}, None

    def run(self, key, body, use_graphs=True, state=()):
        if not use_graphs:
            return body()
        g = self.g.get(key)
        if g is None:
            snap = [t.clone() for t in state]
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    body()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                body()
            self.pool = g.pool(); self.g[key] = g
            for t, v in zip(state, snap):
                t.copy_(v)
        g.replay()


class RavenEngine:
    def _pre(self, s):
        """The coda input of Huginn is ln_f(state); the state itself is carried unnormalized."""
        if not self.pre_coda_norm: return s
        return self.c_rms(s, self.m.ln_f.weight, self.eps) if hasattr(self, "c_rms") else self.m.ln_f(s)

    def __init__(self, m, dr=None, n_rounds=1, tmax=4096, bucket=256, graphs=True, fused=None, attn=None, gemm=None, das=None):
        """das: training-free DAS baselines run in this engine instead of a trained drafter (dr=None):
        {"mode": "wave", "J": 24, "k": 1, "gamma": 32}  DAS-Wave (window of gamma provisional positions after the anchor, J rounds
                                                       of k core iterations with carried latents, prefix read at its final iteration);
        {"mode": "seq", "r": 4, "gamma": 8}             sequential DAS / Mature-Spec (gamma single-position drafts of depth r)."""
        self.m, self.dr, self.n, self.das = m, dr, n_rounds, das
        c = m.c; dev, dt = m.device, m.dtype
        self.dev, self.tmax, self.bucket, self.use_graphs = dev, tmax, bucket, graphs
        self.H, self.KV = c["n_heads"], c["num_key_value_heads"]; self.D = c["n_embd"] // self.H; self.G = self.H // self.KV
        self.d, self.eps = c["n_embd"], c["norm_eps"]
        self.pre_coda_norm = bool(getattr(m, "pre_coda_norm", False))     # Huginn: ln_f on the recurrent state before the coda (never on the carried state)
        self.fused = (os.environ.get("FUSED", "1") == "1") if fused is None else fused
        self.attn_kind = attn or os.environ.get("ATTN", "triton")          # triton (split-KV kernel) | gqa (SDPA math = generate)
        self.gemm_kind = gemm or os.environ.get("GEMM", "triton")          # triton (fused_gemm) | cublas (F.linear)
        self.lm_triton = os.environ.get("LMHEAD", "triton") == "triton" and self.gemm_kind == "triton"   # lm_head GEMM too
        self.profile = os.environ.get("PROFILE", "0") == "1"
        self.sandwich, self.interleaved = m.core[0].sandwich, m.core[0].interleaved
        assert m.adapter.bias is None
        self.keys_p = [("p", i) for i in range(len(m.prelude))]
        self.keys_c = [[("c", it, i) for i in range(len(m.core))] for it in range(m.R)]
        self.keys_o = [("o", i) for i in range(len(m.coda))]
        self.keys = self.keys_p + [k for ks in self.keys_c for k in ks] + self.keys_o
        self.K = {k: torch.zeros(1, self.KV, tmax, self.D, device=dev, dtype=dt) for k in self.keys}
        self.V = {k: torch.zeros(1, self.KV, tmax, self.D, device=dev, dtype=dt) for k in self.keys}
        self.ctx_iter = int(os.environ.get("RAVEN_CTX_ITER", "-1"))     # core iteration the drafter's slots read (-1 = last; drafter_raven.CTX_ITER)
        self.ctx_keys = self.keys_p + self.keys_c[self.ctx_iter] + self.keys_o    # the drafter's context view, in draft-pass block order
        self.nd_dr = (dr.k + (1 if dr.shift else 0)) if dr is not None else 0   # the trained drafter's drafts per pass
        self.k1_dr = (dr.k + 1) if dr is not None else 1                        # the trained drafter's slots
        if das is not None:                                                    # DAS modes (hybrid = wave seeded by the drafter)
            self.nd = das["gamma"]; self.k1 = das["gamma"] + 1 if das["mode"] == "wave" else 1
        else:
            self.nd, self.k1 = self.nd_dr, self.k1_dr
        QM = self.nd + 1
        self.ids = torch.zeros(1, QM, dtype=torch.long, device=dev)        # target-step input ids
        self.pos0 = torch.zeros(1, dtype=torch.long, device=dev)             # first position of the step (= cut)
        self.pred = torch.zeros(QM, dtype=torch.long, device=dev)            # target argmax per input position
        self.res = torch.zeros(1 + QM, dtype=torch.long, device=dev)         # step result: [n_new, new tokens ...]
        self.hres = torch.zeros(1 + QM, dtype=torch.long).pin_memory()
        self.ev = torch.cuda.Event()
        self.state = [self.pos0, self.ids, self.pred, self.res]
        if dr is not None or das is not None:
            self.sid = torch.zeros(1, self.k1, dtype=torch.long, device=dev)         # slot ids (slot 0 = y)
            self.is_clean = torch.zeros(1, self.k1, dtype=torch.bool, device=dev); self.is_clean[0, 0] = True
            self.state.append(self.sid)
            self.sid_dr = torch.zeros(1, self.k1_dr, dtype=torch.long, device=dev)   # the trained drafter's slot ids
            self.is_clean_dr = torch.zeros(1, self.k1_dr, dtype=torch.bool, device=dev); self.is_clean_dr[0, 0] = True
            self.state.append(self.sid_dr)
            self.pos_d = torch.zeros(1, dtype=torch.long, device=dev)             # sequential DAS: position of the current draft
            self.state.append(self.pos_d)
        # adaptive refinement (env ADAPT_MAX = max passes; 0 = off): one graph for the first pass, one for each further pass,
        # and a host check between passes: stop when the stable prefix of the drafted block stops growing
        self.adapt_max = int(os.environ.get("ADAPT_MAX", 0)) if dr is not None else 0
        self.passes_used = []
        if self.adapt_max:
            self.st = torch.zeros(self.k1_dr, c["n_embd"], device=dev, dtype=dt)       # carried core latent of the slots
            self.cur_pred = torch.zeros(self.k1_dr, dtype=torch.long, device=dev)     # last pass's argmax per slot
            self.stable = torch.zeros(1, dtype=torch.long, device=dev)               # leading drafts unchanged by the last pass
            self.hstable = torch.zeros(1, dtype=torch.long).pin_memory(); self.ev2 = torch.cuda.Event()
            self.expacc = torch.zeros(1, device=dev); self.dexp = torch.zeros(1, device=dev)   # drafter's expected accepted length
            self.hdexp = torch.zeros(1).pin_memory()
            self.adapt_rule = os.environ.get("ADAPT_RULE", "conf"); self.adapt_delta = float(os.environ.get("ADAPT_DELTA", 0.8))
            self.state += [self.st, self.cur_pred, self.stable, self.expacc, self.dexp]
        # convergence-stopped seeded rounds (das['stop'] = rounds per chunk, LoopAhead-seeded window only): the rounds run in
        # chunks with one host check in between, and stop once a round leaves the drafted window unchanged (or after J rounds)
        self.wave_stop = int(das.get("stop", 0)) if (das is not None and das["mode"] == "wave") else 0
        self.rounds_used = []
        if self.wave_stop:
            self.w_round = torch.zeros(1, dtype=torch.long, device=dev)        # rounds done before the current chunk (window growth)
            self.state.append(self.w_round)
            self.w_tok = torch.zeros(self.k1, dtype=torch.long, device=dev)
            self.w_state = torch.zeros(self.k1, c["n_embd"], device=dev, dtype=dt)
            self.w_flag = torch.zeros(1, dtype=torch.long, device=dev)
            self.w_stable = torch.zeros(1, dtype=torch.long, device=dev)       # leading window positions unchanged over the last chunk
            self.h_wflag = torch.zeros(1, dtype=torch.long).pin_memory(); self.ev3 = torch.cuda.Event()
            self.state += [self.w_tok, self.w_state, self.w_flag, self.w_stable]
        # carry (das['carry'], LoopAhead-seeded window only): the refinement rounds start from the slot states of the drafting pass
        self.dr_state = None
        if das is not None and das.get("init") == "loopahead" and das.get("carry"):
            self.dr_state = torch.zeros(self.k1_dr, c["n_embd"], device=dev, dtype=dt)
            self.state.append(self.dr_state)
        m.init_table(tmax)                                                   # like-init rows, built before any capture
        self.G_ = _Graphs()
        self.att = DecodeAttention(self.H, self.KV, self.D, tmax, max(QM, self.k1, self.k1_dr), dev) if self.attn_kind == "triton" else None
        if self.fused:
            import torch._inductor.config as icfg
            icfg.emulate_precision_casts = True
            self.c_prep = torch.compile(prep, dynamic=False)
            self.c_finish = torch.compile(finish, dynamic=False)
            self.c_rope = torch.compile(rope_write, dynamic=False)
            self.c_mid = torch.compile(post_mid, dynamic=False)
            self.c_rms = torch.compile(_rms, dynamic=False)
            self.c_silu = torch.compile(silu_mul, dynamic=False)
            if self.gemm_kind == "triton":
                from floodgates.engines import fused_gemm
                self.fg = fused_gemm
        if self.profile:
            self.pev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            self.t_draft = self.t_verify = 0.0; self.n_prof = 0

    # ---- per-layer pieces ----------------------------------------------------------------------------------------------
    def _mm(self, h, W, silu=False):
        """h (T, K) @ W^T -> (T, N) bf16 (silu: W = [gate; up] rows -> silu(g) * u)."""
        if self.gemm_kind == "triton":
            T, K = h.shape; N = W.shape[0] // 2 if silu else W.shape[0]
            cfg = GEMM_CFG.get((max(16, 1 << (T - 1).bit_length()), N, K))
            return self.fg.gemm(h, W, silu, cfg)
        y = F.linear(h, W)
        if silu:
            g, u = y.chunk(2, dim=-1)
            return F.silu(g) * u
        return y

    def _lm(self, h):
        """lm_head on (T, d); the Triton GEMM keeps the rounding independent of T (plain vs verification rows)."""
        return self._mm(h, self.m.lm_head.weight) if self.lm_triton else self.m.lm_head(h)

    @staticmethod
    def _lora(x, l):
        """drafter_raven.LoRA.forward: fp32 x A^T B^T, cast back."""
        return (x.float() @ l.A.t() @ l.B.t()).to(x.dtype)

    def _mm_lora(self, h, W, l):
        """h @ W^T + LoRA delta (drafter block_fwd: base(h) + lora(h)); Triton: the delta's second matmul and the bf16
        roundings / add fused into the base GEMM's epilogue."""
        if self.gemm_kind != "triton" or (l.A.shape[0] & (l.A.shape[0] - 1)):
            return self._mm(h, W) + self._lora(h, l)
        T, K = h.shape
        u = (h.float() @ l.A.t()).contiguous()                                  # (T, R) fp32, as LoRA.forward's first matmul
        return gemm_lora(h, W, u, l.B, GEMM_CFG.get((max(16, 1 << (T - 1).bit_length()), W.shape[0], K)))

    def _attend(self, q, Kb, Vb, Lb, mask, full, base=None):
        """q (1, H, Q, D) over cache rows [0, base + Q) of the buffers Kb/Vb (1, KV, tmax, D): causal by position, or full
        (draft slots).  mask: bool (1, 1, Q, Lb) for the SDPA path.  base defaults to pos0.  Returns (Q, H*D)."""
        if self.att is not None:
            return self.att(q, Kb, Vb, self.pos0 if base is None else base, Lb, full)
        o = F.scaled_dot_product_attention(q, Kb[:, :, :Lb], Vb[:, :, :Lb], attn_mask=mask, enable_gqa=True)   # (1, H, Q, D)
        return o.transpose(1, 2).reshape(o.shape[2], self.d)

    def _stack(self, blocks, keys, x, dn, cos, sin, pos, Lb, mask, lora=None, full=False, base=None):
        """Blocks of one stage on the (T, d) residual stream x (+ pending residual dn); lora: per-block ModuleDicts of
        drafter LoRA deltas (draft pass).  Returns (x_last, dn_last)."""
        sw = self.sandwich; n4 = None                                  # n4: the PREVIOUS block's post-MLP norm (sandwich)
        for i, (b, key) in enumerate(zip(blocks, keys)):
            lo = lora[i] if lora is not None else None
            Kb, Vb = self.K[key], self.V[key]
            x, h = self.c_prep(x, dn, n4, b.norm_1.weight, self.eps, sw)
            y = self._mm(h, b.Wqkv.weight) if lo is None else self._mm_lora(h, b.Wqkv.weight, lo["Wqkv"])
            q = self.c_rope(y, cos, sin, b.qk_bias, Kb, Vb, pos, b.H, b.KV, b.D, b.interleaved)
            a = self._attend(q, Kb, Vb, Lb, mask, full, base)
            o = self._mm(a, b.proj.weight) if lo is None else self._mm_lora(a, b.proj.weight, lo["proj"])
            x, h2 = self.c_mid(x, o, b.norm_2.weight, b.norm_3.weight if sw else None, self.eps, sw)
            if lo is None:
                mm = self._mm(h2, b.fc.weight, silu=True)
                dn = self._mm(mm, b.mlp_proj.weight)
            else:
                mm = self.c_silu(self._mm_lora(h2, b.fc.weight, lo["fc"]))
                dn = self._mm_lora(mm, b.mlp_proj.weight, lo["mlp_proj"])
            n4 = b.norm_4.weight if sw else None
        return x, dn

    def _fin(self, blocks, x, dn):
        return self.c_finish(x, dn, blocks[-1].norm_4.weight if self.sandwich else None, self.eps, self.sandwich)

    def _stack_ref(self, blocks, keys, x, cos, sin, pos, Lb, mask):
        """FUSED=0: raven_min.Block.forward on (1, T, d) with the same cache writes / attention."""
        for b, key in zip(blocks, keys):
            Kb, Vb = self.K[key], self.V[key]
            def attn(q, k, v, Kb=Kb, Vb=Vb):
                Kb.index_copy_(2, pos, k); Vb.index_copy_(2, pos, v)
                T = q.shape[2]
                return self._attend(q.contiguous(), Kb, Vb, Lb, mask, False).view(1, T, self.H, self.D).transpose(1, 2)
            x = b(x, cos, sin, attn)
        return x

    def _mask(self, Q, Lb, allowed_fn):
        """SDPA path only: allowed_fn(j (1, Lb), t (Q, 1)) -> bool (Q, Lb) -> (1, 1, Q, Lb)."""
        if self.att is not None:
            return None
        j = torch.arange(Lb, device=self.dev)[None, :]; t = torch.arange(Q, device=self.dev)[:, None]
        return allowed_fn(j, t)[None, None]

    # ---- graph bodies ----------------------------------------------------------------------------------------------
    def _target(self, Q, Lb, mode):
        """Q tokens ids[:, :Q] at positions pos0 .. pos0+Q-1; mode: plain (advance by one) | spec (accept)."""
        m = self.m
        pos = self.pos0 + torch.arange(Q, device=self.dev)                       # (Q,)
        cos, sin = m.rope(pos[None])
        mask = self._mask(Q, Lb, lambda j, t: j <= self.pos0 + t)
        if self.fused:
            e = m.wte(self.ids[0, :Q]) * m.emb_scale                            # (Q, d)
            e = self._fin(m.prelude, *self._stack(m.prelude, self.keys_p, e, None, cos, sin, pos, Lb, mask))
            s = m.init_table(self.tmax)[pos]
            for it in range(m.R):
                s = self._mm(torch.cat([s, e], -1), m.adapter.weight)
                s = self._fin(m.core, *self._stack(m.core, self.keys_c[it], s, None, cos, sin, pos, Lb, mask))
            s = self._fin(m.coda, *self._stack(m.coda, self.keys_o, self._pre(s), None, cos, sin, pos, Lb, mask))
            logits = self._lm(self.c_rms(s, m.ln_f.weight, self.eps))
        else:
            e = m.wte(self.ids[:, :Q]) * m.emb_scale
            e = self._stack_ref(m.prelude, self.keys_p, e, cos, sin, pos, Lb, mask)
            s = m.init_table(self.tmax)[pos][None]
            for it in range(m.R):
                s = m.adapter(torch.cat([s, e], -1))
                s = self._stack_ref(m.core, self.keys_c[it], s, cos, sin, pos, Lb, mask)
            s = self._stack_ref(m.coda, self.keys_o, m.ln_f(s) if self.pre_coda_norm else s, cos, sin, pos, Lb, mask)
            logits = m.lm_head(m.ln_f(s[0]))
        self.pred[:Q] = logits.argmax(-1)
        if mode == "plain":
            p = self.pred[:1]
            self.res[:1].fill_(1); self.res[1:2].copy_(p); self.ids[0, :1].copy_(p); self.pos0 += 1
        else:
            self._accept()

    def _accept(self):
        """Device-side greedy acceptance: drafts ids[1..nd] vs pred[0..nd-1]; res = [acc+1, pred[0..nd]]; next y = pred[acc]."""
        nd = self.nd; Q = nd + 1
        P = self.pred[:Q]
        acc = (self.ids[0, 1:Q] == P[:nd]).long().cumprod(0).sum()
        y = P.gather(0, acc[None])
        self.res[:1].copy_(acc[None] + 1); self.res[1:Q + 1].copy_(P)
        self.ids[0, :1].copy_(y); self.sid[0, :1].copy_(y); self.sid_dr[0, :1].copy_(y)
        self.pos0 += acc + 1

    def _draft(self, Lb):
        """RavenSelfDrafter.round over the k+1 slots at cut .. cut+k: prelude -> adapter -> one core iteration -> coda with
        the LoRA deltas; the slots' K/V go to the cache rows cut .. cut+k of the prelude / last-iteration / coda buffers
        (overwritten by the verification), and the slots attend to rows < cut + k+1 (context + all slots)."""
        dr, m, k1 = self.dr, self.m, self.k1_dr
        if not self.fused:
            return self._draft_ref(Lb)
        pos = self.pos0 + torch.arange(k1, device=self.dev)
        cos, sin = m.rope(pos[None])
        mask = (self._mask(k1, Lb, lambda j, t: j <= self.pos0 + t) if getattr(dr, "causal", False)
                else self._mask(k1, Lb, lambda j, t: j < self.pos0 + k1))           # causal slots: DAS-Wave-style window
        e_in = dr.slot_inputs(self.sid_dr, self.is_clean_dr)[0]                # (k1, d)
        s0 = m.init_table(self.tmax)[pos]
        nP, nC = dr.nP, dr.nC
        state = None
        fb = getattr(dr, "feedback", False)
        for r_ in range(self.n):
            if fb and r_ > 0:                                                  # trained refinement: drafted tokens fed back
                pred = logits.argmax(-1)                                       # (k1,)
                toks = torch.cat([self.sid_dr[0, :1], pred[:k1 - 1]]) if dr.shift else torch.cat([self.sid_dr[0, :1], pred[1:]])
                e_in = m.wte(toks) * m.emb_scale
                e_in = torch.cat([e_in[:1], e_in[1:] + dr.draft_emb.to(e_in.dtype)])
            x = self._fin(m.prelude, *self._stack(m.prelude, self.keys_p, e_in, None, cos, sin, pos, Lb, mask, dr.lora[:nP], True))
            if state is not None:
                x = x + dr.state_proj(m.ln_f(state).float()).to(x.dtype)
            s = state if (getattr(dr, "carry", False) and state is not None) else s0   # carry: continue the recurrence
            for _ in range(getattr(dr, "iters", 1)):                         # deeper draft pass: d core iterations
                ae = torch.cat([s, x], -1)
                s = self._mm(ae, m.adapter.weight) + dr.lora_adapter(ae)
                s = self._fin(m.core, *self._stack(m.core, self.keys_c[self.ctx_iter], s, None, cos, sin, pos, Lb, mask, dr.lora[nP:nP + nC], True))
            h = self._fin(m.coda, *self._stack(m.coda, self.keys_o, self._pre(s), None, cos, sin, pos, Lb, mask, dr.lora[nP + nC:], True))
            logits = self._lm(self.c_rms(h, m.ln_f.weight, self.eps)).float()
            state = s
        d = logits[(0 if dr.shift else 1):].argmax(-1)                         # (nd_dr,)
        self.ids[0, 1:1 + self.nd_dr].copy_(d); self.ids[0, :1].copy_(self.sid_dr[0, :1])
        if self.dr_state is not None:
            self.dr_state.copy_(state)

    def _one_pass(self, Lb, e_in, state):
        """One drafter pass over the k1 slots (prelude -> [state] -> adapter/core x iters -> coda); returns (logits, core latent)."""
        dr, m, k1 = self.dr, self.m, self.k1_dr
        pos = self.pos0 + torch.arange(k1, device=self.dev)
        cos, sin = m.rope(pos[None])
        mask = (self._mask(k1, Lb, lambda j, t: j <= self.pos0 + t) if getattr(dr, "causal", False)
                else self._mask(k1, Lb, lambda j, t: j < self.pos0 + k1))           # causal slots: DAS-Wave-style window
        nP, nC = dr.nP, dr.nC
        x = self._fin(m.prelude, *self._stack(m.prelude, self.keys_p, e_in, None, cos, sin, pos, Lb, mask, dr.lora[:nP], True))
        if state is not None:
            x = x + dr.state_proj(m.ln_f(state).float()).to(x.dtype)
        s = state if (getattr(dr, "carry", False) and state is not None) else m.init_table(self.tmax)[pos]
        for _ in range(getattr(dr, "iters", 1)):
            ae = torch.cat([s, x], -1)
            s = self._mm(ae, m.adapter.weight) + dr.lora_adapter(ae)
            s = self._fin(m.core, *self._stack(m.core, self.keys_c[self.ctx_iter], s, None, cos, sin, pos, Lb, mask, dr.lora[nP:nP + nC], True))
        h = self._fin(m.coda, *self._stack(m.coda, self.keys_o, self._pre(s), None, cos, sin, pos, Lb, mask, dr.lora[nP + nC:], True))
        return self._lm(self.c_rms(h, m.ln_f.weight, self.eps)).float(), s

    def _exp_accept(self, logits):
        """sum_j prod_{i<=j} max-prob of draft i: the drafter's own estimate of the accepted length of its block."""
        conf = logits.softmax(-1).amax(-1)[(0 if self.dr.shift else 1):]
        return conf.cumprod(0).sum()[None]

    def _draft_first(self, Lb):
        dr = self.dr
        logits, s = self._one_pass(Lb, dr.slot_inputs(self.sid_dr, self.is_clean_dr)[0], None)
        pred = logits.argmax(-1)
        self.st.copy_(s); self.cur_pred.copy_(pred); self.stable.zero_()
        self.expacc.copy_(self._exp_accept(logits)); self.dexp.zero_()
        self.ids[0, 1:1 + self.nd_dr].copy_(pred[(0 if dr.shift else 1):]); self.ids[0, :1].copy_(self.sid_dr[0, :1])

    def _draft_refine(self, Lb):
        dr, m, k1 = self.dr, self.m, self.k1_dr
        pred = self.cur_pred
        toks = torch.cat([self.sid_dr[0, :1], pred[:k1 - 1]]) if dr.shift else torch.cat([self.sid_dr[0, :1], pred[1:]])
        e_in = m.wte(toks) * m.emb_scale
        e_in = torch.cat([e_in[:1], e_in[1:] + dr.draft_emb.to(e_in.dtype)])
        logits, s = self._one_pass(Lb, e_in, self.st)
        new = logits.argmax(-1)
        old_d, new_d = self.ids[0, 1:1 + self.nd_dr], new[(0 if dr.shift else 1):]
        self.stable.copy_((old_d == new_d).long().cumprod(0).sum()[None])
        e = self._exp_accept(logits); self.dexp.copy_(e - self.expacc); self.expacc.copy_(e)
        self.st.copy_(s); self.cur_pred.copy_(new)
        self.ids[0, 1:1 + self.nd_dr].copy_(new_d)

    def _draft_adaptive(self, need):
        """First pass, then up to adapt_max - 1 refinement passes; stop when the stable prefix stops growing or covers the block."""
        self._run("draft1", need)
        best, used = -1, 1
        for _ in range(self.adapt_max - 1):
            self._run("refine", need); used += 1
            if self.adapt_rule == "conf":                                    # stop when a pass adds < delta expected tokens
                self.hdexp.copy_(self.dexp, non_blocking=True); self.ev2.record(); self.ev2.synchronize()
                if float(self.hdexp[0]) < self.adapt_delta:
                    break
                continue
            self.hstable.copy_(self.stable, non_blocking=True); self.ev2.record(); self.ev2.synchronize()
            st = int(self.hstable[0])
            if st >= self.nd_dr or st <= best:
                break
            best = st
        self.passes_used.append(used)

    def _draft_wave(self, Lb, phase="full"):
        """DAS-Wave (Li et al. 2026), training-free, in this engine: window of k1 = gamma+1 positions cut .. cut+gamma (position 0 =
        the anchor y), inputs seeded with the anchor token; J rounds of [prelude -> adapter -> k core iterations -> coda] over
        the whole window in parallel, causal within the window, reading the prefix at its FINAL iteration (deep K/V) and
        carrying the core latent across rounds; the readout of position i becomes position i+1's input.  The paper's window
        growth W_j = min(gamma, 2 + j) is emulated by resetting not-yet-active positions to their initial state (all
        positions are computed every round; at batch size one that costs the same)."""
        m, k1, J, kc = self.m, self.k1, self.das["J"], self.das["k"]
        pos = self.pos0 + torch.arange(k1, device=self.dev)
        cos, sin = m.rope(pos[None])
        mask = self._mask(k1, Lb, lambda j, t: j <= self.pos0 + t)
        s_init = m.init_table(self.tmax)[pos]
        if phase == "chunk":                                                   # continue the window of the previous chunk
            tok = self.w_tok.clone(); state = self.w_state.clone()
        elif self.das.get("init") == "loopahead":                              # hybrid: seed the window with the trained drafter's block
            self._draft(Lb)                                                    # ids[0, 1:1+nd_dr] = LoopAhead drafts
            nd = self.nd_dr
            tok = torch.cat([self.ids[0, :1 + nd], self.ids[0, nd:1 + nd].expand(k1 - 1 - nd)]) if k1 - 1 > nd else self.ids[0, :k1]
            state = s_init
            if self.dr_state is not None:                                      # carry: slots continue from the drafting pass
                n_ = min(k1, self.k1_dr)
                state = torch.cat([self.dr_state[:n_], s_init[n_:]]) if n_ < k1 else self.dr_state[:k1]
        else:
            tok = self.sid[0, :1].expand(k1).clone(); state = s_init
        grow = self.das.get("grow", self.das.get("init") != "loopahead")       # the paper's window growth (off for the hybrid)
        act = (torch.arange(k1, device=self.dev) - 1) if grow else torch.zeros(k1, dtype=torch.long, device=self.dev)
        if phase != "full":                                                    # chunked rounds: growth counts from the first chunk
            act = act - (self.w_round if phase == "chunk" else 0)
        tok_start = tok
        cheap = self.das.get("cheap")      # (topk, F): candidate-vocabulary readouts -- the head is read over the union over positions
        nJ = J if phase == "full" else self.wave_stop   # of the top-topk tokens of the last FULL readout and the window tokens; a full
        for j in range(nJ):                             # readout at the first round and at every F-th round (the last one for J % F == 0)
            e = m.wte(tok) * m.emb_scale
            e = self._fin(m.prelude, *self._stack(m.prelude, self.keys_p, e, None, cos, sin, pos, Lb, mask))
            s = torch.where((act <= j)[:, None], state, s_init)
            for _ in range(kc):
                s = self._mm(torch.cat([s, e], -1), m.adapter.weight)
                s = self._fin(m.core, *self._stack(m.core, self.keys_c[-1], s, None, cos, sin, pos, Lb, mask))
            state = s if self.das.get("carry_rounds", True) else s_init          # Jacobi decoding: every round restarts
            h = self._fin(m.coda, *self._stack(m.coda, self.keys_o, self._pre(s), None, cos, sin, pos, Lb, mask))
            hn = self.c_rms(h, m.ln_f.weight, self.eps)
            if cheap and j > 0 and (j + 1) % cheap[1] != 0:
                z = cand[F.linear(hn, Wc).argmax(-1)]                           # (k1,) over the candidates only
            else:
                lg = self._lm(hn)
                z = lg.argmax(-1)                                               # (k1,)
                if cheap:                                                   # candidates accumulate over the refreshes
                    cand = torch.cat([lg.topk(cheap[0], -1).indices.flatten(), tok, z] + ([cand] if j > 0 else []))
                    Wc = m.lm_head.weight[cand]
            tok = torch.cat([tok[:1], z[:-1]])
        if phase != "full":       # persist the window; continue only while its stable (unchanged-over-the-chunk) prefix still grows
            same = (tok == tok_start)[1:].long()
            stable = torch.cumprod(same, 0).sum()[None]
            last = (torch.zeros_like(stable) - 1) if phase == "seed" else self.w_stable      # the first chunk never stops
            self.w_flag.copy_((stable > last).long()); self.w_stable.copy_(stable)
            self.w_tok.copy_(tok); self.w_state.copy_(state)
            self.w_round.copy_(torch.zeros_like(self.w_round) + (0 if phase == "seed" else self.w_round) + self.wave_stop)
        self.ids[0, 1:1 + self.nd].copy_(tok[1:1 + self.nd]); self.ids[0, :1].copy_(self.sid[0, :1])

    def _wave_stopped(self, need):
        """Seeded rounds in chunks of wave_stop rounds, stopping when a round leaves the window unchanged or after J rounds."""
        self._run("wseed", need); done = self.wave_stop
        while done < self.das["J"]:
            self.h_wflag.copy_(self.w_flag, non_blocking=True); self.ev3.record(); self.ev3.synchronize()
            if int(self.h_wflag[0]) == 0:
                break
            self._run("wchunk", need); done += self.wave_stop
        self.rounds_used.append(done)

    def _draft_seq(self, Lb):
        """Sequential DAS / Mature-Spec (deep K/V reads), training-free, in this engine: gamma single-position drafts one after
        the other, each [prelude -> adapter -> r core iterations -> coda] at position cut+j reading the prefix at its final
        iteration and the earlier drafts' rows (written at the same buffers); the readout seeds the next draft."""
        m, r, g = self.m, self.das["r"], self.das["gamma"]
        tok = self.sid[0, :1]
        for j in range(g):
            self.pos_d.copy_(self.pos0 + j)
            pos = self.pos_d; cos, sin = m.rope(pos[None])
            mask = self._mask(1, Lb, lambda jj, t: jj <= self.pos_d)
            e = m.wte(tok) * m.emb_scale
            e = self._fin(m.prelude, *self._stack(m.prelude, self.keys_p, e, None, cos, sin, pos, Lb, mask, base=self.pos_d))
            s = m.init_table(self.tmax)[pos]
            for _ in range(r):
                s = self._mm(torch.cat([s, e], -1), m.adapter.weight)
                s = self._fin(m.core, *self._stack(m.core, self.keys_c[-1], s, None, cos, sin, pos, Lb, mask, base=self.pos_d))
            h = self._fin(m.coda, *self._stack(m.coda, self.keys_o, self._pre(s), None, cos, sin, pos, Lb, mask, base=self.pos_d))
            z = self._lm(self.c_rms(h, m.ln_f.weight, self.eps)).argmax(-1)      # (1,)
            self.ids[0, 1 + j:2 + j].copy_(z); tok = z
        self.ids[0, :1].copy_(self.sid[0, :1])

    def _draft_ref(self, Lb):
        """Reference draft: drafter_raven.RavenSelfDrafter.round on views of the static buffers (positions < cut masked)."""
        dr, m, k1 = self.dr, self.m, self.k1_dr
        cut = self.pos0
        mask = dr.masks(cut.view(1, 1), Lb, 1, k1, self.dev)                     # (1, 1, k1, Lb + k1)
        slot_pos = (cut + torch.arange(k1, device=self.dev))[None]              # (1, k1)
        e_in = dr.slot_inputs(self.sid_dr, self.is_clean_dr)
        s0 = m.init_table(self.tmax)[slot_pos]
        ctx_kv = [(self.K[key][:, :, :Lb], self.V[key][:, :, :Lb]) for key in self.ctx_keys]
        state = None
        for _ in range(self.n):
            logits, state = dr.round(e_in, s0, slot_pos, ctx_kv, mask, state)
        d = logits[0, (0 if dr.shift else 1):].argmax(-1)                       # (nd_dr,)
        self.ids[0, 1:1 + self.nd_dr].copy_(d); self.ids[0, :1].copy_(self.sid_dr[0, :1])

    # ---- graph management ------------------------------------------------------------------------------------------
    def _run(self, kind, need_len):
        Lb = min(self.tmax, self.bucket * math.ceil(need_len / self.bucket))
        if kind == "plain":
            body = lambda: self._target(1, Lb, "plain")
        elif kind == "spec":
            body = lambda: self._target(self.nd + 1, Lb, "spec")
        elif kind == "draft1":
            body = lambda: self._draft_first(Lb)
        elif kind == "refine":
            body = lambda: self._draft_refine(Lb)
        elif kind == "wseed":
            body = lambda: self._draft_wave(Lb, "seed")
        elif kind == "wchunk":
            body = lambda: self._draft_wave(Lb, "chunk")
        elif self.das is not None and self.das["mode"] == "pld":
            return                                                             # drafts come from the host (prompt lookup)
        elif self.das is not None:
            body = (lambda: self._draft_wave(Lb)) if self.das["mode"] == "wave" else (lambda: self._draft_seq(Lb))
        else:
            body = lambda: self._draft(Lb)
        self.G_.run((kind, Lb), body, self.use_graphs, self.state)

    def warmup(self, max_len):
        """Capture every graph up to max_len (positions) so that timing excludes capture."""
        for Lb in range(self.bucket, min(self.tmax, self.bucket * math.ceil(max_len / self.bucket)) + 1, self.bucket):
            self.pos0.fill_(Lb // 2)
            self._run("plain", Lb)
            if self.adapt_max:
                self._run("draft1", Lb); self._run("refine", Lb); self._run("spec", Lb)
            elif self.wave_stop:
                self._run("wseed", Lb); self._run("wchunk", Lb); self._run("spec", Lb)
            elif self.dr is not None or self.das is not None:
                self._run("draft", Lb); self._run("spec", Lb)
        torch.cuda.synchronize()

    def _result(self):
        """The one host sync per cycle: res -> pinned host buffer -> (n_new, tokens)."""
        self.hres.copy_(self.res, non_blocking=True); self.ev.record(); self.ev.synchronize()
        r = self.hres.tolist()
        return r[0], r[1:1 + r[0]]

    # ---- decoding ----------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, prompt):
        """Eager causal pass over the prompt (RavenMin.generate's first step: is_causal SDPA), filling rows 0 .. P-1 of
        every cache; returns the first generated token."""
        m = self.m; P = len(prompt)
        assert P + 1 < self.tmax
        ids = torch.tensor([prompt], device=self.dev)
        pos = torch.arange(P, device=self.dev)[None]
        cos, sin = m.rope(pos)

        def attn_for(key):
            def attn(q, k, v):
                self.K[key][:, :, :P] = k; self.V[key][:, :, :P] = v
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
        return int(m.lm_head(m.ln_f(s[0, -1:])).argmax(-1))

    @torch.no_grad()
    def generate_plain(self, prompt, max_new, stop=lambda out: False):
        """Greedy decoding; returns (ids, decode seconds) -- the prefill is not timed."""
        y = self.prefill(prompt); out = [y]; cur = len(prompt)                  # y sits at position cur (not yet cached)
        self.ids[0, 0] = y; self.pos0.fill_(cur)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        while len(out) < max_new and not stop(out):
            self._run("plain", cur + 1)
            n, new = self._result(); out += new; cur += n
        torch.cuda.synchronize()
        return out, time.perf_counter() - t0

    @torch.no_grad()
    def tau_teacher_forced(self, prompt, ref):
        """eval_accept_raven.py's offline setting inside the engine: the context is the REFERENCE sequence; at reference
        token ref[m] draft, accept the longest prefix equal to ref[m+1:], continue from the reference (the verification
        step writes the K/V rows of y = ref[m] and of the accepted drafts = reference tokens).  Returns the cycle taus."""
        self.prefill(prompt); cur = len(prompt); m = 0; taus = []
        while m < len(ref) - 1:
            self.ids[0, 0] = ref[m]; self.sid[0, 0] = ref[m]; self.sid_dr[0, 0] = ref[m]; self.pos0.fill_(cur)
            need = cur + self.nd + 1
            self._run("draft", need); self._run("spec", need)
            d = self.ids[0, 1:1 + self.nd].tolist()
            tgt = ref[m + 1: m + 1 + self.nd]
            acc = 0
            while acc < len(tgt) and d[acc] == tgt[acc]:
                acc += 1
            taus.append(acc + 1); m += acc + 1; cur += acc + 1
        return taus

    @torch.no_grad()
    def generate_spec(self, prompt, max_new, stop=lambda out: False):
        """Greedy speculative decoding (draft -> verify -> accept, one host sync per cycle); returns (ids, taus, seconds)."""
        y = self.prefill(prompt); out = [y]; cur = len(prompt); taus = []
        self.ids[0, 0] = y; self.sid[0, 0] = y; self.sid_dr[0, 0] = y; self.pos0.fill_(cur)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        while len(out) < max_new and not stop(out):
            need = cur + self.nd + 1
            if self.profile:
                self.pev[0].record(); self._run("draft", need); self.pev[1].record(); self._run("spec", need); self.pev[2].record()
            elif self.adapt_max:
                self._draft_adaptive(need); self._run("spec", need)
            elif self.wave_stop:
                self._wave_stopped(need); self._run("spec", need)
            elif self.das is not None and self.das["mode"] == "pld":          # prompt lookup: copy the continuation of the
                d = pld_draft(prompt + out, self.das["n"], self.nd)             # latest n-gram match, else a plain step
                if d:
                    self.ids[0, 1:1 + self.nd].copy_(torch.tensor(d, device=self.dev)); self._run("spec", need)
                else:
                    self._run("plain", need)
            else:
                self._run("draft", need); self._run("spec", need)
            n, new = self._result(); out += new; taus.append(n); cur += n
            if self.profile:
                self.t_draft += self.pev[0].elapsed_time(self.pev[1]); self.t_verify += self.pev[1].elapsed_time(self.pev[2])
                self.n_prof += 1
        torch.cuda.synchronize()
        return out, taus, time.perf_counter() - t0
