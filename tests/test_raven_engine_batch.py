"""Kernel-level checks of raven_engine_batch.py (no model needed except for the shapes):
  python test_raven_engine_batch.py
 1. ragged batched attention == raven_engine.DecodeAttention bitwise for the row (B = 1 numerics), causal / full, every NQ
    the batch-1 engine uses; DET tiling: a query's result is the same in a 33-query block (16-query tiles) and alone.
 2. row-wise Triton norm chains vs raven_engine's eager / compiled pieces (error) and batch invariance (bitwise).
 3. det_kernels.det_mm batch invariance on the Recurrent-Llama shapes.
 4. PLDIndex == raven_engine.pld_draft on random growing sequences.
 5. GEMM timing on the Recurrent-Llama shapes: fused_gemm (batch-1 configs) vs cuBLAS vs det_mm (CUDA graphs, L2 rotation).
Model-level checks (identity to plain batched decoding, B = 1 reproduction of the batch-1 engine) are in
bench_raven_engine_batch.py (eq_plain / eq_b1)."""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import math, random, sys, time, os
import torch
import torch.nn.functional as F
from floodgates.engines import raven_engine as RE
from floodgates.engines import raven_engine_batch as RB
from floodgates.engines import fused_gemm as fg
from floodgates.engines.det_kernels import det_mm

torch.manual_seed(0)
dev = "cuda"
print(torch.cuda.get_device_name(), flush=True)
ok_all = True

# ---- 1. attention ------------------------------------------------------------------------------------------------------
H, KV, D, tmax = 32, 8, 64, 1024
B = 3
kc1 = torch.randn(1, KV, tmax, D, device=dev, dtype=torch.bfloat16); vc1 = torch.randn_like(kc1)
Kb = torch.randn(1, KV, B * tmax, D, device=dev, dtype=torch.bfloat16); Vb = torch.randn_like(Kb)
Kb[:, :, tmax:2 * tmax] = kc1; Vb[:, :, tmax:2 * tmax] = vc1
a1 = RE.DecodeAttention(H, KV, D, tmax, 64, dev)
ab = RB.BatchAttention(H, KV, D, tmax, B * 40, dev)
for NQ, full in [(1, False), (14, False), (9, False), (13, False), (33, False), (13, True), (33, True)]:
    for pos0 in [5, 255, 300, 700, 900]:
        q1 = torch.randn(1, NQ, H, D, device=dev, dtype=torch.bfloat16).transpose(1, 2)        # (1, H, NQ, D) strided
        Lb = 256 * math.ceil((pos0 + NQ) / 256)
        o1 = a1(q1, kc1, vc1, torch.tensor([pos0], device=dev), Lb, full)
        # packed: row 0 gets 2 dummy queries, row 1 the NQ queries, row 2 one dummy query
        qd = torch.randn(1, H, 2, D, device=dev, dtype=torch.bfloat16)
        qp = torch.cat([qd, q1, qd[:, :, :1]], 2)
        base = torch.tensor([17, pos0, 40], device=dev); qc = torch.tensor([2, NQ, 1], device=dev); qs = torch.cumsum(qc, 0) - qc
        nqp = max(16, 1 << (NQ - 1).bit_length()) if NQ > 1 else 16
        ob = ab(qp, Kb, Vb, base, qs, qc, B, 1, nqp, full, torch.arange(B, device=dev))
        eq = torch.equal(ob[2:2 + NQ], o1)
        ok_all &= eq
        if not eq or pos0 == 300:
            print(f"attn NQ={NQ:2d} full={int(full)} pos0={pos0:3d}: batched row == batch-1 kernel bitwise: {eq} "
                  f"(max|diff| {(ob[2:2 + NQ].float() - o1.float()).abs().max().item():.2e})", flush=True)
# DET tiling invariance: 33-query causal block in 16-tiles vs each query alone
pos0 = 400
qb = torch.randn(1, 33, H, D, device=dev, dtype=torch.bfloat16).transpose(1, 2).contiguous()
base = torch.tensor([pos0], device=dev); qc = torch.tensor([33], device=dev); qs = torch.zeros(1, dtype=torch.long, device=dev)
Kx = Kb[:, :, tmax:].contiguous(); Vx = Vb[:, :, tmax:].contiguous()
abx = RB.BatchAttention(H, KV, D, tmax, 64, dev)
r0 = torch.zeros(1, dtype=torch.long, device=dev)
oblk = abx(qb, Kx, Vx, base, qs, qc, 1, 3, 16, False, r0).clone()
same = all(torch.equal(abx(qb[:, :, t:t + 1].contiguous(), Kx, Vx, torch.tensor([pos0 + t], device=dev), qs,
                           torch.ones(1, dtype=torch.long, device=dev), 1, 1, 16, False, r0), oblk[t:t + 1]) for t in range(33))
ok_all &= same
print(f"attn DET tiling: each of 33 block queries == the same query alone (16-query tiles): {same}", flush=True)

# ---- 2. norm chains ------------------------------------------------------------------------------------------------------
K = 2048; eps = 1e-5
x = torch.randn(100, K, device=dev, dtype=torch.bfloat16); r = torch.randn_like(x) * 0.5
w1 = (torch.rand(K, device=dev) + 0.5).to(torch.bfloat16); w2 = (torch.rand(K, device=dev) + 0.5).to(torch.bfloat16)
xo, ho = RE.prep(x, r, None, w1, eps, False)
yo, hh = RB._chain(x, r, None, w1, eps, True, False, True, True)
print(f"prep (non-sandwich): x bitwise {torch.equal(xo, yo)}; h max|err| vs eager {(hh.float() - ho.float()).abs().max().item():.2e} "
      f"(differing {(hh != ho).float().mean().item():.4f})", flush=True)
xo, ho = RE.prep(x, r, w2, w1, eps, True)
yo, hh = RB._chain(x, r, w2, w1, eps, True, True, True, True)
print(f"prep (sandwich): x max|err| {(yo.float() - xo.float()).abs().max().item():.2e}; h max|err| {(hh.float() - ho.float()).abs().max().item():.2e}", flush=True)
g1 = [RB._chain(x[i:i + 1].contiguous(), r[i:i + 1].contiguous(), None, w1, eps, True, False, True, True)[1] for i in (0, 37, 99)]
inv = all(torch.equal(g, hh_) for g, hh_ in zip(g1, [RB._chain(x, r, None, w1, eps, True, False, True, True)[1][i:i + 1] for i in (0, 37, 99)]))
ok_all &= inv
print(f"norm chain batch invariance (row alone == row in 100): {inv}", flush=True)
gu = torch.randn(50, 2 * 8192, device=dev, dtype=torch.bfloat16)
print(f"silu*up vs eager: max|err| {(RB.silu_mul_t(gu).float() - RE.silu_mul(gu).float()).abs().max().item():.2e} "
      f"(bitwise {torch.equal(RB.silu_mul_t(gu), RE.silu_mul(gu))})", flush=True)

# ---- 3. det_mm invariance -----------------------------------------------------------------------------------------------
for (Nn, Kk, silu) in [(3072, 2048, False), (2048, 2048, False), (8192, 2048, True), (2048, 8192, False), (2048, 4096, False),
                       (128256, 2048, False)]:
    Wt = torch.randn(Nn * (2 if silu else 1), Kk, device=dev, dtype=torch.bfloat16) * 0.02
    X = torch.randn(1056, Kk, device=dev, dtype=torch.bfloat16)
    ref = det_mm(X, Wt, silu)
    okm = all(torch.equal(det_mm(X[i:i + mm].contiguous(), Wt, silu), ref[i:i + mm]) for mm, i in
              [(1, 0), (1, 7), (14, 20), (32, 64), (33, 100), (52, 100), (448, 200), (1000, 0)])
    ok_all &= okm
    print(f"det_mm N={Nn} K={Kk} silu={silu}: rows bitwise invariant across M: {okm}", flush=True)

for (Nn, Kk, silu) in [(3072, 2048, False), (2048, 2048, False), (8192, 2048, True), (2048, 8192, False), (2048, 4096, False),
                       (128256, 2048, False)]:
    Wt = torch.randn(Nn * (2 if silu else 1), Kk, device=dev, dtype=torch.bfloat16) * 0.02
    X = torch.randn(1056, Kk, device=dev, dtype=torch.bfloat16)
    ref = RB.det_mm_rl(X, Wt, silu)
    okm = all(torch.equal(RB.det_mm_rl(X[i:i + mm].contiguous(), Wt, silu), ref[i:i + mm]) for mm, i in
              [(1, 0), (1, 7), (14, 20), (32, 64), (33, 100), (52, 100), (130, 3), (448, 200), (1000, 0)])
    ok_all &= okm
    print(f"det_mm_rl (tuned) N={Nn} K={Kk} silu={silu}: tuned {(Nn, Kk, silu) in RB.DET_RL}; rows bitwise invariant across M: {okm}", flush=True)

# ---- 4. prompt-lookup index ------------------------------------------------------------------------------------------------
rng = random.Random(0)
okp = True
for trial in range(200):
    V = rng.choice([3, 5, 20, 200])
    seq = [rng.randrange(V) for _ in range(rng.randrange(2, 60))]
    n_max, k = rng.choice([(3, 12), (2, 8), (1, 4)])
    ix = RB.PLDIndex(seq, n_max, k)
    for step in range(30):
        if ix.draft() != RE.pld_draft(seq, n_max, k):
            okp = False
        new = [rng.randrange(V) for _ in range(rng.randrange(1, 6))]
        seq = seq + new; ix.extend(new)
ok_all &= okp
print(f"PLDIndex == pld_draft on 200 x 30 random growing sequences: {okp}", flush=True)


# ---- 5. GEMM timing -----------------------------------------------------------------------------------------------------
def gtime(fn, n=8, reps=10):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    g.replay(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / (reps * n) * 1e6


NC = 6
for (Nn, Kk, silu) in [(3072, 2048, False), (2048, 2048, False), (8192, 2048, True), (2048, 8192, False), (2048, 4096, False),
                       (128256, 2048, False)]:
    Ws = [torch.randn(Nn * (2 if silu else 1), Kk, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(NC if Nn < 100000 else 2)]
    line = []
    for T in [1, 16, 32, 33, 48, 64, 96, 128, 448, 1056]:
        h = torch.randn(T, Kk, device=dev, dtype=torch.bfloat16)
        i = [0]
        def tri():
            i[0] = (i[0] + 1) % len(Ws)
            return fg.gemm(h, Ws[i[0]], silu, RE.GEMM_CFG.get((max(16, 1 << (T - 1).bit_length()), Nn, Kk)))
        def cub():
            i[0] = (i[0] + 1) % len(Ws); y = h @ Ws[i[0]].t()
            return RB.silu_mul_t(y) if silu else y
        def dt_():
            i[0] = (i[0] + 1) % len(Ws); return det_mm(h, Ws[i[0]], silu)
        tt = gtime(tri) if T <= 64 else float("nan")
        line.append(f"T={T}: tri {tt:.0f} cub {gtime(cub):.0f} det {gtime(dt_):.0f}")
    print(f"GEMM N={Nn} K={Kk} silu={int(silu)} (us): " + " | ".join(line), flush=True)
print("ALL KERNEL CHECKS PASSED" if ok_all else "SOME KERNEL CHECK FAILED", flush=True)
