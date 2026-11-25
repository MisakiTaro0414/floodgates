"""Small-M (decode) GEMMs for engine.py in Triton: Y (T, N) = H (T, K) @ W (N, K)^T, bf16 in / fp32 accumulate / bf16 out,
optionally with the gate/up SiLU epilogue (m = silu(g) * u), plus a 'prep' kernel that fuses the sandwich-norm elementwise
chain of ouro_min.Layer between two matmuls:  x <- x + rms(r) * n_r  [; x <- rms(x) * n_s (the per-loop model norm)];
h = rms(x) * n_h.  Rounding sequence replicates engine._rms / post_block (fp32 RMS, bf16 products), so only the fp32
accumulation ORDER of the matmuls differs from cuBLAS."""
import torch
import triton
import triton.language as tl


@triton.jit
def _bf16r(v, EXACT: tl.constexpr):
    """Intermediate bf16 rounding of an fp32 value.  EXACT = 1: round to nearest even via integer arithmetic (the compiler
    cannot fold it; a plain .to(bf16).to(f32) round-trip in front of an add gets contracted into an FMA) = the eager
    ouro_min semantics.  EXACT = 2: no intermediate rounding at all (fp32 chain, one rounding at the store) = what the
    inductor-compiled pre/post blocks compute.  EXACT = 0: the plain round-trip (FMA-contracted)."""
    if EXACT == 1:
        b = v.to(tl.int32, bitcast=True)
        b = (b + 0x7FFF + ((b >> 16) & 1)) & -65536
        return b.to(tl.float32, bitcast=True)
    elif EXACT == 2:
        return v
    else:
        return v.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _prep_kernel(X, R, NR, NS, NH, XO, HO, eps, K: tl.constexpr, HAS_RES: tl.constexpr, HAS_SNORM: tl.constexpr,
                 EXACT: tl.constexpr):
    t = tl.program_id(0)
    rk = tl.arange(0, K)
    x = tl.load(X + t * K + rk).to(tl.float32)
    if HAS_RES:
        r = tl.load(R + t * K + rk).to(tl.float32)
        inv = tl.rsqrt(tl.sum(r * r, 0) / K + eps)
        rn = _bf16r(r * inv, EXACT)
        nr = tl.load(NR + rk).to(tl.float32)
        x = _bf16r(x + _bf16r(nr * rn, EXACT), 1)          # the residual stream is materialized in bf16 in every mode
    if HAS_SNORM:                                            # the per-loop model norm: an eager RMSNorm module in engine._target
        inv = tl.rsqrt(tl.sum(x * x, 0) / K + eps)
        xn = _bf16r(x * inv, 1)
        ns = tl.load(NS + rk).to(tl.float32)
        x = _bf16r(ns * xn, 1)
    tl.store(XO + t * K + rk, x.to(tl.bfloat16))
    inv = tl.rsqrt(tl.sum(x * x, 0) / K + eps)
    hn = _bf16r(x * inv, EXACT)
    nh = tl.load(NH + rk).to(tl.float32)
    tl.store(HO + t * K + rk, (nh * hn).to(tl.bfloat16))


@triton.jit
def _gemm_kernel(H, W, W2, Y, N, K, T, swn, syt,
                 TP: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SILU: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rt = tl.arange(0, TP)
    acc = tl.zeros([TP, BN], tl.float32)
    acc2 = tl.zeros([TP, BN], tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        h = tl.load(H + rt[:, None] * K + rk[None, :], mask=(rt[:, None] < T) & (rk[None, :] < K), other=0.0)
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
    tl.store(Y + rt[:, None] * syt + rn[None, :], out, mask=(rt[:, None] < T) & (rn[None, :] < N))


# (BN, BK, num_warps) per (padded row count TP, N, K), from the L2-defeating benchmark below, per GPU family.
# TP = 128 (T = 72) is slower than cuBLAS on both -> engine.py uses the compiled path there (MAX_T).
CFG_A100 = {(16, 6144, 2048): (32, 128, 4), (16, 2048, 2048): (32, 512, 4), (16, 5632, 2048): (64, 128, 4), (16, 2048, 5632): (32, 512, 4),
            (64, 6144, 2048): (64, 256, 4), (64, 2048, 2048): (32, 256, 4), (64, 5632, 2048): (64, 128, 4), (64, 2048, 5632): (32, 256, 4)}
CFG_H100 = {(16, 6144, 2048): (64, 256, 4), (16, 2048, 2048): (16, 512, 4), (16, 5632, 2048): (64, 128, 4), (16, 2048, 5632): (16, 512, 4),
            (64, 6144, 2048): (64, 256, 4), (64, 2048, 2048): (16, 256, 4), (64, 5632, 2048): (32, 128, 4), (64, 2048, 5632): (16, 256, 4)}
CFG = CFG_H100 if "H100" in torch.cuda.get_device_name() else CFG_A100
MAX_T = 36
EXACT_ROUND = 2          # rounding of the RMSNorm intermediates: 1 = bf16 (eager ouro_min semantics), 2 = none (= the inductor-compiled
                         # pre/post blocks of the original engine), 0 = plain round-trip (FMA-contracted)


def prep(x, r, n_r, n_s, n_h, eps):
    """x (T, K) bf16 (+ r (T, K) residual input normed by n_r) [+ step norm n_s] -> (x_new (T, K), h = rms(x_new) * n_h)."""
    T, K = x.shape
    xo, ho = torch.empty_like(x), torch.empty_like(x)
    _prep_kernel[(T,)](x, r if r is not None else x, n_r if n_r is not None else n_h, n_s if n_s is not None else n_h, n_h,
                       xo, ho, eps, K=K, HAS_RES=r is not None, HAS_SNORM=n_s is not None, EXACT=EXACT_ROUND, num_warps=4)
    return xo, ho


def gemm(h, W, silu=False, cfg=None):
    """h (T, K) bf16 contiguous, W (N, K) [SILU: W = cat(gate, up) rows, N = rows / 2] -> (T, N) bf16."""
    T, K = h.shape
    N = W.shape[0] // 2 if silu else W.shape[0]
    TP = max(16, triton.next_power_of_2(T))
    BN, BK, nw = cfg or CFG.get((TP, N, K), (32, 256, 4))
    y = torch.empty(T, N, device=h.device, dtype=h.dtype)
    _gemm_kernel[(triton.cdiv(N, BN),)](h, W, W[N:] if silu else W, y, N, K, T, W.stride(0), y.stride(0),
                                        TP=TP, BN=BN, BK=BK, SILU=silu, num_warps=nw)
    return y


if __name__ == "__main__":
    import math, time, torch.nn.functional as F
    torch.manual_seed(0)

    def gtime(fn, n=8, reps=10):
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2): fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n): fn()
        g.replay(); torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(reps): g.replay()
        torch.cuda.synchronize(); return (time.perf_counter() - t0) / (reps * n) * 1e6
    print(torch.cuda.get_device_name())
    NC = 12                                                             # weight copies rotated to defeat the L2 cache
    for N, K, silu in [(6144, 2048, False), (2048, 2048, False), (11264, 2048, True), (2048, 5632, False)]:
        Ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(NC)]
        for T in [1, 9, 36, 72]:
            h = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
            ref = h @ Ws[0].t()
            if silu:
                g_, u_ = ref.chunk(2, -1); ref = F.silu(g_) * u_
            i = [0]
            def cub():
                i[0] = (i[0] + 1) % NC; y = h @ Ws[i[0]].t()
                if silu:
                    g_, u_ = y.chunk(2, -1); y = F.silu(g_) * u_
                return y
            tc = gtime(cub)
            best = None
            TP = max(16, triton.next_power_of_2(T))
            for BN in [16, 32, 64]:
                for BK in [128, 256, 512, 1024]:
                    for nw in [4, 8]:
                        if BN * BK * (2 if silu else 1) * 2 > 64 * 1024 * (nw // 4):
                            continue
                        try:
                            def tri():
                                i[0] = (i[0] + 1) % NC; return gemm(h, Ws[i[0]], silu, (BN, BK, nw))
                            t = gtime(tri)
                        except Exception as e:
                            continue
                        if best is None or t < best[0]: best = (t, BN, BK, nw)
            y = gemm(h, Ws[0], silu, best[1:]); torch.cuda.synchronize()
            err = (y.float() - ref.float()).abs().max().item(); neq = (y != ref).sum().item()
            mb = N * K * 2 / 1e6
            print(f"N={N:5d} K={K:4d} T={T:2d} silu={int(silu)} ({mb:5.1f} MB): cuBLAS(+silu) {tc:6.1f} us [{mb / tc * 1e3:5.2f} TB/s]"
                  f"  triton {best[0]:6.1f} us [{mb / best[0] * 1e3:5.2f} TB/s] cfg BN={best[1]} BK={best[2]} nw={best[3]}"
                  f"  max|err| {err:.2e} differing {neq}/{y.numel()}", flush=True)
    # prep kernel vs the reference elementwise chain
    from floodgates.engines.engine import _rms
    K = 2048; eps = 1e-6
    for T in [1, 9, 72]:
        x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16); r = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
        nr, ns, nh = [torch.rand(K, device="cuda", dtype=torch.bfloat16) + 0.5 for _ in range(3)]
        xr = x + _rms(r, nr, eps); xs = ns * (xr.float() * torch.rsqrt(xr.float().pow(2).mean(-1, keepdim=True) + eps)).to(torch.bfloat16)
        href = _rms(xs, nh, eps)
        xo, ho = prep(x, r, nr, ns, nh, eps)
        print(f"prep T={T}: x_new max|err| {(xo.float() - xs.float()).abs().max().item():.2e} (differing {(xo != xs).sum().item()}), "
              f"h max|err| {(ho.float() - href.float()).abs().max().item():.2e} (differing {(ho != href).sum().item()}), "
              f"time {gtime(lambda: prep(x, r, nr, ns, nh, eps)):.1f} us")
