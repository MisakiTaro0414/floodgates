"""Batch-1 decoding engine for Ouro (plain PyTorch, static per-(loop, layer) KV cache, CUDA graphs) used to measure WALL-CLOCK
speedups of self-drafting against plain decoding in the same engine.
  - target step: Q tokens through all R loops; K/V written into the cache rows cut .. cut+Q-1; attention over the committed
    context [0, cut) plus the step's own tokens (causal, or block-causal for a tree of S candidate blocks); last-loop logits.
  - draft: n rounds of the drafter over S blocks of k+1 slots at positions cut .. cut+k (block b = candidate b); the slots
    attend to the final-loop cache < cut and to the slots of their own block (bidirectional), as drafter.LoopFlowDrafter.round.
    S > 1: candidate 0 = argmax draft, candidates >= 1 sampled from the last round's slot distributions (Gumbel-max, T=1),
    i.e. drafter.draft_many(sample_readout=True) semantics; the noise drafter also gets its own noise per candidate.
  - spec cycle (greedy): draft -> verify the S*(k+1) tokens [y, d^b_1..d^b_k]_b in ONE target step -> candidate with the
    longest accepted prefix wins (ties: lowest index) -> its accepted K/V rows are moved to the canonical rows cut .. cut+acc
    -> emit; rejected rows are simply overwritten later.
Attention (ATTN=triton, default): decode_attn.DecodeAttention = split-KV kernel that reads the context length from a DEVICE
tensor (pos0), so one CUDA graph per step kind serves every context length.  ATTN=sdpa: the former path (masked SDPA over a
256-length bucket of the cache, one graph per (kind, bucket)).
Control flow lives on the device: the step graph writes its result (n_new, tokens) into `res`, advances pos0 / the next
input token and (spec) does the acceptance; the host reads results back through pinned memory one cycle late (PIPE=1,
default: the next cycle is launched before the current result is read, so the GPU never idles on the host; PIPE=0: serial,
one host sync per cycle).  Greedy outputs are identical in both modes.
Matmuls of the graph-captured target steps (GEMM=triton, default): fused_gemm.py small-M Triton GEMMs with the sandwich-norm
chains fused in (fewer, faster kernels than cuBLAS M<=72 + inductor elementwise); GEMM=cublas: the torch.compile'd blocks.
Env knobs: FUSED (1), ATTN (triton | sdpa), FP32P (1: the Q=1 attention keeps the probabilities in fp32 on CUDA cores = the
numerics of flashinfer's decode kernel / the SGLang server, best agreement with the server's greedy outputs; 0: tensor-core
kernel with bf16 P, ~8-10% faster plain decoding), GEMM (triton | cublas), PIPE (1 | 0)."""
import math, os, time
import torch
import torch.nn.functional as F
from floodgates.engines.decode_attn import DecodeAttention
from floodgates.engines import fused_gemm as fg


# ---- fused per-layer blocks (torch.compile; same math as ouro_min.Layer, incl. the fp32 RMSNorm) ------------------------
def _rms(x, w, eps):
    dt = x.dtype
    xf = x.float()
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(dt)


def _rope(x, cos, sin):
    h = x.shape[-1] // 2
    return x * cos[:, None] + torch.cat([-x[..., h:], x[..., :h]], dim=-1) * sin[:, None]


def _lora(x, A, B):
    return (x.float() @ A.t() @ B.t()).to(x.dtype)


def pre_block(x, cos, sin, n1, qkv_w, eps: float, H: int, lq=None):
    """input norm -> fused q,k,v projection (+ LoRA) -> RoPE.  Returns q, k, v (B, H, T, D)."""
    B, T, _ = x.shape
    h = _rms(x, n1, eps)
    y = h @ qkv_w.t()
    if lq is not None:
        y = y + torch.cat([_lora(h, lq[0], lq[1]), _lora(h, lq[2], lq[3]), _lora(h, lq[4], lq[5])], -1)
    q, k, v = y.view(B, T, 3, H, -1).permute(2, 0, 3, 1, 4)
    return _rope(q, cos, sin), _rope(k, cos, sin), v


def rope_split(y, cos, sin, H: int):
    """Fused q,k,v projection output (T, 3*H*D) -> q, k (RoPE'd), v as (1, H, T, D)."""
    T = y.shape[0]
    q, k, v = y.view(1, T, 3, H, -1).permute(2, 0, 3, 1, 4)
    return _rope(q, cos, sin), _rope(k, cos, sin), v


def post_block(x, a, o_w, n2, n3, gu_w, d_w, n4, eps: float, lo=None):
    """o_proj -> norm_2 -> residual; norm -> fused gate/up -> SiLU * up -> down -> norm_2 -> residual (+ LoRA)."""
    B, T, _ = x.shape
    a = a.transpose(1, 2).reshape(B, T, -1)
    o = a @ o_w.t()
    if lo is not None:
        o = o + _lora(a, lo[0], lo[1])
    x = x + _rms(o, n2, eps)
    h = _rms(x, n3, eps)
    gu = h @ gu_w.t()
    if lo is not None:
        gu = gu + torch.cat([_lora(h, lo[2], lo[3]), _lora(h, lo[4], lo[5])], -1)
    g, u = gu.chunk(2, -1)
    m = F.silu(g) * u
    dn = m @ d_w.t()
    if lo is not None:
        dn = dn + _lora(m, lo[6], lo[7])
    return x + _rms(dn, n4, eps)


class Engine:
    def __init__(self, ouro, dr=None, n_rounds=1, tmax=4096, bucket=256, graphs=True, S=1, attn=None, pipeline=None):
        self.o, self.dr, self.n = ouro, dr, n_rounds
        c = ouro.c; R, L, H, D = ouro.R, ouro.L, c["num_attention_heads"], c["head_dim"]
        dev, dt = ouro.device, ouro.dtype
        self.R, self.L, self.dev, self.tmax, self.bucket, self.use_graphs = R, L, dev, tmax, bucket, graphs
        self.cl = R - 1 if getattr(dr, "ctx_loop", None) is None else dr.ctx_loop   # loop whose cache the slots read (drafter.ctx_loop; final loop by default, 0 = first-loop ablation)
        self.attn_kind = attn or os.environ.get("ATTN", "triton")
        self.gemm_kind = os.environ.get("GEMM", "triton")                      # triton: fused small-M GEMM stack (target steps)
        self.pipe = (os.environ.get("PIPE", "1") == "1") if pipeline is None else pipeline
        self.att = DecodeAttention(H, D, tmax, dev, fp32p=os.environ.get("FP32P", "1") == "1") if self.attn_kind == "triton" else None
        self.kc = torch.zeros(R, L, 1, H, tmax, D, device=dev, dtype=dt)
        self.vc = torch.zeros_like(self.kc)
        self.k = dr.k if dr is not None else 0
        self.S = S if dr is not None else 1                                   # draft candidates per cycle (tree width)
        k1 = self.k + 1; QM = max(self.S * k1, 1)
        self.k1, self.QM = k1, QM
        self.ids = torch.zeros(1, QM, dtype=torch.long, device=dev)          # target-step input ids
        self.pos0 = torch.zeros(1, dtype=torch.long, device=dev)               # first position of the step (= cut)
        self.pred = torch.zeros(QM, dtype=torch.long, device=dev)              # target argmax per input position
        self.res = torch.zeros(1 + QM, dtype=torch.long, device=dev)           # step result: [n_new, new tokens ...]
        self.hres = torch.zeros(2, 1 + QM, dtype=torch.long).pin_memory()      # host copies (double-buffered)
        self.ev = [torch.cuda.Event() for _ in range(2)]; self.slot = 0
        if dr is not None:
            self.sid = torch.zeros(1, QM, dtype=torch.long, device=dev)          # slot 0 of every block = y_m
            self.is_clean = torch.zeros(1, QM, dtype=torch.bool, device=dev); self.is_clean.view(self.S, k1)[:, 0] = True
            self.eps = torch.zeros(1, QM, dr.V, device=dev)                      # drafter noise (filled before replay)
            self.dout = torch.zeros(self.S * self.k, dtype=torch.long, device=dev)
            self.gum = torch.zeros(self.S, self.k, dr.V, device=dev) if self.S > 1 else None   # Gumbel noise, row 0 = 0
        # adaptive refinement (env ADAPT_MAX = max passes, 0 = off; needs a feedback drafter, S = 1): stop refining when a pass
        # adds less than ADAPT_DELTA tokens to the drafter's own expected accepted length (sum_j prod_{i<=j} max-prob_i)
        self.adapt_max = int(os.environ.get("ADAPT_MAX", 0)) if (dr is not None and self.S == 1 and getattr(dr, "feedback", False)) else 0
        self.adapt_delta = float(os.environ.get("ADAPT_DELTA", 2.5)); self.passes_used = []
        if self.adapt_max:
            self.pipe = False                                                  # the host decides between passes
            self.st = torch.zeros(1, k1, c["hidden_size"], device=dev, dtype=dt)
            self.cur_pred = torch.zeros(k1, dtype=torch.long, device=dev)
            self.expacc = torch.zeros(1, device=dev); self.dexp = torch.zeros(1, device=dev)
            self.hdexp = torch.zeros(1).pin_memory(); self.ev3 = torch.cuda.Event()
        self.graphs, self.pool = {}, None
        # fused weights + compiled blocks (FUSED=0 -> the plain ouro_min.Layer path)
        self.fused = os.environ.get("FUSED", "1") == "1"
        if self.fused:
            self.rms_eps, self.H = c["rms_norm_eps"], H
            self.W = []
            for ly in ouro.layers:
                self.W.append(dict(qkv=torch.cat([ly.q_proj.weight, ly.k_proj.weight, ly.v_proj.weight]).contiguous(),
                                   o=ly.o_proj.weight, gu=torch.cat([ly.gate_proj.weight, ly.up_proj.weight]).contiguous(),
                                   d=ly.down_proj.weight, n1=ly.input_layernorm.weight, n2=ly.input_layernorm_2.weight,
                                   n3=ly.post_attention_layernorm.weight, n4=ly.post_attention_layernorm_2.weight))
            if dr is not None:
                self.LQ, self.LO = [], []
                for lm in dr.lora:
                    self.LQ.append(tuple(t for n in ("q_proj", "k_proj", "v_proj") for t in (lm[n].A, lm[n].B)))
                    self.LO.append(tuple(t for n in ("o_proj", "gate_proj", "up_proj", "down_proj") for t in (lm[n].A, lm[n].B)))
            self.pre = torch.compile(pre_block, dynamic=False)
            self.post = torch.compile(post_block, dynamic=False)
            self.rope_split = torch.compile(rope_split, dynamic=False)

    def _layer(self, li, x, cos, sin, attn, lora=False):
        """One decoder layer, fused path or ouro_min.Layer; attn(q, k, v) handles cache/mask."""
        if not self.fused:
            return self.o.layers[li](x, cos, sin, attn, lora=self.dr.lora[li] if lora else None)[0]
        W = self.W[li]
        q, k, v = self.pre(x, cos, sin, W["n1"], W["qkv"], self.rms_eps, self.H, self.LQ[li] if lora else None)
        a = attn(q, k, v)
        return self.post(x, a, W["o"], W["n2"], W["n3"], W["gu"], W["d"], W["n4"], self.rms_eps, self.LO[li] if lora else None)

    # ---- graph bodies ----------------------------------------------------------------------------------------------
    def _target(self, Q, Lb, mode="prefill"):
        """mode: prefill (eager, no post-processing) | plain (advance by one) | spec (accept) | tree (S blocks of k+1)."""
        o, k1 = self.o, self.k1
        blk = k1 if mode == "tree" else Q
        ar = torch.arange(Q, device=self.dev)
        pos = self.pos0 + ar % blk                                             # RoPE positions (tree: cut + r per block)
        cos, sin = o.rope(pos[None])
        tri = self.att is not None and mode != "prefill"
        if not tri:
            wpos = self.pos0 + ar                                              # cache rows written (blocks at distinct rows)
            j = torch.arange(Lb, device=self.dev)[None, :]
            if mode == "tree":
                rel = j - self.pos0
                allowed = (j < self.pos0) | ((rel >= 0) & (rel < Q) & (rel // blk == (ar // blk)[:, None])
                                             & (rel % blk <= (ar % blk)[:, None]))
            else:
                allowed = j <= pos[:, None]
            allowed = allowed[None, None]                                      # (1, 1, Q, Lb)
        if tri and self.fused and self.gemm_kind == "triton" and Q <= fg.MAX_T:
            self._stack_fast(Q, cos, sin, blk)
        else:
            x = o.embed_tokens(self.ids[:, :Q])
            for ut in range(self.R):
                for li, layer in enumerate(o.layers):
                    kc, vc = self.kc[ut, li], self.vc[ut, li]
                    if tri:
                        def attn(q, k, v, kc=kc, vc=vc):
                            return self.att(q, k, v, kc, vc, self.pos0, blk, False, True)
                    else:
                        def attn(q, k, v, kc=kc, vc=vc):
                            kc.index_copy_(2, wpos, k); vc.index_copy_(2, wpos, v)
                            return F.scaled_dot_product_attention(q, kc[:, :, :Lb], vc[:, :, :Lb], attn_mask=allowed)
                    x = layer(x, cos, sin, attn)[0] if mode == "prefill" else self._layer(li, x, cos, sin, attn)
                x = o.norm(x)
            self.pred[:Q] = o.lm_head(x[0]).argmax(-1)
        if mode == "plain":
            p = self.pred[:1]
            self.res[:1].fill_(1); self.res[1:2].copy_(p)
            self.ids[0, :1].copy_(p); self.pos0 += 1
        elif mode in ("spec", "tree"):
            self._accept(Q // k1)

    def _stack_fast(self, Q, cos, sin, blk):
        """The R x L stack for Q tokens with Triton small-M GEMMs; the sandwich-norm elementwise chains are fused into
        fused_gemm.prep (between matmuls) and the SiLU*up epilogue; same math/rounding as pre_block/post_block."""
        o, eps, ns = self.o, self.rms_eps, self.o.norm.weight
        res, dn, n_prev = o.embed_tokens(self.ids[0, :Q]), None, None
        for ut in range(self.R):
            for li in range(self.L):
                W = self.W[li]
                xo, h = fg.prep(res, dn, n_prev, ns if (li == 0 and ut > 0) else None, W["n1"], eps)
                q, k, v = self.rope_split(fg.gemm(h, W["qkv"]), cos, sin, self.H)
                a = self.att(q, k, v, self.kc[ut, li], self.vc[ut, li], self.pos0, blk, False, True)
                x1, h2 = fg.prep(xo, fg.gemm(a.transpose(1, 2).reshape(Q, -1), W["o"]), W["n2"], None, W["n3"], eps)
                dn = fg.gemm(fg.gemm(h2, W["gu"], silu=True), W["d"])
                res, n_prev = x1, W["n4"]
        xf, _ = fg.prep(res, dn, n_prev, ns, ns, eps)                      # last residual add + model norm
        self.pred[:Q] = o.lm_head(xf).argmax(-1)

    def _accept(self, S):
        """Device-side acceptance over S candidate blocks: longest accepted prefix wins; writes res = [acc+1, tokens],
        the next y into ids/sid, moves the winner's K/V rows to the canonical rows, advances pos0."""
        k, k1 = self.k, self.k1
        P = self.pred[:S * k1].view(S, k1); Dr = self.ids[0, :S * k1].view(S, k1)[:, 1:]
        accv = (Dr == P[:, :k]).long().cumprod(1).sum(1)                       # (S,) accepted drafts per candidate
        b = accv.argmax(); acc = accv.max()                                    # first longest (candidate 0 on ties)
        Pb = P.index_select(0, b[None])[0]                                     # (k1,): accepted drafts == these preds
        y = Pb.gather(0, acc[None])                                            # the correction token = next y
        self.res[:1].copy_(acc[None] + 1); self.res[1:k1 + 1].copy_(Pb)
        self.ids[0, :S * k1].view(S, k1)[:, 0].copy_(y.expand(S)); self.sid.view(S, k1)[:, 0].copy_(y.expand(S))
        if S > 1:
            ar = torch.arange(k1, device=self.dev)
            src, dst = self.pos0 + b * k1 + ar, self.pos0 + ar
            for c in (self.kc, self.vc):
                f = c.view(-1, self.tmax, c.shape[-1])
                f.index_copy_(1, dst, f.index_select(1, src))
        self.pos0 += acc + 1

    def _draft(self, Lb):
        o, dr, S, k, k1 = self.o, self.dr, self.S, self.k, self.k1
        Q = S * k1
        cut = self.pos0
        ar = torch.arange(Q, device=self.dev)
        slot_pos = (cut + ar % k1)[None]
        cos, sin = o.rope(slot_pos)
        if self.att is None:
            ctx = (torch.arange(Lb, device=self.dev) < cut)[None, :].expand(Q, Lb)
            own = (ar[:, None] // k1) == (ar[None, :] // k1)
            mask = torch.cat([ctx, own], -1)[None, None]
        x_v = self.eps * dr.sigma
        state = None
        for i in range(self.n):
            t0, t1 = i / self.n, (i + 1) / self.n
            t = torch.full((1, Q), t0, device=self.dev); t_in = torch.where(self.is_clean, torch.ones_like(t), t)
            t_end = torch.where(self.is_clean, torch.ones_like(t), torch.full_like(t, t1))
            if getattr(dr, "feedback", False) and i > 0:                     # trained refinement: drafted tokens fed back
                pred = logits.view(S, k1, -1).argmax(-1)
                h = dr.feedback_inputs(self.sid.view(S, k1)[:, 0], pred, k1, state.view(S, k1, -1)).reshape(1, Q, -1)
            else:
                h = dr.slot_inputs(x_v, self.sid, self.is_clean, t_in, state, t_end)
            for li, layer in enumerate(o.layers):
                kc, vc = self.kc[self.cl, li], self.vc[self.cl, li]
                if self.att is not None:
                    def attn(q, k_, v_, kc=kc, vc=vc):
                        return self.att(q, k_, v_, kc, vc, self.pos0, k1, True, False)
                else:
                    def attn(q, k_, v_, kc=kc, vc=vc):
                        return F.scaled_dot_product_attention(q, torch.cat([kc[:, :, :Lb], k_], 2), torch.cat([vc[:, :, :Lb], v_], 2),
                                                              attn_mask=mask)
                h = self._layer(li, h, cos, sin, attn, lora=True)
            logits = o.lm_head(o.norm(h)).float()
            if i < self.n - 1 and dr.use_noise:
                x_v = x_v + (t1 - t0) / (1 - t0) * (logits.softmax(-1) - x_v)
            state = h
        if getattr(self, "dr_state", None) is not None:                    # engine_baselines: keep the slots' normalized states
            self.dr_state.copy_(o.norm(state)[0])
        lg = logits.view(S, k1, -1)[:, 1:]
        if S > 1:
            lg = lg + self.gum                                                 # Gumbel-max readout sampling (row 0: argmax)
        if getattr(dr, "markov_rank", 0):                                     # left-to-right readout with the Markov bias
            assert S == 1 and not dr.shift
            toks = dr.markov_decode(logits.view(S, k1, -1), self.sid.view(S, k1)[:, 0])
        else:
            toks = lg.argmax(-1)                                               # (S, k)
        self.dout.copy_(toks.reshape(-1))
        ids = self.ids.view(S, k1)
        ids[:, 1:].copy_(toks); ids[:, 0].copy_(self.sid.view(S, k1)[:, 0])

    def _draft_pass(self, Lb, first):
        """One drafter pass (adaptive mode, S = 1): the mask pass (first) or a feedback pass from the stored state/pred."""
        o, dr, k1 = self.o, self.dr, self.k1
        Q = k1
        slot_pos = (self.pos0 + torch.arange(Q, device=self.dev))[None]
        cos, sin = o.rope(slot_pos)
        if self.att is None:
            ctx = (torch.arange(Lb, device=self.dev) < self.pos0)[None, :].expand(Q, Lb)
            mask = torch.cat([ctx, torch.ones(Q, Q, dtype=torch.bool, device=self.dev)], -1)[None, None]
        if first:
            t = torch.zeros(1, Q, device=self.dev); t_in = torch.where(self.is_clean, torch.ones_like(t), t)
            h = dr.slot_inputs(self.eps * dr.sigma, self.sid, self.is_clean, t_in, None, t_in)
        else:
            h = dr.feedback_inputs(self.sid[:, 0], self.cur_pred[None], k1, self.st).reshape(1, Q, -1)
        for li, layer in enumerate(o.layers):
            kc, vc = self.kc[self.cl, li], self.vc[self.cl, li]
            if self.att is not None:
                def attn(q, k_, v_, kc=kc, vc=vc):
                    return self.att(q, k_, v_, kc, vc, self.pos0, k1, True, False)
            else:
                def attn(q, k_, v_, kc=kc, vc=vc):
                    return F.scaled_dot_product_attention(q, torch.cat([kc[:, :, :Lb], k_], 2), torch.cat([vc[:, :, :Lb], v_], 2), attn_mask=mask)
            h = self._layer(li, h, cos, sin, attn, lora=True)
        logits = o.lm_head(o.norm(h)).float()[0]                               # (k1, V)
        pred = logits.argmax(-1)
        e = logits[1:].softmax(-1).amax(-1).cumprod(0).sum()[None]             # drafter's expected accepted length
        if first:
            self.dexp.zero_()
        else:
            self.dexp.copy_(e - self.expacc)
        self.expacc.copy_(e); self.st.copy_(h); self.cur_pred.copy_(pred)
        self.dout.copy_(pred[1:]); self.ids[0, 1:].copy_(pred[1:]); self.ids[0, :1].copy_(self.sid[0, :1])

    # ---- graph management --------------------------------------------------------------------------------------------
    def _run(self, kind, Q, need_len, mode="plain"):
        Lb = self.tmax if self.att is not None else min(self.tmax, self.bucket * math.ceil(need_len / self.bucket))
        if kind == "t":
            body = lambda: self._target(Q, Lb, mode)
        elif kind in ("d1", "dr"):
            body = lambda: self._draft_pass(Lb, kind == "d1")
        else:
            body = lambda: self._draft(Lb)
        if not self.use_graphs:
            return body()
        key = (kind, Q, Lb, mode)
        g = self.graphs.get(key)
        if g is None:
            saved = [t.clone() for t in self._state()]                         # the body advances the device state
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    body()                                                     # warm-up (allocations, autotune)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                body()
            self.pool = g.pool(); self.graphs[key] = g
            for t, v in zip(self._state(), saved):
                t.copy_(v)
        g.replay()

    def _state(self):
        return [self.pos0, self.ids, self.res] + ([self.sid] if self.dr is not None else []) + \
            ([self.st, self.cur_pred, self.expacc, self.dexp] if getattr(self, "adapt_max", 0) else [])

    def warmup(self, max_len):
        """Capture every graph up to max_len so timing excludes capture."""
        S, k1 = self.S, self.k1
        Lbs = [self.tmax] if self.att is not None else range(self.bucket, min(self.tmax, max_len) + 1, self.bucket)
        for Lb in Lbs:
            self.pos0.fill_(Lb // 2)
            self._run("t", 1, Lb, "plain")
            if self.adapt_max:
                self._run("d1", 0, Lb); self._run("dr", 0, Lb); self._run("t", S * k1, Lb, "spec")
            elif self.dr is not None:
                self._run("d", 0, Lb); self._run("t", S * k1, Lb, "tree" if S > 1 else "spec")
        torch.cuda.synchronize()

    # ---- host <-> device result passing --------------------------------------------------------------------------
    def _readback(self):
        """Enqueue the copy of `res` into a pinned slot; returns the slot (wait with _wait)."""
        s = self.slot; self.slot ^= 1
        self.hres[s].copy_(self.res, non_blocking=True); self.ev[s].record()
        return s

    def _wait(self, s):
        self.ev[s].synchronize()
        r = self.hres[s].tolist()
        return r[0], r[1:1 + r[0]]

    # ---- decoding ------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, prompt):
        """Eager (no graph) causal pass over the prompt; returns the first generated token (int)."""
        P = len(prompt)
        ids_save = self.ids
        self.ids = torch.tensor([prompt], device=self.dev); self.pos0.fill_(0)
        pred_save = self.pred; self.pred = torch.zeros(P, dtype=torch.long, device=self.dev)
        Lb = min(self.tmax, self.bucket * math.ceil(P / self.bucket))
        self._target(P, Lb, "prefill")
        y = int(self.pred[P - 1])
        self.ids, self.pred = ids_save, pred_save
        return y

    @torch.no_grad()
    def generate_plain(self, prompt, max_new, stop):
        y = self.prefill(prompt); out = [y]; cur = len(prompt)          # y sits at position cur (not yet in the cache)
        self.ids[0, 0] = y; self.pos0.fill_(cur)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        go = lambda: len(out) < max_new and not stop(out)
        if not go():
            return out, 0.0
        launch = lambda c: (self._run("t", 1, c + 1, "plain"), self._readback())[1]
        s = launch(cur)
        while True:
            if self.pipe:
                s_next = launch(cur + 1)                                # look-ahead: input token produced on the device
            y = self._wait(s)[1][0]; out.append(y); cur += 1
            if not go():
                break
            s = s_next if self.pipe else launch(cur)
        t1 = time.perf_counter(); torch.cuda.synchronize()                 # drain the look-ahead step (not timed)
        return out, t1 - t0

    @torch.no_grad()
    def generate_spec(self, prompt, max_new, stop, gen=None):
        S, k1 = self.S, self.k1
        y = self.prefill(prompt); out = [y]; cur = len(prompt); taus = []
        self.sid.view(S, k1)[:, 0] = y; self.ids.view(S, k1)[:, 0] = y; self.pos0.fill_(cur)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        go = lambda: len(out) < max_new and not stop(out)
        if not go():
            return out, taus, 0.0
        mode = "tree" if S > 1 else "spec"

        def launch(c):                                                     # c: upper bound of cur (SDPA buckets only)
            if self.dr.use_noise:
                self.eps.normal_(generator=gen)
            if S > 1:
                self.gum[1:].exponential_(generator=gen).log_().neg_()      # Gumbel(0, 1); candidate 0 keeps zeros
            if self.adapt_max:                                                 # adaptive number of drafter passes
                self._run("d1", 0, c + k1); used = 1
                for _ in range(self.adapt_max - 1):
                    self._run("dr", 0, c + k1); used += 1
                    self.hdexp.copy_(self.dexp, non_blocking=True); self.ev3.record(); self.ev3.synchronize()
                    if float(self.hdexp[0]) < self.adapt_delta:
                        break
                self.passes_used.append(used)
            else:
                self._run("d", 0, c + S * k1)
            self._run("t", S * k1, c + S * k1, mode)
            return self._readback()
        s = launch(cur)
        while True:
            if self.pipe:
                s_next = launch(cur + k1)                               # the in-flight cycle adds at most k+1 tokens
            n, new = self._wait(s); out += new; taus.append(n); cur += n
            if not go():
                break
            s = s_next if self.pipe else launch(cur)
        t1 = time.perf_counter(); torch.cuda.synchronize()
        return out, taus, t1 - t0
