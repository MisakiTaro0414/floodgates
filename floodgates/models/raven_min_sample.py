"""Sampling versions of RavenMin.generate (plain decoding) and RavenMin.df_generate (our reimplementation of the parallel /
diffusion-forcing sampler of Geiping et al. 2026, arXiv 2510.14961), for df_eval_sample.py.  raven_min.py is NOT modified: the two
methods are copied here unchanged except that every argmax becomes a draw from the target distribution of our sampling runs,
  p = top-p(softmax(logits / T)) with the tie-symmetric nucleus of sampling_ops.probs_cand (= raven_engine_sample.probs_cand: keep
  x iff p_x >= the smallest probability of the sorted-prefix nucleus, threshold searched among the K = 4096 most likely tokens),
  drawn by the exponential race of sampling_ops.sample (torch.rand on the default CUDA generator, so torch.cuda.manual_seed(s)
  before a generation makes it reproducible).  T = 0 is exact greedy (one-hot argmax, lowest index on ties = torch.argmax), so
  temp=0 reproduces RavenMin.generate / RavenMin.df_generate token for token.  The per-position initial states are the same
  seed-0 table as the greedy runs (raven_min.init_table); the sampling noise is the only new randomness.

How the parallel sampler samples -- as specified by Geiping et al.: Sec. 3.3, Algorithm 1 l.12-13 ("y_hat <- Sample(p)",
"y_current <- [y_frozen, y_hat]") and App. A.1, Algorithm 2 l.12-13 and l.17 ("y_frozen <- y_current[1:k*]"); their code
(seal-rg/recurrent-pretraining, recpre/raven_modeling_minimal.py, generate_diffusion_style) calls _sample_next_token on the
logits of EVERY wavefront position at EVERY step, and their runs sample (App. A.3: temperature 0.2, top-p 0.95; we use the
T = 1, top-p 0.7 of our other sampling runs):
  * every forward pass draws a fresh token y_i ~ p_i for every wavefront position from its current readout (independent draws
    across passes and positions);
  * y_i is the input of position i+1 in the next pass, through the embedding momentum of Eq. 1 (e = eta e_prev + (1-eta) P(y));
  * a position that freezes commits the token drawn in the pass in which it freezes.
The freeze rule is unchanged and never looks at tokens: the relative latent change delta_i = ||z_i - z_prev,i|| / ||z_i|| of
Eq. 3 below eps, frozen as the longest such prefix (or after max_iters = R core iterations, as in RavenMin.df_generate).  Under
sampling it stays meaningful because a re-drawn predecessor token changes the input embedding of position i and therefore
moves z_i, which keeps delta_i above eps until the conditioning settles; positions whose predecessors have broad
distributions therefore freeze only after their predecessors (prefix rule) or at the iteration cap.  Two definitions of z_prev:
  delta="iter" (default, = RavenMin.df_generate and the greedy df3_* rows): z_prev = the latent before the LAST of the `inner`
    core iterations of the pass, i.e. the change of one iteration at fixed input;
  delta="step" (the paper's: Alg. 2 l.14 with z_prev <- z at l.25, the end of the previous step; their code compares with
    `previous_states` saved at the end of the previous step): z_prev = the latent at the START of the pass (a new position: its
    initial state), i.e. the change over a whole sampler step, which includes the response to a re-drawn input token.
Diagnostics in self.stats (accumulated over calls, reset by reset_stats()): committed tokens, tokens frozen by the iteration cap
rather than by delta < eps ("forced"), and "mismatch" = committed tokens that differ from the input their successor consumed in
the freezing pass (when several positions freeze in one pass, the successor's K/V was computed from the previous pass's draw;
the paper's algorithm and code commit the new draw all the same; under greedy decoding this is rare)."""
import torch
import torch.nn.functional as F
from floodgates.models.raven_min import RavenMin
from floodgates.engines.sampling_ops import probs_cand, sample


class RavenMinSample(RavenMin):
    def __init__(self, model_dir, temp=1.0, top_p=0.7, k_cand=4096, delta="iter", **kw):
        super().__init__(model_dir, **kw)
        assert delta in ("iter", "step"), delta
        self.temp, self.top_p, self.k_cand, self.delta = float(temp), float(top_p), int(k_cand), delta
        self.reset_stats()

    def reset_stats(self):
        self.stats = {"committed": 0, "forced": 0, "mismatch": 0}
        self._uncov = torch.zeros((), dtype=torch.long, device=self.device)

    @property
    def uncovered(self):
        """Rows whose top-k_cand mass was below top_p (the nucleus then keeps everything >= the k-th probability); 0 = exact."""
        return int(self._uncov)

    def draw(self, logits):
        """logits (N, V) -> (N,) long, one draw per row from top-p(softmax(logits / T)) (T = 0: argmax)."""
        p, unc = probs_cand(logits, self.temp, self.top_p, self.k_cand)
        if unc is not None:
            self._uncov += unc.sum()
        return sample(p)

    @torch.no_grad()
    def generate(self, prompt, max_new, stop=lambda out: False, n_iter=None, seed=0):
        """RavenMin.generate with y ~ top-p(softmax(logits / T)) instead of the argmax (batch 1, per-(stage, iteration, layer)
        KV caches; the initial state of position j is row j of the seed-`seed` table, as in the greedy loop)."""
        n_iter = n_iter or self.R
        cache = {}

        def attn_for(key):
            def attn(q, k, v):
                if key in cache:
                    pk, pv = cache[key]
                    k, v = torch.cat([pk, k], 2), torch.cat([pv, v], 2)
                cache[key] = (k, v)
                T, S = q.shape[2], k.shape[2]
                if T == S:
                    return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
                m = torch.ones(T, S, dtype=torch.bool, device=q.device).tril(S - T)
                return F.scaled_dot_product_attention(q, k, v, attn_mask=m, enable_gqa=True)
            return attn

        out, ids, start = [], list(prompt), 0
        x_ids = torch.tensor([ids], device=self.device)
        while len(out) < max_new:
            T = x_ids.shape[1]
            cos, sin = self.rope(torch.arange(start, start + T, device=self.device)[None])
            e = self.wte(x_ids) * self.emb_scale
            for li, b in enumerate(self.prelude):
                e = b(e, cos, sin, attn_for(("p", li)))
            s = self.init_state_at(range(start, start + T), seed)
            for it in range(n_iter):
                s = self.adapter(torch.cat([s, e], -1))
                for li, b in enumerate(self.core):
                    s = b(s, cos, sin, attn_for(("c", it, li)))
            s = self.pre(s)
            for li, b in enumerate(self.coda):
                s = b(s, cos, sin, attn_for(("o", li)))
            nxt = int(self.draw(self.lm_head(self.ln_f(s[:, -1])))[0])       # greedy: .argmax(-1)
            out.append(nxt)
            if stop(out):
                break
            x_ids, start = torch.tensor([[nxt]], device=self.device), start + T
        return out

    @torch.no_grad()
    def df_generate(self, prompt, max_new, stop=lambda out: False, inner=4, eta=0.1, eps=0.03, max_wave=128,
                    max_iters=None, seed=0, delta=None):
        """RavenMin.df_generate under sampling (module docstring): each forward pass, one new position joins the wavefront (its
        input = the predecessor's current draw); input embeddings of in-flight positions are refreshed from the predecessors'
        current draws with momentum eta; `inner` recurrence steps run for all in-flight positions; the coda reads out every
        position and a fresh token is drawn for each; the longest prefix with delta < eps (or max_iters reached) is frozen and
        commits this pass's draws.  KV: committed positions keep only their latest (final) K/V.
        Returns (generated ids, number of forward passes, total core iterations)."""
        max_iters = max_iters or self.R
        delta = delta or self.delta
        W = self.wte.weight
        ck = {}

        def attn_with(key, commit):
            def attn(q, k, v):
                if key in ck:
                    pk, pv = ck[key]
                    K, V = torch.cat([pk, k], 2), torch.cat([pv, v], 2)
                else:
                    K, V = k, v
                if commit is not None:
                    commit[key] = (k, v)
                T, S = q.shape[2], K.shape[2]
                m = torch.ones(T, S, dtype=torch.bool, device=q.device).tril(S - T)
                return F.scaled_dot_product_attention(q, K, V, attn_mask=m, enable_gqa=True)
            return attn

        P = len(prompt)
        cos, sin = self.rope(torch.arange(P, device=self.device)[None])
        new = {}
        e = self.wte(torch.tensor([prompt], device=self.device)) * self.emb_scale
        for li, b in enumerate(self.prelude):
            e = b(e, cos, sin, attn_with(("p", li), new))
        s = self.init_state_at(range(P), seed)
        for it in range(self.R):
            s = self.adapter(torch.cat([s, e], -1))
            for li, b in enumerate(self.core):
                s = b(s, cos, sin, attn_with(("c", li), new if it == self.R - 1 else None))
        s = self.pre(s)
        for li, b in enumerate(self.coda):
            s = b(s, cos, sin, attn_with(("o", li), new))
        ck.update(new)
        out = [int(self.draw(self.lm_head(self.ln_f(s[:, -1])))[0])]          # greedy: .argmax(-1)
        wave = []                                      # dict(pos, z, e, pred, iters)
        passes = iters_total = 0
        st = self.stats
        while len(out) < max_new and not stop(out):
            nxt_pos = (wave[-1]["pos"] + 1) if wave else P + len(out) - 1
            if len(wave) < max_wave:
                wave.append({"pos": nxt_pos, "z": self.init_state_at([nxt_pos], seed), "e": None, "pred": None, "iters": 0})
            pos = torch.tensor([[w["pos"] for w in wave]], device=self.device)
            cos, sin = self.rope(pos)
            toks = [out[-1]] + [w["pred"] for w in wave[:-1]]       # input of each position = predecessor's current draw
            toks = [t if t is not None else out[-1] for t in toks]
            x_in = W[torch.tensor(toks, device=self.device)][None] * self.emb_scale
            rec = {}                                   # this pass's K/V of all wave positions (prelude, last inner iter, coda)
            e = x_in
            for li, b in enumerate(self.prelude):
                e = b(e, cos, sin, attn_with(("p", li), rec))
            e_mix = torch.cat([(eta * w["e"] + (1 - eta) * e[:, k:k + 1]) if w["e"] is not None else e[:, k:k + 1]
                               for k, w in enumerate(wave)], 1)
            z = torch.cat([w["z"] for w in wave], 1); z_prev = z           # delta="step": the latent at the start of the pass
            for r_ in range(inner):
                if delta == "iter":
                    z_prev = z
                z = self.adapter(torch.cat([z, e_mix], -1))
                for li, b in enumerate(self.core):
                    z = b(z, cos, sin, attn_with(("c", li), rec if r_ == inner - 1 else None))
            h = self.pre(z)
            for li, b in enumerate(self.coda):
                h = b(h, cos, sin, attn_with(("o", li), rec))
            preds = self.draw(self.lm_head(self.ln_f(h))[0]).tolist()     # a fresh draw for every position (greedy: argmax)
            dz = ((z - z_prev).float().norm(dim=-1) / z.float().norm(dim=-1).clamp(min=1e-6))[0].tolist()
            passes += 1; iters_total += inner * len(wave)
            for k, w in enumerate(wave):
                w["z"], w["e"], w["pred"], w["iters"] = z[:, k:k + 1], e_mix[:, k:k + 1], preds[k], w["iters"] + inner
            # freeze the longest converged prefix: recompute its K/V once with its final latent and commit
            nfz = nforced = 0
            for w, dl in zip(wave, dz):
                if dl < eps or w["iters"] >= max_iters:
                    nfz += 1
                    nforced += int(not dl < eps)
                else:
                    break
            if nfz:
                fz = wave[:nfz]
                for key, (k, v) in rec.items():                           # freeze: keep their latest K/V
                    pk, pv = ck[key]
                    ck[key] = (torch.cat([pk, k[:, :, :nfz]], 2), torch.cat([pv, v[:, :, :nfz]], 2))
                st["forced"] += nforced
                for j, w in enumerate(fz):
                    out.append(w["pred"])
                    st["committed"] += 1
                    st["mismatch"] += int(j + 1 < nfz and w["pred"] != toks[j + 1])
                    if stop(out) or len(out) >= max_new:
                        break
                wave = wave[nfz:]
        return out, passes, iters_total
