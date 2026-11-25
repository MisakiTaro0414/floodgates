"""Lossless speculative SAMPLING on a Raven-family looped LM (Recurrent-Llama-3.2, R=32), offline: a real per-problem
generation loop with RavenMin as the exact verifier (all R core iterations), Leviathan et al. / Chen et al. rejection
sampling as in eval_sample.py (the Ouro version).  Target p = softmax(final logits / T) restricted to the top-p nucleus
(LoopSpec's sampling setting: T=1, top-p 0.7).  A cycle starts after the verifier emitted y (known, not yet processed; its
position cut = cache length); the proposer drafts d_1..d_n for positions cut+1..cut+n with proposals q_1..q_n; ONE target
pass over [y, d_1..d_n] (all R iterations; K/V appended to the per-(block, iteration) cache) gives p_1..p_{n+1}; d_j is
accepted w.p. min(1, p_j(d_j)/q_j(d_j)); at the first rejection a sample of norm(max(p_j - q_j, 0)) is emitted, if all n
are accepted a bonus sample of p_{n+1}; the cache is truncated to prefix + y + accepted drafts.  tau = tokens emitted per
verification.  Lossless for any proposer whose d_j is distributed as q_j given everything it used (auxiliary randomness,
earlier passes' samples): every q_j here is fixed before the d's are drawn and d_j ~ q_j independently, so the output is
distributed exactly as autoregressive sampling from p.
Proposers (all read the verified prefix's K/V of the prelude, the FINAL core iteration and the coda, as the offline evals):
  ours <run_dir> <passes...>  RavenSelfDrafter (drafter_raven.py; args from <run>/args.json) with n passes; feedback passes
      take tokens SAMPLED from the previous pass's proposal softmax; q_j = softmax(final-pass slot logits / T) (= the plain
      softmax at T=1); shift: slot j predicts position cut+j+1 and all k+1 slots are drafts (draft length k+1).
  daswave-argmax [J k gamma]  DAS-Wave exactly as eval_daswave_raven.py (paper settings J 24, k 1, gamma 32; window growth
      W_j = min(gamma, 2 + j) -> 25 drafts at J 24); drafts = final readout argmax, q = one-hot (accept prob = p(d_j)).
  daswave-sampled [J k gamma] same deterministic rounds; d_j ~ q_j = softmax(final readout logits / T).
env: TASK (gsm8k | math500 -> trace, MAX_NEW 512 | 2048, stop "Q:" | "Problem:"), N_PROBLEMS (100), MAX_NEW, STOP_STRS
     ('|'-separated), TEMP (1.0), TOP_P (0.7), Q_TOPP (0; 1 = apply the target's top-p nucleus to q too), OUT
     (runs/raven/sampling), EVAL_TAG (extra filename suffix).  Seeds: problem i uses generator seed 1000 + i.
     TEMP=0: exact greedy (p, q one-hot on the argmax, lowest-index tie-break as RavenMin.generate) = greedy speculative
     decoding; GEN_CHECK=k additionally decodes the first k problems with RavenMin.generate and reports the common prefix.
     FB=argmax (ours): feedback passes take the previous pass's argmax tokens (as in training) instead of samples (label
     gets "_fbargmax"); q is still the final pass's softmax, so the proposal stays valid.
Writes <OUT>/<label>_T<T>_p<P>[_qtopp]_<task><tag>.json (keyed by passes / J) and one *_outputs.jsonl per config."""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import json, os, sys, time
import torch
import torch.nn.functional as F
import os
from floodgates.paths import LOOPSPEC, model_dir, trace
from floodgates.models.raven_min import RavenMin
from floodgates.drafters.drafter_raven import RavenSelfDrafter
from transformers import PreTrainedTokenizerFast

L = LOOPSPEC
MODEL = os.environ.get("MODEL", f"{L}/models/Recurrent-Llama-3.2-train-recurrence-32")
TASK = os.environ.get("TASK", "gsm8k")
TASK_DEF = {"gsm8k": (512, "Q:"), "math500": (2048, "Problem:")}[TASK]
TR = os.environ.get("TRACE", trace(MODEL, TASK))
N_PROB = int(os.environ.get("N_PROBLEMS", 100)); MAX_NEW = int(os.environ.get("MAX_NEW", TASK_DEF[0]))
STOPS = [x for x in os.environ.get("STOP_STRS", TASK_DEF[1]).split("|") if x]
TEMP = float(os.environ.get("TEMP", 1.0)); TOP_P = float(os.environ.get("TOP_P", 0.7))
Q_TOPP = os.environ.get("Q_TOPP", "0") == "1"
FB = os.environ.get("FB", "sample")            # ours, feedback passes: "sample" (default) or "argmax" (as in training)
OUT = os.environ.get("OUT", "runs/raven/sampling"); TAG = os.environ.get("EVAL_TAG", "")
W_INIT = 2
torch.backends.cuda.enable_cudnn_sdp(False)   # cuDNN SDPA builds a graph per new (q, k) shape: ~40 ms each, 24 per DAS-Wave cycle

kind = sys.argv[1]
m = RavenMin(MODEL); dev = m.device; c = m.c; R = m.R
nP, nC, nO = len(m.prelude), len(m.core), len(m.coda)
NB = nP + R * nC + nO                                      # cache blocks: prelude, core x every iteration, coda
FINAL = list(range(nP)) + [nP + (R - 1) * nC + li for li in range(nC)] + [nP + R * nC + li for li in range(nO)]
ar_layers = nP + nC * R + nO
if kind == "ours":
    run = sys.argv[2]; configs = [int(x) for x in sys.argv[3:]] or [1]
    a = json.load(open(os.path.join(run, "args.json")))
    assert os.path.realpath(model_dir(a["model"])) == os.path.realpath(MODEL), (model_dir(a["model"]), MODEL)
    dr = RavenSelfDrafter(m, k=a["k"], rank=a["rank"], shift=a.get("shift", False), feedback=a.get("feedback", False),
                          iters=a.get("iters", 1), carry=a.get("carry", False))
    missing, unexpected = dr.load_state_dict(torch.load(os.path.join(run, "drafter.pt")), strict=False)
    assert not unexpected and all(x.startswith("m.") or x == "draft_emb" for x in missing), (unexpected, [x for x in missing if not x.startswith("m.")])
    dr.eval().requires_grad_(False)
    NDRAFT = a["k"] + (1 if dr.shift else 0)
    label = f"ours_{os.path.basename(run.rstrip('/'))}" + ("_fbargmax" if FB == "argmax" and dr.feedback else "")
    draft_layers = lambda n: nP + nC * n * dr.iters + nO
elif kind in ("daswave-argmax", "daswave-sampled"):
    J, KIT, GAMMA = (int(x) for x in sys.argv[2:5]) if len(sys.argv) >= 5 else (24, 1, 32)
    configs = [J]
    NDRAFT = min(GAMMA, W_INIT + J - 1)
    label = f"{kind}_J{J}_k{KIT}_g{GAMMA}"
    draft_layers = lambda n: J * (nP + nC * KIT + nO)
else:
    raise SystemExit(f"unknown proposer {kind}")
tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(MODEL, "tokenizer.json"))
tc = json.load(open(os.path.join(MODEL, "tokenizer_config.json")))
eos_tok = tc.get("eos_token"); eos_tok = eos_tok["content"] if isinstance(eos_tok, dict) else eos_tok
EOS = tok.convert_tokens_to_ids(eos_tok) if eos_tok else None
LB = max(len(tok(x, add_special_tokens=False)["input_ids"]) for x in STOPS) + NDRAFT + 12   # stop-string lookback


def nucleus(p, top_p):
    """p restricted to its top-p nucleus (smallest prefix of the sorted probs reaching top_p), renormalised (last dim)."""
    sp, si = p.sort(-1, descending=True)
    keep = (sp.cumsum(-1) - sp) < top_p
    p = torch.zeros_like(p).scatter_(-1, si, sp * keep)
    return p / p.sum(-1, keepdim=True)


def greedy(logits):
    return F.one_hot(logits.argmax(-1), logits.shape[-1]).float()


def target_probs(logits):
    if TEMP == 0:
        return greedy(logits)
    p = (logits.float() / TEMP).softmax(-1)
    return nucleus(p, TOP_P) if TOP_P < 1.0 else p


def proposal_probs(logits):
    if TEMP == 0:
        return greedy(logits)
    p = (logits.float() / TEMP).softmax(-1)
    return nucleus(p, TOP_P) if (Q_TOPP and TOP_P < 1.0) else p


class Cache:
    """K/V of the processed positions [0, n) for every block and every core iteration (as RavenMin.generate), preallocated
    (NB, 1, KV, smax, D); truncation on rejection = moving n back."""
    def __init__(self, smax):
        KV, D = c["num_key_value_heads"], c["n_embd"] // c["n_heads"]
        self.K = torch.empty(NB, 1, KV, smax, D, dtype=m.dtype, device=dev)
        self.V = torch.empty_like(self.K)
        self.n = 0

    def final_ctx(self):
        """Draft-pass block order [prelude..., core LAST iteration..., coda...] of (k, v) (1, KV, n, D) (= context_kv)."""
        return [(self.K[b][:, :, :self.n], self.V[b][:, :, :self.n]) for b in FINAL]


@torch.no_grad()
def extend(cache, ids, last_only=False):
    """Exact target pass of ids (T,) at positions [n, n+T): causal over the cache + themselves, R core iterations from the
    per-position like-init state init_table[pos] (as RavenMin.generate); appends K/V.  Returns final logits (T or 1, V)."""
    n0 = cache.n; T = ids.shape[0]; S = n0 + T
    pos = torch.arange(n0, S, device=dev)[None]
    cos, sin = m.rope(pos)
    mask = torch.ones(T, S, dtype=torch.bool, device=dev).tril(n0)

    def attn_for(b):
        def attn(q, k, v):
            cache.K[b][:, :, n0:S] = k; cache.V[b][:, :, n0:S] = v
            return F.scaled_dot_product_attention(q, cache.K[b][:, :, :S], cache.V[b][:, :, :S], attn_mask=mask, enable_gqa=True)
        return attn

    e = m.wte(ids[None]) * m.emb_scale
    for li, b in enumerate(m.prelude):
        e = b(e, cos, sin, attn_for(li))
    s = m.init_table(S + 1, 0)[pos]
    for it in range(R):
        s = m.adapter(torch.cat([s, e], -1))
        for li, b in enumerate(m.core):
            s = b(s, cos, sin, attn_for(nP + it * nC + li))
    for li, b in enumerate(m.coda):
        s = b(s, cos, sin, attn_for(nP + R * nC + li))
    cache.n = S
    if last_only:
        s = s[:, -1:]
    return m.lm_head(m.ln_f(s))[0].float()


@torch.no_grad()
def draft_ours(cache, y, n_pass, gen):
    """RavenSelfDrafter.draft with sampling: returns drafts d (NDRAFT,) and proposals q (NDRAFT, V)."""
    cut = cache.n; ctx = cache.final_ctx(); k1 = dr.k + 1
    mask = dr.masks(torch.tensor([[cut]], device=dev), cut, 1, k1, dev)
    sid = torch.zeros(1, k1, dtype=torch.long, device=dev); sid[0, 0] = y[0]
    is_clean = torch.zeros(1, k1, dtype=torch.bool, device=dev); is_clean[0, 0] = True
    slot_pos = (cut + torch.arange(k1, device=dev))[None]
    e_in = dr.slot_inputs(sid, is_clean); s0 = dr.init_states(slot_pos); state = None
    for i in range(n_pass):
        logits, state = dr.round(e_in, s0, slot_pos, ctx, mask, state)
        if dr.feedback and i < n_pass - 1:                  # feed back tokens sampled from this pass's proposal softmax
            fb = logits[0].argmax(-1) if FB == "argmax" else torch.multinomial(proposal_probs(logits[0]), 1, generator=gen)[:, 0]
            e_in = dr.feedback_inputs(sid[:, 0], fb[None], k1).to(e_in.dtype)
    q = proposal_probs(logits[0, (0 if dr.shift else 1):])
    return torch.multinomial(q, 1, generator=gen)[:, 0], q


@torch.no_grad()
def draft_wave(cache, y, gen, sampled):
    """eval_daswave_raven.wave_draft on the cache (prefix K/V at the final iteration).  Drafts for positions cut+1..cut+W."""
    cut = cache.n; ctx = cache.final_ctx()
    pre, core, coda = ctx[:nP], ctx[nP:nP + nC], ctx[nP + nC:]
    tab = m.init_table(cut + GAMMA + 2, 0)
    tokens = torch.cat([y, y])                               # anchor + one provisional input seeded with the anchor
    states = tab[cut:cut + 2][None]                          # (1, W, d) carried latents
    last = y
    for j in range(J):
        Wj = min(GAMMA, W_INIT + j)
        while tokens.shape[0] < Wj:                          # grow the window, new position seeded by the last readout
            tokens = torch.cat([tokens, last])
            states = torch.cat([states, tab[cut + tokens.shape[0] - 1][None, None]], 1)
        W = tokens.shape[0]; pos = torch.arange(cut, cut + W, device=dev)[None]; cos, sin = m.rope(pos)
        mask = torch.ones(W, cut + W, dtype=torch.bool, device=dev).tril(cut)

        def attn_with(kv):
            ck, cv = kv
            def attn(q, kk, vv):
                return F.scaled_dot_product_attention(q, torch.cat([ck, kk], 2), torch.cat([cv, vv], 2), attn_mask=mask,
                                                      enable_gqa=True)
            return attn
        e = m.wte(tokens[None]) * m.emb_scale
        for bi, b in enumerate(m.prelude):
            e = b(e, cos, sin, attn_with(pre[bi]))
        s = states
        for _ in range(KIT):
            s = m.adapter(torch.cat([s, e], -1))
            for bi, b in enumerate(m.core):
                s = b(s, cos, sin, attn_with(core[bi]))
        states = s
        h = s
        for bi, b in enumerate(m.coda):
            h = b(h, cos, sin, attn_with(coda[bi]))
        logits = m.lm_head(m.ln_f(h))[0]                     # (W, V); readout of position i -> input of position i+1
        z = logits.argmax(-1)
        tokens = torch.cat([y, z[:W - 1]]); last = z[W - 1:W]
    if sampled:
        q = proposal_probs(logits)
        return torch.multinomial(q, 1, generator=gen)[:, 0], q
    return z, F.one_hot(z, logits.shape[-1]).float()


def propose(cache, y, cfg, gen):
    if kind == "ours":
        return draft_ours(cache, y, cfg, gen)
    return draft_wave(cache, y, gen, kind == "daswave-sampled")


rows = [json.loads(l) for l in open(TR)][:N_PROB]
cache = Cache(max(len(r["input"]) for r in rows) + MAX_NEW + NDRAFT + 4)
os.makedirs(OUT, exist_ok=True)
base = os.path.join(OUT, f"{label}_T{TEMP}_p{TOP_P}" + ("_qtopp" if Q_TOPP else "") + f"_{TASK}{TAG}")
print(f"{label} task={TASK} T={TEMP} top_p={TOP_P} q_topp={Q_TOPP} N={len(rows)} MAX_NEW={MAX_NEW} stops={STOPS} "
      f"draft_len={NDRAFT}", flush=True)
res = {}
for cfg in configs:
    t0 = time.time(); t_draft = t_verify = 0.0; gen_check = []
    taus, n_tokens, ends = [], 0, {"stop": 0, "eos": 0, "max": 0}
    tried, accd = [0] * NDRAFT, [0] * NDRAFT; alpha_sum = [0.0] * NDRAFT
    fo = open(f"{base}_{cfg}_outputs.jsonl", "w")
    for pi, r in enumerate(rows):
        gen = torch.Generator(device=dev).manual_seed(1000 + pi)
        cache.n = 0
        lg = extend(cache, torch.tensor(r["input"], device=dev), last_only=True)
        y = torch.multinomial(target_probs(lg[-1]), 1, generator=gen)      # (1,) first token from the prefill
        out = [int(y)]; ptaus = []; end = "max"
        while len(out) < MAX_NEW:
            cut = cache.n
            torch.cuda.synchronize(); ta = time.time()
            d, q = propose(cache, y, cfg, gen)
            torch.cuda.synchronize(); tb = time.time(); t_draft += tb - ta
            n = d.shape[0]
            p = target_probs(extend(cache, torch.cat([y, d])))              # (n+1, V): p for d_1..d_n and the bonus
            ar = torch.arange(n, device=dev)
            ratio = (p[ar, d] / q[ar, d].clamp(min=1e-20)).clamp(max=1.0)
            rej = torch.rand(n, device=dev, generator=gen) >= ratio
            acc_t = torch.where(rej.any(), rej.float().argmax(), torch.full((), n, device=dev))   # first rejection
            alpha = torch.minimum(p[:n], q).sum(-1)                         # P(accept | position reached) = 1 - TV(p_j, q_j)
            pa = p[acc_t]; qa = q[acc_t.clamp(max=n - 1)] * (acc_t < n)     # rejection: residual; all accepted: bonus p_{n+1}
            resid = (pa - qa).clamp(min=0); rs = resid.sum()
            dist = torch.where(rs > 0, resid / rs.clamp(min=1e-30), pa)
            emitted = torch.multinomial(dist, 1, generator=gen)
            vals = torch.cat([acc_t[None], emitted, d]).tolist(); al = alpha.tolist(); t_verify += time.time() - tb
            acc = vals[0]
            for j in range(min(acc + 1, n)):
                tried[j] += 1; accd[j] += int(j < acc); alpha_sum[j] += al[j]
            cache.n = cut + 1 + acc                                         # keep y and the accepted drafts
            new = vals[2:2 + acc] + [vals[1]]
            out += new; ptaus.append(acc + 1); y = emitted
            if any(s in tok.decode(out[-LB:]) for s in STOPS):
                end = "stop"; break
            if EOS is not None and EOS in new:
                end = "eos"; break
        taus += ptaus; n_tokens += len(out); ends[end] += 1
        if pi < int(os.environ.get("GEN_CHECK", 0)):           # reference: plain greedy RavenMin.generate (TEMP=0 only)
            ref = m.generate(r["input"], MAX_NEW, lambda o: any(s in tok.decode(o[-LB:]) for s in STOPS) or o[-1] == EOS)
            k_ = 0
            while k_ < min(len(ref), len(out)) and ref[k_] == out[k_]:
                k_ += 1
            gen_check.append({"idx": pi, "common_prefix": k_, "spec_len": len(out), "generate_len": len(ref)})
            print(f"    gen check problem {pi}: common prefix {k_} (spec {len(out)}, generate {len(ref)} tokens)", flush=True)
        fo.write(json.dumps({"idx": pi, "output_ids": out, "taus": ptaus, "end": end}) + "\n"); fo.flush()
        if pi % 10 == 9 or pi == len(rows) - 1:
            print(f"  [{label} cfg={cfg}] {pi + 1}/{len(rows)} problems, tau so far {sum(taus) / len(taus):.3f}, "
                  f"{len(taus)} cycles, {time.time() - t0:.0f}s", flush=True)
    fo.close()
    tau = sum(taus) / len(taus); dl = draft_layers(cfg)
    res[cfg] = {"tau": tau, "cycles": len(taus), "tokens": n_tokens, "n_problems": len(rows), "draft_len": NDRAFT,
                "accept_rate_by_slot": [round(x / t, 4) if t else None for x, t in zip(accd, tried)],
                "alpha_by_slot": [round(x / t, 4) if t else None for x, t in zip(alpha_sum, tried)],
                "reached_by_slot": tried, "draft_layers": dl, "ar_layers": ar_layers,
                "ideal_speedup": tau * ar_layers / (ar_layers + dl), "temp": TEMP, "top_p": TOP_P, "q_topp": Q_TOPP,
                "max_new": MAX_NEW, "stops": STOPS, "ends": ends, "mean_out_len": n_tokens / len(rows),
                "seconds": time.time() - t0, "draft_s_per_cycle": t_draft / len(taus), "verify_s_per_cycle": t_verify / len(taus),
                "trace": TR, "seed": "1000 + problem index"}
    if gen_check:
        res[cfg]["gen_check"] = gen_check
    print(f"{label} cfg={cfg} task={TASK} T={TEMP} top_p={TOP_P}: tau={tau:.3f} cycles={len(taus)} tokens={n_tokens} "
          f"ends={ends} draft {t_draft / len(taus) * 1e3:.0f} ms + verify {t_verify / len(taus) * 1e3:.0f} ms per cycle, accept by slot={res[cfg]['accept_rate_by_slot'][:12]} alpha={res[cfg]['alpha_by_slot'][:8]}", flush=True)
    json.dump(res, open(f"{base}.json", "w"), indent=1)
