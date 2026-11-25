"""Exact speculative-sampling ops shared by engine_sample.py and its tests (no CUDA needed at import): tie-symmetric top-p nucleus,
exponential-race draws, Leviathan et al. / Chen et al. rejection sampling.  Same code as raven_engine_sample.py's helpers."""
import torch


def nucleus(p, top_p):
    """p (..., V) restricted to its top-p nucleus (smallest prefix of the sorted probs reaching top_p), renormalised."""
    sp, si = p.sort(-1, descending=True)
    keep = (sp.cumsum(-1) - sp) < top_p
    p = torch.zeros_like(p).scatter_(-1, si, sp * keep)
    return p / p.sum(-1, keepdim=True)


def one_hot_argmax(logits):
    return torch.zeros(logits.shape, dtype=torch.float32, device=logits.device).scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)


def probs(logits, temp, top_p):
    """Reference (sort-based) target / proposal probabilities: (..., V) logits -> float32."""
    if temp == 0:
        return one_hot_argmax(logits)
    p = (logits.float() / temp).softmax(-1)
    return nucleus(p, top_p) if top_p < 1.0 else p


def nucleus_tie(p, top_p, K=0):
    """Tie-symmetric top-p nucleus: keep x iff p_x >= t, t = the smallest probability in the sorted-prefix nucleus (keep x iff the
    mass strictly above p_x is < top_p, plus its ties).  t is found among the K most likely tokens (K = 0: full sort); exact
    whenever the top-K mass reaches top_p, else the row keeps everything >= the K-th probability, flagged in `uncovered`.
    Returns (renormalised p, uncovered (N,) bool or None)."""
    sp = p.topk(min(K, p.shape[-1]), -1).values if K else p.sort(-1, descending=True).values
    cs = sp.cumsum(-1)
    m = ((cs - sp) < top_p).sum(-1, keepdim=True)                       # sorted-prefix nucleus size (>= 1)
    t = sp.gather(-1, m - 1)
    q = p * (p >= t)
    return q / q.sum(-1, keepdim=True), ((cs[..., -1] < top_p) if K else None)


def probs_cand(logits, temp, top_p, K):
    """The engine's target / proposal probabilities: probs() with the tie-symmetric nucleus searched among the top K tokens.
    Returns (p (N, V) float32, uncovered (N,) bool or None)."""
    if temp == 0:
        return one_hot_argmax(logits), None
    p = (logits.float() / temp).softmax(-1)
    if top_p >= 1.0:
        return p, None
    return nucleus_tie(p, top_p, K)


def sample(p):
    """One draw per row of the non-negative (possibly unnormalised) rows p (N, V) -> (N,) long, P(i) = p_i / sum(p):
    argmax_i p_i / E_i with E_i = -log(U_i) ~ Exp(1).  Zero entries score 0 and are never chosen."""
    e = -torch.log(torch.rand(p.shape, device=p.device, dtype=torch.float32).clamp_(min=1e-30))
    return (p / e).argmax(-1)


def verify(p, d, q=None, u=None):
    """Rejection sampling for one cycle (vectorised, no host sync).  p (n+1, V) target probabilities at the inputs [y, d_1..d_n]
    (row j = distribution of the token after input j); d (n,) drafts; q (n, V) proposals or None = one-hot on d (argmax drafts);
    u (n,) uniforms (default: torch.rand).  Returns (acc, e): number of accepted drafts (0-d long, 0..n) and the emitted token
    (0-d long): the residual draw at the first rejection, or the bonus draw from p[n]."""
    n = d.shape[0]
    pd = p[:n].gather(1, d[:, None])[:, 0]
    qd = torch.ones_like(pd) if q is None else q.gather(1, d[:, None])[:, 0]
    ratio = (pd / qd.clamp(min=1e-20)).clamp(max=1.0)
    if u is None:
        u = torch.rand(n, device=p.device, dtype=torch.float32)
    acc = (u < ratio).long().cumprod(0).sum()                            # accepted prefix length
    pa = p.index_select(0, acc[None])[0]                                 # p at the first rejection / the bonus row
    ia = acc.clamp(max=n - 1)[None]
    qa = torch.zeros_like(pa).scatter_(0, d.index_select(0, ia), 1.0) if q is None else q.index_select(0, ia)[0]
    qa = qa * (acc < n)                                                  # all accepted: plain p[n] (bonus)
    resid = (pa - qa).clamp(min=0); rs = resid.sum()
    dist = torch.where(rs > 0, resid / rs.clamp(min=1e-30), pa)
    return acc, sample(dist[None])[0]
