"""External drafter (CONTROL for the self-drafter of drafter.py): a separate small transformer that does NOT reuse Ouro's
looped block, trained on the same anchors with the same prefix-weighted CE and evaluated with the same offline acceptance
protocol -- "is the target's own loop a better drafter than a separate network of comparable size?" (DFlash/EAGLE-style).

Inputs.  Context position i (< cut): E[y_i] + W_f h_i, with h_i = Ouro's normalized FINAL-loop hidden state at i (frozen,
computed without grad; the input of Ouro's lm_head) and W_f a trainable Linear(d, d) (zero-init).  Canvas: slot 0 = E[y_m]
(the just-emitted token), slots 1..k = a learned mask embedding (as the mask variant of LoopFlowDrafter).
Attention.  Context positions causal; the slots of an anchor see the context < cut and each other (bidirectional): the
same mask as LoopFlowDrafter.masks.  The drafter computes its OWN per-layer K/V of the context (it never reads Ouro's).
Trunk.  n_layers fresh ouro_min.Layer of Ouro's width; init "copy" = Ouro's first n_layers layers, "random" = N(0, 0.02).
Readout.  Trainable copy of Ouro's final norm + Ouro's FROZEN lm_head (tied readout keeps the comparison about the trunk).
Trainable: layers + W_f + norm + mask embedding (55.6M for 1 layer, 209.8M for 4; fp32 master weights, bf16 autocast).
Inference.  The context K/V are computed once per sequence (teacher-forced, causal) by prepare(ids) and sliced to < cut
per cycle, exactly as eval_accept.py reuses Ouro's full-sequence K/V for the self-drafter."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from floodgates.models.ouro_min import Layer, RMSNorm
from floodgates.drafters.drafter import LoopFlowDrafter

build_batch = LoopFlowDrafter.build_batch     # same anchors (uses only self.k)
masks = LoopFlowDrafter.masks                 # same attention-mask semantics (pure function of cut, T, A, k1)


class ExtDrafter(nn.Module):
    def __init__(self, ouro, k=8, n_layers=1, init="random", d_model=None):
        super().__init__()
        assert init in ("random", "copy"), init
        self.ouro, self.k, self.n_layers, self.init_mode = ouro, k, n_layers, init
        self.shift = False      # mask-filling slots, as the Ouro drafter (build_batch reads it)
        c = ouro.c
        d, V = c["hidden_size"], c["vocab_size"]
        dm = d if d_model is None else d_model
        self.d, self.dm, self.V = d, dm, V
        cfg = dict(c)
        if dm != d:   # narrower trunk: projections in/out of Ouro's width (head_dim kept, so ouro.rope applies)
            assert init == "random", "init='copy' needs d_model == Ouro's hidden size"
            cfg.update(hidden_size=dm, num_attention_heads=dm // c["head_dim"],
                       intermediate_size=max(64, round(c["intermediate_size"] * dm / d / 64) * 64))
        self.layers = nn.ModuleList([Layer(cfg) for _ in range(n_layers)])
        self.in_proj = nn.Linear(d, dm, bias=False) if dm != d else nn.Identity()
        self.out_proj = nn.Linear(dm, d, bias=False) if dm != d else nn.Identity()
        self.feat_proj = nn.Linear(d, dm, bias=False)    # W_f: zero-init => at init the trunk sees token embeddings only
        nn.init.zeros_(self.feat_proj.weight)
        self.mask_emb = nn.Parameter(torch.zeros(dm))
        self.norm = RMSNorm(d, c["rms_norm_eps"])        # trainable copy of Ouro's final norm (both init modes)
        with torch.no_grad():
            self.norm.weight.copy_(ouro.norm.weight)
            if init == "copy":
                for mine, theirs in zip(self.layers, ouro.layers):
                    mine.load_state_dict(theirs.state_dict())            # bf16 -> fp32 master copy
            else:
                for m in self.layers.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, std=c.get("initializer_range", 0.02))
        self.to(ouro.device)                             # fp32 params on the GPU; the forward runs under bf16 autocast

    def n_trainable(self):
        return sum(p.numel() for n, p in self.named_parameters() if not n.startswith("ouro."))

    # ---- context: the drafter's own causal pass -----------------------------------------------------------------
    def context_kv(self, ids, hidden):
        """ids (B, T) long; hidden (B, T, d) Ouro's normalized final-loop hidden states of the same (teacher-forced) ids.
        Returns the drafter's per-layer (k, v) (B, H, T, D) of the context (causal; RoPE at positions 0..T-1)."""
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)[None].expand(B, T)
        cos, sin = self.ouro.rope(pos)
        causal = lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = self.in_proj(self.ouro.embed_tokens(ids)) + self.feat_proj(hidden)
            kv = []
            for layer in self.layers:
                x, k, v = layer(x, cos, sin, causal)
                kv.append((k, v))
        return kv

    @torch.no_grad()
    def prepare(self, ids):
        """Offline evaluation: Ouro's final-loop hidden states + this drafter's context K/V for a full sequence (B, T),
        computed once; draft() then slices positions < cut (the per-sequence cache of eval_accept.py)."""
        _, _, hidden = self.ouro.forward_context(ids, keep_kv_loop=-1, logits=False, return_hidden=True)
        return self.context_kv(ids, hidden)

    # ---- slots: one pass through the trunk, reading the context K/V --------------------------------------------
    def slot_inputs(self, sid, is_clean):
        """sid (N, S) long (slot 0 = y_m), is_clean (N, S) bool: E[y] on clean slots, the mask embedding elsewhere."""
        e = self.in_proj(self.ouro.embed_tokens(sid))
        return torch.where(is_clean[..., None], e, self.mask_emb.to(e.dtype).expand_as(e))

    def slots(self, e, slot_pos, ctx_kv, mask):
        """e (B, Q, dm) slot inputs; slot_pos (B, Q) positions; ctx_kv per layer (k, v) (B, H, T, D) from context_kv;
        mask (B, 1, Q, T+Q) bool (True = may attend) or None (= all).  Returns logits (B, Q, V) float."""
        cos, sin = self.ouro.rope(slot_pos)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = e
            for li, layer in enumerate(self.layers):
                ck, cv = ctx_kv[li]
                def attn(q, k, v, ck=ck, cv=cv):
                    return F.scaled_dot_product_attention(q, torch.cat([ck, k], 2), torch.cat([cv, v], 2), attn_mask=mask)
                x, _, _ = layer(x, cos, sin, attn)
            h = self.norm(self.out_proj(x))
            return self.ouro.lm_head(h.to(self.ouro.dtype)).float()

    # ---- training: LoopFlowDrafter.loss with one pass and no noise/time/state --------------------------------------
    def loss(self, seqs, prompt_lens, anchors_per_seq=16, rounds_max=None, gen=None, prefix_decay=4.0):
        """Same anchors (build_batch) and prefix-weighted CE (exp(-(j-1)/4) over slots 1..k) as LoopFlowDrafter.loss;
        a single pass, so rounds_max is ignored.  Ouro's hidden states are computed without grad; the drafter's context
        pass and slot pass are both trained."""
        dev = self.ouro.device; k1 = self.k + 1; A = anchors_per_seq
        ids, slot_ids, slot_valid, cut = build_batch(self, seqs, prompt_lens, A, gen)
        ids, slot_ids, slot_valid, cut = ids.to(dev), slot_ids.to(dev), slot_valid.to(dev), cut.to(dev)
        B, T = ids.shape; Q = A * k1
        _, _, hidden = self.ouro.forward_context(ids, keep_kv_loop=-1, logits=False, return_hidden=True)
        ctx_kv = self.context_kv(ids, hidden)
        mask = masks(self, cut, T, A, k1, dev)
        sid = slot_ids.view(B, Q); valid = slot_valid.view(B, Q)
        slot_pos = (cut[:, :, None] + torch.arange(k1, device=dev)[None, None]).view(B, Q)
        is_clean = torch.zeros(B, A, k1, dtype=torch.bool, device=dev); is_clean[:, :, 0] = True
        is_clean = is_clean.view(B, Q)
        logits = self.slots(self.slot_inputs(sid, is_clean), slot_pos, ctx_kv, mask)
        w = torch.exp(-(torch.arange(k1, device=dev).float() - 1).clamp(min=0) / prefix_decay)
        w = w[None, None].expand(B, A, k1).reshape(B, Q) * valid.float() * (~is_clean).float()
        ce = F.cross_entropy(logits.view(-1, self.V), sid.view(-1), reduction="none").view(B, Q)
        loss = (ce * w).sum() / w.sum().clamp(min=1)
        with torch.no_grad():
            acc = ((logits.argmax(-1) == sid).float() * (w > 0)).sum() / (w > 0).sum().clamp(min=1)
        return loss, {"acc_r0": float(acc), "rounds": 1}

    # ---- inference --------------------------------------------------------------------------------------------
    @torch.no_grad()
    def draft(self, ctx_kv, cut, y_m, n_rounds=1, gen=None):
        """Single sequence.  ctx_kv = self.prepare(ids) over the full sequence; positions < cut are used (sliced).
        One pass (n_rounds and gen are ignored).  Returns the k drafted ids (list)."""
        dev = self.ouro.device; k1 = self.k + 1
        kv = [(k[:, :, :cut], v[:, :, :cut]) for k, v in ctx_kv]
        sid = torch.zeros(1, k1, dtype=torch.long, device=dev); sid[0, 0] = y_m
        is_clean = torch.zeros(1, k1, dtype=torch.bool, device=dev); is_clean[0, 0] = True
        slot_pos = (cut + torch.arange(k1, device=dev))[None]
        logits = self.slots(self.slot_inputs(sid, is_clean), slot_pos, kv, None)
        return logits[0, 1:].argmax(-1).tolist()
