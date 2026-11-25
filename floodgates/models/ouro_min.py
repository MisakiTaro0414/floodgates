"""Minimal Ouro-1.4B (ByteDance) in plain PyTorch, matching modeling_ouro.py: 24 sandwich-norm decoder layers looped
total_ut_steps=4 times, the model norm applied after EVERY loop (and fed to the next loop), untied lm_head on the last loop
(LoopSpec serves fixed depth R=4 and never evaluates the early-exit gate).  Exposes the last loop's per-layer K/V so draft
slots can attend to the verified context (see drafter.py)."""
import json, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dt)


def rope_cos_sin(positions, head_dim, theta, dtype):
    """positions: (..., T) long -> cos, sin (..., T, head_dim) in fp32 then cast, as OuroRotaryEmbedding."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32) / head_dim))
    f = positions.float()[..., None] * inv
    emb = torch.cat([f, f], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(x, cos, sin):
    """x (B, H, T, D); cos/sin (B, T, D)."""
    return x * cos[:, None] + rotate_half(x) * sin[:, None]


class Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        d, f, eps = c["hidden_size"], c["intermediate_size"], c["rms_norm_eps"]
        self.H, self.D = c["num_attention_heads"], c["head_dim"]
        self.q_proj = nn.Linear(d, self.H * self.D, bias=False)
        self.k_proj = nn.Linear(d, self.H * self.D, bias=False)
        self.v_proj = nn.Linear(d, self.H * self.D, bias=False)
        self.o_proj = nn.Linear(self.H * self.D, d, bias=False)
        self.gate_proj = nn.Linear(d, f, bias=False)
        self.up_proj = nn.Linear(d, f, bias=False)
        self.down_proj = nn.Linear(f, d, bias=False)
        self.input_layernorm = RMSNorm(d, eps)
        self.input_layernorm_2 = RMSNorm(d, eps)
        self.post_attention_layernorm = RMSNorm(d, eps)
        self.post_attention_layernorm_2 = RMSNorm(d, eps)

    def qkv(self, h, cos, sin, lora=None):
        B, T, _ = h.shape
        def proj(name):
            y = getattr(self, name)(h)
            if lora is not None and name in lora:
                y = y + lora[name](h)
            return y.view(B, T, self.H, self.D).transpose(1, 2)
        q, k, v = proj("q_proj"), proj("k_proj"), proj("v_proj")
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

    def out(self, name, x, lora=None):
        y = getattr(self, name)(x)
        if lora is not None and name in lora:
            y = y + lora[name](x)
        return y

    def mlp(self, h, lora=None):
        return self.out("down_proj", F.silu(self.out("gate_proj", h, lora)) * self.out("up_proj", h, lora), lora)

    def forward(self, x, cos, sin, attn_fn, lora=None):
        """attn_fn(q, k, v) -> (B, H, T, D) with any cache / mask handling; returns (x_out, k, v)."""
        h = self.input_layernorm(x)
        q, k, v = self.qkv(h, cos, sin, lora)
        a = attn_fn(q, k, v)
        a = a.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        x = x + self.input_layernorm_2(self.out("o_proj", a, lora))
        x = x + self.post_attention_layernorm_2(self.mlp(self.post_attention_layernorm(x), lora))
        return x, k, v


class OuroMin(nn.Module):
    def __init__(self, model_dir, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        c = json.load(open(os.path.join(model_dir, "config.json")))
        self.c, self.R, self.L = c, c["total_ut_steps"], c["num_hidden_layers"]
        d, V = c["hidden_size"], c["vocab_size"]
        self.embed_tokens = nn.Embedding(V, d)
        self.layers = nn.ModuleList([Layer(c) for _ in range(self.L)])
        self.norm = RMSNorm(d, c["rms_norm_eps"])
        self.lm_head = nn.Linear(d, V, bias=False)
        sd = load_file(os.path.join(model_dir, "model.safetensors"))
        own = {}
        for k, v in sd.items():
            if k.startswith("model.early_exit_gate"):
                continue
            k = k.replace("model.", "", 1) if k.startswith("model.") else k
            own[k.replace(".self_attn.", ".").replace(".mlp.", ".")] = v
        missing, unexpected = self.load_state_dict(own, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        self.to(device=device, dtype=dtype).eval().requires_grad_(False)
        self.device, self.dtype = device, dtype
        self.theta, self.D = c["rope_theta"], c["head_dim"]

    def rope(self, positions):
        return rope_cos_sin(positions, self.D, self.theta, self.dtype)

    @torch.no_grad()
    def forward_context(self, ids, keep_kv_loop=None, logits=True, return_hidden=False):
        """Teacher-forced pass over ids (B, T), R loops, causal. Returns (logits of the last loop (B, T, V),
        kv: list over layers of (k, v) (B, H, T, D) from loop keep_kv_loop (default: last loop; -1 = keep none)) and,
        with return_hidden, also the normalized last-loop hidden states (B, T, d) (the input of lm_head)."""
        keep = self.R - 1 if keep_kv_loop is None else keep_kv_loop
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)[None].expand(B, T)
        cos, sin = self.rope(pos)
        x = self.embed_tokens(ids)
        kv = [None] * self.L
        causal = lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)
        for ut in range(self.R):
            for li, layer in enumerate(self.layers):
                x, k, v = layer(x, cos, sin, causal)
                if ut == keep:
                    kv[li] = (k, v)
            x = self.norm(x)
        out = (self.lm_head(x).float() if logits else None), kv
        return out + (x,) if return_hidden else out

    @torch.no_grad()
    def greedy(self, prompt_ids, max_new, stop_ids=()):
        """Plain greedy decoding with a per-(loop, layer) KV cache, batch 1. Returns the generated ids (list)."""
        ids = torch.tensor([prompt_ids], device=self.device)
        cache = [[None] * self.L for _ in range(self.R)]
        out = []
        x_ids, start = ids, 0
        for step in range(max_new):
            T = x_ids.shape[1]
            pos = torch.arange(start, start + T, device=self.device)[None]
            cos, sin = self.rope(pos)
            x = self.embed_tokens(x_ids)
            for ut in range(self.R):
                for li, layer in enumerate(self.layers):
                    def attn(q, k, v, ut=ut, li=li):
                        if cache[ut][li] is not None:
                            pk, pv = cache[ut][li]
                            k, v = torch.cat([pk, k], 2), torch.cat([pv, v], 2)
                        cache[ut][li] = (k, v)
                        return F.scaled_dot_product_attention(q, k, v, is_causal=(start == 0))
                    x, _, _ = layer(x, cos, sin, attn)
                x = self.norm(x)
            nxt = int(self.lm_head(x[:, -1]).float().argmax(-1))
            out.append(nxt)
            if nxt in stop_ids:
                break
            x_ids, start = torch.tensor([[nxt]], device=self.device), start + T
        return out
