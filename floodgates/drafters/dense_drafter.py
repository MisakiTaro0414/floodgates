"""LAL's drafting pass on a DENSE model (the LayerSkip control): k+1 slots after the last token (slot 0 = the token just
emitted, slots 1..k = a learned mask embedding) run once through the FIRST E layers of the model, with LoRA (zero-init B) on the slot
rows of those layers, and are read out by the model's final norm and LM head (LayerSkip's early-exit head).  In every layer the slots
attend to the context's K/V of the SAME layer from the plain teacher-forced pass and bidirectionally to the slots of their anchor; a
dense model has no deeper cache for its shallow layers, so this is the dense counterpart of LAL with a shallow prefix cache.
Training as drafter_raven.py with shift=True: random anchors in the output region, slot j predicts the token at cut+j+1 of the model's
own greedy output, prefix-weighted CE exp(-j/4).  Offline evaluation: drafts at every output position of the model's own greedy
continuation of the test prompts, then decoding cycles simulated from the per-position accepted lengths (tokens per step tau).
  python dense_drafter.py train --model <hf dir> --data <jsonl> --out <dir> --E 2 [--k 12 --rank 64 --steps 16000 --batch 8 --anchors 16]
  python dense_drafter.py eval --run <dir> --prompts <jsonl with "context"> --task gsm8k|math500 [--n 100]"""
import argparse, json, math, os, random, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
CODE = ["<|endoftext|>", "<|endofmask|>", "</s>", "\nif __name__", "\ndef main(", "\nprint(", "\n# Test", "\nassert ", "\n```"]   # the stops of tools/stopcut.py
STOPS = {"gsm8k": ["Q:"], "math500": ["Problem:"], "humaneval": CODE, "mbpp": CODE, "bbh": ["</s>", "Q", "\n\n"]}
MAXNEW = {"gsm8k": 512, "math500": 1024, "humaneval": 1024, "mbpp": 1024, "bbh": 1024}


class LoRA(nn.Module):
    def __init__(self, din, dout, r):
        super().__init__()
        self.A = nn.Parameter(torch.randn(r, din) / math.sqrt(din)); self.B = nn.Parameter(torch.zeros(dout, r))

    def forward(self, x):
        return (x.float() @ self.A.t() @ self.B.t()).to(x.dtype)


def load(model_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16).cuda().eval().requires_grad_(False)
    return m, AutoTokenizer.from_pretrained(model_dir)


def layer_fwd(layer, x, cos, sin, attn_fn, lora=None):
    """One LlamaDecoderLayer with an explicit attention function (q, k, v after RoPE -> (B, H, T, D)) and optional LoRA deltas."""
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    at = layer.self_attn; B, T, _ = x.shape; D = at.head_dim
    lin = lambda name, mod, h: mod(h) + (lora[name](h) if lora is not None else 0)
    h = layer.input_layernorm(x)
    q = lin("q_proj", at.q_proj, h).view(B, T, -1, D).transpose(1, 2)
    k = lin("k_proj", at.k_proj, h).view(B, T, -1, D).transpose(1, 2)
    v = lin("v_proj", at.v_proj, h).view(B, T, -1, D).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    a = attn_fn(q, k, v).transpose(1, 2).reshape(B, T, -1)
    x = x + lin("o_proj", at.o_proj, a)
    h = layer.post_attention_layernorm(x); mlp = layer.mlp
    g = lin("gate_proj", mlp.gate_proj, h); u = lin("up_proj", mlp.up_proj, h)
    mm = F.silu(g) * u
    return x + lin("down_proj", mlp.down_proj, mm)


@torch.no_grad()
def context_kv(m, ids, E):
    """Teacher-forced causal pass of the first E layers over ids (B, T): per layer (k, v) (B, KV, T, D) after RoPE."""
    B, T = ids.shape; pos = torch.arange(T, device=ids.device)[None].expand(B, T)
    x = m.model.embed_tokens(ids); cos, sin = m.model.rotary_emb(x, pos); kv = []
    for layer in m.model.layers[:E]:
        def attn(q, k, v):
            kv.append((k, v)); return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        x = layer_fwd(layer, x, cos, sin, attn)
    return kv


class DenseDrafter(nn.Module):
    def __init__(self, m, E, k=12, rank=64):
        super().__init__()
        self.m, self.E, self.k = m, E, k
        c = m.config; d = c.hidden_size; f = c.intermediate_size; D = getattr(c, "head_dim", None) or d // c.num_attention_heads
        H, KV = c.num_attention_heads, c.num_key_value_heads
        dims = {"q_proj": (d, H * D), "k_proj": (d, KV * D), "v_proj": (d, KV * D), "o_proj": (H * D, d), "gate_proj": (d, f),
                "up_proj": (d, f), "down_proj": (f, d)}
        self.lora = nn.ModuleList([nn.ModuleDict({n: LoRA(*dims[n], rank) for n in TARGETS}) for _ in range(E)])
        self.mask_emb = nn.Parameter(torch.zeros(d)); self.V = c.vocab_size
        self.to(m.device)

    def forward(self, sid, is_clean, slot_pos, ctx_kv, mask):
        """sid (B, Q) slot input ids (slot 0 clean), is_clean (B, Q), slot_pos (B, Q), mask (B, 1, Q, T+Q) -> logits (B, Q, V)."""
        m = self.m; e = m.model.embed_tokens(sid)
        x = torch.where(is_clean[..., None], e, self.mask_emb.to(e.dtype).expand_as(e))
        cos, sin = m.model.rotary_emb(x, slot_pos)
        for i, layer in enumerate(m.model.layers[:self.E]):
            ck, cv = ctx_kv[i]
            def attn(q, k, v, ck=ck, cv=cv):
                return F.scaled_dot_product_attention(q, torch.cat([ck, k], 2), torch.cat([cv, v], 2), attn_mask=mask, enable_gqa=True)
            x = layer_fwd(layer, x, cos, sin, attn, self.lora[i])
        return m.lm_head(m.model.norm(x)).float()

    def batch(self, seqs, plens, A, gen, cuts=None):
        """Anchors (random in the output region, or the given cuts per sequence) -> tensors of drafter_raven.build_batch (shift)."""
        B = len(seqs); T = max(len(s) for s in seqs); k1 = self.k + 1
        ids = torch.zeros(B, T, dtype=torch.long)
        for b, s in enumerate(seqs): ids[b, :len(s)] = torch.tensor(s)
        if cuts is None:
            cuts = [[plens[b] + int(x) for x in torch.randint(0, max(len(seqs[b]) - plens[b] - 1, 1), (A,), generator=gen)] for b in range(B)]
        A = len(cuts[0]); sid = torch.zeros(B, A, k1, dtype=torch.long); tgt = torch.zeros(B, A, k1, dtype=torch.long)
        valid = torch.zeros(B, A, k1, dtype=torch.bool); cut = torch.tensor(cuts, dtype=torch.long)
        for b, s in enumerate(seqs):
            for a, c in enumerate(cuts[b]):
                sid[b, a, 0] = s[c]
                for j in range(k1):
                    if c + j + 1 < len(s): tgt[b, a, j] = s[c + j + 1]; valid[b, a, j] = True
        return ids, sid, tgt, valid, cut

    def run(self, ids, sid, cut):
        dev = self.m.device; B, T = ids.shape; A, k1 = sid.shape[1], sid.shape[2]; Q = A * k1
        ctx = context_kv(self.m, ids.to(dev), self.E)
        cutd = cut.to(dev)
        c = (torch.arange(T, device=dev)[None, None, :] < cutd[:, :, None])[:, :, None, :].expand(B, A, k1, T).reshape(B, Q, T)
        own = torch.block_diag(*[torch.ones(k1, k1, dtype=torch.bool, device=dev)] * A)[None].expand(B, Q, Q)
        mask = torch.cat([c, own], -1)[:, None]
        is_clean = torch.zeros(B, A, k1, dtype=torch.bool, device=dev); is_clean[:, :, 0] = True
        slot_pos = (cutd[:, :, None] + torch.arange(k1, device=dev)[None, None]).view(B, Q)
        sin_ = sid.to(dev).clone(); sin_[:, :, 1:] = 0
        return self(sin_.view(B, Q), is_clean.view(B, Q), slot_pos, ctx, mask).view(B, A, k1, -1)


def train(a):
    m, tok = load(a.model); dr = DenseDrafter(m, a.E, a.k, a.rank)
    rows = [json.loads(l) for l in open(a.data)]; os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(f"{a.out}/args.json", "w"), indent=1)
    opt = torch.optim.AdamW([p for p in dr.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * max(0.0, 1 - s / a.steps))
    gen = torch.Generator().manual_seed(0); rng = random.Random(0); k1 = a.k + 1
    w = torch.exp(-torch.arange(k1).float() / 4.0).cuda()
    log = open(f"{a.out}/train_log.jsonl", "a"); t0 = time.time()
    for step in range(a.steps):
        bt = rng.sample(rows, a.batch)
        seqs = [r["prompt_ids"] + r["output_ids"] for r in bt]; plens = [len(r["prompt_ids"]) for r in bt]
        ids, sid, tgt, valid, cut = dr.batch(seqs, plens, a.anchors, gen)
        logits = dr.run(ids, sid, cut)
        ce = F.cross_entropy(logits.reshape(-1, dr.V), tgt.cuda().view(-1), reduction="none").view(tgt.shape)
        ww = w[None, None] * valid.cuda().float()
        loss = (ce * ww).sum() / ww.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(dr.parameters(), 1.0); opt.step(); sched.step()
        if step % 25 == 0:
            with torch.no_grad():
                hit = ((logits.argmax(-1) == tgt.cuda()) & valid.cuda()).float().sum((0, 1)) / valid.cuda().float().sum((0, 1)).clamp(min=1)
            rec = {"step": step, "loss": float(loss), "slot_acc": [round(float(x), 4) for x in hit], "s": round(time.time() - t0)}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
    torch.save({k: v for k, v in dr.state_dict().items() if not k.startswith("m.")}, f"{a.out}/drafter.pt")


@torch.no_grad()
def evaluate(a):
    ar = json.load(open(f"{a.run}/args.json")); m, tok = load(ar["model"]); dr = DenseDrafter(m, ar["E"], ar["k"], ar["rank"])
    dr.load_state_dict(torch.load(f"{a.run}/drafter.pt"), strict=False); dr.eval()
    rows = [json.loads(l) for l in open(a.prompts)][:a.n]; tot_tok = tot_cyc = 0; per = []
    for r in rows:
        ids = tok(r["context"], return_tensors="pt").input_ids.cuda()
        out = m.generate(ids, max_new_tokens=MAXNEW[a.task], do_sample=False, stop_strings=STOPS[a.task], tokenizer=tok)[0].tolist()
        P = ids.shape[1]; s = out; n = len(s) - P
        if n < 2: continue
        cuts = list(range(P, len(s) - 1)); acc = []
        for i in range(0, len(cuts), 64):                       # drafts at every output position, teacher-forced on the trace
            ids_, sid, tgt, valid, cut = dr.batch([s], [P], None, None, cuts=[cuts[i:i + 64]])
            lg = dr.run(ids_, sid, cut)[0]; ok = ((lg.argmax(-1) == tgt[0].cuda()) & valid[0].cuda())
            for row in ok:                                          # slot j drafts the token at cut+j+1 (k+1 drafts, as the engine's shift)
                a_ = 0
                for j in range(row.shape[0]):
                    if not row[j]: break
                    a_ += 1
                acc.append(a_)
        # a cycle at cut accepts the leading correct drafts and the verifier adds one token: it emits acc+1 and moves cut by acc+1
        mpos, cyc = 0, 0
        while mpos < len(acc):
            mpos += 1 + acc[mpos]; cyc += 1
        tot_tok += len(acc); tot_cyc += cyc; per.append({"n": len(acc), "cycles": cyc})
    res = {"run": a.run, "task": a.task, "n": len(per), "tau": tot_tok / max(tot_cyc, 1), "E": ar["E"], "layers": m.config.num_hidden_layers}
    json.dump(res, open(f"{a.run}/accept_{a.task}.json", "w"), indent=1); print(json.dumps(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--model"); ap.add_argument("--data"); ap.add_argument("--out"); ap.add_argument("--E", type=int, default=2)
    ap.add_argument("--k", type=int, default=12); ap.add_argument("--rank", type=int, default=64); ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--anchors", type=int, default=16); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--run"); ap.add_argument("--prompts"); ap.add_argument("--task", default="gsm8k"); ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    train(a) if a.mode == "train" else evaluate(a)
