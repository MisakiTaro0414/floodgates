"""Future-token probes (Future Lens style, Pal et al. 2023): how well does a linear reader of a model's state at a given depth
predict the token j positions ahead, without the tokens in between?  Loops of looped models (Recurrent-Llama, Ouro) against
layers of dense models (Llama-3.2-1B, LayerSkip-Llama-3.2-1B).

For each depth, K maps W_j (d x d, initialized to the identity) are trained on frozen states, and token t+j is read out with the
model's own frozen final norm and head: logits_j = head(norm(W_j h_t + b_j)) (Ouro: no extra norm, its loop states are already
normalized).  W_j = I at the final depth reproduces the model's own next-token prediction (logit lens).  Score: top-1 accuracy
of token t+j on held-out sequences (split by sequence).

Texts: "own" = the model's own greedy outputs (data/mix4c_<model>_greedy.jsonl, positions in the output), what its drafter must
predict; "math" = human-written MATH solutions (EleutherAI/hendrycks_math train), the same text for every model.
  python probe_future.py --model rl32|llama|layerskip|ouro14 --text own|math --out runs/probe/<name>.json"""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import argparse, json, os, random, time
import torch, torch.nn as nn, torch.nn.functional as F
import os
from floodgates.paths import LOOPSPEC, PARENT

S = PARENT
M = {"rl32": f"{LOOPSPEC}/models/Recurrent-Llama-3.2-train-recurrence-32",
     "ouro14": f"{LOOPSPEC}/models/Ouro-1.4B", "ouro26": f"{LOOPSPEC}/models/Ouro-2.6B",
     "huginn": f"{LOOPSPEC}/models/huginn-0125", "tl32": f"{LOOPSPEC}/models/Recurrent-TinyLlama-3T-train-recurrence-32",
     "llama": "meta-llama/Llama-3.2-1B", "layerskip": "facebook/layerskip-llama3.2-1B"}          # gated Hugging Face repositories
OWN = {"rl32": "data/mix4c_rl32_greedy.jsonl", "ouro14": "data/mix4c_ouro14_greedy.jsonl", "ouro26": "data/mix4c_ouro26_greedy.jsonl",
       "huginn": "data/mix4c_huginn_greedy.jsonl", "tl32": "data/mix4c_tl32_greedy.jsonl",
       "llama": "data/mix4c_dense_llama_greedy.jsonl", "layerskip": "data/mix4c_dense_layerskip_greedy.jsonl"}
MATH_CFGS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus")


class Extractor:
    """states(ids (1, T)) -> {depth name: (T, d) tensor}; plus the frozen readout (norm or None, head)."""

    def __init__(self, name):
        self.name = name
        if name in ("rl32", "huginn", "tl32"):
            from floodgates.models.raven_min import RavenMin
            self.m = RavenMin(M[name]); self.norm, self.head = self.m.ln_f, self.m.lm_head
            self.depths = ["embed", "prelude"] + [f"loop{i}" for i in (1, 2, 4, 8, 16, 32)] + ["coda"]
        elif name in ("ouro14", "ouro26"):
            from floodgates.models.ouro_min import OuroMin
            self.m = OuroMin(M[name]); self.norm, self.head = None, self.m.lm_head
            self.depths = ["embed"] + [f"loop{i}" for i in range(1, self.m.R + 1)]
        else:
            from transformers import AutoModelForCausalLM
            self.m = AutoModelForCausalLM.from_pretrained(M[name], torch_dtype=torch.bfloat16).cuda().eval().requires_grad_(False)
            self.norm, self.head = self.m.model.norm, self.m.lm_head
            self.keep = (1, 2, 4, 8, 12, 16)
            self.depths = ["embed"] + [f"layer{i}" for i in self.keep]
            self._buf = {}
            for i in self.keep:
                self.m.model.layers[i - 1].register_forward_hook(
                    lambda mod, inp, out, i=i: self._buf.__setitem__(f"layer{i}", (out[0] if isinstance(out, tuple) else out)[0]))

    @torch.no_grad()
    def states(self, ids):
        out = {}
        if self.name in ("rl32", "huginn", "tl32"):
            from floodgates.models.raven_min import causal_attn
            m = self.m; T = ids.shape[1]
            cos, sin = m.rope(torch.arange(T, device=ids.device)[None])
            out["embed"] = (m.wte(ids) * m.emb_scale)[0]
            e = m.embed(ids, cos, sin); out["prelude"] = e[0]
            s = m.init_state(e.shape, torch.Generator(device=ids.device).manual_seed(0))
            for i in range(1, m.R + 1):
                s = m.core_step(s, e, cos, sin)
                if f"loop{i}" in self.depths:
                    out[f"loop{i}"] = s[0]
            c = m.pre(s)
            for b in m.coda:
                c = b(c, cos, sin, causal_attn)
            out["coda"] = c[0]
        elif self.name in ("ouro14", "ouro26"):
            m = self.m; T = ids.shape[1]
            cos, sin = m.rope(torch.arange(T, device=ids.device)[None])
            x = m.embed_tokens(ids); out["embed"] = x[0]
            causal = lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)
            for ut in range(m.R):
                for layer in m.layers:
                    x, _, _ = layer(x, cos, sin, causal)
                x = m.norm(x); out[f"loop{ut + 1}"] = x[0]
        else:
            self._buf.clear()
            out["embed"] = self.m.model.embed_tokens(ids)[0]
            self.m(input_ids=ids, use_cache=False)
            out.update(self._buf)
        return out

    def readout(self, h):
        return self.head(self.norm(h) if self.norm is not None else h)


def sequences(model, text, n, seed=0):
    """List of (ids, first position whose next token is in the region of interest)."""
    if text == "own":
        rows = [json.loads(l) for l in open(OWN[model])]
        random.Random(seed).shuffle(rows)
        return [(r["prompt_ids"] + r["output_ids"], len(r["prompt_ids"]) - 1) for r in rows[:n] if len(r["output_ids"]) > 16]
    import datasets
    from floodgates.tokenizer import load_tokenizer
    src = M["llama" if model == "rl32" else model]                 # Recurrent-Llama keeps Llama-3.2's tokenizer (same vocab and merges)
    try: tok = load_tokenizer(src)
    except Exception:                                               # retrofitted models with a custom config: the tokenizer file alone
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(src, "tokenizer.json"))
    first = json.loads(open(OWN[model]).readline())["prompt_ids"][0]
    bos = [first] if first == tok.bos_token_id or first == 128000 else []   # a BOS token exactly when the model's own prompts start with one
    rows = [r for c in MATH_CFGS for r in datasets.load_dataset("EleutherAI/hendrycks_math", c, split="train")]
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows[:n]:
        p = bos + tok(f"Problem: {r['problem']}\nSolution:", add_special_tokens=False)["input_ids"]
        s = tok(" " + r["solution"], add_special_tokens=False)["input_ids"]
        out.append((p + s, len(p) - 1))
    return out


TASKS = ("gsm8k", "math500", "humaneval", "mbpp", "bbh")


def task_texts(model, n_train=2000, n_test=500, seed=0):
    """Human-written text of the five benchmarks, the same for every model of a tokenizer.  Train: GSM8K train solutions, MATH train
    solutions, MBPP programs (train/validation/prompt splits, without the MBPP+ problems) and CodeSearchNet Python functions, and BBH
    questions beyond the first 20 of each subtask.  Test: GSM8K test solutions, MATH-500 solutions, HumanEval+ and MBPP+ canonical
    solutions, and the first 20 questions of each BBH subtask (as the evaluation).  Positions: in the answer, solution or code; BBH: all."""
    import datasets
    from floodgates.tokenizer import load_tokenizer
    src = M["llama" if model == "rl32" else model]
    try: tok = load_tokenizer(src)
    except Exception:
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(src, "tokenizer.json"))
    first = json.loads(open(OWN[model]).readline())["prompt_ids"][0]
    bos = [first] if first == tok.bos_token_id or first == 128000 else []
    enc = lambda a, b: (bos + tok(a, add_special_tokens=False)["input_ids"], tok(b, add_special_tokens=False)["input_ids"])
    def mk(a, b, whole=False):
        x, y = enc(a, b); return (x + y, len(bos) if whole else len(x) - 1)
    g = random.Random(seed); tr, te = {}, {}
    L = lambda *a, **k: datasets.load_dataset(*a, **k)
    tr["gsm8k"] = [mk(f"Question: {r['question']}\nAnswer:", " " + r["answer"]) for r in L("openai/gsm8k", "main", split="train")]
    te["gsm8k"] = [mk(f"Question: {r['question']}\nAnswer:", " " + r["answer"]) for r in L("openai/gsm8k", "main", split="test")]
    tr["math500"] = [mk(f"Problem: {r['problem']}\nSolution:", " " + r["solution"]) for c in MATH_CFGS for r in L("EleutherAI/hendrycks_math", c, split="train")]
    te["math500"] = [mk(f"Problem: {r['problem']}\nSolution:", " " + r["solution"]) for r in L("HuggingFaceH4/MATH-500", split="test")]
    plus = L("evalplus/mbppplus", split="test"); plus_ids = {int(str(r["task_id"]).split("/")[-1]) for r in plus}
    code = [mk(f'"""\n{r["text"]}\n"""\n', r["code"]) for sp in ("train", "validation", "prompt") for r in L("google-research-datasets/mbpp", "full", split=sp) if r["task_id"] not in plus_ids]
    try:
        csn = L("code-search-net/code_search_net", "python", split="train")
        code += [mk("", r["whole_func_string"], whole=True) for r in csn.select(range(min(3000, len(csn))))]
    except Exception as e:
        print("CodeSearchNet unavailable:", str(e)[:120], flush=True)
    tr["code"] = code
    te["humaneval"] = [mk(r["prompt"], r["canonical_solution"]) for r in L("evalplus/humanevalplus", split="test")]
    te["mbpp"] = [mk(f'"""\n{r["prompt"]}\n"""\n', r["code"]) for r in plus]
    bbh_tr, bbh_te = [], []
    for c in ("boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa", "dyck_languages", "formal_fallacies",
              "geometric_shapes", "hyperbaton", "logical_deduction_five_objects", "logical_deduction_seven_objects", "logical_deduction_three_objects",
              "movie_recommendation", "multistep_arithmetic_two", "navigate", "object_counting", "penguins_in_a_table",
              "reasoning_about_colored_objects", "ruin_names", "salient_translation_error_detection", "snarks", "sports_understanding",
              "temporal_sequences", "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects",
              "tracking_shuffled_objects_three_objects", "web_of_lies", "word_sorting"):
        rows = [mk(f"Q: {r['input']}\nA:", " " + r["target"], whole=True) for r in L("SaylorTwift/bbh", c, split="test")]
        bbh_te += rows[:20]; bbh_tr += rows[20:]
    tr["bbh"], te["bbh"] = bbh_tr, bbh_te
    train = []
    for k, v in tr.items():
        g.shuffle(v); train += v[:n_train]
    g.shuffle(train)
    test = {k: v[:n_test] if k != "bbh" else v for k, v in te.items()}
    print({k: len(v) for k, v in tr.items()}, {k: len(v) for k, v in test.items()}, flush=True)
    return train, test


def collect(ex, seqs, K, max_pos, max_len, seed):
    """Stack states at sampled positions t (depth -> (N, d) fp16 on CPU) and targets ids[t+1..t+K] (N, K)."""
    g = random.Random(seed); feats = {d: [] for d in ex.depths}; tgt = []
    for ids, start in seqs:
        ids = ids[:max_len]
        pos = list(range(start, len(ids) - K))
        if not pos:
            continue
        pos = sorted(g.sample(pos, min(max_pos, len(pos))))
        st = ex.states(torch.tensor([ids], device="cuda"))
        p = torch.tensor(pos, device="cuda")
        for d in ex.depths:
            feats[d].append(st[d][p].half().cpu())
        tgt.append(torch.tensor([[ids[t + j] for j in range(1, K + 1)] for t in pos]))
    return {d: torch.cat(v) for d, v in feats.items()}, torch.cat(tgt)


@torch.no_grad()
def evaluate(ex, D, b, X, Y, K):
    """Per-j top-1 accuracy and CE of the maps I + D[j], b[j] on (X, Y)."""
    I = torch.eye(X.shape[1], device="cuda"); acc, ce = [], []
    for j in range(K):
        c = l = 0
        for s in range(0, X.shape[0], 1024):
            h = X[s:s + 1024].cuda().float(); y = Y[s:s + 1024, j].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = ex.readout(h @ (I + D[j]).t() + b[j]).float()
            c += (logits.argmax(-1) == y).sum().item(); l += F.cross_entropy(logits, y, reduction="sum").item()
        acc.append(c / X.shape[0]); ce.append(l / X.shape[0])
    return acc, ce


def fit(ex, Xtr, Ytr, K, epochs, bs, lr, val_frac=0.1):
    """K identity-initialized affine maps on frozen states, early-stopped per j on a validation split (the last val_frac of the
    training positions, which come from the last training sequences): the identity map (logit lens) is evaluated first and a map
    is kept only if it beats the best so far.  Returns per-j test accuracy and CE of the kept maps, and the epoch kept per j."""
    d = Xtr.shape[1]; nv = int(Xtr.shape[0] * val_frac)
    Xv, Yv, Xtr, Ytr = Xtr[-nv:], Ytr[-nv:], Xtr[:-nv], Ytr[:-nv]
    D = nn.Parameter(torch.zeros(K, d, d, device="cuda")); b = nn.Parameter(torch.zeros(K, d, device="cuda"))
    opt = torch.optim.AdamW([D, b], lr=lr, weight_decay=0.01)
    I = torch.eye(d, device="cuda")
    n = Xtr.shape[0]; per = n // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(epochs * per, 1), pct_start=0.05)
    best = evaluate(ex, D, b, Xv, Yv, K)[0]; keep = [0] * K
    bestD, bestb = D.detach().clone(), b.detach().clone()
    for ep in range(1, epochs + 1):
        for _ in range(per):
            idx = torch.randint(0, n, (bs,))
            h = Xtr[idx].cuda().float(); y = Ytr[idx].cuda()
            opt.zero_grad(set_to_none=True)
            for j in range(K):     # independent maps: per-j backward gives exact gradients with 1/K of the logits memory
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = ex.readout(h @ (I + D[j]).t() + b[j])
                F.cross_entropy(logits.float(), y[:, j]).backward()
            opt.step(); sched.step()
        va = evaluate(ex, D, b, Xv, Yv, K)[0]
        for j in range(K):
            if va[j] > best[j]:
                best[j], keep[j] = va[j], ep; bestD[j].copy_(D[j]); bestb[j].copy_(b[j])
    return bestD, bestb, keep


def train_probe(ex, Xtr, Ytr, Xte, Yte, K, epochs, bs, lr, val_frac=0.1):
    bestD, bestb, keep = fit(ex, Xtr, Ytr, K, epochs, bs, lr, val_frac)
    acc, ce = evaluate(ex, bestD, bestb, Xte, Yte, K)
    return acc, ce, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(M)); ap.add_argument("--text", default="own", choices=("own", "math", "tasks"))
    ap.add_argument("--out", required=True); ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=1600); ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--max-pos", type=int, default=128); ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--bs", type=int, default=512); ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    torch.manual_seed(0); t0 = time.time()
    ex = Extractor(a.model)
    if a.text == "tasks":
        train, test = task_texts(a.model)
        Xtr, Ytr = collect(ex, train, a.k, a.max_pos, a.max_len, 1)
        T = {t: collect(ex, v, a.k, a.max_pos, a.max_len, 2) for t, v in test.items()}
        res = {"model": a.model, "text": "tasks", "k": a.k, "n_train_pos": Ytr.shape[0], "n_test_pos": {t: v[1].shape[0] for t, v in T.items()},
               "args": vars(a), "gpu": torch.cuda.get_device_name(), "depths": {}}
        print(f"{a.model}/tasks: {Ytr.shape[0]} train positions, test {res['n_test_pos']}, {time.time() - t0:.0f}s", flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        for d in ex.depths:
            D_, b_, keep = fit(ex, Xtr[d], Ytr, a.k, a.epochs, a.bs, a.lr)
            res["depths"][d] = {"epoch_kept": keep, **{t: {"acc": evaluate(ex, D_, b_, X[d], Y, a.k)[0]} for t, (X, Y) in T.items()}}
            print(f"  {d:8s} " + "  ".join(f"{t} " + " ".join(f"{x:.3f}" for x in res["depths"][d][t]["acc"][:3]) for t in T) + f"  ({time.time() - t0:.0f}s)", flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
        return
    seqs = sequences(a.model, a.text, a.n_train + a.n_test)
    Xtr, Ytr = collect(ex, seqs[:a.n_train], a.k, a.max_pos, a.max_len, 1)
    Xte, Yte = collect(ex, seqs[a.n_train:], a.k, a.max_pos, a.max_len, 2)
    print(f"{a.model}/{a.text}: {Ytr.shape[0]} train / {Yte.shape[0]} test positions, depths {ex.depths}, {time.time() - t0:.0f}s", flush=True)
    res = {"model": a.model, "text": a.text, "k": a.k, "n_train_pos": Ytr.shape[0], "n_test_pos": Yte.shape[0],
           "args": vars(a), "gpu": torch.cuda.get_device_name(), "depths": {}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for d in ex.depths:
        acc, ce, keep = train_probe(ex, Xtr[d], Ytr, Xte[d], Yte, a.k, a.epochs, a.bs, a.lr)
        res["depths"][d] = {"acc": acc, "ce": ce, "epoch_kept": keep}
        print(f"  {d:8s} acc " + " ".join(f"{x:.3f}" for x in acc) + f"   kept {keep}  ({time.time() - t0:.0f}s)", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
