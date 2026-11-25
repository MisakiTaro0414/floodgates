"""Speedup and tokens per step counted up to the stop string, as reported in the paper.  Plain decoding checks its stop strings after
every token; the speculative benchmarks check them only after each verification cycle (and the parallel sampler after each pass), so
a speculative output can carry up to a block of tokens past the stop, which the benchmarks' own speedups count as generated tokens.
For every problem we cut the speculative output at the first length where the per-token stop check of plain decoding fires (a stop
string in the decoded tail, or EOS), and recompute speedup = (sum cut spec tokens / sum spec seconds) / (sum plain tokens / sum plain
seconds) and tokens per step = sum cut spec tokens / number of cycles.  Timings are unchanged (the extra tokens did cost time).
Sanity check: the plain outputs must recount to their own length (plain_ok = 1.0).
    python evaluation/stopcut.py <result .json or _outputs.jsonl> ... [--model NAME] [--task TASK]
The model is read from the args.json next to the result (drafter runs) or from the file name, the task from the file name."""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import argparse, glob, json
from floodgates.paths import LOOPSPEC, model_dir

MODELS = ["Recurrent-Llama-3.2-train-recurrence-32", "Recurrent-TinyLlama-3T-train-recurrence-32", "huginn-0125", "Ouro-1.4B-Thinking",
          "Ouro-2.6B-Thinking", "Ouro-1.4B", "Ouro-2.6B"]                                   # longest names first
SHORT = {"rl32": MODELS[0], "tl32": MODELS[1], "huginn": MODELS[2]}
CODE = ["<|endoftext|>", "<|endofmask|>", "</s>", "\nif __name__", "\ndef main(", "\nprint(", "\n# Test", "\nassert ", "\n```"]
STOPS = {"gsm8k": ["Q:", "</s>", "<|im_end|>"], "math500": ["Problem:"], "humaneval": CODE, "mbpp": CODE, "bbh": ["</s>", "Q", "\n\n"],
         "gsm8kchat": ["Q:", "</s>", "<|im_end|>"], "math500chat": ["Problem:", "<|im_end|>"]}
CAP = {"gsm8k": 512, "math500": 2048, "humaneval": 1024, "mbpp": 1024, "bbh": 1024, "gsm8kchat": 4096, "math500chat": 8192}
TASKS = ["gsm8kchat", "math500chat", "gsm8k", "math500", "humaneval", "mbpp", "bbh"]
_tok = {}


def infer(path):
    b, d, model = os.path.basename(path), os.path.dirname(path), None
    if os.path.exists(os.path.join(d, "args.json")):
        model = os.path.basename(json.load(open(os.path.join(d, "args.json")))["model"].rstrip("/"))
    if model is None:
        model = next((m for m in MODELS if m in b), None)
    if model is None:
        model = next((m for s, m in SHORT.items() if f"_{s}_" in b or b.startswith(f"{s}_")), None)
    task = next((t for t in TASKS if f"_{t}_" in b or b.endswith(f"_{t}.json") or b.endswith(f"_{t}_outputs.jsonl") or f"_{t}." in b), None)
    return model, task


def tokenizer(model):
    """The tokenizer as the paper's recount loaded it (Recurrent-Llama: the bare tokenizer.json, so no EOS id)."""
    if model not in _tok:
        from transformers import AutoTokenizer, PreTrainedTokenizerFast
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        p = os.path.join(LOOPSPEC, "models", model)
        p = p if os.path.isdir(p) else model_dir(model)
        try: _tok[model] = AutoTokenizer.from_pretrained(p, trust_remote_code=False)
        except Exception: _tok[model] = PreTrainedTokenizerFast(tokenizer_file=os.path.join(p, "tokenizer.json"))
    return _tok[model]


def _cut(tok, out, stops, eos, start, cap, LB=40):
    for L in range(max(1, start), len(out) + 1):
        if (eos is not None and out[L - 1] == eos) or any(x in tok.decode(out[max(0, L - LB):L]) for x in stops): return L
    return min(len(out), cap + 1)


def _common(a, b):
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]: i += 1
    return i


def compute(path, model=None, task=None):
    """path: a result .json (its _outputs.jsonl is read), the outputs file itself, or a json with per_problem records."""
    m0, t0 = infer(path); model = model or m0; task = task or t0
    if model is None or task is None:
        raise ValueError(f"{path}: cannot tell the model / task from the path (pass --model / --task)")
    js = path if path.endswith(".json") else None
    outp = path if path.endswith("_outputs.jsonl") else path[:-5] + "_outputs.jsonl"
    J = json.load(open(js)) if js and os.path.exists(js) else {}
    if "per_problem" in J:
        recs = [{"plain": p["ar_out"], "spec": p["df_out"], "dt_plain": p["ar_sec"], "dt_spec": p["df_sec"]} for p in J["per_problem"]]; plus1 = 0
    elif os.path.exists(outp):
        recs = [json.loads(l) for l in open(outp)]; plus1 = 1         # the first token comes from the (untimed) prefill
    else:
        raise FileNotFoundError(outp)
    timed = all(k in recs[0] for k in ("dt_plain", "dt_spec"))
    if not timed and "speedup" not in J:
        raise ValueError(f"{path}: no per-problem times and no stored speedup")
    tok = tokenizer(model); eos = tok.eos_token_id; stops = STOPS[task]
    cap = J["max_new"] if isinstance(J.get("max_new"), int) else (1024 if (model == MODELS[1] and task == "math500") else CAP[task])
    np_ = tp = ts = raw = cut = cyc = plain_ok = 0
    for r in recs:
        p, s = r["plain"], r["spec"]
        c = _common(p, s)
        L = len(p) if c >= len(p) and len(s) >= len(p) else _cut(tok, s, stops, eos, c, cap)
        Lp = _cut(tok, p, stops, eos, max(1, len(p) - 3), cap)          # plain must stop where it stopped
        plain_ok += int(Lp == len(p) or len(p) >= cap)
        np_ += len(p) - plus1; raw += len(s) - plus1; cut += L - plus1
        if timed: tp += r["dt_plain"]; ts += r["dt_spec"]
        if isinstance(r.get("tau"), list): cyc += len(r["tau"])
    # without per-problem times: the stored aggregate speedup scaled by the share of tokens kept (exact for that aggregate)
    sr = (raw / ts) / (np_ / tp) if timed else J["speedup"]
    return {"model": model, "task": task, "n": len(recs), "speedup_raw": sr, "speedup": sr * cut / raw, "tau": (cut / cyc) if cyc else None,
            "spec_tokens_raw": raw, "spec_tokens_cut": cut, "plain_tokens": np_, "plain_ok": plain_ok / len(recs)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+"); ap.add_argument("--model"); ap.add_argument("--task")
    a = ap.parse_args()
    for pat in a.paths:
        for f in sorted(glob.glob(pat)) or [pat]:
            r = compute(f, a.model, a.task)
            tau = f"{r['tau']:.2f}" if r["tau"] is not None else "-"
            print(f"{f}: {r['model']} {r['task']} n={r['n']}  speedup {r['speedup_raw']:.2f} -> {r['speedup']:.2f} up to the stop, "
                  f"tokens per step {tau}, plain_ok {r['plain_ok']:.2f}")
