"""LayerSkip's own self-speculative decoding on LayerSkip-Llama-3.2-1B, as a dense baseline: the first E layers
draft autoregressively and the full model verifies (Hugging Face generate with assistant_early_exit=E, greedy), against plain greedy
generate of the same model, at batch size 1 on the same prompts, stops and length limits as dense_drafter.py eval (first 100 GSM8K
test problems; the MATH-500 problems of the harness).  Wall-clock per problem with CUDA synchronisation, two warm-up problems, tokens
counted up to the stop string; checks that the outputs equal plain decoding.
  python baselines/dense/layerskip_selfspec.py <model dir> <prompts jsonl> <gsm8k|math500> <out.json> [E,E,...]"""
import json, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StopStringCriteria, StoppingCriteriaList
MD, PR, TASK, OUT = sys.argv[1:5]
ES = [int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "2,4,6,8").split(",")]
STOPS = {"gsm8k": ["Q:"], "math500": ["Problem:"]}[TASK]; MAXNEW = {"gsm8k": 512, "math500": 1024}[TASK]
tok = AutoTokenizer.from_pretrained(MD)
m = AutoModelForCausalLM.from_pretrained(MD, torch_dtype=torch.bfloat16).cuda().eval()
rows = [json.loads(l) for l in open(PR)][:100]


def cut(ids):
    """Tokens up to and including the first stop string (as SGLang's completion_tokens)."""
    for n in range(1, len(ids) + 1):
        if any(s in tok.decode(ids[max(0, n - 8):n]) for s in STOPS): return ids[:n]
    return ids


def run(E):
    # stop strings as an explicit criterion: generate's stop_strings is copied into the early-exit assistant's generate, which
    # then has no tokenizer and fails; the same criterion is used for plain decoding
    kw = dict(max_new_tokens=MAXNEW, do_sample=False, stopping_criteria=StoppingCriteriaList([StopStringCriteria(tok, STOPS)]))
    if E: kw["assistant_early_exit"] = E
    outs, secs = [], []
    for i, r in enumerate([rows[0], rows[1]] + rows):          # two warm-up problems, not timed
        ids = tok(r["context"], return_tensors="pt").input_ids.cuda()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad(): o = m.generate(ids, **kw)[0, ids.shape[1]:].tolist()
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        if i >= 2: outs.append(cut(o)); secs.append(dt)
    return outs, secs


res = {"model": MD, "task": TASK, "n": len(rows), "gpu": torch.cuda.get_device_name()}
plain, tp = run(0)
res["plain"] = {"tok_s": sum(map(len, plain)) / sum(tp), "tokens": sum(map(len, plain)), "seconds": sum(tp)}
print(json.dumps({"plain": res["plain"]}), flush=True)
for E in ES:
    o, t = run(E)
    tps = sum(map(len, o)) / sum(t)
    res[f"E{E}"] = {"tok_s": tps, "speedup": tps / res["plain"]["tok_s"], "tokens": sum(map(len, o)), "seconds": sum(t),
                    "identical_to_plain": sum(a == b for a, b in zip(o, plain))}
    print(json.dumps({f"E{E}": res[f"E{E}"]}), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
