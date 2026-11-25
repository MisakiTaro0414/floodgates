"""Same-GPU numbers for Geiping, Yang & Su (ICML 2026) on Huginn-0125 with THEIR code (recurrent-pretraining/recpre): plain AR
generation (num_steps=32), the diffusion-forcing sampler with the paper's defaults (inner_recurrence 4, exit_t 0.03,
ema_embeds 0.1, max_wavefront 128), and the model's built-in self-speculative mode (draft with fewer steps, exact
verification).  Prompts = the 100 GSM8K problems of our Huginn trace; max_new_tokens 512; throughput = generated tokens /
wall-clock of the generate call (prefill included, same for all methods); accuracy from the first answer before "Q:".
  N_PROBLEMS=100 METHODS=ar,diffusion,spec python bench_geiping.py <out_dir>"""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import glob, json, os, re, sys, time
import torch
import os
from floodgates.paths import LOOPSPEC, PARENT, trace
from floodgates.tokenizer import load_tokenizer
from transformers import GenerationConfig
from floodgates.models.ravenhf.raven_config_minimal import RavenConfig
from floodgates.models.ravenhf.raven_modeling_minimal import RavenForCausalLM

S = PARENT; MD = os.environ.get("MODEL_DIR", f"{LOOPSPEC}/models/huginn-0125")   # any Raven-architecture checkpoint
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
N = int(os.environ.get("N_PROBLEMS", 100)); METHODS = os.environ.get("METHODS", "ar,diffusion,spec").split(",")
MAX_NEW = int(os.environ.get("MAX_NEW", 512))
tok = load_tokenizer(MD)
cfg = RavenConfig.from_pretrained(MD)
RavenForCausalLM._tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}   # transformers 5.x wants a dict (tie_embeddings=True)
if os.environ.get("USE_REMOTE") == "1":                                   # retrofitted models: their own modeling file (grouped-query attention)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MD, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
else:
    model = RavenForCausalLM.from_pretrained(MD, config=cfg, torch_dtype=torch.bfloat16).cuda().eval()
print(torch.cuda.get_device_name(0), "| params", sum(p.numel() for p in model.parameters()) / 1e9, "B", flush=True)
rows = [json.loads(l) for l in open(os.environ.get("TRACE", trace("huginn-0125", "gsm8k")))][:N]
gold = {}
for f in glob.glob(f"{LOOPSPEC}/runs/Ouro-1.4B-gsm8k-n100/baseline/samples_*.jsonl"):
    for l in open(f):
        d = json.loads(l)["doc"]; gold[d["question"].strip()] = d["answer"].split("####")[-1].strip().replace(",", "")
print(len(rows), "prompts,", len(gold), "gold answers", flush=True)
gc = GenerationConfig(max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id, eos_token_id=tok.eos_token_id)
KW = {"ar": dict(num_steps=32),
      "diffusion": dict(headway=1, inner_recurrence=4, num_steps=32, exit_t=0.03, ema_embeds=0.1, max_wavefront=128),
      "spec": dict(draft_steps=8, lookahead_for_draft=8, verification_threshold=1, num_steps=32)}


def answer_of(text):
    body = text.split("Q:")[0]
    m = re.findall(r"answer is\s*\$?(-?[\d,]*\.?\d+)", body)
    return (m[-1] if m else "").replace(",", "")


res = {}
for meth in METHODS:
    toks = secs = correct = 0; per = []
    for i, r in enumerate(rows):
        ids = torch.tensor([r["input"]], device="cuda")
        if i == 0:                                                        # warm-up (compile/caches), not timed
            model.generate(ids, generation_config=GenerationConfig(max_new_tokens=16, do_sample=False), max_new_tokens=16, **KW[meth])
        torch.cuda.synchronize(); t = time.time()
        o = model.generate(ids, generation_config=gc, max_new_tokens=MAX_NEW, **KW[meth])   # their samplers read max_new_tokens from kwargs
        torch.cuda.synchronize(); dt = time.time() - t
        o = o[0] if isinstance(o, torch.Tensor) else o["sequences"][0] if isinstance(o, dict) else o[0]
        gen = o[ids.shape[1]:].tolist(); text = tok.decode(gen)
        q = re.findall(r"Q: (.*?)\nA:", tok.decode(r["input"]), re.S)[-1].strip()
        ok = gold.get(q) is not None and answer_of(text) == gold[q]
        toks += len(gen); secs += dt; correct += int(ok); per.append({"tokens": len(gen), "sec": dt, "correct": ok})
        if i % 10 == 9:
            print(f"{meth}: {i + 1}/{len(rows)}  {toks / secs:.1f} tok/s  acc {correct / (i + 1):.2f}", flush=True)
    res[meth] = {"tok_s": toks / secs, "accuracy": correct / len(rows), "tokens": toks, "sec": secs, "kwargs": KW[meth], "n": len(rows), "per_problem": per}
    print(f"== {meth}: {toks / secs:.1f} tok/s, accuracy {correct / len(rows):.3f}, {toks} tokens in {secs:.0f} s", flush=True)
    json.dump(res, open(f"{out}/geiping_results.json", "w"), indent=1)
if "ar" in res:
    for meth in res:
        print(f"{meth}: {res[meth]['tok_s'] / res['ar']['tok_s']:.2f}x over their AR", flush=True)
print("done", flush=True)
