"""Does our Huginn forward match the official implementation?  (a) teacher-forced argmax agreement of RavenMin.forward_full vs the
HF model (trust_remote_code) on prompt + continuation, (b) greedy generation of RavenMin vs HF for 48 tokens, (c) our CUDA-graph
engine's plain decoding vs RavenMin.generate.   python verify_pre_coda.py <model_dir> <trace.jsonl> [N=3]"""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import json, os, sys, time
import torch
from floodgates.models.raven_min import RavenMin
M, TR = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 3
rows = [json.loads(l) for l in open(TR)][:N]
m = RavenMin(M); print("RavenMin pre_coda_norm =", m.pre_coda_norm, "R =", m.R, flush=True)
from transformers import AutoModelForCausalLM
hf = AutoModelForCausalLM.from_pretrained(M, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
for i, r in enumerate(rows):
    ids = r["input"] + (r.get("output_ids") or [])[:64]
    x = torch.tensor([ids], device="cuda")
    with torch.no_grad():
        lg_hf = hf(input_ids=x, num_steps=torch.tensor([m.R])).logits[0].float()
        lg_rm = m.forward_full(x)[0][0].float()                       # (T, V); teacher-forced, same ids
    a_hf, a_rm = lg_hf.argmax(-1), lg_rm.argmax(-1)
    P = len(r["input"])
    print(f"[a] row {i}: argmax agreement {int((a_hf == a_rm).sum())}/{len(ids)} (continuation {int((a_hf[P:] == a_rm[P:]).sum())}/{len(ids) - P}); "
          f"max |logit diff| {float((lg_hf - lg_rm).abs().max()):.2f}; mean |diff| {float((lg_hf - lg_rm).abs().mean()):.3f}", flush=True)
for i, r in enumerate(rows):
  try:
    x = torch.tensor([r["input"]], device="cuda")
    with torch.no_grad():
        g_hf = hf.generate(x, max_new_tokens=48, do_sample=False, num_steps=m.R)[0, len(r["input"]):].tolist()
    g_rm = m.generate(r["input"], 48)
    k = next((j for j, (a, b) in enumerate(zip(g_hf, g_rm)) if a != b), min(len(g_hf), len(g_rm)))
    print(f"[b] row {i}: HF vs RavenMin greedy: first divergence at token {k} of {min(len(g_hf), len(g_rm))}", flush=True)
  except Exception as e:
    print("[b] HF generate failed:", repr(e)[:200], flush=True); break
try:
    from floodgates.engines.raven_engine import RavenEngine
    eng = RavenEngine(m, None); eng.warmup(max(len(r["input"]) for r in rows) + 64)
    for i, r in enumerate(rows):
        o, dt = eng.generate_plain(r["input"], 48, lambda out: False)
        g_rm = m.generate(r["input"], 48)
        k = next((j for j, (a, b) in enumerate(zip(o, g_rm)) if a != b), min(len(o), len(g_rm)))
        print(f"[c] row {i}: engine plain vs RavenMin greedy: first divergence at token {k} of {min(len(o), len(g_rm))}", flush=True)
except Exception as e:
    print("[c] engine check skipped:", repr(e)[:200], flush=True)
print("VERIFY DONE", flush=True)
