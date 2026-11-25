"""Validate RavenMin against the checkpoint's own HF modeling code: next-token argmax agreement over a GSM8K prompt+answer.
The two use different random initial states (like-init), so agreement is high but not exactly 100%.
  python check_raven_hf.py <model_dir>      (run in a venv with transformers 4.5x for the remote code)"""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
from floodgates.paths import ROOT
import json, os, sys
import torch
from floodgates.models.raven_min import RavenMin
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
M = sys.argv[1]
tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(M, "tokenizer.json"))
text = open(os.path.join(ROOT, "datagen", "prompts", "math_4shot_prefix.txt")).read()[:3000]
ids = torch.tensor([tok(text, add_special_tokens=False)["input_ids"][:400]], device="cuda")
m = RavenMin(M)
with torch.no_grad():
    lg_min, _ = m.forward_full(ids)
del m; torch.cuda.empty_cache()
hf = AutoModelForCausalLM.from_pretrained(M, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
with torch.no_grad():
    lg_hf = hf(ids, num_steps=32).logits
a, b = lg_min[0].float().argmax(-1), lg_hf[0].float().argmax(-1)
agree = float((a == b).float().mean())
nxt = ids[0, 1:]; acc_min = float((a[:-1] == nxt).float().mean()); acc_hf = float((b[:-1] == nxt).float().mean())
print(json.dumps({"model": M, "T": ids.shape[1], "argmax_agreement": agree, "next_token_acc_min": acc_min, "next_token_acc_hf": acc_hf}), flush=True)
assert agree > 0.9, "RavenMin disagrees with the reference implementation"
