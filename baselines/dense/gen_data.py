"""Four-domain drafter data for a dense model (the LayerSkip control): the model's own greedy outputs on exactly the prompts of
data/mix4_rl32_greedy.jsonl (same prompt ids, Llama-3.2 tokenizer), with the stop strings and length limits of gen_data.py per domain
(gsm8k: Q: / 512, math: Problem: / 1024 as the observed cap, code: the code stops / 1024, general: 256), vLLM, greedy.
  python baselines/dense/gen_data.py <hf model dir> <out.jsonl>"""
import json, sys
from vllm import LLM, SamplingParams
MD, OUT = sys.argv[1], sys.argv[2]
CODE = ["<|endoftext|>", "<|endofmask|>", "</s>", "\nif __name__", "\ndef main(", "\nprint(", "\n# Test", "\nassert ", "\n```"]
DOM = {"gsm8k": (["Q:", "</s>", "<|im_end|>"], 512), "math": (["Problem:"], 1024), "code": (CODE, 1024), "general": (["</s>"], 256)}
rows = [json.loads(l) for l in open("data/mix4_rl32_greedy.jsonl")]
llm = LLM(MD, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.9, seed=0)
out = [None] * len(rows)
for dom, (stops, mx) in DOM.items():
    idx = [i for i, r in enumerate(rows) if r["domain"] == dom]
    sp = SamplingParams(temperature=0.0, max_tokens=mx, stop=stops, include_stop_str_in_output=False)
    res = llm.generate([{"prompt_token_ids": rows[i]["prompt_ids"]} for i in idx], sp)
    for i, r in zip(idx, res): out[i] = list(r.outputs[0].token_ids)
    print(dom, len(idx), "mean length", sum(len(out[i]) for i in idx) / len(idx), flush=True)
with open(OUT, "w") as f:
    for r, o in zip(rows, out):
        if o: f.write(json.dumps({"prompt_ids": r["prompt_ids"], "output_ids": o, "domain": r["domain"]}) + "\n")
print("wrote", OUT)
