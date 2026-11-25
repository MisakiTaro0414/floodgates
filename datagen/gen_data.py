"""Self-distillation data: Ouro's greedy continuations of GSM8K train questions in the lm-eval gsm8k_cot 3-shot format
(prefix = lm-eval's first 3 exemplars, 'Q: {question}\nA:', stop at 'Q:', max 512 new tokens), via a running SGLang server.
Writes one JSON line per question: prompt_ids, output_ids, text.   python gen_data.py <out.jsonl> [concurrency] [limit]"""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import asyncio, json, os, sys
import aiohttp
from datasets import load_dataset
from floodgates.tokenizer import load_tokenizer
from floodgates.paths import LOOPSPEC
OUT, CONC = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 64
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else None
TEMP = float(os.environ.get("GEN_TEMP", 0)); NS = int(os.environ.get("GEN_SAMPLES", 1))   # T>0: NS samples per prompt
MODEL = os.environ.get("GEN_MODEL", LOOPSPEC + '/models/Ouro-1.4B')
TASK = os.environ.get("GEN_TASK", "gsm8k")
if TASK == "math":   # lm-eval minerva_math 4-shot format (prefix extracted from a LoopSpec MATH-500 trace)
    prefix = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "math_4shot_prefix.txt")).read()
    FMT, STOP, MAXNEW = (lambda q: "Problem:\n" + q + "\n\nSolution:"), ["Problem:"], 2048
elif TASK == "code":   # LoopSpec's custom 0-shot HumanEval+/MBPP+ format (evaluations/tasks/*_custom.yaml)
    prefix = ""
    CODE_STOP = ["<|endoftext|>", "<|endofmask|>", "</s>", "\nif __name__", "\ndef main(", "\nprint(", "\n# Test", "\nassert ", "\n```"]
    FMT, STOP, MAXNEW = (lambda q: q), CODE_STOP, 1024
elif TASK == "gsm8kchat":   # Ouro-*-Thinking: lm-eval gsm8k_cot 3-shot with the chat template + thinking (prefix from a LoopSpec trace)
    import re as _re
    _tr = json.loads(open(os.environ["CHAT_TRACE"]).readline())
    _tok0 = load_tokenizer(MODEL)
    _txt = _tok0.decode(_tr["input"])
    prefix = _txt[: _txt.rindex("<|im_start|>user\n")]
    FMT = lambda q: "<|im_start|>user\nQ: " + q + "\nA:<|im_end|>\n<|im_start|>assistant\n<think>\n"
    STOP, MAXNEW = ["<|im_end|>"], 4096
    _tail = _txt[_txt.rindex("<|im_start|>user\n"):]                      # the LAST user turn = the test question
    _q0 = _re.match(r"<\|im_start\|>user\nQ: (.*)\nA:<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n$", _tail, _re.S).group(1)
    assert _tok0(prefix + FMT(_q0), add_special_tokens=False)["input_ids"] == _tr["input"], "chat prompt reconstruction mismatch"
    print("chat prompt format verified against the trace", flush=True)
elif TASK == "general":   # FineWeb-Edu document continuations (prefix = first 256 tokens, continue 256)
    prefix = ""
    FMT, STOP, MAXNEW = (lambda q: q), ["</s>"], 256
else:                # lm-eval gsm8k_cot 3-shot format
    prefix = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "gsm8k_cot_3shot_prefix.txt")).read()
    FMT, STOP, MAXNEW = (lambda q: "Q: " + q + "\nA:"), ["Q:", "</s>", "<|im_end|>"], 512
try:
    tok = load_tokenizer(MODEL)
except Exception:   # Raven-family dirs (Recurrent-Llama, Huginn) only load through their tokenizer.json
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(MODEL, "tokenizer.json"))
CODE_HEAD = "Please provide a self-contained Python script that solves the following problem in a markdown code block:\n```\n"
CODE_TAIL = "\n```\nBelow is a Python script with a self-contained function that solves the problem and passes corresponding tests:\n```python\n"
def code_prompts():
    import re
    out = []
    for split in ("train", "validation", "prompt"):           # MBPP (not the test split used by MBPP+)
        for r in load_dataset("google-research-datasets/mbpp", "full", split=split):
            body = '"""\n' + re.sub(r"https://\S+", "", r["text"]) + "\n" + r["test_list"][0] + '\n"""'
            out.append(CODE_HEAD + body + CODE_TAIL)
    # HumanEval-style function stubs (signature + docstring) from CodeSearchNet python, completed by the model
    n = 0
    for r in load_dataset("code-search-net/code_search_net", "python", split="train", streaming=True):
        code, doc = r["func_code_string"], r["func_documentation_string"].strip()
        m = re.match(r"(def [^\n]*:\n)", code)
        if not m or not doc or len(doc) < 40 or len(code) > 1500 or "self" in m.group(1):
            continue
        stub = m.group(1) + '    """' + doc.split("\n\n")[0].strip() + '\n    """\n'
        out.append(CODE_HEAD + stub.strip() + CODE_TAIL)
        n += 1
        if n >= int(os.environ.get("GEN_CODE_N", 3000)):
            break
    return out
def general_prompts():
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    out = []
    for r in ds:
        ids = tok(r["text"], add_special_tokens=False)["input_ids"]
        if len(ids) < 512:
            continue
        out.append(tok.decode(ids[:256]))
        if len(out) >= int(os.environ.get("GEN_GENERAL_N", 4000)):
            break
    return out
if TASK == "code":
    qs = code_prompts()
elif TASK == "general":
    qs = general_prompts()
elif TASK == "math":
    qs = [r["problem"] for sub in ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
                                   "prealgebra", "precalculus") for r in load_dataset("EleutherAI/hendrycks_math", sub, split="train")]
else:
    qs = [r["question"] for r in load_dataset("openai/gsm8k", "main", split="train")]
if LIMIT: qs = qs[:LIMIT]
SH, NSH = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))   # shard i % NSHARD == SHARD (default: all)
def _sglang_main():
    import os
    done_ids = set()
    if os.path.exists(OUT):                      # resume: keep finished questions, append the rest
        for l in open(OUT):
            try: done_ids.add(json.loads(l)["i"])
            except Exception: pass
    print(f"resume: {len(done_ids)} already done", flush=True)
    async def one(sess, sem, i, q, f):
        ids = tok(prefix + FMT(q), add_special_tokens=False)["input_ids"]
        payload = {"input_ids": ids, "sampling_params": {"temperature": TEMP, "top_p": 1.0, "max_new_tokens": MAXNEW, "stop": STOP}}
        async with sem:
            for attempt in range(3):
                try:
                    async with sess.post("http://127.0.0.1:30000/generate", json=payload, timeout=aiohttp.ClientTimeout(total=3600)) as r:
                        out = await r.json()
                        break
                except Exception as e:
                    if attempt == 2: print("fail", i, e, flush=True); return
        rec = {"i": i, "prompt_ids": ids, "output_ids": out.get("output_ids"), "text": out.get("text")}
        f.write(json.dumps(rec) + "\n"); f.flush()
    async def main():
        sem = asyncio.Semaphore(CONC)
        with open(OUT, "a") as f:
            async with aiohttp.ClientSession() as sess:
                await asyncio.gather(*[one(sess, sem, i * NS + j, q, f) for i, q in enumerate(qs) for j in range(NS) if i * NS + j not in done_ids and i % NSH == SH])
        print("done", len(qs), flush=True)
    asyncio.run(main())

if __name__ == "__main__":
    _sglang_main()
