"""Four-domain training mix of a model's own greedy continuations: data/mix4c_<tag>_greedy.jsonl from the four domain files written by
gen_raven_data2.py / gen_data.py (data/{gsm8k,math,code,general}_train_<tag>_greedy.jsonl).  Rows without an output are dropped, as are the
first 474 code prompts (the MBPP train/validation/prompt problems, so that MBPP+ stays unseen).  The order is the paper's: the rows are
shuffled with seed 0, filtered, and shuffled again with seed 0.
    python datagen/build_mix.py <tag>            # e.g. rl32, tl32, huginn, ouro14, ouro26
Environment: CODE_FILE overrides the code file.
    python datagen/build_mix.py <tag> --with-mbpp   # data/mix4_<tag>_greedy.jsonl: all rows after the first shuffle (dense-model control)
    python datagen/build_mix.py <tag> --chat     # Thinking models: data/mix_chat_<tag>_greedy.jsonl = the GSM8K-chat rows, then the
                                                  MATH-chat rows (data/{gsm8kchat,mathchat}_train_<tag>_greedy.jsonl), with an output"""
import json, os, random, sys

tag = sys.argv[1]
if "--chat" in sys.argv[2:]:
    out = f"data/mix_chat_{tag}_greedy.jsonl"
    rows = [l for f in (f"data/gsm8kchat_train_{tag}_greedy.jsonl", f"data/mathchat_train_{tag}_greedy.jsonl") for l in open(f)
            if json.loads(l).get("output_ids")]
    open(out, "w").writelines(rows)
    print(f"{out}: {len(rows)} rows")
    sys.exit()
files = [(f"data/gsm8k_train_{tag}_greedy.jsonl", "gsm8k"), (f"data/math_train_{tag}_greedy.jsonl", "math"),
         (os.environ.get("CODE_FILE", f"data/code_train_{tag}_greedy.jsonl"), "code"), (f"data/general_train_{tag}_greedy.jsonl", "general")]
rows = []
for f, domain in files:
    n = 0
    for l in open(f):
        r = json.loads(l)
        if r.get("output_ids"):
            rows.append({"prompt_ids": r["prompt_ids"], "output_ids": r["output_ids"], "domain": domain, "mbpp": domain == "code" and r.get("i", 474) < 474}); n += 1
    print(domain, n)
random.seed(0); random.shuffle(rows)
if "--with-mbpp" in sys.argv[2:]:
    with open(f"data/mix4_{tag}_greedy.jsonl", "w") as f:
        for r in rows: f.write(json.dumps({k: v for k, v in r.items() if k != "mbpp"}) + "\n")
    print(f"data/mix4_{tag}_greedy.jsonl: {len(rows)} rows")
keep = [{k: v for k, v in r.items() if k != "mbpp"} for r in rows if not r["mbpp"]]
random.seed(0); random.shuffle(keep)
out = f"data/mix4c_{tag}_greedy.jsonl"
with open(out, "w") as f:
    for r in keep: f.write(json.dumps(r) + "\n")
print(f"{out}: {len(rows)} rows -> {len(keep)} after dropping {len(rows) - len(keep)} MBPP prompts")
