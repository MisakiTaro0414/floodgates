"""Self-distillation data for Ouro-1.4B-Thinking on MATH: the model's greedy continuations of MATH TRAIN problems in the lm-eval
minerva_math 4-shot format WITH the chat template and thinking enabled (the format of LoopSpec's MATH-500-chat runs; prefix = the
four exemplar turns of a LoopSpec math500chat trace, verified token for token against the trace), via a running SGLang server
(as gen_data.py, whose gsm8kchat branch made data/gsm8kchat_train_ouro14t_greedy.jsonl).  Stop: EOS (<|im_end|>) / "Problem:"
(the lm-eval 'until'); at most MAX_NEW (4096, as the GSM8K-chat data) new tokens.  Resumable by i; one JSON line per problem:
i, prompt_ids, output_ids, text.
  CHAT_TRACE=<math500chat trace> GEN_MODEL=<model dir> python gen_data_chat.py <out.jsonl> [concurrency=64] [limit]
The paper's data: the 2,032 problems of ONLY_FROM=datagen/prompts/mathchat_train_problems.jsonl (indices into MATH train)."""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import asyncio, json, os, re, sys
import aiohttp
from datasets import load_dataset
from floodgates.tokenizer import load_tokenizer
import os
from floodgates.paths import LOOPSPEC

OUT, CONC = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 64
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else None
MODEL = os.environ.get("GEN_MODEL", LOOPSPEC + '/models/Ouro-1.4B-Thinking')
MAXNEW = int(os.environ.get("MAX_NEW", 4096)); STOP = ["Problem:", "<|im_end|>"]
tok = load_tokenizer(MODEL)
tr = json.loads(open(os.environ["CHAT_TRACE"]).readline())
txt = tok.decode(tr["input"])
prefix = txt[: txt.rindex("<|im_start|>user\n")]
FMT = lambda q: "<|im_start|>user\nProblem:\n" + q + "\n\nSolution:<|im_end|>\n<|im_start|>assistant\n<think>\n"
tail = txt[txt.rindex("<|im_start|>user\n"):]
q0 = re.match(r"<\|im_start\|>user\nProblem:\n(.*)\n\nSolution:<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n$", tail, re.S).group(1)
assert tok(prefix + FMT(q0), add_special_tokens=False)["input_ids"] == tr["input"], "chat prompt reconstruction mismatch"
print("chat prompt format verified against the trace", flush=True)
qs = [r["problem"] for sub in ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
                               "prealgebra", "precalculus") for r in load_dataset("EleutherAI/hendrycks_math", sub, split="train")]
if os.environ.get("SHUFFLE", "1") == "1":                 # interleave subjects so a partial run covers all of them
    import random
    order = list(range(len(qs))); random.Random(0).shuffle(order)
else:
    order = list(range(len(qs)))
if LIMIT:
    order = order[:LIMIT]
if os.environ.get("ONLY_FROM"):                          # the problems of an earlier run, e.g. the 1.4B-Thinking MATH-chat data
    keep = {json.loads(l)["i"] for l in open(os.environ["ONLY_FROM"])}
    order = [i for i in range(len(qs)) if i in keep]
SH, NSH = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
order = [i for i in order if i % NSH == SH]
print(f"{len(qs)} MATH train problems, generating {len(order)}", flush=True)


def main():
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try:
                done.add(json.loads(l)["i"])
            except Exception:
                pass
    print(f"resume: {len(done)} already done", flush=True)

    async def one(sess, sem, i, f):
        ids = tok(prefix + FMT(qs[i]), add_special_tokens=False)["input_ids"]
        payload = {"input_ids": ids, "sampling_params": {"temperature": 0, "top_p": 1.0, "max_new_tokens": MAXNEW, "stop": STOP}}
        async with sem:
            for attempt in range(3):
                try:
                    async with sess.post("http://127.0.0.1:30000/generate", json=payload, timeout=aiohttp.ClientTimeout(total=7200)) as r:
                        out = await r.json()
                        break
                except Exception as e:
                    if attempt == 2:
                        print("fail", i, e, flush=True); return
        f.write(json.dumps({"i": i, "prompt_ids": ids, "output_ids": out.get("output_ids"), "text": out.get("text")}) + "\n"); f.flush()

    async def run():
        sem = asyncio.Semaphore(CONC)
        with open(OUT, "a") as f:
            async with aiohttp.ClientSession() as sess:
                await asyncio.gather(*[one(sess, sem, i, f) for i in order if i not in done])
        print("done", len(order), flush=True)
    asyncio.run(run())


if __name__ == "__main__":
    main()
