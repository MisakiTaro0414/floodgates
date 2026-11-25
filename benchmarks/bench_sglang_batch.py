"""Real wall-clock throughput of an SGLang server (plain baseline or LoopSpec) at client concurrency B, on the prompts of a
LoopSpec trace (exact token ids), with the lm-eval request payload (greedy, max_new_tokens, stop strings).
  python bench_sglang_batch.py <generate_url> <trace.jsonl> <B> <out.json> [cont|static]
env: N_PROBLEMS (100), MAX_NEW (512), STOP_STRS ("Q:;;</s>;;<|im_end|>", separator ";;"), PROBE (1: static mode first times
     each wave with max_new_tokens=1 = prefill + first token, so decode time = wave time - probe time; needs the radix
     cache off, else the probe pre-caches the prompts), SERVER_LOG (LoopSpec / baseline server log with generation_metrics
     lines: their sum(decode_tokens) / sum(decode_seconds) is added to the result).
cont   : B client threads pull prompts from one queue, so the server always has up to B requests in flight (continuous
         batching; LoopSpec serves one request at a time whatever B is).
static : waves of B prompts sent together; the next wave starts when the whole wave completed (= engine_batch.py waves).
Tokens = completion_tokens (SGLang output_ids, through the stop string) = the count engine_batch.py uses."""
import json, os, re, sys, threading, time, urllib.request

url, trace, B, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
mode = sys.argv[5] if len(sys.argv) > 5 else "cont"
N = int(os.environ.get("N_PROBLEMS", 100)); MAX_NEW = int(os.environ.get("MAX_NEW", 512))
STOPS = [x for x in os.environ.get("STOP_STRS", "Q:;;</s>;;<|im_end|>").split(";;") if x]
PROBE = os.environ.get("PROBE", "1") == "1"
rows = [json.loads(l) for l in open(trace)][:N]


def gen(ids, max_new):
    payload = {"input_ids": ids, "sampling_params": {"max_new_tokens": max_new, "temperature": 0, "stop": STOPS}}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        res = json.loads(r.read())
    return res, time.perf_counter() - t0


def run_parallel(items, max_new, width):
    """items: list of (index, ids); at most `width` requests in flight; returns {index: (response, latency)}."""
    results, lock, it = {}, threading.Lock(), iter(items)

    def worker():
        while True:
            with lock:
                nxt = next(it, None)
            if nxt is None:
                return
            i, ids = nxt
            r = gen(ids, max_new)
            with lock:
                results[i] = r
    ths = [threading.Thread(target=worker) for _ in range(width)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return results


# warm-up (not timed): two short requests
for r in rows[:2]:
    gen(r["input"], 8)

items = [(i, r["input"]) for i, r in enumerate(rows)]
waves = []
t0 = time.perf_counter()
if mode == "cont":
    res = run_parallel(items, MAX_NEW, B)
    wall = time.perf_counter() - t0
else:
    res, wall, probe_total = {}, 0.0, 0.0
    for w in range(0, len(items), B):
        wave = items[w:w + B]
        tp = None
        if PROBE:
            tq = time.perf_counter(); run_parallel(wave, 1, len(wave)); tp = time.perf_counter() - tq
            probe_total += tp
        tw = time.perf_counter(); res.update(run_parallel(wave, MAX_NEW, len(wave))); dtw = time.perf_counter() - tw
        wall += dtw
        waves.append({"size": len(wave), "t_wave": dtw, "t_probe": tp})
tot = sum(res[i][0]["meta_info"]["completion_tokens"] for i in res)
outs = [res[i][0]["output_ids"] for i in range(len(rows))]
summary = {"mode": mode, "B": B, "n": len(rows), "tokens": tot, "wall_s": wall, "tok_s": tot / wall,
           "mean_latency_s": sum(res[i][1] for i in res) / len(res), "url": url, "max_new": MAX_NEW, "stops": STOPS}
if mode == "static" and PROBE:
    dec = wall - probe_total
    summary.update(probe_s=probe_total, decode_s_est=dec, decode_tok_s_est=(tot - len(rows)) / dec)
same_trace = sum(int(o == r["output_ids"]) for o, r in zip(outs, rows))
summary["identical_to_trace"] = same_trace
log = os.environ.get("SERVER_LOG")
if log and os.path.exists(log):
    ms = [json.loads(m.group(1)) for l in open(log) for m in [re.search(r"generation_metrics\s+(\{.*\})$", l.strip())] if m]
    ms = [m for m in ms if m.get("decode_tokens", 0) > 8]           # drop the startup / warm-up generations
    if ms:
        summary["server_decode_tokens"] = sum(m["decode_tokens"] for m in ms)
        summary["server_decode_s"] = sum(m["decode_seconds"] for m in ms)
        summary["server_per_request_decode_tok_s"] = summary["server_decode_tokens"] / summary["server_decode_s"]
summary["waves"] = waves
print(json.dumps({k: v for k, v in summary.items() if k != "waves"}), flush=True)
json.dump(summary, open(out, "w"), indent=1)
with open(out.replace(".json", "_outputs.jsonl"), "w") as f:
    for i, o in enumerate(outs):
        f.write(json.dumps({"i": i, "output_ids": o, "latency": res[i][1]}) + "\n")
