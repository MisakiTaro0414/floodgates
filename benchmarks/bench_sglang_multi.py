"""Wall-clock throughput of N independent single-request SGLang servers on one GPU (LoopSpec's best showing under
concurrent load: its server serves one request at a time, so concurrency = N server processes).
Same prompts (exact token ids of a LoopSpec trace) and lm-eval request payload as bench_sglang_batch.py (greedy,
max_new_tokens, stop strings). B = N client threads; thread t always talks to server t % N (round-robin over the
servers, so every server has exactly one request in flight) and pulls the next prompt from one shared queue.
  python bench_sglang_multi.py --urls u0,u1,.. --logs l0,l1,.. --trace trace.jsonl --out out.json
         [--ls-trace loopspec_trace.jsonl] [--ref single_outputs.jsonl ...] [--mps 0|1] [--B B]
env: N_PROBLEMS (100), MAX_NEW (512), STOP_STRS ("Q:;;</s>;;<|im_end|>", separator ";;") as in bench_sglang_batch.py;
     KV_TOKENS / CTX_LEN / MEMFRAC are only recorded (server flags set by runs/batch/sglang_multi.sh).
Server-side decode: generation_metrics lines written to each server log during the timed run (byte offset taken after
warm-up; decode_tokens > 8 as in bench_sglang_batch.py); per server decode_tokens / decode_seconds, summed over servers
(server_decode_tok_s_sum) and pooled (server_per_request_decode_tok_s = sum tokens / sum seconds).
identical_to_single: requests whose output_ids equal those of the single-server run (--ref, first existing file that is
not this run's own output; none -> this is the single-server run, all identical by definition)."""
import argparse, json, os, re, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--urls", required=True)
ap.add_argument("--logs", default="")
ap.add_argument("--trace", required=True)
ap.add_argument("--ls-trace", default="")
ap.add_argument("--ref", nargs="*", default=[])
ap.add_argument("--out", required=True)
ap.add_argument("--mps", type=int, default=0)
ap.add_argument("--B", type=int, default=0)
a = ap.parse_args()
urls = a.urls.split(",")
logs = [x for x in a.logs.split(",") if x]
NS = len(urls)
B = a.B or NS
N = int(os.environ.get("N_PROBLEMS", 100)); MAX_NEW = int(os.environ.get("MAX_NEW", 512))
STOPS = [x for x in os.environ.get("STOP_STRS", "Q:;;</s>;;<|im_end|>").split(";;") if x]
rows = [json.loads(l) for l in open(a.trace)][:N]
out_jsonl = a.out.replace(".json", "_outputs.jsonl")


def gen(url, ids, max_new):
    payload = {"input_ids": ids, "sampling_params": {"max_new_tokens": max_new, "temperature": 0, "stop": STOPS}}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        res = json.loads(r.read())
    return res, time.perf_counter() - t0


# warm-up (not timed): two short requests per server
for u in urls:
    for r in rows[:2]:
        gen(u, r["input"], 8)
offsets = [os.path.getsize(l) if os.path.exists(l) else 0 for l in logs]

results, errors, lock = {}, {}, threading.Lock()
it = iter(list(enumerate(rows)))
t_first = {}


def worker(t):
    url = urls[t % NS]
    while True:
        with lock:
            nxt = next(it, None)
        if nxt is None:
            return
        i, r = nxt
        ts = time.perf_counter()
        try:
            res, lat = gen(url, r["input"], MAX_NEW)
        except Exception as e:                      # a dead server: record, stop this thread
            with lock:
                errors[i] = f"{url}: {e!r}"
            return
        with lock:
            results[i] = (res, lat, url, ts)


load0 = open("/proc/loadavg").read().split()[:3]
t0 = time.perf_counter()
ths = [threading.Thread(target=worker, args=(t,)) for t in range(B)]
for th in ths:
    th.start()
for th in ths:
    th.join()
wall = time.perf_counter() - t0
load1 = open("/proc/loadavg").read().split()[:3]
time.sleep(2)                                       # let the servers flush their generation_metrics lines

ok = sorted(results)
tot = sum(results[i][0]["meta_info"]["completion_tokens"] for i in ok)
lats = sorted(results[i][1] for i in ok)
outs = {i: results[i][0]["output_ids"] for i in ok}


def same(ref_outs):
    return sum(int(i in ref_outs and outs[i] == ref_outs[i]) for i in ok)


ref_path = next((p for p in a.ref if os.path.exists(p) and os.path.abspath(p) != os.path.abspath(out_jsonl)), None)
if ref_path:
    ref = {d["i"]: d["output_ids"] for d in map(json.loads, open(ref_path))}
    ident_single = same(ref)
else:
    ident_single = len(ok)                          # this run is the single-server reference
summary = {"mode": "multi", "n_servers": NS, "B": B, "mps": bool(a.mps), "n": len(rows), "n_ok": len(ok),
           "n_failed": len(errors), "errors": errors, "tokens": tot, "wall_s": wall, "tok_s": tot / wall,
           "mean_latency_s": sum(lats) / max(len(lats), 1),
           "p50_latency_s": lats[len(lats) // 2] if lats else None,
           "p90_latency_s": lats[int(0.9 * (len(lats) - 1))] if lats else None,
           "identical_to_single": ident_single, "single_ref": ref_path or "self",
           "identical_to_trace": same({i: r["output_ids"] for i, r in enumerate(rows)})}
if a.ls_trace and os.path.exists(a.ls_trace):
    lsr = [json.loads(l) for l in open(a.ls_trace)][:N]
    summary["identical_to_ls_trace"] = same({i: r["output_ids"] for i, r in enumerate(lsr)})
    summary["ls_trace"] = a.ls_trace
summary.update(urls=urls, max_new=MAX_NEW, stops=STOPS, trace=a.trace,
               kv_tokens=os.environ.get("KV_TOKENS"), context_length=os.environ.get("CTX_LEN"),
               mem_fraction_static=os.environ.get("MEMFRAC"), loadavg_before=load0, loadavg_after=load1)

per = []
for k, u in enumerate(urls):
    mine = [i for i in ok if results[i][2] == u]
    d = {"url": u, "n_requests": len(mine), "tokens": sum(results[i][0]["meta_info"]["completion_tokens"] for i in mine)}
    if k < len(logs) and os.path.exists(logs[k]):
        with open(logs[k], "rb") as f:
            f.seek(offsets[k])
            text = f.read().decode(errors="replace")
        ms = [json.loads(m.group(1)) for l in text.splitlines()
              for m in [re.search(r"generation_metrics\s+(\{.*\})$", l.strip())] if m]
        ms = [m for m in ms if m.get("decode_tokens", 0) > 8]
        if ms:
            d.update(log=logs[k], n_metrics=len(ms), decode_tokens=sum(m["decode_tokens"] for m in ms),
                     decode_s=sum(m["decode_seconds"] for m in ms))
            d["decode_tok_s"] = d["decode_tokens"] / d["decode_s"]
            for key in ("first_accept", "first_reject", "second_accept", "second_reject", "pipeline_cycles"):
                d[key] = sum(m.get(key, 0) for m in ms)
    per.append(d)
have = [d for d in per if "decode_tokens" in d]
if have:
    summary["server_decode_tokens"] = sum(d["decode_tokens"] for d in have)
    summary["server_decode_s"] = sum(d["decode_s"] for d in have)
    summary["server_per_request_decode_tok_s"] = summary["server_decode_tokens"] / summary["server_decode_s"]
    summary["server_decode_tok_s_sum"] = sum(d["decode_tok_s"] for d in have)
summary["per_server"] = per
print(json.dumps({k: v for k, v in summary.items() if k not in ("per_server", "urls", "errors")}), flush=True)
json.dump(summary, open(a.out, "w"), indent=1)
with open(out_jsonl, "w") as f:
    for i in ok:
        f.write(json.dumps({"i": i, "output_ids": outs[i], "latency": results[i][1], "server": results[i][2],
                            "t_start": results[i][3] - t0}) + "\n")
