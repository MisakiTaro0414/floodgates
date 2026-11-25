"""Task accuracy of any decoded output, scored exactly as LoopSpec's lm-eval harness runs (looped_refs/loopspec/runs/<run>/<cfg>/).
Everything is driven by the harness run's own files: the task configs in results_*.json (filter_list, metric_list, process_results),
the per-doc samples_*.jsonl (doc, target, logged metrics) and the 'until' / max_gen_toks of each request.
  gsm8k_cot (also the chat variant): the strict-match / flexible-extract filter pipelines rebuilt with lm-eval's own filter registry
      + lm-eval's exact_match with the task's kwargs -> 'exact_match,strict-match', 'exact_match,flexible-extract' (report flexible)
  minerva_math500 (also chat): lm_eval.tasks.minerva_math.utils.process_results -> 'exact_match,none', 'math_verify,none'
      (report math_verify); its source is checked against the one stored in the results file
  humaneval_plus_custom / mbpp_plus_custom: LoopSpec's evaluations/tasks/utils.py pass_at_1, called per doc as lm-eval does
      (references=[target], predictions=[response]) -> 'pass_at_1,none'.  Executes model code: only with ACC_ALLOW_CODE=1 (a CPU pod),
      in worker subprocesses, with a persistent cache keyed by sha1(target, response) (runs/accuracy_cache/code_exec.jsonl)
  bbh_cot_fewshot_*: the 'get-answer' filter ((?<=the answer is )(.*)(?=.), take_first) + exact_match -> 'exact_match,get-answer'
Rows are aligned to the harness samples by prompt text (as score_outputs.py): a trace row's 'context', or its 'input' ids decoded
without special tokens (Huginn traces: BOS dropped).  Token ids -> response text as the SGLang server + lm-eval produce it:
keep at most max_gen_toks ids, cut at the first EOS id (special tokens among tokenizer / config / generation-config EOS),
decode with special tokens kept (the Thinking models' <think> tags) and with the last 5 prompt tokens as left context (SGLang's
incremental detokenizer; keeps the leading space of SentencePiece outputs, TinyLlama), cut at the first 'until' string.  The server's own texts
additionally keep a partial stop string when the stop completes on a 50-token detokenizer boundary (e.g. '...\n' for the stop
'\n\n'); pass_at_1 strips such suffixes and the regex filters ignore them, so the scores agree (checked by 'validate').
  python score_any.py validate [--code] [<harness_cfg_dir> ...]   (default: every harness run dir; writes paper/accuracy_validation.json)
  python score_any.py outputs <outputs.jsonl> <harness_baseline_dir> <trace.jsonl> [sides=plain,spec]
  python score_any.py _code_worker <in.jsonl> <out.jsonl>          (internal)"""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import glob, hashlib, inspect, json, os, re, subprocess, sys, tempfile, time
import os
from floodgates.paths import LOOPSPEC, PARENT, RUNS, ROOT

S = PARENT; L = f"{LOOPSPEC}"; LR = f"{L}/runs"; MD = f"{L}/models"
HERE = ROOT   # repository root
CACHE_DIR = os.environ.get("ACC_CACHE_DIR", f"{RUNS}/accuracy_cache")
PRIMARY = {"gsm8k_cot": "exact_match,flexible-extract", "minerva_math500": "math_verify,none", "humaneval_plus_custom": "pass_at_1,none",
           "mbpp_plus_custom": "pass_at_1,none", "bbh": "exact_match,get-answer"}
CODE_TASKS = ("humaneval_plus_custom", "mbpp_plus_custom")
SGLANG_DETOK_CTX = 5        # sglang/srt/managers/schedule_batch.py INIT_INCREMENTAL_DETOKENIZATION_OFFSET


def task_family(name):
    return "bbh" if name.startswith("bbh_") else name


# ---- tokenizers -------------------------------------------------------------------------------------------------------------
_TOK = {}


def model_of(run_name):
    """Model directory name of a harness run name (longest model name that prefixes it)."""
    names = [m for m in os.listdir(MD) if run_name == m or run_name.startswith(m + "-")]
    return max(names, key=len) if names else None


def tokenizer(model):
    """(tokenizer, EOS ids) for a model dir name.  Ouro: AutoTokenizer; Raven family: PreTrainedTokenizerFast(tokenizer.json)."""
    if model in _TOK:
        return _TOK[model]
    import warnings
    warnings.filterwarnings("ignore")
    from transformers import AutoTokenizer, PreTrainedConfig, PreTrainedTokenizerFast
    d = f"{MD}/{model}"
    tok = (AutoTokenizer.from_pretrained(d, config=PreTrainedConfig()) if model.startswith("Ouro")      # as the harness loads it
           else PreTrainedTokenizerFast(tokenizer_file=f"{d}/tokenizer.json"))                            # (no custom model config)
    tc = json.load(open(f"{d}/tokenizer_config.json")); cfg = json.load(open(f"{d}/config.json"))
    gc = json.load(open(f"{d}/generation_config.json")) if os.path.exists(f"{d}/generation_config.json") else {}
    special = {int(k) for k, v in tc.get("added_tokens_decoder", {}).items() if v.get("special")}
    e = tc.get("eos_token"); e = e["content"] if isinstance(e, dict) else e
    cand = set()
    if e:
        cand.add(tok.convert_tokens_to_ids(e))
    for x in (cfg.get("eos_token_id"), gc.get("eos_token_id")):
        cand.update(x if isinstance(x, list) else [x] if x is not None else [])
    eos = {i for i in cand if i is not None and (i in special or tok.convert_ids_to_tokens(i) == e)}   # Recurrent-Llama's config
    _TOK[model] = (tok, eos)                                                                           # eos 65505 is ' creek'
    return _TOK[model]


def cut_until(text, until):
    return text[:min([text.find(s) for s in until if s in text] + [len(text)])]


# ---- one harness run ---------------------------------------------------------------------------------------------------------
class Harness:
    """A harness run directory (<run>/<cfg>): its samples (by prompt text), task configs, trace, tokenizer."""
    _cache = {}

    @classmethod
    def get(cls, d):
        d = os.path.normpath(d)
        if d not in cls._cache:
            cls._cache[d] = cls(d)
        return cls._cache[d]

    def __init__(self, d):
        self.dir = d; self.run = os.path.basename(os.path.dirname(d)); self.cfg = os.path.basename(d)
        self.model = model_of(self.run)
        res = sorted(glob.glob(f"{d}/results_*.json"))
        assert res, f"no results file in {d}"
        self.configs = json.load(open(res[-1]))["configs"]
        self.by_prompt = {}; self.docs = []
        for f in sorted(glob.glob(f"{d}/samples_*.jsonl")):
            sub = re.match(r"samples_(.+)_\d{4}-\d\d-\d\dT", os.path.basename(f)).group(1)
            per = {}
            for l in open(f):
                r = json.loads(l); per.setdefault(r["doc_id"], []).append(r)
            for doc_id, recs in sorted(per.items()):
                r0 = recs[0]; prompt = r0["arguments"]["gen_args_0"]["arg_0"]; gk = r0["arguments"]["gen_args_0"]["arg_1"]
                logged = {f"{m},{r.get('filter', 'none')}": float(r[m]) for r in recs for m in r["metrics"]}
                row = {"task": sub, "doc_id": doc_id, "doc": r0["doc"], "target": r0["target"], "prompt": prompt, "resp": r0["resps"][0][0],
                       "until": gk.get("until", []), "max_gen_toks": gk.get("max_gen_toks"), "logged": logged}
                self.docs.append(row); self.by_prompt.setdefault(prompt, []).append(row)
        self.dups = sum(len(v) - 1 for v in self.by_prompt.values())
        self.trace = [json.loads(l) for l in open(f"{d}/trace.jsonl")] if os.path.exists(f"{d}/trace.jsonl") else []

    def lookup(self, prompt):
        rows = self.by_prompt.get(prompt)
        if rows is None:
            cands = [k for k in self.by_prompt if k.strip() == prompt.strip()]
            rows = self.by_prompt[cands[0]] if len(cands) == 1 else None
        return rows[0] if rows else None

    def tok(self):
        return tokenizer(self.model)

    def response(self, ids, row, prompt_ids=None):
        """Token ids -> the response text lm-eval scores (see the module docstring).  With the prompt ids, decoded as SGLang's
        incremental detokenizer does (the last 5 prompt tokens as left context: keeps SentencePiece's leading space, TinyLlama)."""
        tok, eos = self.tok()
        ids = list(ids)
        if row["max_gen_toks"]:
            ids = ids[:row["max_gen_toks"]]
        for k, t in enumerate(ids):
            if t in eos:
                ids = ids[:k]; break
        if prompt_ids:
            ctx = list(prompt_ids[-SGLANG_DETOK_CTX:])
            text = tok.decode(ctx + ids, skip_special_tokens=False)[len(tok.decode(ctx, skip_special_tokens=False)):]
        else:
            text = tok.decode(ids, skip_special_tokens=False)
        return cut_until(text, row["until"])

    def prompt_of(self, trace_row, tok=None):
        if "context" in trace_row:
            return trace_row["context"]
        tok = tok or self.tok()[0]
        return tok.decode(trace_row["input"], skip_special_tokens=True)

    def align(self, trace_rows, tok=None):
        """Sample rows for engine trace rows (by prompt text); None where no sample matches."""
        return [self.lookup(self.prompt_of(r, tok)) for r in trace_rows]


# ---- metrics ----------------------------------------------------------------------------------------------------------------
_FILTERS = {}


def _filters(h, task):
    key = (h.dir, task)
    if key not in _FILTERS:
        from lm_eval.filters import build_filter_ensemble
        from lm_eval.api.registry import get_metric
        c = h.configs[task]
        if c.get("filter_list"):
            fl = [(f["name"], build_filter_ensemble(f["name"], [[g["function"], {k: v for k, v in g.items() if k != "function"}] for g in f["filter"]]))
                  for f in c["filter_list"]]
        else:
            fl = [("none", build_filter_ensemble("none", [["take_first", None]]))]
        ms = []
        for m in c["metric_list"]:
            name = m["metric"]
            if not name.startswith("def "):
                ms.append((name, get_metric(name), {k: v for k, v in m.items() if k not in ("metric", "aggregation", "higher_is_better")}))
        _FILTERS[key] = (fl, ms)
    return _FILTERS[key]


def _math_process_results(h, task):
    from lm_eval.tasks.minerva_math import utils as mu
    src = h.configs[task].get("process_results")
    assert src is None or src.strip() == inspect.getsource(mu.process_results).strip(), "minerva_math process_results differs from the harness"
    return mu.process_results


def score_text_nocode(h, row, text):
    """Per-doc metrics of one response text for a non-code task (keys as lm-eval's results: '<metric>,<filter>')."""
    task = row["task"]; c = h.configs[task]
    if c.get("process_results"):
        if "minerva_math" in task or task.startswith("minerva"):
            r = _math_process_results(h, task)(row["doc"], [text])
            return {f"{k},none": float(v) for k, v in r.items()}
        raise NotImplementedError(f"process_results task {task}")
    fl, ms = _filters(h, task)
    out = {}
    for fname, ens in fl:
        resps = [[text]]
        for f in ens.filters:
            resps = list(f().apply(resps, [row["doc"]]))
        result = resps[0]
        gold = row["target"]
        if type(gold) is not type(result) and not isinstance(result, list):
            gold = type(result)(gold)
        for mname, fn, kw in ms:
            v = fn(references=[gold], predictions=[result], **kw)
            out[f"{mname},{fname}"] = float(v[mname] if isinstance(v, dict) else v)
    return out


class CodeExec:
    """pass_at_1 per (target, response), as lm-eval calls it per doc; persistent cache; subprocess workers.  Refuses to run model
    code unless ACC_ALLOW_CODE=1 (set by the CPU-pod script)."""

    def __init__(self, workers=None):
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.path = f"{CACHE_DIR}/code_exec.jsonl"; self.cache = {}; self.load = {}
        if os.path.exists(self.path):
            for l in open(self.path):
                try:
                    r = json.loads(l); self.cache[r["key"]] = r["pass"]; self.load[r["key"]] = r.get("w", 99)
                except Exception:
                    pass
        self.workers = workers or int(os.environ.get("ACC_CODE_WORKERS", 8))
        self.retry_workers = int(os.environ.get("ACC_CODE_RETRY_WORKERS", 4))
        self.allowed = os.environ.get("ACC_ALLOW_CODE") == "1"
        self.recover()

    def recover(self):
        """Results of workers of an interrupted run (acc_code_*/out*.jsonl) -> the cache."""
        new = []
        for fo in glob.glob(f"{CACHE_DIR}/acc_code_*/out*.jsonl"):
            for l in open(fo):
                try:
                    r = json.loads(l)
                except Exception:
                    continue
                if r["key"] not in self.cache:
                    self.cache[r["key"]] = r["pass"]; self.load[r["key"]] = 99; new.append(r)
        if new and self.allowed:                                       # only the code-running process writes the cache
            with open(self.path, "a") as fc:
                for r in new:
                    fc.write(json.dumps({"key": r["key"], "pass": r["pass"], "w": 99}) + "\n")
            print(f"[code] recovered {len(new)} results of an interrupted run", flush=True)

    @staticmethod
    def key(ref, pred):
        return hashlib.sha1((ref + "\x00\x01\x02" + pred).encode()).hexdigest()

    def get(self, ref, pred):
        return self.cache.get(self.key(ref, pred))

    def run(self, pairs):
        """pairs: [(ref, pred)] -> fills the cache for the missing ones (if allowed).  Every failure is then re-executed at low
        concurrency (ACC_CODE_RETRY_WORKERS, 4), and that result is kept: the harness ran one program at a time, and under a
        24-way load timing-sensitive EvalPlus tests (HumanEval/15, Mbpp/160 ...) exceed the 5 s timeout."""
        todo = {}
        for ref, pred in pairs:
            k = self.key(ref, pred)
            if k not in self.cache:
                todo[k] = (ref, pred)
        if not self.allowed:
            return len(todo)
        if todo:
            self._exec(list(todo.items()), self.workers)
        retry = {}
        for ref, pred in pairs:
            k = self.key(ref, pred)
            if self.cache.get(k) == 0.0 and self.load.get(k, 99) > self.retry_workers:
                retry[k] = (ref, pred)
        if retry:
            print(f"[code] re-executing {len(retry)} failed programs at low load", flush=True)
            self._exec(list(retry.items()), self.retry_workers)
        return sum(1 for ref, pred in pairs if self.key(ref, pred) not in self.cache)

    def _exec(self, items, workers):
        n = len(items); W = min(workers, n)
        print(f"[code] executing {n} programs with {W} workers", flush=True)
        tmp = tempfile.mkdtemp(prefix="acc_code_", dir=CACHE_DIR); procs = []
        for w in range(W):
            fi = f"{tmp}/in{w}.jsonl"
            with open(fi, "w") as f:
                for k, (ref, pred) in items[w::W]:
                    f.write(json.dumps({"key": k, "ref": ref, "pred": pred}) + "\n")
            procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), "_code_worker", fi, f"{tmp}/out{w}.jsonl"]))
        t0 = time.time()
        for p in procs:
            p.wait()
        done = 0
        with open(self.path, "a") as fc:
            for w in range(W):
                fo = f"{tmp}/out{w}.jsonl"
                if not os.path.exists(fo):
                    continue
                for l in open(fo):
                    r = json.loads(l); self.cache[r["key"]] = r["pass"]; self.load[r["key"]] = W
                    fc.write(json.dumps({"key": r["key"], "pass": r["pass"], "w": W}) + "\n"); done += 1
        print(f"[code] {done}/{n} programs scored in {time.time() - t0:.0f}s", flush=True)
        if done == n:
            import shutil; shutil.rmtree(tmp, ignore_errors=True)


def code_worker(fi, fo):
    assert os.environ.get("ACC_ALLOW_CODE") == "1", "code execution is only allowed with ACC_ALLOW_CODE=1 (CPU pod)"
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    os.environ["HF_METRICS_CACHE"] = tempfile.mkdtemp(prefix="acc_metrics_")   # evaluate's per-call arrow/lock files: local disk, not NFS
    done = set()
    if os.path.exists(fo):                                            # resume
        done = {json.loads(l)["key"] for l in open(fo) if l.strip()}
    sys.path.insert(0, L)
    from evaluations.tasks.utils import pass_at_1
    with open(fo, "a") as f:
        for l in open(fi):
            r = json.loads(l)
            if r["key"] in done:
                continue
            try:
                v = float(pass_at_1([r["ref"]], [r["pred"]]))
            except Exception as e:                          # lm-eval would crash; record as a failure and say so
                print(f"[code worker] {r['key']}: {type(e).__name__}: {e}", flush=True); v = 0.0
            f.write(json.dumps({"key": r["key"], "pass": v}) + "\n"); f.flush()


_CODE = None


def code_exec():
    global _CODE
    if _CODE is None:
        _CODE = CodeExec()
    return _CODE


_VERDICTS = None


def harness_verdicts():
    """sha1(target, response) -> lm-eval's own pass_at_1 for every program a harness run scored (max over runs: on HumanEval/15 the
    harness itself gives the identical program both verdicts).  Four CPU-bound EvalPlus problems (HumanEval/15, Mbpp/160, 271, 392)
    pass within the 5 s timeout on the harness node but time out on the CPU pods (V100 nodes), so programs identical to a harness
    response take the harness verdict; only novel programs are executed.  Not used by 'validate' (raw execution there)."""
    global _VERDICTS
    if _VERDICTS is None:
        _VERDICTS = {}
        for d in all_harness_dirs():
            if any(t in d for t in ("humaneval", "mbpp")):
                for r in Harness.get(d).docs:
                    k = CodeExec.key(r["target"], r["resp"]); _VERDICTS[k] = max(_VERDICTS.get(k, 0.0), r["logged"]["pass_at_1,none"])
    return _VERDICTS


def score_texts(h, rows, texts, run_code=True, verdicts=None):
    """Per-doc metric dicts for (sample row, response text) pairs; code tasks need the code cache (None where not yet executed).
    verdicts (harness_verdicts()): lm-eval's verdict for programs the harness scored ('_exec' keeps our execution's)."""
    out = [None] * len(rows)
    code = [(i, r, t) for i, (r, t) in enumerate(zip(rows, texts)) if r is not None and r["task"] in CODE_TASKS]
    if code:
        ce = code_exec()
        if run_code:
            ce.run([(r["target"], t) for _, r, t in code])
        for i, r, t in code:
            v = ce.get(r["target"], t); k = CodeExec.key(r["target"], t)
            if verdicts is not None and k in verdicts:
                out[i] = {"pass_at_1,none": float(verdicts[k]), "_exec": v}
            else:
                out[i] = None if v is None else {"pass_at_1,none": float(v)}
    for i, (r, t) in enumerate(zip(rows, texts)):
        if r is not None and r["task"] not in CODE_TASKS:
            out[i] = score_text_nocode(h, r, t)
    return out


def primary_key(row):
    return PRIMARY[task_family(row["task"])]


# ---- engine outputs -----------------------------------------------------------------------------------------------------------
def score_ids(h, rows, id_lists, run_code=True, prompts=None, verdicts=None):
    """Responses + per-doc metrics for token-id lists aligned to sample rows (prompts: the prompt ids, for the decode context)."""
    prompts = prompts or [None] * len(rows)
    texts = [h.response(ids, r, p) if r is not None else None for ids, r, p in zip(id_lists, rows, prompts)]
    return texts, score_texts(h, rows, texts, run_code, verdicts)


# ---- validation ---------------------------------------------------------------------------------------------------------------
def validate_dir(d, run_code):
    """V1: rescoring the harness's own responses (resps) reproduces lm-eval's per-doc metrics.  V2: our id->text procedure on the
    trace's output_ids, scored, reproduces them too (and how often the text itself is identical)."""
    h = Harness.get(d)
    rows = h.docs
    out = {"dir": d.split("/runs/")[-1], "model": h.model, "n": len(rows), "dup_prompts": h.dups}
    sc = score_texts(h, rows, [r["resp"] for r in rows], run_code)
    mism = {}; pend = 0
    for r, s in zip(rows, sc):
        if s is None:
            pend += 1; continue
        for k, v in r["logged"].items():
            if k in s and abs(s[k] - v) > 1e-9:
                mism.setdefault(k, []).append(r["doc_id"] if "bbh" not in r["task"] else f"{r['task']}:{r['doc_id']}")
            elif k not in s:
                mism.setdefault(k + " (missing)", []).append(r["doc_id"])
    out["v1_pending"] = pend; out["v1_mismatch"] = {k: len(v) for k, v in mism.items()}; out["v1_mismatch_docs"] = {k: v[:10] for k, v in mism.items()}
    out["v1_metrics"] = sorted({k for r in rows for k in r["logged"]})
    if h.trace:
        al = h.align(h.trace)
        ok = [(t, r) for t, r in zip(h.trace, al) if r is not None]
        out["v2_unaligned"] = len(h.trace) - len(ok)
        texts, sc2 = score_ids(h, [r for _, r in ok], [t["output_ids"] for t, _ in ok], run_code, [t["input"] for t, _ in ok])
        out["v2_text_identical"] = sum(x == r["resp"] for x, (_, r) in zip(texts, ok))
        mm = {}; pend2 = 0
        for (t, r), s in zip(ok, sc2):
            if s is None:
                pend2 += 1; continue
            for k, v in r["logged"].items():
                if k in s and abs(s[k] - v) > 1e-9:
                    mm.setdefault(k, []).append(r["doc_id"])
        out["v2_pending"] = pend2; out["v2_mismatch"] = {k: len(v) for k, v in mm.items()}; out["v2_mismatch_docs"] = {k: v[:10] for k, v in mm.items()}
    return out


def all_harness_dirs():
    ds = []
    for run in sorted(os.listdir(LR)):
        if "INVALID" in run or not os.path.isdir(f"{LR}/{run}") or model_of(run) is None:
            continue
        for cfg in sorted(os.listdir(f"{LR}/{run}")):
            d = f"{LR}/{run}/{cfg}"
            if glob.glob(f"{d}/results_*.json") and glob.glob(f"{d}/samples_*.jsonl"):
                ds.append(d)
    return ds


if __name__ == "__main__":
    if sys.argv[1] == "_code_worker":
        code_worker(sys.argv[2], sys.argv[3]); sys.exit(0)
    if sys.argv[1] == "validate":
        args = [a for a in sys.argv[2:] if not a.startswith("--")]; run_code = "--code" in sys.argv
        dirs = args or all_harness_dirs()
        is_code = lambda d: any(t in d for t in ("humaneval", "mbpp"))
        if not run_code:
            dirs = [d for d in dirs if not is_code(d)]
        else:                                                           # code: the code dirs, all programs first in one parallel batch
            dirs = [d for d in dirs if is_code(d)]
            pairs = []
            for d in dirs:
                if is_code(d):
                    h = Harness.get(d)
                    pairs += [(r["target"], r["resp"]) for r in h.docs]
                    if h.trace:
                        al = h.align(h.trace)
                        pairs += [(r["target"], h.response(t["output_ids"], r, t["input"])) for t, r in zip(h.trace, al) if r is not None]
            code_exec().run(pairs)
        rep = []
        for d in dirs:
            try:
                r = validate_dir(d, run_code)
            except Exception as e:
                r = {"dir": d.split("/runs/")[-1], "error": f"{type(e).__name__}: {e}"}
            rep.append(r)
            print(json.dumps({k: v for k, v in r.items() if not k.endswith("_docs") and k != "v1_metrics"}), flush=True)
        out = os.environ.get("ACC_VALIDATION_OUT", f"{RUNS}/accuracy_validation{'_code' if run_code else ''}.json")
        old = json.load(open(out)) if os.path.exists(out) else {}
        old.update({r["dir"]: r for r in rep}); json.dump(old, open(out, "w"), indent=1)
        print("wrote", out)
    elif sys.argv[1] == "outputs":
        f, hd, tr = sys.argv[2], sys.argv[3], sys.argv[4]; sides = (sys.argv[5] if len(sys.argv) > 5 else "plain,spec").split(",")
        h = Harness.get(hd); trace = [json.loads(l) for l in open(tr)]; outs = [json.loads(l) for l in open(f)]
        rows = h.align(trace[:len(outs)])
        for side in sides:
            texts, sc = score_ids(h, rows, [o[side] for o in outs], True, [t["input"] for t in trace[:len(outs)]], harness_verdicts())
            vals = [s[primary_key(r)] for r, s in zip(rows, sc) if s is not None]
            print(f"{side}: {sum(vals) / max(len(vals), 1):.4f} over {len(vals)} of {len(outs)} ({sum(r is None for r in rows)} unaligned)")
