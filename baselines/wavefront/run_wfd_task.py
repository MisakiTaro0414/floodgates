"""WaveFront Decoding (Ha & Kim 2026, arXiv 2609.23033; github.com/summerbro-hhj/wavefront-decoding @4345209) on Huginn-0125,
run with THEIR code (unmodified, imported from looped_refs/wavefront-decoding) on our 100 GSM8K prompts.

Per problem: their AR baseline (ar_generate_parcae_blocks, eager, same P/R/C primitives) then their WFD
(generate_wavefront_dynamic, T_total 32, T_draft 4 = paper default), greedy, max 512 new tokens, stop strings 'Q:' and
'</s>' (+ model EOS). The only additions are outside their code: (1) prefill is timed via a HuginnBlocks subclass that
wraps prefill() with cuda syncs, so decode time = their walltime - prefill; (2) stop strings via an object passed as
eos_token_id whose __eq__ sees each committed token once (their loops compare `tok == eos_token_id` per commit).
Real decoding has no CUDA-graph path in their repo (graphs exist only in the forced alpha=1 exp5 harness).

  [S0=ours] python baselines/wavefront/run_wfd_task.py <out.json> <kv_budget_s: 0 = off (lossless) | 4 | 1> [N]
"""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # repository root
import json, os, re, sys, time, glob
from pathlib import Path
from floodgates.paths import WAVEFRONT, LOOPSPEC, PARENT, trace

WF = WAVEFRONT
sys.path.insert(0, WF)
assert os.path.isdir(os.path.join(WF, "src")), f"WaveFront code not found at {WF}: clone github.com/summerbro-hhj/wavefront-decoding (commit 4345209) and set WAVEFRONT_DIR"
import torch  # noqa: E402
from src.decoding.huginn_blocks import HuginnBlocks  # noqa: E402
from src.decoding.policy import SsdPolicy  # noqa: E402
from src.decoding.scheduler import generate_wavefront_dynamic  # noqa: E402
from src.recursive_models.huginn_wrapper import HuginnWrapper  # noqa: E402

S = PARENT
MD = f"{LOOPSPEC}/models/huginn-0125"
out_path, kv_s = sys.argv[1], int(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 100
TASK = os.environ.get("TASK", "gsm8k"); STRIDE = int(os.environ.get("STRIDE", 1))
CODE_STOPS = ("<|endoftext|>", "<|endofmask|>", "</s>", "\nif __name__", "\ndef main(", "\nprint(", "\n# Test", "\nassert ", "\n```")
STOPS, MAX_NEW = {"gsm8k": (("Q:", "</s>"), 512), "math500": (("Problem:",), 2048), "humaneval": (CODE_STOPS, 1024), "mbpp": (CODE_STOPS, 1024), "bbh": (("</s>", "Q", "\n\n"), 1024)}[TASK]
T_TOTAL, T_DRAFT = 32, 4
if os.environ.get("NOSTOP"): STOPS = ()   # NOSTOP=1: their protocol (EOS-only)


class TimedHuginnBlocks(HuginnBlocks):
    last_prefill_s = 0.0

    def prefill(self, input_ids, T_total, T_draft):
        torch.cuda.synchronize(); t = time.perf_counter()
        r = super().prefill(input_ids, T_total, T_draft)
        torch.cuda.synchronize(); self.last_prefill_s = time.perf_counter() - t
        return r


class OurS0Blocks(TimedHuginnBlocks):
    """Diagnostic (S0=ours): replace their seeded s0 draws by OUR engine's per-position like-init table
    (raven_min.init_table, seed 0: row j = randn(cuda gen seeded j) * std, clamped +-3 std, * embed_scale, bf16),
    so their AR can be compared token-for-token with our plain greedy reference."""

    def _table(self):
        if getattr(self, "_tab", None) is None:
            iv = self.config.init_values; d = self.config.n_embd
            rows = [(torch.randn((1, d), device=self.device, generator=torch.Generator(device=self.device).manual_seed(j))
                     * iv["std"]).clamp(-3 * iv["std"], 3 * iv["std"]) * iv["embed_scale"] for j in range(4096)]
            self._tab = torch.cat(rows, 0).to(next(self.model.parameters()).dtype)
        return self._tab

    def _init_state_tokens(self, positions):
        return self._table()[torch.tensor(positions, device=self.device)]

    def _init_state_prefill(self, batch, length):
        return self._table()[:length].unsqueeze(0).expand(batch, -1, -1).contiguous()


class StopCheck:
    """Passed as eos_token_id: `committed_tok == stop` -> reflected __eq__ records the token and checks stop strings."""

    def __init__(self, tok, eos):
        self.tok, self.eos, self.seen = tok, eos, []

    def reset(self):
        self.seen = []

    def __eq__(self, t):
        self.seen.append(int(t))
        if self.eos is not None and t == self.eos:
            return True
        tail = self.tok.decode(self.seen[-12:])
        return any(s in tail for s in STOPS)

    __hash__ = object.__hash__


def answer_of(text):
    m = re.findall(r"answer is\s*\$?(-?[\d,]*\.?\d+)", text.split("Q:")[0])
    return (m[-1] if m else "").replace(",", "")


wrapper = HuginnWrapper.from_pretrained(MD, dtype=torch.bfloat16, device="cuda")
model, tok = wrapper.model, wrapper.tokenizer
S0 = os.environ.get("S0", "theirs")
blocks = (OurS0Blocks if S0 == "ours" else TimedHuginnBlocks)(model, T_total=T_TOTAL, max_seq_len=4096)   # BBH prompts reach 2461 tokens, MATH-500 generates 2048
blocks.state_init, blocks.state_seed = "random", 0          # their default (exp2/exp3 build_blocks)
blocks.kv_budget_s = kv_s or None
policy = SsdPolicy(T_total=T_TOTAL, T_draft=T_DRAFT, mode="greedy")
sampler = policy.make_sampler()
stop = StopCheck(tok, wrapper.eos_token_id)
print(torch.cuda.get_device_name(0), "| kv_budget_s", blocks.kv_budget_s, "| eos", wrapper.eos_token_id, flush=True)

TR = trace(MD, TASK)
rows = [json.loads(l) for l in open(TR)][::STRIDE][:N]
print(f"task {TASK}: {len(rows)} prompts (stride {STRIDE}), max_new {MAX_NEW}, stops {STOPS}", flush=True)
gold = {}
for f in glob.glob(f"{LOOPSPEC}/runs/Ouro-1.4B-gsm8k-n100/baseline/samples_*.jsonl"):
    for l in open(f):
        d = json.loads(l)["doc"]; gold[d["question"].strip()] = d["answer"].split("####")[-1].strip().replace(",", "")

warm = torch.tensor(rows[0]["input"], device="cuda")         # untimed warm-up of both paths
blocks.ar_generate(warm, max_new_tokens=8, sampler=sampler)
generate_wavefront_dynamic(blocks, warm, max_new_tokens=16, policy=policy)

per, agg = [], {m: dict(tok=0, wall=0.0, pre=0.0, correct=0) for m in ("ar", "wfd")}
n_acc = n_prop = 0
for i, r in enumerate(rows):
    ids = torch.tensor(r["input"], device="cuda")
    q = re.findall(r"Q: (.*?)\nA:", tok.decode(r["input"]), re.S)[-1].strip() if TASK == "gsm8k" else None
    rec = {"i": r["i"], "ref_len": len(r.get("output_ids") or [])}
    stop.reset(); torch.cuda.synchronize()
    g_ar, wall_ar = blocks.ar_generate(ids, max_new_tokens=MAX_NEW, eos_token_id=stop, sampler=sampler)
    assert stop.seen == g_ar, "stop checker out of sync (AR)"
    pre_ar = blocks.last_prefill_s
    stop.reset(); torch.cuda.synchronize()
    g_wf, tr = generate_wavefront_dynamic(blocks, ids, max_new_tokens=MAX_NEW, policy=policy, eos_token_id=stop)
    assert stop.seen == g_wf, "stop checker out of sync (WFD)"
    pre_wf = blocks.last_prefill_s
    n_acc += tr.n_drafts_accepted; n_prop += tr.n_drafts_proposed
    for m, g, wall, pre in (("ar", g_ar, wall_ar, pre_ar), ("wfd", g_wf, tr.walltime_s, pre_wf)):
        ok = gold.get(q) is not None and answer_of(tok.decode(g)) == gold[q]
        a = agg[m]; a["tok"] += len(g); a["wall"] += wall; a["pre"] += pre; a["correct"] += int(ok)
        rec.update({f"{m}_len": len(g), f"{m}_wall": wall, f"{m}_prefill": pre, f"{m}_correct": ok})
    rec.update(wfd_eq_ar=g_wf == g_ar, ar_eq_ref=g_ar == r.get("output_ids"), wfd_eq_ref=g_wf == r.get("output_ids"),
               first_diff_wfd_ar=next((k for k, (a, b) in enumerate(zip(g_wf, g_ar)) if a != b), None),
               wfd_tokens=g_wf, ar_tokens=g_ar)
    per.append(rec)
    if i % 10 == 9 or i == len(rows) - 1:
        A, W = agg["ar"], agg["wfd"]
        print(f"{i + 1}/{len(rows)}  AR {A['tok'] / (A['wall'] - A['pre']):.2f} tok/s  WFD {W['tok'] / (W['wall'] - W['pre']):.2f} tok/s"
              f"  WFD==AR {sum(p['wfd_eq_ar'] for p in per)}/{len(per)}  AR==ref {sum(p['ar_eq_ref'] for p in per)}"
              f"  WFD==ref {sum(p['wfd_eq_ref'] for p in per)}  alpha {n_acc / max(n_prop, 1):.3f}", flush=True)

res = {"setup": dict(code="summerbro-hhj/wavefront-decoding@4345209 (unmodified)", model=MD, gpu=torch.cuda.get_device_name(0),
                     T_total=T_TOTAL, T_draft=T_DRAFT, kv_budget_s=kv_s, max_new=MAX_NEW, stops=STOPS, n=len(rows),
                     state_init=("our engine per-position table (seed 0)" if S0 == "ours" else "random seed 0 (theirs)"), attn_backend="torch (their default)", engine="eager (no CUDA graph)")}
for m, a in agg.items():
    res[m] = dict(tokens=a["tok"], decode_s=a["wall"] - a["pre"], prefill_s=a["pre"], wall_s=a["wall"],
                  decode_tok_s=a["tok"] / (a["wall"] - a["pre"]), e2e_tok_s=a["tok"] / a["wall"], accuracy=a["correct"] / len(rows))
res["speedup_decode"] = res["wfd"]["decode_tok_s"] / res["ar"]["decode_tok_s"]
res["speedup_e2e"] = res["wfd"]["e2e_tok_s"] / res["ar"]["e2e_tok_s"]
res["acceptance"] = n_acc / max(n_prop, 1)
res["wfd_eq_ar"] = sum(p["wfd_eq_ar"] for p in per); res["ar_eq_ref"] = sum(p["ar_eq_ref"] for p in per)
res["wfd_eq_ref"] = sum(p["wfd_eq_ref"] for p in per)
res["per_problem"] = per
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(out_path, "w"), indent=1)
print("== RESULT", json.dumps({k: v for k, v in res.items() if k != "per_problem"}), flush=True)
