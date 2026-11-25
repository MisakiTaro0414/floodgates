"""Train the Raven self-drafter (drafter_raven.RavenSelfDrafter) on the model's own greedy continuations (frozen base; LoRA on
the slot rows, mask embedding, state projection).
  python train_raven.py --data data/gsm8k_train_rl32_greedy.jsonl --out runs/raven/<name> [--steps 6000 --batch 8 --anchors 16 ...]
Outputs in --out: args.json, drafter.pt (trainable params only; also every --save-every steps), train_log.jsonl (every
--log-every steps: loss, grad norm, lr, mean and per-slot accuracy at round 0)."""
import argparse, json, os, random, sys, time
import torch
from floodgates.models.raven_min import RavenMin
from floodgates.drafters.drafter_raven import RavenSelfDrafter, MODEL_DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--model", default=MODEL_DEFAULT)
ap.add_argument("--shift", action="store_true", help="next-token-aligned slots (k+1 drafts per block)")
ap.add_argument("--feedback", action="store_true", help="trained refinement: rounds > 1 take the previous round's drafted tokens as inputs; all rounds_max rounds trained")
ap.add_argument("--prefix-decay", type=float, default=4.0)
ap.add_argument("--iters", type=int, default=1, help="core iterations inside one draft pass")
ap.add_argument("--carry", action="store_true", help="rounds continue the core recurrence from the previous round's latent")
ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--anchors", type=int, default=16); ap.add_argument("--k", type=int, default=8)
ap.add_argument("--rank", type=int, default=32); ap.add_argument("--rounds-max", type=int, default=1)
ap.add_argument("--causal", action="store_true", help="slots attend causally to earlier slots (DAS-Wave-style window) instead of bidirectionally")
ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--warmup", type=int, default=100)
ap.add_argument("--init", default=None, help="run dir whose drafter.pt initialises the drafter")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--log-every", type=int, default=25)
ap.add_argument("--save-every", type=int, default=1000)
ap.add_argument("--plain-rounds", type=int, default=0, help="train the adapter pass THROUGH this many rounds of the model's own recurrence over the drafted window (continuation; loss after every round)")
ap.add_argument("--no-plain-carry", action="store_true", help="plain rounds start from the like-init state instead of the adapter pass's latent")
ap.add_argument("--plain-weight", type=float, default=1.0, help="loss weight of each plain round")
ap.add_argument("--local-target", action="store_true", help="feedback passes are trained toward the model's own full-depth prediction given the provisional prefix (consistent target) instead of the ground-truth token")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)
torch.manual_seed(args.seed); random.seed(args.seed)

rows = []
for l in open(args.data):                      # the file may still be growing: skip a partial last line
    try:
        r = json.loads(l)
    except json.JSONDecodeError:
        continue
    if len(r.get("output_ids", [])) >= 2:
        rows.append(r)
random.shuffle(rows)
print(f"{len(rows)} training sequences", flush=True)

m = RavenMin(args.model)
drafter = RavenSelfDrafter(m, k=args.k, rank=args.rank, shift=args.shift, feedback=args.feedback, iters=args.iters, carry=args.carry, causal=args.causal)
if args.init:
    missing, unexpected = drafter.load_state_dict(torch.load(os.path.join(args.init, "drafter.pt")), strict=False)
    assert not unexpected, unexpected
    print("init from", args.init, "| not in checkpoint:", sorted({n.split(".")[0] for n in missing if not n.startswith("m.")}), flush=True)
params = [p for n, p in drafter.named_parameters() if p.requires_grad and not n.startswith("m.")]
print(f"trainable params {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)
opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) * max(0.1, 1 - s / args.steps))
gen = torch.Generator().manual_seed(args.seed)
log = open(os.path.join(args.out, "train_log.jsonl"), "a" if os.path.exists(os.path.join(args.out, "ckpt.pt")) else "w")


def save():
    sd = {n: p.detach().cpu() for n, p in drafter.named_parameters() if not n.startswith("m.")}
    torch.save(sd, os.path.join(args.out, "drafter.pt"))


t0 = time.time(); ptr = 0
SKIP = 'm.'

def _save_ckpt(step):
    torch.save({"step": step, "drafter": {n: p.detach().cpu() for n, p in drafter.named_parameters() if not n.startswith(SKIP)},
                "opt": opt.state_dict(), "sched": sched.state_dict(), "gen": gen.get_state(), "random": random.getstate(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(), "ptr": ptr, "rows_order": [r["i"] for r in rows] if rows and "i" in rows[0] else None,
                "elapsed": time.time() - t0}, os.path.join(args.out, "ckpt.pt.tmp"))
    os.replace(os.path.join(args.out, "ckpt.pt.tmp"), os.path.join(args.out, "ckpt.pt"))

start = 0; _ck = os.path.join(args.out, "ckpt.pt")
if os.path.exists(_ck):
    ck = torch.load(_ck, weights_only=False)
    missing, unexpected = drafter.load_state_dict(ck["drafter"], strict=False); assert not unexpected, unexpected
    opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); gen.set_state(ck["gen"]); random.setstate(ck["random"])
    torch.set_rng_state(ck["torch_rng"]); torch.cuda.set_rng_state(ck["cuda_rng"]); ptr = ck["ptr"]; start = ck["step"] + 1
    if ck.get("rows_order") is not None:
        by = {r["i"]: r for r in rows if "i" in r}; rows = [by[i] for i in ck["rows_order"] if i in by] or rows
    t0 = time.time() - ck["elapsed"]
    print(f"resumed from {_ck} at step {start}", flush=True)
for step in range(start, args.steps):
    batch = rows[ptr: ptr + args.batch]; ptr += args.batch
    if len(batch) < args.batch:
        random.shuffle(rows); ptr = args.batch; batch = rows[: args.batch]
    seqs = [r["prompt_ids"] + r["output_ids"] for r in batch]
    plens = [len(r["prompt_ids"]) for r in batch]
    loss, logs = drafter.loss(seqs, plens, anchors_per_seq=args.anchors, rounds_max=args.rounds_max, gen=gen, prefix_decay=args.prefix_decay, plain_rounds=args.plain_rounds, plain_carry=not args.no_plain_carry, plain_weight=args.plain_weight, local_target=args.local_target)
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); sched.step()
    if step % args.log_every == 0 or step == args.steps - 1:
        rec = {"step": step, "loss": float(loss), "grad_norm": float(gn), "lr": sched.get_last_lr()[0], "sec": time.time() - t0, **logs}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
    if args.save_every and step and step % args.save_every == 0:
        save()
        _save_ckpt(step)
save()
print("saved", os.path.join(args.out, "drafter.pt"), f"| peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
