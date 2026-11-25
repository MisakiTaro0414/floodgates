"""Train the looped-flow drafter on Ouro's own greedy continuations (frozen Ouro; LoRA on slots, time MLP, state proj).
  python train.py --data data/gsm8k_train_ouro14_greedy.jsonl --out runs/<name> [--steps 2000 --batch 8 --anchors 16 ...]
Control (--ext-layers N > 0): the external drafter of drafter_ext.py (N fresh layers, same data/anchors/loss)."""
import argparse, json, os, random, sys, time
import torch
import os
from floodgates.paths import LOOPSPEC
from floodgates.models.ouro_min import OuroMin
from floodgates.drafters.drafter import LoopFlowDrafter, loss_commit, loss_gumbel
from floodgates.drafters.drafter_ext import ExtDrafter

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--model", default=LOOPSPEC + '/models/Ouro-1.4B')
ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--anchors", type=int, default=16); ap.add_argument("--k", type=int, default=8)
ap.add_argument("--rank", type=int, default=32); ap.add_argument("--kappa", type=float, default=4.0)
ap.add_argument("--prefix-decay", type=float, default=4.0, help="loss weights w_j ~ exp(-(j-1)/prefix_decay) along the block")
ap.add_argument("--rounds-max", type=int, default=3); ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--no-state", action="store_true"); ap.add_argument("--no-noise", action="store_true")
ap.add_argument("--commit", action="store_true"); ap.add_argument("--flowmap", action="store_true")
ap.add_argument("--shift", action="store_true", help="next-token-aligned slots (k+1 drafts per block)")
ap.add_argument("--gumbel", action="store_true", help="shared-noise coupled drafter (requires --no-noise); --uncoupled = control")
ap.add_argument("--uncoupled", action="store_true")
ap.add_argument("--acc-weight", action="store_true", help="expected-acceptance loss weighting instead of exponential prefix decay")
ap.add_argument("--feedback", action="store_true", help="trained refinement: rounds > 1 take the previous round's drafted tokens as inputs; all rounds_max rounds trained")
ap.add_argument("--markov-rank", type=int, default=0, help="> 0: low-rank Markov bias on slot logits from the previous slot's token")
ap.add_argument("--ctx-loop", type=int, default=None, help="loop whose K/V the slots read as context (default: final loop; 0 = first loop, ablation)")
ap.add_argument("--save-every", type=int, default=0, help="also save drafter_step<N>.pt every N steps (training curve)")
ap.add_argument("--ext-layers", type=int, default=0, help="control: external drafter with N fresh layers (0 = our self-drafter)")
ap.add_argument("--ext-init", default="random", choices=["random", "copy"], help="external drafter layers: fresh N(0, 0.02) or Ouro's first N layers")
ap.add_argument("--init", default=None, help="run dir whose drafter.pt initialises the drafter (new modules stay zero-init)"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--log-every", type=int, default=25)
args = ap.parse_args()
assert not args.gumbel or args.no_noise, "--gumbel uses the mask input in round 1 (pass --no-noise)"
os.makedirs(args.out, exist_ok=True)
json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)
torch.manual_seed(args.seed); random.seed(args.seed)

rows = [json.loads(l) for l in open(args.data)]
rows = [r for r in rows if r.get("output_ids")]
random.shuffle(rows)
print(f"{len(rows)} training sequences", flush=True)

ouro = OuroMin(args.model)
if args.ext_layers > 0:
    assert not (args.gumbel or args.commit or args.flowmap), "the external drafter is a plain one-pass drafter"
    drafter = ExtDrafter(ouro, k=args.k, n_layers=args.ext_layers, init=args.ext_init)
else:
    drafter = LoopFlowDrafter(ouro, k=args.k, rank=args.rank, kappa=args.kappa, use_state=not args.no_state, use_noise=not args.no_noise, flowmap=args.flowmap, shift=args.shift, ctx_loop=args.ctx_loop, feedback=args.feedback, markov_rank=args.markov_rank)
if args.init:
    missing, unexpected = drafter.load_state_dict(torch.load(os.path.join(args.init, "drafter.pt")), strict=False)
    assert not unexpected, unexpected
    print("init from", args.init, "| not in checkpoint:", sorted({m.split(".")[0] for m in missing if not m.startswith("ouro.")}), flush=True)
params = [p for n, p in drafter.named_parameters() if p.requires_grad and not n.startswith("ouro.")]
print(f"trainable params {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)
opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * max(0.1, 1 - s / args.steps))
gen = torch.Generator().manual_seed(args.seed)
log = open(os.path.join(args.out, "train_log.jsonl"), "a" if os.path.exists(os.path.join(args.out, "ckpt.pt")) else "w")
t0 = time.time(); ptr = 0
SKIP = 'ouro.'

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
    if args.gumbel:
        loss, logs = loss_gumbel(drafter, seqs, plens, anchors_per_seq=args.anchors, coupled=not args.uncoupled, gen=gen)
    elif args.commit:
        loss, logs = loss_commit(drafter, seqs, plens, anchors_per_seq=args.anchors, rounds_max=args.rounds_max, gen=gen)
    else:
        if args.ext_layers > 0:                                                # external drafter (drafter comparison): its own loss signature
            loss, logs = drafter.loss(seqs, plens, anchors_per_seq=args.anchors, gen=gen, prefix_decay=args.prefix_decay)
        else:
            loss, logs = drafter.loss(seqs, plens, anchors_per_seq=args.anchors, rounds_max=args.rounds_max, gen=gen, prefix_decay=args.prefix_decay, acc_weight=args.acc_weight)
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); sched.step()
    if args.save_every and step > 0 and step % args.save_every == 0:
        torch.save({n: p.detach().cpu() for n, p in drafter.named_parameters() if not n.startswith("ouro.")}, os.path.join(args.out, f"drafter_step{step}.pt"))
        _save_ckpt(step)
    if step % args.log_every == 0 or step == args.steps - 1:
        rec = {"step": step, "loss": float(loss), "grad_norm": float(gn), "lr": sched.get_last_lr()[0], "sec": time.time() - t0, **logs}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
sd = {n: p.detach().cpu() for n, p in drafter.named_parameters() if not n.startswith("ouro.")}
torch.save(sd, os.path.join(args.out, "drafter.pt"))
print("saved", os.path.join(args.out, "drafter.pt"), f"| peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
