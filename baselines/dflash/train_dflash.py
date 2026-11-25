"""Train a DFlash drafter for Ouro-1.4B on our data (frozen Ouro; see dflash_ouro.py for the architecture/recipe).
  python baselines/dflash/train_dflash.py --out runs/dflash/<name> [--layers unroll|last --anchors 512 --steps 16000 --batch 8]
DFlash recipe: AdamW lr 6e-4, grad clip 1.0, cosine schedule with 4% warmup, block 16, gamma 7, max length 3072."""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # repository root
import argparse, json, os, random, sys, time
import torch
from floodgates.paths import LOOPSPEC, DATA
from floodgates.models.ouro_min import OuroMin
from dflash_ouro import DFlashOuro, dflash_loss

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=DATA + '/mix4_ouro14_greedy.jsonl')
ap.add_argument("--out", required=True)
ap.add_argument("--model", default=LOOPSPEC + '/models/Ouro-1.4B')
ap.add_argument("--layers", default="unroll", choices=["unroll", "last"])
ap.add_argument("--n-layers", type=int, default=5); ap.add_argument("--block", type=int, default=16)
ap.add_argument("--gamma", type=float, default=7.0); ap.add_argument("--anchors", type=int, default=512)
ap.add_argument("--steps", type=int, default=16000); ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--lr", type=float, default=6e-4); ap.add_argument("--warmup", type=float, default=0.04)
ap.add_argument("--max-len", type=int, default=3072); ap.add_argument("--mask-id", type=int, default=0)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--log-every", type=int, default=25)
ap.add_argument("--save-every", type=int, default=4000)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)
torch.manual_seed(args.seed); random.seed(args.seed)

rows = [json.loads(l) for l in open(args.data)]
rows = [r for r in rows if r.get("output_ids") and len(r["prompt_ids"]) + 1 < args.max_len]
random.shuffle(rows)
print(f"{len(rows)} training sequences", flush=True)
ouro = OuroMin(args.model)
dr = DFlashOuro(ouro, layers=args.layers, n_layers=args.n_layers, block=args.block, mask_id=args.mask_id, gamma=args.gamma)
print("target layers (effective, loop*24+layer):", dr.target_layer_ids, flush=True)
params = [p for _, p in dr.trainable()]
print(f"trainable params {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
W = max(1, int(args.warmup * args.steps))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / W if s < W else 0.5 * (1 + __import__("math").cos(__import__("math").pi * (s - W) / max(1, args.steps - W))))
gen = torch.Generator().manual_seed(args.seed)
nb = None if args.anchors <= 32 else 64
log = open(os.path.join(args.out, "train_log.jsonl"), "w")
t0 = time.time(); ptr = 0
for step in range(args.steps):
    batch = rows[ptr: ptr + args.batch]; ptr += args.batch
    if len(batch) < args.batch:
        random.shuffle(rows); ptr = args.batch; batch = rows[: args.batch]
    seqs = [(r["prompt_ids"] + r["output_ids"])[: args.max_len] for r in batch]
    plens = [len(r["prompt_ids"]) for r in batch]
    loss, logs = dflash_loss(dr, seqs, plens, args.anchors, gen, n_bucket=nb)
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); sched.step()
    if args.save_every and step > 0 and step % args.save_every == 0:
        torch.save(dr.state(), os.path.join(args.out, f"drafter_step{step}.pt"))
    if step % args.log_every == 0 or step == args.steps - 1:
        rec = {"step": step, "loss": float(loss), "grad_norm": float(gn), "lr": sched.get_last_lr()[0], "sec": time.time() - t0, **logs}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
torch.save(dr.state(), os.path.join(args.out, "drafter.pt"))
print("saved", os.path.join(args.out, "drafter.pt"), f"| peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
