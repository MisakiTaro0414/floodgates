"""OuroMin vs the SGLang server's deterministic greedy decoding (the LoopSpec baseline outputs in traces/): teacher-forced argmax of the
last loop must reproduce the server's next tokens.  Reports per-token agreement and how many sequences match fully."""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # repository root
import json, sys, torch
import os
from floodgates.paths import LOOPSPEC, trace
from floodgates.models.ouro_min import OuroMin
TR = trace("Ouro-1.4B", "gsm8k")
m = OuroMin(LOOPSPEC + '/models/Ouro-1.4B')
rows = [json.loads(l) for l in open(TR)][: int(sys.argv[1]) if len(sys.argv) > 1 else 20]
agree = tot = full = 0; first_bad = []
for r in rows:
    p, o = r["input"], r["output_ids"]
    ids = torch.tensor([p + o], device="cuda")
    logits, _ = m.forward_context(ids)
    pred = logits[0, len(p) - 1 : len(p) + len(o) - 1].argmax(-1).tolist()
    ok = [a == b for a, b in zip(pred, o)]
    agree += sum(ok); tot += len(ok); full += all(ok)
    if not all(ok): first_bad.append(ok.index(False))
print(f"token agreement {agree}/{tot} = {agree / tot:.4f}; fully matching sequences {full}/{len(rows)}; first mismatch idx {first_bad[:10]}")
