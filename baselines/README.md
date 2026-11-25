# Baselines with their own code

DAS, DAS-Wave, Jacobi decoding and prompt lookup decoding run in our engines from a spec (see the main README).  This folder holds
the baselines that need a separate drafter or their authors' code.  Every script documents its options in its docstring.

## DFlash (Ouro-1.4B)

A five-layer DFlash drafter reimplemented from the official sources (`dflash_ouro.py`), trained on our Ouro-1.4B data and evaluated
offline on the same traces as `evaluation/eval_accept.py`:

```bash
python baselines/dflash/train_dflash.py --out runs/dflash/df-last-a512 --layers last --anchors 512 --steps 16000 --batch 8
python baselines/dflash/eval_dflash.py runs/dflash/df-last-a512 gsm8k math500
```

`baselines/dense/ouro_spec_hf.py` times it against our drafter in one eager implementation (`dflash:<run>:<K>`).

## Dense-model control (LayerSkip)

Our drafting pass on the first E layers of a dense Llama-3.2-1B, plain and trained with LayerSkip's early-exit recipe, the base model
of Recurrent-Llama-3.2.  It needs access to the gated repositories `meta-llama/Llama-3.2-1B` and `facebook/layerskip-llama3.2-1B`,
and vLLM for the data, on the prompts of the Recurrent-Llama mix (`python datagen/build_mix.py rl32 --with-mbpp`):

```bash
python baselines/dense/gen_data.py <model dir> data/mix4_dense_layerskip_greedy.jsonl
python floodgates/drafters/dense_drafter.py train --model <model dir> --data data/mix4_dense_layerskip_greedy.jsonl --out runs/dense/layerskip-E2 --E 2
python floodgates/drafters/dense_drafter.py eval --run runs/dense/layerskip-E2 --prompts traces/Recurrent-Llama-3.2-train-recurrence-32-gsm8k-full.jsonl --task gsm8k
python baselines/dense/dense_spec_hf.py runs/dense/layerskip-E2 <prompts> gsm8k <out.json> lal,ls:2:4     # wall-clock, Hugging Face eager
python baselines/dense/layerskip_selfspec.py <model dir> <prompts> gsm8k <out.json> 2                       # LayerSkip's own self-speculation
```

`baselines/dense/raven_spec_hf.py` runs our Recurrent-Llama drafter in the same eager setting.

## WaveFront (Huginn-3.5B, Ouro)

WaveFront Decoding with its authors' code (github.com/summerbro-hhj/wavefront-decoding at 4345209, checked out at `$WAVEFRONT_DIR`,
default `../looped_refs/wavefront-decoding`), unmodified, on our prompts, stops and limits:

```bash
S0=ours MODEL=huginn TASK=gsm8k python baselines/wavefront/run_wfd_task2.py runs/wavefront/huginn_gsm8k.json 0 1000     # draft depth 4 (their default)
S0=ours MODEL=huginn TASK=gsm8k WF_T=32 WF_TD=8 python baselines/wavefront/run_wfd_depth.py runs/wavefront/huginn_gsm8k_td8.json 0 1000
S0=ours MODEL=huginn TASK=gsm8k TEMP=1.0 python baselines/wavefront/run_wfd_task_sample.py runs/wavefront/huginn_gsm8k_T1.json 0 1000
```

`0` keeps the full KV cache (lossless) and `1000` runs every prompt of the task.  `S0=ours` starts their loops from our engine's
initial state, so that their plain decoding matches ours token for token, as in the paper.  The paper reports the better of draft
depths 4 and 8.
