# Training data

Drafters are trained on the base model's own greedy continuations of 7,473 GSM8K and 7,500 MATH training problems, 3,474 code
prompts (MBPP and CodeSearchNet function signatures) and 4,000 FineWeb-Edu prefixes, mixed by `build_mix.py`.

**Recurrent-Llama, Recurrent-TinyLlama, Huginn.** The continuations come from our engine:

```bash
MODEL=Recurrent-Llama-3.2-train-recurrence-32 TAG=rl32 bash scripts/gen_data.sh      # data/mix4c_rl32_greedy.jsonl
```

**Ouro.** The continuations come from the SGLang server of [LoopSpec](https://github.com/kaist-flexml-lab/loopspec):

```bash
# in the LoopSpec checkout
CUDA_VISIBLE_DEVICES=0 uv run --locked --no-dev python -m sglang_recurrent.server --model models/Ouro-1.4B --host 127.0.0.1 --port 30000 \
  --mem-fraction-static 0.7 --random-seed 1234 --max-running-requests 64
# in this repository
for t in gsm8k math code general; do
  GEN_TASK=$t GEN_MODEL=$LOOPSPEC_DIR/models/Ouro-1.4B python datagen/gen_data.py data/${t}_train_ouro14_greedy.jsonl 64
done
python datagen/build_mix.py ouro14                                                    # data/mix4c_ouro14_greedy.jsonl
```

**Ouro Thinking models.** GSM8K and 2,032 MATH problems in the chat format, from a server started with `--model models/Ouro-1.4B-Thinking`:

```bash
CHAT_TRACE=traces/Ouro-1.4B-Thinking-gsm8kchat.jsonl GEN_TASK=gsm8kchat GEN_MODEL=$LOOPSPEC_DIR/models/Ouro-1.4B-Thinking \
  python datagen/gen_data.py data/gsm8kchat_train_ouro14t_greedy.jsonl 64
CHAT_TRACE=traces/Ouro-1.4B-Thinking-math500chat.jsonl ONLY_FROM=datagen/prompts/mathchat_train_problems.jsonl \
  GEN_MODEL=$LOOPSPEC_DIR/models/Ouro-1.4B-Thinking python datagen/gen_data_chat.py data/mathchat_train_ouro14t_greedy.jsonl 64
python datagen/build_mix.py ouro14t --chat                                            # data/mix_chat_ouro14t_greedy.jsonl
```
