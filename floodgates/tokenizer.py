"""Tokenizers of the looped models.  Their model repositories ship custom modeling code, and AutoTokenizer asks on the terminal
whether to run it (waiting 15 seconds).  The tokenizers themselves are standard; an empty config skips the question, as the
LoopSpec harness loads them (same token ids as the default load and as the bare tokenizer.json, checked on every model)."""
from transformers import AutoTokenizer, PreTrainedConfig


def load_tokenizer(path, **kw):
    return AutoTokenizer.from_pretrained(path, config=PreTrainedConfig(), **kw)
