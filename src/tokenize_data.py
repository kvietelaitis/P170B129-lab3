"""Task 4: tokenize text and pack into fixed-length blocks for causal LM training.

Documents are joined with the EOS token, then cut into `block_size` chunks.
The same function is used for every model/split so comparisons stay fair.
"""
from itertools import chain

from datasets import Dataset
from transformers import AutoTokenizer


def get_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.eos_token   # GPT2 has no pad token
    return tok


def pack(ds: Dataset, tok, block_size: int) -> Dataset:
    def _tok(batch):
        return tok([t + tok.eos_token for t in batch["text"]])

    def _group(batch):
        ids = list(chain.from_iterable(batch["input_ids"]))
        n = (len(ids) // block_size) * block_size
        blocks = [ids[i:i + block_size] for i in range(0, n, block_size)]
        return {"input_ids": blocks, "attention_mask": [[1] * block_size for _ in blocks],
                "labels": [b[:] for b in blocks]}

    tokenized = ds.map(_tok, batched=True, remove_columns=ds.column_names)
    return tokenized.map(_group, batched=True, remove_columns=tokenized.column_names)


def show_tokens(text: str, tok) -> list[str]:
    """Handy for the qualitative tokenization analysis in the notebook."""
    return [tok.decode([i]) for i in tok(text)["input_ids"]]
