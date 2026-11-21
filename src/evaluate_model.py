"""Tasks 5, 9, 12: cross-entropy, perplexity and prediction entropy on a split.

Same tokenizer, same blocks, same block_size for every model (fair comparison).

Usage:
  python -m src.evaluate_model                         # base model from config
  python -m src.evaluate_model --model outputs/<run>/model
  python -m src.evaluate_model --split general
"""
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python src/x.py` as well as `python -m src.x`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import math

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, default_data_collator

from src.tokenize_data import get_tokenizer, pack
from src.utils import DATA_DIR, OUT_DIR, get_device, load_config, set_seed


@torch.no_grad()
def evaluate(model, blocks, batch_size=8, device=None) -> dict:
    device = device or get_device()
    model.eval().to(device)
    loader = DataLoader(blocks, batch_size=batch_size, collate_fn=default_data_collator)
    tot_ce, tot_ent, n_tok = 0.0, 0.0, 0
    for batch in tqdm(loader, desc="eval"):
        ids = batch["input_ids"].to(device)
        logits = model(input_ids=ids).logits[:, :-1].float()
        targets = ids[:, 1:]
        logp = torch.log_softmax(logits, dim=-1)
        ce = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)       # cross-entropy per token
        ent = -(logp.exp() * logp).sum(-1)                              # prediction entropy per token
        tot_ce += ce.sum().item()
        tot_ent += ent.sum().item()
        n_tok += targets.numel()
    loss = tot_ce / n_tok
    return {"cross_entropy": loss, "perplexity": math.exp(loss),
            "prediction_entropy": tot_ent / n_tok, "tokens": n_tok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="HF name or path; default = config base model")
    ap.add_argument("--split", default="test", choices=["train", "validation", "test", "general"])
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    base = cfg["model"]["name"]
    tok = get_tokenizer(base)               # always the base tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model or base)

    if args.split == "general":
        ds = load_from_disk(str(DATA_DIR / "general"))
    else:
        ds = load_from_disk(str(DATA_DIR / "domain"))[args.split]
    blocks = pack(ds, tok, cfg["model"]["block_size"])

    res = {"model": args.model or base, "split": args.split, **evaluate(model, blocks)}
    print(json.dumps(res, indent=2))
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "eval_results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
