"""Task 14: does the model reproduce training text verbatim?

Takes the first `prefix_tokens` of training documents, generates greedily, and
measures the longest common token run with the true continuation.

Run on --split train (seen texts) AND --split test (unseen control): memorization
shows up as train matches being clearly longer than test matches.

Usage: python -m src.memorization --model outputs/<run>/model [--split test] [--n 300]
"""
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python src/x.py` as well as `python -m src.x`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from difflib import SequenceMatcher

import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from src.tokenize_data import get_tokenizer
from src.utils import DATA_DIR, OUT_DIR, get_device, load_config, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--prefix-tokens", type=int, default=32)
    ap.add_argument("--gen-tokens", type=int, default=50)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    tok = get_tokenizer(cfg["model"]["name"])
    model = AutoModelForCausalLM.from_pretrained(args.model).to(get_device()).eval()
    docs = load_from_disk(str(DATA_DIR / "domain"))[args.split]
    docs = docs.select(range(min(args.n, len(docs))))
    need = args.prefix_tokens + args.gen_tokens

    print(f"device={model.device} model={args.model} split={args.split}", flush=True)

    # Every prefix is exactly prefix_tokens long, so they batch without padding.
    pairs = []
    for text in docs["text"]:
        ids = tok(text, truncation=True, max_length=need)["input_ids"]
        if len(ids) == need:
            pairs.append((ids[:args.prefix_tokens], ids[args.prefix_tokens:]))

    results = []
    for i in tqdm(range(0, len(pairs), args.batch_size), desc="memorization"):
        batch = pairs[i:i + args.batch_size]
        inp = torch.tensor([p for p, _ in batch], device=model.device)
        with torch.no_grad():
            out = model.generate(inp, attention_mask=torch.ones_like(inp), do_sample=False,
                                 max_new_tokens=args.gen_tokens, pad_token_id=tok.eos_token_id)
        for (prefix, truth), row in zip(batch, out[:, args.prefix_tokens:].tolist()):
            gen = row
            if tok.eos_token_id in gen:                 # cut at end-of-text, like single generation
                gen = gen[:gen.index(tok.eos_token_id)]
            m = SequenceMatcher(None, gen, truth).find_longest_match(0, len(gen), 0, len(truth))
            results.append({"longest_match_tokens": m.size, "prefix": tok.decode(prefix),
                            "generated": tok.decode(gen), "truth": tok.decode(truth)})

    results.sort(key=lambda r: -r["longest_match_tokens"])
    summary = {"model": args.model, "split": args.split, "checked": len(results),
               "mean_longest_match": sum(r["longest_match_tokens"] for r in results) / max(len(results), 1),
               "docs_with_match_>=10_tokens": sum(r["longest_match_tokens"] >= 10 for r in results),
               "docs_with_match_>=20_tokens": sum(r["longest_match_tokens"] >= 20 for r in results)}
    print(json.dumps(summary, indent=2))
    OUT_DIR.mkdir(exist_ok=True)
    tag = args.model.replace("outputs/", "").replace("/model", "").replace("/", "_")
    out_file = OUT_DIR / f"memorization_{tag}_{args.split}.json"   # one file per model+split
    out_file.write_text(json.dumps({"summary": summary, "top": results[:10]}, indent=2))
    with open(OUT_DIR / "memorization_results.jsonl", "a") as f:     # all summaries, for the table
        f.write(json.dumps(summary) + "\n")
    print(f"Details: {out_file}")


if __name__ == "__main__":
    main()
