"""Task 14: does the model reproduce training text verbatim?

Takes the first `prefix_tokens` of training documents, generates greedily, and
measures the longest common token run with the true continuation.

Usage: python -m src.memorization --model outputs/<run>/model
"""
import argparse
import json
from difflib import SequenceMatcher

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM

from src.tokenize_data import get_tokenizer
from src.utils import DATA_DIR, OUT_DIR, get_device, load_config, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--prefix-tokens", type=int, default=32)
    ap.add_argument("--gen-tokens", type=int, default=50)
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    tok = get_tokenizer(cfg["model"]["name"])
    model = AutoModelForCausalLM.from_pretrained(args.model).to(get_device()).eval()
    train = load_from_disk(str(DATA_DIR / "domain"))["train"].select(range(args.n))

    results = []
    for text in train["text"]:
        ids = tok(text)["input_ids"]
        if len(ids) < args.prefix_tokens + args.gen_tokens:
            continue
        prefix, truth = ids[:args.prefix_tokens], ids[args.prefix_tokens:args.prefix_tokens + args.gen_tokens]
        with torch.no_grad():
            out = model.generate(torch.tensor([prefix], device=model.device), do_sample=False,
                                 max_new_tokens=args.gen_tokens, pad_token_id=tok.eos_token_id)
        gen = out[0][len(prefix):].tolist()
        m = SequenceMatcher(None, gen, truth).find_longest_match(0, len(gen), 0, len(truth))
        results.append({"longest_match_tokens": m.size, "prefix": tok.decode(prefix),
                        "generated": tok.decode(gen), "truth": tok.decode(truth)})

    results.sort(key=lambda r: -r["longest_match_tokens"])
    summary = {"model": args.model, "checked": len(results),
               "mean_longest_match": sum(r["longest_match_tokens"] for r in results) / max(len(results), 1),
               "docs_with_match_>=20_tokens": sum(r["longest_match_tokens"] >= 20 for r in results)}
    print(json.dumps(summary, indent=2))
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "memorization.json").write_text(json.dumps({"summary": summary, "top": results[:10]}, indent=2))


if __name__ == "__main__":
    main()
