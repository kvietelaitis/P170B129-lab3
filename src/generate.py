"""Tasks 7, 10: generate with a grid of temperature / top-k; measure repetition.

Usage:
  python -m src.generate                               # base model
  python -m src.generate --model outputs/<run>/model
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM

from src.tokenize_data import get_tokenizer
from src.utils import OUT_DIR, get_device, load_config, set_seed


def distinct_n(text: str, n: int) -> float:
    """Share of unique n-grams: low value = repetitive text."""
    w = text.split()
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    cfg = load_config()
    g = cfg["generate"]
    tok = get_tokenizer(cfg["model"]["name"])
    model = AutoModelForCausalLM.from_pretrained(args.model or cfg["model"]["name"]).to(get_device()).eval()

    rows = []
    for prompt in g["prompts"]:
        for temp in g["temperatures"]:
            for k in g["top_ks"]:
                set_seed(cfg["seed"])         # same seed per setting -> comparable
                enc = tok(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**enc, do_sample=True, temperature=temp, top_k=k,
                                         max_new_tokens=g["max_new_tokens"],
                                         pad_token_id=tok.eos_token_id)
                text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                rows.append({"model": args.model or cfg["model"]["name"], "prompt": prompt,
                             "temperature": temp, "top_k": k, "text": text,
                             "distinct_1": distinct_n(text, 1), "distinct_2": distinct_n(text, 2)})
                print(f"\n[T={temp} k={k}] {prompt}{text}")

    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "generations.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
