"""Tasks 2-3: download corpus, clean, deduplicate, split into train/val/test.

Deduplication happens BEFORE splitting so the same text cannot land in both
train and test (no train/test overlap).

Usage: python -m src.prepare_data
"""
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python src/x.py` as well as `python -m src.x`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hashlib
import json
import re

from datasets import Dataset, DatasetDict, load_dataset

from src.utils import DATA_DIR, load_config, set_seed


def clean(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", text)      # IMDB contains HTML line breaks
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def norm_hash(text: str) -> str:
    return hashlib.md5(re.sub(r"\W+", "", text.lower()).encode()).hexdigest()


def main():
    cfg = load_config()
    dcfg = cfg["data"]
    set_seed(cfg["seed"])

    raw = load_dataset(dcfg["hf_name"], split=dcfg["hf_split"])
    texts = [clean(t) for t in raw[dcfg["text_field"]]]
    n_raw = len(texts)

    seen, uniq = set(), []
    for t in texts:
        if len(t) < dcfg["min_chars"]:
            continue
        h = norm_hash(t)
        if h not in seen:
            seen.add(h)
            uniq.append(t)

    ds = Dataset.from_dict({"text": uniq}).shuffle(seed=cfg["seed"])
    if dcfg.get("max_docs"):
        ds = ds.select(range(min(dcfg["max_docs"], len(ds))))

    holdout = dcfg["val_frac"] + dcfg["test_frac"]
    s1 = ds.train_test_split(test_size=holdout, seed=cfg["seed"])
    s2 = s1["test"].train_test_split(test_size=dcfg["test_frac"] / holdout, seed=cfg["seed"])
    splits = DatasetDict(train=s1["train"], validation=s2["train"], test=s2["test"])

    # Sanity check: no overlap between splits
    hashes = {k: {norm_hash(t) for t in v["text"]} for k, v in splits.items()}
    assert not (hashes["train"] & hashes["test"]), "train/test overlap!"
    assert not (hashes["train"] & hashes["validation"]), "train/val overlap!"

    out = DATA_DIR / "domain"
    splits.save_to_disk(str(out))

    # General-domain text (task 12)
    gen = load_dataset(dcfg["general_hf_name"], dcfg["general_hf_config"], split=dcfg["general_split"])
    gen = gen.filter(lambda x: len(x["text"].strip()) > 0)
    Dataset.from_dict({"text": ["\n".join(gen["text"])]}).save_to_disk(str(DATA_DIR / "general"))

    stats = {
        "source": dcfg["hf_name"],
        "raw_docs": n_raw,
        "after_clean_dedup": len(uniq),
        **{f"{k}_docs": len(v) for k, v in splits.items()},
    }
    (DATA_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
