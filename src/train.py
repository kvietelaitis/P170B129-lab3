"""Tasks 6, 8, 11, 13, 16: fine-tune (full or LoRA) with train/val loss logging.

Usage:
  python -m src.train                                  # settings from config.yaml
  python -m src.train --method lora
  python -m src.train --fraction 0.25 --run-name frac25
  python -m src.train --model gpt2                     # extra experiment
"""
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python src/x.py` as well as `python -m src.x`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import copy
import json
from datetime import datetime

from datasets import load_from_disk
from transformers import (AutoModelForCausalLM, Trainer, TrainingArguments,
                          default_data_collator)

from src.tokenize_data import get_tokenizer, pack
from src.utils import (DATA_DIR, OUT_DIR, load_config, mixed_precision,
                       save_run_metadata, set_seed)


def check_inputs(model, tok, train_blocks, val_blocks, block_size):
    """Catch the usual causes of CUDA 'device-side assert' before training starts."""
    vocab = model.get_input_embeddings().num_embeddings
    max_pos = model.config.n_positions
    print(f"model={model.name_or_path} vocab={vocab} n_positions={max_pos} "
          f"tokenizer_len={len(tok)} block_size={block_size} "
          f"train_blocks={len(train_blocks)} val_blocks={len(val_blocks)}")
    assert block_size <= max_pos, f"block_size {block_size} > model context {max_pos}"
    assert len(train_blocks) > 0 and len(val_blocks) > 0, "empty dataset - rerun prepare_data"
    for name, blocks in [("train", train_blocks), ("validation", val_blocks)]:
        for col in ("input_ids", "labels"):
            hi = max(max(row) for row in blocks[col])
            lo = min(min(row) for row in blocks[col])
            assert lo >= 0 and hi < vocab, f"{name}.{col} ids out of range [{lo}, {hi}], vocab={vocab}"
        lens = {len(r) for r in blocks["input_ids"]}
        assert lens == {block_size}, f"{name} blocks have lengths {lens}, expected {block_size}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["full", "lora"])
    ap.add_argument("--fraction", type=float)
    ap.add_argument("--epochs", type=float)
    ap.add_argument("--model")
    ap.add_argument("--run-name")
    args = ap.parse_args()

    cfg = copy.deepcopy(load_config())
    t = cfg["train"]
    if args.method: t["method"] = args.method
    if args.fraction: t["train_fraction"] = args.fraction
    if args.epochs: t["epochs"] = args.epochs
    if args.model: cfg["model"]["name"] = args.model
    set_seed(cfg["seed"])

    name = cfg["model"]["name"]
    run_name = args.run_name or f"{name}_{t['method']}_f{t['train_fraction']}_{datetime.now():%m%d-%H%M}"
    run_dir = OUT_DIR / run_name

    tok = get_tokenizer(name)
    ds = load_from_disk(str(DATA_DIR / "domain"))
    train_ds = ds["train"]
    if t["train_fraction"] < 1.0:
        train_ds = train_ds.select(range(int(len(train_ds) * t["train_fraction"])))
    bs = cfg["model"]["block_size"]
    train_blocks, val_blocks = pack(train_ds, tok, bs), pack(ds["validation"], tok, bs)

    model = AutoModelForCausalLM.from_pretrained(name)
    check_inputs(model, tok, train_blocks, val_blocks, bs)
    lr = t["lr"]
    if t["method"] == "lora":
        from peft import LoraConfig, get_peft_model
        lc = t["lora"]
        model = get_peft_model(model, LoraConfig(
            r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
            target_modules=lc["target_modules"], fan_in_fan_out=True, task_type="CAUSAL_LM"))
        model.print_trainable_parameters()
        lr = lc["lr"]

    targs = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=t["epochs"],
        learning_rate=lr,
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        weight_decay=t["weight_decay"],
        warmup_ratio=t["warmup_ratio"],
        eval_strategy="steps", eval_steps=t["eval_steps"],
        logging_steps=50,
        save_strategy="steps", save_steps=t["eval_steps"], save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        seed=cfg["seed"], data_seed=cfg["seed"],
        report_to="none",
        dataloader_num_workers=2,
        **mixed_precision(),              # bf16/fp16 on CUDA
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_blocks,
                      eval_dataset=val_blocks, data_collator=default_data_collator)
    trainer.train()

    if t["method"] == "lora":
        model = model.merge_and_unload()     # plain model -> same eval/generate code works
    model.save_pretrained(run_dir / "model")
    tok.save_pretrained(run_dir / "model")
    (run_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    save_run_metadata(run_dir, cfg, {"train_blocks": len(train_blocks), "val_blocks": len(val_blocks)})
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
