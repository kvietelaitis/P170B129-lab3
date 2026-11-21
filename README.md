# GPT2 fine-tuning: movie reviews (IMDB)

Project for the Generative AI course: fine-tune DistilGPT2 / GPT2 Small on a domain corpus
and test how fine-tuning changed the model's behaviour.

**Domain:** film reviews (IMDB, Maas et al. 2011, free for research use).
To switch to news, change `data.hf_name` in `config.yaml` to e.g. `fancyzhx/ag_news`.

**Research question (draft, refine it):** *How does fine-tuning DistilGPT2 on IMDB reviews change
perplexity and generation style on film reviews compared with general text (WikiText-2), and
how does LoRA compare to full fine-tuning at different training-data sizes?*

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Runs on an NVIDIA GPU (CUDA), using bf16/fp16 mixed precision for training; falls back to CPU.
Check the GPU is visible: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.

Run every command **from the project root** (`~/P170B129-lab3`), either as a module (`python -m src.train`, with a dot and no `.py`) or as a script (`python src/train.py`).
All settings live in `config.yaml`. Each training run saves `config_used.yaml`, `env.json`
(library versions) and `log_history.json` to `outputs/<run>/`.

## Pipeline (run from the project root)

| Step | Command | PDF task |
|---|---|---|
| 1. Data: download, clean, dedup, split 80/10/10 | `python -m src.prepare_data` | 2, 3 |
| 2. Tokenization exploration | `notebooks/01_explore.ipynb` | 4 |
| 3. Baseline eval (test + general) | `python -m src.evaluate_model` / `--split general` | 5, 12 |
| 4. Baseline generations | `python -m src.generate` | 7 |
| 5. Full fine-tune | `python -m src.train --run-name full` | 6, 8, 13 |
| 6. LoRA fine-tune | `python -m src.train --method lora --run-name lora` | 11 |
| 7. Data size | `python -m src.train --fraction 0.25 --run-name f25` (and 0.5) | 16 |
| 8. Extra: GPT2 Small | `python -m src.train --model gpt2 --run-name gpt2_full` | extra |
| 9. Evaluate fine-tuned | `python -m src.evaluate_model --model outputs/full/model` | 9, 12 |
| 10. Generate after FT | `python -m src.generate --model outputs/full/model` | 7, 10 |
| 11. Memorization | `python -m src.memorization --model outputs/full/model` | 14 |

Results are appended to `outputs/eval_results.jsonl` and `outputs/generations.jsonl`.
For task 13 (where validation loss starts rising), train with more epochs, e.g. `--epochs 8`,
and plot `log_history.json`.

**Fair comparison:** all models use the base tokenizer, the same test blocks and the same
`block_size`. Use only the validation set to choose hyperparameters. The test set is for the final numbers.

## Metrics
- **Cross-entropy:** mean −log p(true next token).
- **Perplexity:** exp(cross-entropy).
- **Prediction entropy:** entropy of the full next-token distribution (how uncertain the model is, not whether it is right).
- **distinct-1/2:** share of unique uni/bigrams in generations. Lower means more repetition.
