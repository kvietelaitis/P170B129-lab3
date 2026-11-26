# GPT2 fine-tuning: movie reviews (IMDB)

Project for the Generative AI course: fine-tune DistilGPT2 / GPT2 Small on a specific domain and test how fine-tuning changed the model's behaviour.

**Domain:** film reviews (IMDB, Andrew L. Maas 2011, free for research use).

@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.
               and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for
               Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}

**Research question:** *How does fine-tuning DistilGPT2 on IMDB reviews change perplexity and generation style on film reviews compared with general text (WikiText-2), and how does LoRA compare to full fine-tuning at different training-data sizes?*

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run every command **from the project root** (`~/P170B129-lab3`), either as a module (`python -m src.train`, with a dot and no `.py`) or as a script (`python src/train.py`).

All settings live in `config.yaml`. Each training run saves `config_used.yaml`, `env.json` (library versions) and `log_history.json` to `outputs/<run>/`.

## Pipeline (run from the project root)

| Step                                            | Command                                                        |
| ----------------------------------------------- | -------------------------------------------------------------- |
| 1. Data: download, clean, dedup, split 80/10/10 | `python -m src.prepare_data `                                  |
| 2. Tokenization exploration                     | `notebooks/01_explore.ipynb`                                   |
| 3. Baseline eval (test + general)               | `python -m src.evaluate_model` / `--split general`             |
| 4. Baseline generations                         | `python -m src.generate`                                       |
| 5. Full fine-tune                               | `python -m src.train --run-name full`                          |
| 6. LoRA fine-tune                               | `python -m src.train --method lora --run-name lora`            |
| 7. Data size                                    | `python -m src.train --fraction 0.25 --run-name f25` (and 0.5) |
| 8. Extra: GPT2 Small                            | `python -m src.train --model gpt2 --run-name gpt2_full`        |
| 9. Evaluate fine-tuned                          | `python -m src.evaluate_model --model outputs/full/model`      |
| 10. Generate after FT                           | `python -m src.generate --model outputs/full/model`            |
| 11. Memorization                                | `python -m src.memorization --model outputs/full/model`        |

Results are appended to `outputs/eval_results.jsonl` and `outputs/generations.jsonl`.

**Fair comparison:** all models use the base tokenizer, the same test blocks and the same `block_size`. Use only the validation set to choose hyperparameters. The test set is for the final numbers.

## Metrics
- **Cross-entropy:** mean −log p(true next token).
- **Perplexity:** exp(cross-entropy).
- **Prediction entropy:** entropy of the full next-token distribution (how uncertain the model is, not whether it is right).
- **distinct-1/2:** share of unique uni/bigrams in generations. Lower means more repetition.
