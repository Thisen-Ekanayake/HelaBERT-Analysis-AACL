# HelaBERT Analysis

Fine-tuning and evaluation of HelaBERT — a BERT-based language model for Sinhala — across four downstream classification tasks, with two classification head variants and a complete data analysis pipeline.

## Overview

This repository benchmarks two HelaBERT model sizes (`HelaBERT_small`, `HelaBERT_large`) on Sinhala NLP tasks following the paper methodology: **5 randomly-seeded runs**, 4:1 train/test split, reported as macro-F1 mean ± std.

Two classification head variants are implemented for every task and model size:

| Variant | Description |
|---|---|
| **Standard** | `[CLS]` → dropout → Linear |
| **Co-Attention** | Parallel co-attention between `[CLS]` and the token sequence → LayerNorm → concat → MLP |

## Tasks

| Task | Classes | Data path |
|---|---|---|
| Sentiment Analysis | variable | `data/sinhala-sentiment-analysis/` |
| News Category Classification | 5 (Business, Political, Entertainment, Sci&Tech, Sports) | `data/Sinhala-News-Category-classification/` |
| News Source Classification | 9 sources | `data/Sinhala-News-Source-classification/` |
| Writing Style Classification | 4 (News, Academic, Blog, Creative) | `data/Writing-style-classification/` |

## Repository Structure

```
HelaBERT-Analysis/
├── tokenizer/
│   ├── unigram_32000_0.9995.model   # SentencePiece unigram model (32k vocab)
│   └── unigram_32000_0.9995.vocab
├── data_analysis/
│   ├── data_analyze.ipynb           # EDA notebook (label distribution, text length, char freq)
│   ├── tsv_to_csv.ipynb             # TSV → CSV conversion utility
│   ├── news_category_analysis.py
│   ├── news_source_analysis.py
│   ├── sentiment_train_test_split.py
│   └── writing_style_analysis.py
├── finetune/
│   ├── helabert_small/              # Fine-tuning scripts for HelaBERT_small
│   │   ├── sentiment.py
│   │   ├── sentiment_co_attn.py
│   │   ├── news_category.py
│   │   ├── news_category_co_attn.py
│   │   ├── news_source.py
│   │   ├── news_source_co_attn.py
│   │   ├── writing_style.py
│   │   └── writing_style_co_attn.py
│   └── helabert_large/              # Fine-tuning scripts for HelaBERT_large
│       └── (same structure as above)
└── inference/
    ├── helabert_small/              # Inference scripts for HelaBERT_small
    │   └── (same structure as finetune)
    └── helabert_large/              # Inference scripts for HelaBERT_large
        └── (same structure as finetune)
```

## Prerequisites

```bash
pip install -r requirements.txt
```

Place the pretrained model weights in the root of this repo:

```
HelaBERT_small/    # small model weights + config.json
HelaBERT_large/    # large model weights + config.json
```

## Training

Each fine-tuning script is self-contained. Run from the repo root so that relative paths (`tokenizer/`, `data/`, model dirs) resolve correctly.

```bash
# Example: fine-tune HelaBERT_small on sentiment (standard head)
python finetune/helabert_small/sentiment.py

# Example: fine-tune HelaBERT_large on news category (co-attention head)
python finetune/helabert_large/news_category_co_attn.py
```

Training runs 5 seeds (`[42, 123, 456, 789, 1024]`) sequentially and saves each run under its output directory (e.g. `HelaBERT_paper_sentiment/run_1/`). A `results.csv` summarising per-run macro-F1 is written on completion.


## Inference

Inference scripts automatically select the best checkpoint (highest macro-F1) from the corresponding training output directory.

```bash
# Example: run inference with HelaBERT_small on sentiment test set
python inference/helabert_small/sentiment.py
```

Results are saved under `results_test/<model>_<task>/`.

## Co-Attention Architecture

The co-attention head implements parallel co-attention between the `[CLS]` token and the full token sequence:

```
affinity = v( tanh( W_cls([CLS]) + W_token(token_seq) ) )   # (B, T)

Direction 1 — [CLS] attends over tokens:
  alpha        = softmax(affinity)
  attended_cls = alpha @ token_seq                            # (B, H)

Direction 2 — tokens attend over [CLS]:
  beta             = sigmoid(affinity) * token_mask
  attended_tokens  = normalize(beta) @ token_seq             # (B, H)

Final: LayerNorm([attended_cls; attended_tokens]) → MLP → logits
```

## Data Analysis

The `data_analysis/` directory contains scripts and notebooks for exploratory analysis of each dataset: label distribution, text length statistics, character frequency, and train/test split generation.

```bash
jupyter notebook data_analysis/data_analyze.ipynb
```

## Tokenizer

A SentencePiece unigram tokenizer trained on Sinhala text:

- Vocabulary size: 32,000
- Coverage: 0.9995
- Model file: `tokenizer/unigram_32000_0.9995.model`
