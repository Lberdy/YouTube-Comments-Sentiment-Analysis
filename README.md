# Fine-Tuning DistilBERT for YouTube Comment Sentiment Analysis

Binary sentiment classification (Positive / Negative) of YouTube comments by fully fine-tuning `distilbert-base-uncased`. The video title is fed to the model together with the comment to give it the context needed to interpret ambiguous comments.

**Final test accuracy: 86.97% | Weighted F1: 0.87**

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model](#model)
- [Training Configuration](#training-configuration)
- [Training Results](#training-results)
- [Test Results](#test-results)
- [Limitations](#limitations)
- [Future Work](#future-work)

---

## Overview

YouTube comments are informal, full of abbreviations and emojis, and often only make sense relative to the video they refer to. For example, "This is so sad!" can be positive (emotion about a moving song) or negative (reaction to bad news), depending on the video.

To address this, each input combines the comment with the video title, and a pre-trained transformer (DistilBERT) is fully fine-tuned on the resulting binary classification task.

## Dataset

English YouTube comments, each paired with the title of its video and a sentiment label. The original data had three classes (Positive, Negative, Neutral). The **Neutral class was removed** because it was ambiguous and noticeably degraded performance in early experiments.

| Property | Value |
|---|---|
| Total size | 60,000 examples (after removing Neutral) |
| Positive | 30,000 (50%) |
| Negative | 30,000 (50%) |
| Train split | 48,600 (81%) |
| Validation split | 5,400 (9%) |
| Test split | 6,000 (10%) |
| Model input | `FullText = comment + " [SEP] " + title` |

## Preprocessing

Cleaning is applied to both comments and titles:

- Convert emojis to descriptive text with the `emoji` library (e.g. 😊 → `:smiling_face:`)
- Remove leftover HTML tags (regex)
- Remove URLs and `@user` mentions
- Normalize hashtags (`#word` → `word`)
- Collapse multiple spaces and line breaks

Tokenization uses HuggingFace's `DistilBertTokenizerFast`, with truncation/padding to a maximum of 512 tokens. `[CLS]` and `[SEP]` special tokens are inserted automatically.

## Model

**DistilBERT** is a distilled version of BERT-base: 40% fewer parameters (110M → 66M) while retaining about 97% of BERT's performance on GLUE.

| Property | Value |
|---|---|
| Base checkpoint | `distilbert-base-uncased` |
| Transformer layers | 6 |
| Attention heads per layer | 12 |
| Hidden size | 768 |
| Vocabulary | 30,522 (WordPiece) |
| Max sequence length | 512 |
| Pre-training data | English Wikipedia + BookCorpus |

**Strategy:** full fine-tuning. All pre-trained weights and a new linear classification head (2 output neurons) on top of the `[CLS]` token are updated during training.

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Learning rate | 3e-5 (linear decay to 0) |
| Batch size (train / eval) | 32 / 32 |
| Epochs | 4 |
| Weight decay | 0.01 |
| Optimizer | AdamW (`adamw_torch_fused`) |
| Max length | 512 tokens (padding / truncation) |
| Precision | BF16 |
| DataLoader workers | 6 (with `pin_memory`) |
| Evaluation | Every epoch (`eval_strategy="epoch"`) |
| Best-model criterion | Weighted F1 (`load_best_model_at_end=True`) |
| Total steps | 6,076 (1,519 per epoch) |

## Training Results

| Epoch | Train Loss | Val Loss | Val Accuracy | Val F1 (weighted) | Steps |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.311 | 0.3115 | 86.11% | 0.8607 | 1,519 |
| **2** ★ | 0.224 | 0.3171 | 86.70% | **0.8667** | 3,038 |
| 3 | 0.119 | 0.4197 | 86.33% | 0.8633 | 4,557 |
| 4 | 0.076 | 0.5919 | 86.61% | 0.8661 | 6,076 |

★ Best checkpoint (step 3038), automatically restored by the HuggingFace `Trainer` for final evaluation.

**Overfitting:** training loss keeps decreasing, while validation loss rises after epoch 2 (0.3171 → 0.4197 → 0.5919). Validation accuracy stays stable around 86–87%, which suggests the model learns robust representations early and the later overfitting only marginally affects predictions. Selecting the best checkpoint by validation F1 mitigates this.

## Test Results

Evaluated on the held-out test set (6,000 examples: 3,000 Positive, 3,000 Negative) using the epoch-2 checkpoint.

| Metric | Value |
|---|---|
| Accuracy | **86.97%** |
| Macro F1 | 0.87 |
| Weighted F1 | 0.87 |

**Per-class metrics**

| Class | Precision | Recall | F1 |
|---|:---:|:---:|:---:|
| Negative (0) | 0.84 | 0.91 | 0.87 |
| Positive (1) | 0.90 | 0.83 | 0.86 |

**Confusion matrix**

| | Predicted Negative | Predicted Positive |
|---|:---:|:---:|
| **Actual Negative** | 2,736 | 264 |
| **Actual Positive** | 518 | 2,482 |

The model is slightly biased toward predicting Negative: high recall on negatives, but 518 positive comments are misclassified as negative.