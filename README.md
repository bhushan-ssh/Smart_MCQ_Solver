# Smart MCQ Solver — Answer Ranking

**Author:** Bhushan Dattatray Sonawane  
**Roll No.:** 23f2003210  
**Course:** Deep Learning & Generative AI Project

## Overview

This project predicts the **top 3 ranked answers** for multiple-choice questions with five options (A–E).

### Models
- **BiLSTM** — built from scratch
- **BiGRU + Attention** — built from scratch
- **BERT (`bert-base-uncased`)** — pretrained and fine-tuned

The final predictions use a **validation-optimized ensemble**.

## Results

| Model | MAP@3 | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| BiLSTM | 0.6001 | 0.4710 | 0.4809 |
| BiGRU | 0.5944 | 0.4198 | 0.4262 |
| BERT | **0.7201** | **0.5392** | **0.5664** |
| **Ensemble** | **0.7503** | — | — |

The best ensemble weights were:

**LSTM: 0.15 | BiGRU: 0.00 | BERT: 0.85**

## Methodology

- Group-aware train/validation split to reduce data leakage
- Deep learning models implemented using PyTorch
- BERT fine-tuned using Hugging Face Transformers
- Evaluation using **MAP@3, Accuracy, and Macro-F1**
- Ensemble weights optimized using validation MAP@3 only
- Final top-3 predictions generated for the test set
- Kaggle `submission.csv` generated

## Experiment Tracking

Experiments are tracked using **Weights & Biases (W&B)**, including:

`model1_lstm` · `model2_bigru` · `model3_bert` · `model_comparison` · `ensemble_final`

## Tech Stack

Python · PyTorch · Transformers · scikit-learn · NumPy · Pandas · Weights & Biases

## Author

**Bhushan Dattatray Sonawane**  
**23f2003210**
