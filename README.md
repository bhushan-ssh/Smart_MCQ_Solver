# Smart MCQ Solver — Answer Ranking

**Deep Learning & Generative AI Project**

---

## Overview

**Smart MCQ Solver** is a deep learning-based system that predicts and ranks the **top 3 answers** for multiple-choice questions containing five options (**A–E**).

The project compares three different deep learning approaches:

* **BiLSTM** — built from scratch using PyTorch
* **BiGRU + Attention** — built from scratch using PyTorch
* **BERT (`bert-base-uncased`)** — pretrained Transformer fine-tuned for multiple-choice classification

The final predictions are generated using a **validation-optimized weighted ensemble**.

---

## Objective

The main objective is not only to predict the correct answer, but to **rank it within the top 3 positions**.

The project focuses on:

* Building different deep learning architectures
* Preventing data leakage using a group-aware split
* Fine-tuning a pretrained Transformer
* Comparing model performance
* Optimizing an ensemble using validation MAP@3
* Generating final Kaggle predictions

---

## Dataset

Each question contains:

| Column   | Description    |
| -------- | -------------- |
| `prompt` | Question       |
| `A`      | Option A       |
| `B`      | Option B       |
| `C`      | Option C       |
| `D`      | Option D       |
| `E`      | Option E       |
| `answer` | Correct answer |

The labelled dataset contains approximately **2,000 examples**.

A **group-aware train/validation split** was used:

* Training: **1,707 examples**
* Validation: **293 examples**

This helps reduce leakage from duplicate or closely related questions.

---

## Models

### 1. BiLSTM

A bidirectional LSTM was implemented from scratch.

```text
Question + Option
       ↓
Embedding (200)
       ↓
BiLSTM (192 hidden per direction)
       ↓
Mean Pool + Max Pool + Final Hidden State
       ↓
MLP Scorer
       ↓
Option Score
       ↓
Softmax
```

The same scorer is applied to all five answer options.

---

### 2. BiGRU + Attention

The second model uses a bidirectional GRU with an attention mechanism.

```text
Question + Option
       ↓
Embedding
       ↓
BiGRU
       ↓
Attention Pooling
       ↓
MLP Scorer
       ↓
Option Score
       ↓
Softmax
```

Attention allows the model to assign more importance to informative words in the question and answer.

---

### 3. BERT

The third model uses pretrained:

```text
bert-base-uncased
```

with Hugging Face's `AutoModelForMultipleChoice`.

For each question, BERT evaluates all five question-option pairs and produces a score for each option.

BERT was fine-tuned end-to-end on the MCQ dataset.

---

## Evaluation Metrics

### MAP@3

The primary competition metric is **Mean Average Precision at 3**.

The correct answer receives:

| Rank          | Score |
| ------------- | ----: |
| 1st           |   1.0 |
| 2nd           |   0.5 |
| 3rd           | 0.333 |
| Outside Top 3 |     0 |

MAP@3 was used for model selection, early stopping, and ensemble optimization.

### Accuracy

Measures whether the model's top-ranked answer is correct.

### Macro-F1

Calculates F1-score independently for classes A–E and averages them.

---

## Results

| Model             |      MAP@3 |   Accuracy |   Macro-F1 |
| ----------------- | ---------: | ---------: | ---------: |
| BiLSTM            |     0.6001 |     0.4710 |     0.4809 |
| BiGRU + Attention |     0.5944 |     0.4198 |     0.4262 |
| **BERT**          | **0.7201** | **0.5392** | **0.5664** |
| **Ensemble**      | **0.7503** |          — |          — |

### Best Ensemble

The final ensemble weights were:

```text
BiLSTM  → 0.15
BiGRU   → 0.00
BERT    → 0.85
```

The ensemble improved validation MAP@3 from **0.7201 with BERT alone** to **0.7503**.

---

## Key Findings

* **BERT performed best** because it uses pretrained language representations and self-attention.
* The from-scratch RNN models performed reasonably well despite the relatively small dataset.
* Both BiLSTM and BiGRU showed signs of **overfitting**, making early stopping important.
* The BiGRU received zero ensemble weight because it did not improve the validation MAP@3 when combined with the other models.
* Ensemble learning provided an additional improvement over the best individual model.

---

## Tech Stack

* **Python**
* **PyTorch**
* **Hugging Face Transformers**
* **scikit-learn**
* **NumPy**
* **Pandas**
* **Weights & Biases**
* **Kaggle**

---

## Experiment Tracking

Experiments were tracked using **Weights & Biases (W&B)**.

Runs include:

```text
model1_lstm
model2_bigru
model3_bert
model_comparison
ensemble_final
```

---

## Workflow

```text
Dataset
   ↓
Data Preprocessing
   ↓
Group-Aware Train/Validation Split
   ↓
 ┌──────────┬──────────────┬─────────┐
 ↓          ↓              ↓
BiLSTM     BiGRU         BERT
 ↓          ↓              ↓
 └──────────┴──────────────┘
              ↓
       Model Evaluation
              ↓
     Ensemble Weight Search
              ↓
       Final Top-3 Ranking
              ↓
        submission.csv
```

---

## Project Output

The final system generates:

```text
submission.csv
```

containing the ranked **top-3 predictions** for each test question in the required Kaggle format.

---

## Future Improvements

* Experiment with RoBERTa or DeBERTa
* Perform cross-validation
* Improve ensemble strategies
* Perform detailed error analysis
* Tune hyperparameters more extensively
* Explore larger pretrained language models

---

## Author

**Bhushan Dattatray Sonawane**
**Roll No.: 23f2003210**

> Built as part of the **Deep Learning & Generative AI Project**.
