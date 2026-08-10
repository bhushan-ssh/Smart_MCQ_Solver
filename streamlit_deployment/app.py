import json
import re
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)

from transformers import AutoTokenizer, AutoModelForMultipleChoice


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart MCQ Solver",
    page_icon="🧠",
    layout="centered",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent

RNN_DIR = BASE_DIR / "models" / "rnn"

BERT_REPO = "23f2003210/smart-mcq-solver-bert"

MAX_LEN = 96

OPTS = ["A", "B", "C", "D", "E"]


# ============================================================
# TOKENIZATION
# ============================================================

TOKEN_RE = re.compile(
    r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[^\sA-Za-z0-9]"
)


def tokenize(text):
    return TOKEN_RE.findall(str(text).lower())


# ============================================================
# LOAD VOCABULARY
# ============================================================

@st.cache_resource
def load_vocab():

    with open(
        RNN_DIR / "vocab.json",
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    return data


vocab_data = load_vocab()

VOCAB = vocab_data["stoi"]

PAD_IDX = vocab_data["pad_idx"]
UNK_IDX = vocab_data["unk_idx"]
SEP_IDX = vocab_data["sep_idx"]

MAX_LEN = vocab_data.get("max_len", MAX_LEN)


# ============================================================
# ENCODE QUESTION + OPTION
# ============================================================

def encode_pair(question, option, max_len=MAX_LEN):

    q_tokens = tokenize(question)
    o_tokens = tokenize(option)

    tokens = q_tokens + ["<sep>"] + o_tokens

    tokens = tokens[:max_len]

    ids = [
        VOCAB.get(token, UNK_IDX)
        for token in tokens
    ]

    length = len(ids)

    if length < max_len:

        ids = ids + [
            PAD_IDX
        ] * (max_len - length)

    return ids, length


# ============================================================
# RNN MODEL 1 — BiLSTM
# ============================================================

class RecurrentScorer(nn.Module):

    def __init__(
        self,
        vocab_size,
        emb_dim=200,
        hidden_dim=192,
        rnn_type="lstm",
        num_layers=1,
        dropout=0.3,
        pad_idx=0,
    ):

        super().__init__()

        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(
            vocab_size,
            emb_dim,
            padding_idx=pad_idx,
        )

        self.emb_dropout = nn.Dropout(dropout)

        rnn_cls = (
            nn.LSTM
            if rnn_type == "lstm"
            else nn.GRU
        )

        self.rnn = rnn_cls(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        self.rnn_type = rnn_type

        enc_dim = hidden_dim * 2

        pooled_dim = (
            enc_dim * 2 + enc_dim
        )

        self.scorer = nn.Sequential(

            nn.Linear(
                pooled_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim // 2,
                1,
            ),
        )


    def encode(self, ids, lengths):

        emb = self.emb_dropout(
            self.embedding(ids)
        )

        lengths_clamped = (
            lengths
            .clamp(min=1)
            .cpu()
        )

        packed = pack_padded_sequence(
            emb,
            lengths_clamped,
            batch_first=True,
            enforce_sorted=False,
        )

        packed_out, hidden = self.rnn(
            packed
        )

        outputs, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=ids.size(1),
        )

        mask = (
            ids != self.pad_idx
        ).unsqueeze(-1).float()

        masked_out = outputs * mask

        summed = masked_out.sum(1)

        counts = (
            mask.sum(1)
            .clamp(min=1)
        )

        mean_pool = summed / counts

        very_neg = torch.finfo(
            outputs.dtype
        ).min

        max_pool = (
            masked_out
            .masked_fill(
                mask == 0,
                very_neg,
            )
            .max(1)
            .values
        )

        h_n = hidden[0]

        h_last = torch.cat(
            [
                h_n[-2],
                h_n[-1],
            ],
            dim=-1,
        )

        pooled = torch.cat(
            [
                mean_pool,
                max_pool,
                h_last,
            ],
            dim=-1,
        )

        return pooled


    def forward(self, ids, lengths):

        B, K, L = ids.shape

        ids_flat = ids.reshape(
            B * K,
            L,
        )

        len_flat = lengths.reshape(
            B * K
        )

        pooled = self.encode(
            ids_flat,
            len_flat,
        )

        scores = self.scorer(
            pooled
        ).reshape(B, K)

        return scores


# ============================================================
# ATTENTION POOL
# ============================================================

class AttentionPool(nn.Module):

    def __init__(self, enc_dim):

        super().__init__()

        self.attn = nn.Linear(
            enc_dim,
            1,
        )


    def forward(self, outputs, mask):

        scores = self.attn(
            outputs
        ).squeeze(-1)

        scores = scores.masked_fill(
            mask.squeeze(-1) == 0,
            torch.finfo(
                scores.dtype
            ).min,
        )

        weights = torch.softmax(
            scores,
            dim=1,
        ).unsqueeze(-1)

        return (
            outputs * weights
        ).sum(1)


# ============================================================
# RNN MODEL 2 — Attention BiGRU
# ============================================================

class AttnGRUScorer(nn.Module):

    def __init__(
        self,
        vocab_size,
        emb_dim=200,
        hidden_dim=192,
        dropout=0.3,
        pad_idx=0,
    ):

        super().__init__()

        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(
            vocab_size,
            emb_dim,
            padding_idx=pad_idx,
        )

        self.emb_dropout = nn.Dropout(
            dropout
        )

        self.gru = nn.GRU(
            emb_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        enc_dim = hidden_dim * 2

        self.pool = AttentionPool(
            enc_dim
        )

        self.scorer = nn.Sequential(

            nn.Linear(
                enc_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                1,
            ),
        )


    def encode(self, ids, lengths):

        emb = self.emb_dropout(
            self.embedding(ids)
        )

        lengths_clamped = (
            lengths
            .clamp(min=1)
            .cpu()
        )

        packed = pack_padded_sequence(
            emb,
            lengths_clamped,
            batch_first=True,
            enforce_sorted=False,
        )

        packed_out, _ = self.gru(
            packed
        )

        outputs, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=ids.size(1),
        )

        mask = (
            ids != self.pad_idx
        ).unsqueeze(-1).float()

        pooled = self.pool(
            outputs,
            mask,
        )

        return pooled


    def forward(self, ids, lengths):

        B, K, L = ids.shape

        ids_flat = ids.reshape(
            B * K,
            L,
        )

        len_flat = lengths.reshape(
            B * K
        )

        pooled = self.encode(
            ids_flat,
            len_flat,
        )

        scores = self.scorer(
            pooled
        ).reshape(B, K)

        return scores


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_rnn_models():

    vocab_size = len(VOCAB)

    lstm = RecurrentScorer(
        vocab_size=vocab_size,
        emb_dim=200,
        hidden_dim=192,
        rnn_type="lstm",
        num_layers=1,
        dropout=0.3,
        pad_idx=PAD_IDX,
    )

    bigru = AttnGRUScorer(
        vocab_size=vocab_size,
        emb_dim=200,
        hidden_dim=192,
        dropout=0.3,
        pad_idx=PAD_IDX,
    )

    lstm.load_state_dict(
        torch.load(
            RNN_DIR / "lstm_model.pt",
            map_location=DEVICE,
        )
    )

    bigru.load_state_dict(
        torch.load(
            RNN_DIR / "bigru_model.pt",
            map_location=DEVICE,
        )
    )

    lstm.to(DEVICE)
    bigru.to(DEVICE)

    lstm.eval()
    bigru.eval()

    return lstm, bigru


@st.cache_resource
def load_bert():

    tokenizer = AutoTokenizer.from_pretrained(
        BERT_REPO
    )

    model = AutoModelForMultipleChoice.from_pretrained(
        BERT_REPO
    )

    model.to(DEVICE)

    model.eval()

    return tokenizer, model


lstm_model, bigru_model = load_rnn_models()

bert_tok, bert_model = load_bert()


# ============================================================
# LOAD ENSEMBLE WEIGHTS
# ============================================================

with open(
    RNN_DIR / "ensemble_weights.json",
    "r",
    encoding="utf-8",
) as f:

    weights = json.load(f)


W_LSTM = weights["lstm"]
W_BIGRU = weights["bigru"]
W_BERT = weights["bert"]


# ============================================================
# RNN PREDICTION
# ============================================================

def predict_rnn(model, question, options):

    ids_list = []
    len_list = []

    for option in options:

        ids, length = encode_pair(
            question,
            option,
        )

        ids_list.append(ids)
        len_list.append(length)

    ids_tensor = torch.tensor(
        [ids_list],
        dtype=torch.long,
        device=DEVICE,
    )

    lengths_tensor = torch.tensor(
        [len_list],
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.no_grad():

        scores = model(
            ids_tensor,
            lengths_tensor,
        )

        probs = torch.softmax(
            scores,
            dim=1,
        )

    return probs.cpu().numpy()[0]


# ============================================================
# BERT PREDICTION
# ============================================================

def predict_bert(question, options):

    first_sentences = [
        question
    ] * 5

    tokenized = bert_tok(
        first_sentences,
        options,
        truncation=True,
        max_length=256,
        padding=True,
        return_tensors="pt",
    )

    batch = {
        key: value.unsqueeze(0).to(DEVICE)
        for key, value in tokenized.items()
    }

    with torch.no_grad():

        outputs = bert_model(
            **batch
        )

        probs = torch.softmax(
            outputs.logits,
            dim=1,
        )

    return probs.cpu().numpy()[0]


# ============================================================
# ENSEMBLE
# ============================================================

def predict_ensemble(question, options):

    lstm_probs = predict_rnn(
        lstm_model,
        question,
        options,
    )

    bigru_probs = predict_rnn(
        bigru_model,
        question,
        options,
    )

    bert_probs = predict_bert(
        question,
        options,
    )

    ensemble_probs = (
        W_LSTM * lstm_probs
        + W_BIGRU * bigru_probs
        + W_BERT * bert_probs
    )

    return (
        lstm_probs,
        bigru_probs,
        bert_probs,
        ensemble_probs,
    )


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("🧠 Smart MCQ Solver")

st.write(
    "Enter a multiple-choice question with five options. "
    "The ensemble predicts and ranks the top 3 answers."
)

question = st.text_area(
    "Question",
    placeholder="Enter your question here...",
)

st.subheader("Options")

options = []

for letter in OPTS:

    option = st.text_input(
        f"Option {letter}",
        placeholder=f"Enter option {letter}...",
    )

    options.append(option)


if st.button(
    "Solve MCQ",
    type="primary",
):

    if not question.strip():

        st.error(
            "Please enter a question."
        )

    elif any(
        not option.strip()
        for option in options
    ):

        st.error(
            "Please enter all five options."
        )

    else:

        with st.spinner(
            "Running LSTM + BiGRU + BERT ensemble..."
        ):

            (
                lstm_probs,
                bigru_probs,
                bert_probs,
                ensemble_probs,
            ) = predict_ensemble(
                question,
                options,
            )

        ranking = np.argsort(
            ensemble_probs
        )[::-1]

        st.success(
            "Prediction complete!"
        )

        st.subheader(
            "🏆 Top 3 Answers"
        )

        for rank, idx in enumerate(
            ranking[:3],
            start=1,
        ):

            st.write(
                f"**{rank}. Option {OPTS[idx]}** — "
                f"{ensemble_probs[idx] * 100:.2f}%"
            )

        st.subheader(
            "📊 Ensemble Scores"
        )

        for idx in ranking:

            st.write(
                f"**Option {OPTS[idx]}**: "
                f"{ensemble_probs[idx] * 100:.2f}%"
            )

        with st.expander(
            "View individual model predictions"
        ):

            st.write("### BiLSTM")

            for idx in ranking:

                st.write(
                    f"Option {OPTS[idx]}: "
                    f"{lstm_probs[idx] * 100:.2f}%"
                )

            st.write("### BiGRU")

            for idx in ranking:

                st.write(
                    f"Option {OPTS[idx]}: "
                    f"{bigru_probs[idx] * 100:.2f}%"
                )

            st.write("### BERT")

            for idx in ranking:

                st.write(
                    f"Option {OPTS[idx]}: "
                    f"{bert_probs[idx] * 100:.2f}%"
                )

        st.caption(
            f"Ensemble weights — "
            f"LSTM: {W_LSTM:.2f}, "
            f"BiGRU: {W_BIGRU:.2f}, "
            f"BERT: {W_BERT:.2f}"
        )