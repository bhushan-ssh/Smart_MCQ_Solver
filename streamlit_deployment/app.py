import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoTokenizer, AutoModelForMultipleChoice

# ---------------------------------------------------------
# Smart MCQ Solver - Streamlit inference app
# Models:
#   1. BiLSTM
#   2. BiGRU + Attention
#   3. bert-base-uncased fine-tuned for multiple choice
# The three probability distributions are combined using
# the weights saved during notebook training.
# ---------------------------------------------------------

st.set_page_config(
    page_title="Smart MCQ Solver",
    page_icon="🧠",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
RNN_DIR = MODEL_DIR / "rnn"
BERT_DIR = MODEL_DIR / "bert_model"

OPTS = ["A", "B", "C", "D", "E"]
OPT2IDX = {o: i for i, o in enumerate(OPTS)}

TOKEN_RE = __import__("re").compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[^\sA-Za-z0-9]")
def tokenize(text):
    return TOKEN_RE.findall(str(text).lower())

PAD = "<pad>"
UNK = "<unk>"
SEP = "<sep>"


class RecurrentScorer(nn.Module):
    def __init__(self, vocab_size, emb_dim=200, hidden_dim=192,
                 rnn_type="lstm", num_layers=1, dropout=0.3, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout)

        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        enc_dim = hidden_dim * 2
        pooled_dim = enc_dim * 2 + enc_dim
        self.rnn_type = rnn_type
        self.scorer = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode(self, ids, lengths):
        emb = self.emb_dropout(self.embedding(ids))
        lengths_clamped = lengths.clamp(min=1).cpu()
        packed = pack_padded_sequence(
            emb, lengths_clamped, batch_first=True, enforce_sorted=False
        )
        packed_out, hidden = self.rnn(packed)
        outputs, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=ids.size(1)
        )

        mask = (ids != self.pad_idx).unsqueeze(-1).float()
        masked_out = outputs * mask
        summed = masked_out.sum(1)
        counts = mask.sum(1).clamp(min=1)
        mean_pool = summed / counts

        very_neg = torch.finfo(outputs.dtype).min
        max_pool = masked_out.masked_fill(mask == 0, very_neg).max(1).values

        h_n = hidden[0] if self.rnn_type == "lstm" else hidden
        h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)

        return torch.cat([mean_pool, max_pool, h_last], dim=-1)

    def forward(self, ids, lengths):
        b, k, length = ids.shape
        ids_flat = ids.reshape(b * k, length)
        len_flat = lengths.reshape(b * k)
        pooled = self.encode(ids_flat, len_flat)
        return self.scorer(pooled).reshape(b, k)


class AttentionPool(nn.Module):
    def __init__(self, enc_dim):
        super().__init__()
        self.attn = nn.Linear(enc_dim, 1)

    def forward(self, outputs, mask):
        scores = self.attn(outputs).squeeze(-1)
        scores = scores.masked_fill(
            mask.squeeze(-1) == 0, torch.finfo(scores.dtype).min
        )
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (outputs * weights).sum(1)


class AttnGRUScorer(nn.Module):
    def __init__(self, vocab_size, emb_dim=200, hidden_dim=192,
                 dropout=0.3, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            emb_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        enc_dim = hidden_dim * 2
        self.pool = AttentionPool(enc_dim)
        self.scorer = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, ids, lengths):
        emb = self.emb_dropout(self.embedding(ids))
        lengths_clamped = lengths.clamp(min=1).cpu()
        packed = pack_padded_sequence(
            emb, lengths_clamped, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        outputs, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=ids.size(1)
        )
        mask = (ids != self.pad_idx).unsqueeze(-1).float()
        return self.pool(outputs, mask)

    def forward(self, ids, lengths):
        b, k, length = ids.shape
        ids_flat = ids.reshape(b * k, length)
        len_flat = lengths.reshape(b * k)
        pooled = self.encode(ids_flat, len_flat)
        return self.scorer(pooled).reshape(b, k)


@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_path = RNN_DIR / "vocab.json"
    lstm_path = RNN_DIR / "lstm_model.pt"
    bigru_path = RNN_DIR / "bigru_model.pt"
    weights_path = RNN_DIR / "ensemble_weights.json"

    required = [vocab_path, lstm_path, bigru_path, weights_path]
    missing = [str(p.relative_to(BASE_DIR)) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model files: " + ", ".join(missing)
        )
    if not BERT_DIR.exists():
        raise FileNotFoundError(
            "Missing models/bert_model directory."
        )

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    vocab = vocab_data["stoi"]
    max_len = int(vocab_data.get("max_len", 96))
    pad_idx = int(vocab_data["pad_idx"])

    lstm = RecurrentScorer(
        vocab_size=len(vocab),
        emb_dim=200,
        hidden_dim=192,
        rnn_type="lstm",
        num_layers=1,
        dropout=0.3,
        pad_idx=pad_idx,
    )
    bigru = AttnGRUScorer(
        vocab_size=len(vocab),
        emb_dim=200,
        hidden_dim=192,
        dropout=0.3,
        pad_idx=pad_idx,
    )

    lstm.load_state_dict(torch.load(lstm_path, map_location=device))
    bigru.load_state_dict(torch.load(bigru_path, map_location=device))
    lstm.to(device).eval()
    bigru.to(device).eval()

    with open(weights_path, "r", encoding="utf-8") as f:
        weights = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
    bert = AutoModelForMultipleChoice.from_pretrained(BERT_DIR)
    bert.to(device).eval()

    return vocab, max_len, pad_idx, lstm, bigru, tokenizer, bert, weights, device


def encode_pair(question, option, vocab, max_len, pad_idx, unk_idx, sep_idx):
    tokens = tokenize(question) + [SEP] + tokenize(option)
    tokens = tokens[:max_len]
    ids = [vocab.get(t, unk_idx) for t in tokens]
    length = len(ids)
    ids += [pad_idx] * (max_len - length)
    return ids, length


def rnn_predict(question, options, model, vocab, max_len, pad_idx, device):
    unk_idx = vocab.get(UNK, 1)
    sep_idx = vocab.get(SEP, 2)

    ids_list, lengths = [], []
    for option in options:
        ids, length = encode_pair(
            question, option, vocab, max_len, pad_idx, unk_idx, sep_idx
        )
        ids_list.append(ids)
        lengths.append(length)

    ids_tensor = torch.tensor([ids_list], dtype=torch.long, device=device)
    len_tensor = torch.tensor([lengths], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(ids_tensor, len_tensor)
        return torch.softmax(logits, dim=1).cpu().numpy()[0]


def bert_predict(question, options, tokenizer, model, device):
    first = [question] * 5
    encoded = tokenizer(
        first,
        options,
        truncation=True,
        max_length=256,
        padding=True,
        return_tensors="pt",
    )
    encoded = {k: v.unsqueeze(0).to(device) for k, v in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits
        return torch.softmax(logits, dim=1).cpu().numpy()[0]


def solve(question, options):
    (
        vocab, max_len, pad_idx, lstm, bigru,
        tokenizer, bert, weights, device
    ) = load_models()

    p_lstm = rnn_predict(
        question, options, lstm, vocab, max_len, pad_idx, device
    )
    p_bigru = rnn_predict(
        question, options, bigru, vocab, max_len, pad_idx, device
    )
    p_bert = bert_predict(question, options, tokenizer, bert, device)

    w1 = float(weights["lstm"])
    w2 = float(weights["bigru"])
    w3 = float(weights["bert"])

    ensemble = w1 * p_lstm + w2 * p_bigru + w3 * p_bert
    order = np.argsort(-ensemble)

    return {
        "lstm": p_lstm,
        "bigru": p_bigru,
        "bert": p_bert,
        "ensemble": ensemble,
        "order": order,
        "weights": (w1, w2, w3),
    }


st.title("🧠 Smart MCQ Solver")
st.caption("DL & GenAI Project — BiLSTM + BiGRU + fine-tuned BERT ensemble")

question = st.text_area(
    "Question",
    placeholder="Enter your multiple-choice question here...",
    height=130,
)

cols = st.columns(5)
options = []
for i, letter in enumerate(OPTS):
    with cols[i]:
        options.append(
            st.text_area(
                f"Option {letter}",
                placeholder=f"Enter option {letter}",
                height=130,
                key=f"option_{letter}",
            )
        )

if st.button("🚀 Predict Answer", type="primary", use_container_width=True):
    if not question.strip():
        st.error("Please enter a question.")
        st.stop()

    if any(not x.strip() for x in options):
        st.error("Please enter all five options (A–E).")
        st.stop()

    try:
        with st.spinner("Running LSTM + BiGRU + BERT ensemble..."):
            result = solve(question.strip(), [x.strip() for x in options])

        order = result["order"]
        probs = result["ensemble"]

        winner = OPTS[int(order[0])]
        st.success(
            f"Predicted answer: **{winner}** — {options[int(order[0])]}"
        )

        st.subheader("Top 3 Answer Ranking")
        for rank, idx in enumerate(order[:3], start=1):
            st.write(
                f"**{rank}. {OPTS[int(idx)]}** — "
                f"{probs[int(idx)] * 100:.2f}% — {options[int(idx)]}"
            )

        st.subheader("Ensemble probabilities")
        chart_data = {
            "Option": OPTS,
            "Probability": [float(x) for x in probs],
        }
        st.bar_chart(chart_data, x="Option", y="Probability")

        with st.expander("Model details"):
            w1, w2, w3 = result["weights"]
            st.write(
                f"Ensemble weights — LSTM: {w1:.2f}, "
                f"BiGRU: {w2:.2f}, BERT: {w3:.2f}"
            )
            st.write("The displayed ranking is produced from the weighted probability ensemble.")

    except Exception as exc:
        st.error("The model could not be loaded or executed.")
        st.exception(exc)
