# RUN THIS CELL AT THE END OF THE TRAINING NOTEBOOK
# before deleting trainer / bert_model.
#
# It creates the exact artifacts expected by the Streamlit deployment.

from pathlib import Path
import json
import torch

DEPLOY_DIR = Path("deployment_artifacts")
RNN_DIR = DEPLOY_DIR / "rnn"
BERT_DIR = DEPLOY_DIR / "bert_model"
RNN_DIR.mkdir(parents=True, exist_ok=True)
BERT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Save recurrent-model vocabulary + preprocessing metadata
with open(RNN_DIR / "vocab.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "stoi": VOCAB,
            "max_len": MAX_LEN,
            "pad_idx": PAD_IDX,
            "unk_idx": UNK_IDX,
            "sep_idx": SEP_IDX,
        },
        f,
        ensure_ascii=False,
    )

# 2. Save trained LSTM and BiGRU weights
torch.save(lstm_model.state_dict(), RNN_DIR / "lstm_model.pt")
torch.save(bigru_model.state_dict(), RNN_DIR / "bigru_model.pt")

# 3. Save the optimized ensemble weights
with open(RNN_DIR / "ensemble_weights.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "lstm": float(w1),
            "bigru": float(w2),
            "bert": float(w3),
        },
        f,
        indent=2,
    )

# 4. Save the fine-tuned BERT model and tokenizer.
# IMPORTANT: do this BEFORE `del trainer, bert_model`.
trainer.save_model(str(BERT_DIR))
bert_tok.save_pretrained(str(BERT_DIR))

print("Deployment artifacts created in:", DEPLOY_DIR.resolve())
print("LSTM:", RNN_DIR / "lstm_model.pt")
print("BiGRU:", RNN_DIR / "bigru_model.pt")
print("BERT:", BERT_DIR)
