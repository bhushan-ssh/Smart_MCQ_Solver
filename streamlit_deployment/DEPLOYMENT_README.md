# Smart MCQ Solver — Streamlit Deployment

This folder contains the deployment files for the trained Smart MCQ Solver ensemble.

## Expected model files

```text
models/
├── rnn/
│   ├── vocab.json
│   ├── lstm_model.pt
│   ├── bigru_model.pt
│   └── ensemble_weights.json
└── bert_model/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    ├── tokenizer.json
    ├── vocab.txt
    └── ...
```

Run `app.py` with:

```bash
streamlit run app.py
```

Do not commit Kaggle credentials, W&B API keys, `.env` files, or private secrets.

The BERT model file may be too large for normal GitHub Git. If `model.safetensors` exceeds GitHub's normal file limit, host the fine-tuned BERT model on Hugging Face Hub and modify `app.py` to load it from the Hub instead of `models/bert_model`.
