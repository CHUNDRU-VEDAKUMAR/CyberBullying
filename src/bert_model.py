# src/bert_model.py
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.preprocessing import clean_text

# Device autodetection: use CUDA when available unless forced to CPU
# Set environment variable FORCE_CPU=1 to force CPU-only behavior
DETECTED_DEVICE = torch.device('cpu') if os.environ.get("FORCE_CPU", "0") == "1" else torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AdvancedContextModel:
    """Context-aware, severity-based, explainable toxicity detector.
    
    Supports multiple pretrained models:
    - 'unitary/toxic-bert' (default): BERT fine-tuned on Jigsaw toxicity
    - 'roberta-base': RoBERTa base model (better contextual understanding)
    - Any HF sequence classification model
    
    All models run on CPU by design for accessibility.
    Integrates with negation handling, context analysis, and LIME explainability.
    """
    def __init__(self, model_name='unitary/toxic-bert', device=None, labels=None):
        self.model_name = model_name
        # allow explicit device override, otherwise use detected device
        self.device = torch.device(device) if device is not None else DETECTED_DEVICE

        print(f"\n[CONTEXT-AWARE] Loading model ({model_name}) on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # load model and move to selected device
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, torch_dtype=torch.float32)
        self.model.to(self.device)
        self.model.eval()

        # Default labels used by Jigsaw models
        self.labels = labels or ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def _prepare(self, text_or_texts):
        if isinstance(text_or_texts, str):
            texts = [text_or_texts]
        else:
            texts = list(text_or_texts)
        texts = [clean_text(t) for t in texts]
        return self.tokenizer(texts, return_tensors='pt', truncation=True, padding=True)

    def predict_proba(self, text_or_texts):
        # Accept single text or list of texts and process in batches for CPU efficiency
        if isinstance(text_or_texts, str):
            texts = [text_or_texts]
        else:
            texts = list(text_or_texts)

        batch_size = 8
        all_probs = []
        # Use inference_mode for slightly better perf on CPU
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = self._prepare(batch)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                probs = torch.sigmoid(outputs.logits).cpu().numpy()
                # probs shape: (batch_size, num_labels)
                all_probs.append(probs)

        if not all_probs:
            return np.zeros((0, len(self.labels)))

        return np.vstack(all_probs)

    def predict(self, text):
        probs = self.predict_proba(text).squeeze()
        if probs.ndim == 0:
            probs = [float(probs)]
        return {label: float(score) for label, score in zip(self.labels, probs)}