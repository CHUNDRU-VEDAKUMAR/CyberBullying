import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.preprocessing import clean_text

# Device autodetection: honor FORCE_CPU env var, otherwise use CUDA when available
DETECTED_DEVICE = torch.device('cpu') if os.environ.get("FORCE_CPU", "0") == "1" else torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelManager:
    """Flexible model loader for different pretrained classifiers with context-awareness.
    
    Supports:
    - 'unitary/toxic-bert' (default): BERT + Jigsaw toxicity fine-tuning
    - 'roberta-base': RoBERTa base model (better contextual understanding for negations, sarcasm)
    - Any HuggingFace sequence classification model
    
    Features:
    - Context-aware processing (negation handling, target type detection)
    - Severity-based scoring and intervention mapping
    - Explainability via LIME with fallback perturbation
    - Actionable interventions based on severity and confidence
    - CPU-only execution for accessibility
    """
    def __init__(self, model_name='unitary/toxic-bert', device=None, labels=None):
        self.model_name = model_name
        self.device = torch.device(device) if device is not None else DETECTED_DEVICE

        print(f"[MODEL MANAGER] Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.float32)
        self.model.to(self.device)
        self.model.eval()

        # Default labels used by Jigsaw models
        self.labels = labels or ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def _prepare(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        texts = [clean_text(t) for t in texts]
        return self.tokenizer(texts, return_tensors='pt', truncation=True, padding=True)

    def predict_proba(self, texts):
        """Return NxL array of probabilities for L labels."""
        inputs = self._prepare(texts)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()
        # If single sample, return shape (1, L)
        return probs

    def predict(self, text):
        probs = self.predict_proba(text).squeeze()
        if probs.ndim == 0:
            probs = [float(probs)]
        return {label: float(score) for label, score in zip(self.labels, probs)}
