import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from src.bert_model import AdvancedContextModel


def evaluate_on_csv(csv_path, text_col='comment_text', label_cols=None, model_name=None, batch_size=32):
    df = pd.read_csv(csv_path)
    if label_cols is None:
        label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    model = AdvancedContextModel(model_name=model_name) if model_name else AdvancedContextModel()

    texts = df[text_col].astype(str).tolist()
    y_true = df[label_cols].fillna(0).values

    # Batch inference for speed
    probs = model.predict_proba(texts)
    preds = (probs > 0.5).astype(int)

    # Compute per-label metrics
    precisions, recalls, f1s, _ = precision_recall_fscore_support(y_true, preds, average=None, zero_division=0)
    results = {l: {'precision': float(p), 'recall': float(r), 'f1': float(f)} for l, p, r, f in zip(label_cols, precisions, recalls, f1s)}
    return results

if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/test.csv'
    res = evaluate_on_csv(path)
    for k, v in res.items():
        print(f"{k}: P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}")
