import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from src.bert_model import AdvancedContextModel

def evaluate_on_csv(csv_path, text_col='comment_text', label_cols=None, model_name=None):
    df = pd.read_csv(csv_path)
    if label_cols is None:
        label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    model = AdvancedContextModel(model_name=model_name) if model_name else AdvancedContextModel()

    y_true = df[label_cols].fillna(0).values
    y_pred = []

    for _, row in df.iterrows():
        text = row[text_col]
        scores = model.predict(text)
        preds = [1 if scores.get(l, 0) > 0.5 else 0 for l in label_cols]
        y_pred.append(preds)

    y_pred = pd.np.array(y_pred)

    # Compute per-label metrics
    precisions, recalls, f1s, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    results = {l: {'precision': float(p), 'recall': float(r), 'f1': float(f)} for l, p, r, f in zip(label_cols, precisions, recalls, f1s)}
    return results

if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/test.csv'
    res = evaluate_on_csv(path)
    for k, v in res.items():
        print(f"{k}: P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}")
