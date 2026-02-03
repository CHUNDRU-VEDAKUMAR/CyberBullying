import os
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

# Device autodetection: allow FORCE_CPU=1 to force CPU, otherwise use CUDA when available
DEVICE = torch.device('cpu') if os.environ.get("FORCE_CPU", "0") == "1" else torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(csv_path, text_col='text', label_cols=None, sample_frac=None):
    df = pd.read_csv(csv_path)
    if sample_frac:
        df = df.sample(frac=sample_frac, random_state=42)

    if label_cols is None:
        # assume a single binary label column 'label' or use 'toxic'
        if 'label' in df.columns:
            df['labels'] = df['label']
        elif 'toxic' in df.columns:
            df['labels'] = df['toxic']
        else:
            raise ValueError('No label column found. Provide label_cols.')
        texts = df[text_col].astype(str).tolist()
        labels = df['labels'].astype(int).tolist()
        return texts, labels

    # multi-label: return dict of lists
    texts = df[text_col].astype(str).tolist()
    labels = df[label_cols].fillna(0).astype(int).values.tolist()
    return texts, labels


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        lbl = self.labels[idx]
        # support multi-label (list/array) as float for BCEWithLogitsLoss
        if isinstance(lbl, (list, tuple)) or (hasattr(lbl, 'shape') and getattr(lbl, 'ndim', 0) >= 1):
            item['labels'] = torch.tensor(lbl, dtype=torch.float)
        else:
            item['labels'] = torch.tensor(lbl, dtype=torch.long)
        return item


def train_model(model_name, train_csv, output_dir, text_col='text', label_cols=None, epochs=1, batch_size=8, lr=5e-5, sample_frac=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(DEVICE)

    # if multi-label, set problem_type so Trainer uses BCEWithLogitsLoss
    if label_cols is not None and isinstance(label_cols, (list, tuple)):
        model.config.problem_type = 'multi_label_classification'

    texts, labels = load_data(train_csv, text_col=text_col, label_cols=label_cols, sample_frac=sample_frac)
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)

    enc_train = tokenizer(train_texts, truncation=True, padding=True)
    enc_val = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = SimpleDataset(enc_train, train_labels)
    val_dataset = SimpleDataset(enc_val, val_labels)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=10,
        save_total_limit=2,
        fp16=True if DEVICE.type == 'cuda' else False,
        evaluation_strategy='epoch'
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    trainer.save_model(output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='roberta-base')
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--output_dir', default='./models/finetuned')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--sample_frac', type=float, default=None)
    args = parser.parse_args()

    train_model(args.model, args.train_csv, args.output_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, sample_frac=args.sample_frac)
