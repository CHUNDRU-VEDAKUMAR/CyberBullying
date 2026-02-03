#!/usr/bin/env python3
"""
Export HF sequence-classification model to ONNX for CPU inference.

Usage:
    python scripts/export_onnx.py --model unitary/toxic-bert --output model.onnx

Note: This script requires `transformers` and `onnx`/`onnxruntime`.
"""
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def export(model_name, output_path, seq_len=128):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    sample = "This is a test"
    inputs = tokenizer(sample, return_tensors='pt', truncation=True, padding='max_length', max_length=seq_len)
    input_names = list(inputs.keys())
    input_tuple = (inputs['input_ids'], inputs.get('attention_mask', None))

    # Build dynamic axes
    dynamic_axes = {k: {0: 'batch_size', 1: 'seq_len'} for k in input_names}
    dynamic_axes['output'] = {0: 'batch_size'}

    with torch.no_grad():
        torch.onnx.export(
            model,
            (inputs['input_ids'], inputs.get('attention_mask', None)),
            output_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            opset_version=13,
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='unitary/toxic-bert')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    export(args.model, args.output)
