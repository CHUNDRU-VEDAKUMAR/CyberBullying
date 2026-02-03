# Model & Stack Rationale

This document explains why this project chooses the particular models and tooling in the stack, how those choices support the project title "Context-Aware, Severity-Based and Explainable Cyberbullying Detection with Actionable Interventions", and how they compare with other common alternatives.

**Key design goals**
- High contextual understanding (negation, sarcasm, target detection)
- Multi-label severity-aware outputs with calibrated confidences
- Explainability (word-level attribution per label)
- Production-friendly performance (low-latency inference, CPU friendliness)
- Practicality: reproducible training, tuning, and deployment

---

## Chosen Models & Tools

- `unitary/toxic-bert` (default in code): a BERT-base model fine-tuned on Jigsaw toxicity datasets. Chosen as a reliable, well-known baseline that yields strong baseline detection for toxicity labels and is small enough to run on constrained infrastructure.
- `roberta-base` (recommended alternative): a stronger contextual encoder for negations, sarcasm, and subtle phrasing compared with BERT-uncased; good trade-off between accuracy and compute.
- Hugging Face Transformers + Trainer: standard ecosystem for reproducible fine-tuning, easy multi-label configuration, and access to tokenizers and export utilities.
- Lightweight explainers (perturbation fallback) + optional LIME/SHAP/Captum: ensures explainability across environments (LIME fallback for CPU-only, Captum/SHAP for model-aware attributions when available).

---

## Why this stack fits the project goals

- Context awareness: Transformer encoders (BERT/RoBERTa) model token interactions and long-range dependencies, enabling correct handling of negations ("I don't kill you"), positive achievements ("you killed it"), and sarcasm when fine-tuned with contextual labels.
- Severity & multi-label outputs: fine-tuned sequence classification heads with sigmoid activations naturally support multi-label outputs required to map to severity categories (toxic, insult, threat, etc.). Correct multi-label losses (BCEWithLogitsLoss) and post-hoc calibration let the system map probabilities to actionable severity thresholds.
- Explainability: model-aware attributions (Integrated Gradients, SHAP) or perturbation-based methods expose word-level impacts per label; combined with the ontology, this provides transparent, actionable decisions for moderators.
- Deployment practicality: using moderately sized transformer models (base-size) balances accuracy with CPU inference time. The codebase also supports exporting to ONNX and quantization for faster CPU serving.

---

## Performance & Metrics Considerations

- Accuracy / F1: `roberta-base` and DeBERTa variants typically outperform vanilla BERT on contextual tasks (negation, sarcasm), which translates to higher recall and F1 for rare but critical labels (e.g., `threat`).
- Latency & Throughput: smaller distilled models (DistilBERT / tiny RoBERTa) provide much lower latency and memory use, but at the cost of lower accuracy for nuanced contexts. For production, export to ONNX and apply dynamic quantization to get large speedups while keeping base models.
- Calibration: raw model probabilities are not well calibrated. The stack includes temperature scaling or isotonic regression on a held-out validation set to produce reliable confidence scores used by the ontology for thresholded, severity-based interventions.
- Explainability fidelity: LIME is model-agnostic but can be slow and sometimes unstable; Captum (Integrated Gradients) or SHAP (GradientExplainer) provide more faithful attributions for deep models.

---

## Comparison with Alternatives

1. Baselines: TF-IDF + SVM / RandomForest
   - Pros: Extremely fast to train and serve, interpretable features, cheap resource usage.
   - Cons: Poor at context-sensitive language (negation, sarcasm, positive achievements). Lower recall for nuanced labels. Cannot easily provide robust per-label attributions aligned with deep models.
   - Use-case: fast prototypes, fallback when no GPUs and labeled data is tiny.

2. Distilled / Tiny Transformer Models (DistilBERT, Tiny RoBERTa)
   - Pros: Much faster inference and smaller memory footprint; good for high-throughput CPU deployments.
   - Cons: Reduced ability to model subtle contextual cues; can miss threats phrased indirectly. Still significantly better than classical ML for semantics, but lower than full base-size models.
   - Use-case: latency-sensitive serving, on-device inference.

3. Large-state-of-the-art encoders (RoBERTa-large, DeBERTa v3, ELECTRA-large)
   - Pros: Best accuracy and contextual understanding; improved F1 on small/rare classes.
   - Cons: High compute and memory cost; longer latency and more expensive inference (often needing GPUs). Heavier to fine-tune and deploy.
   - Use-case: batch scoring, offline analysis, or when serving budget allows GPU-backed inference.

4. LLMs (GPT-family / instruction-tuned models)
   - Pros: Flexible zero-shot and few-shot capability; can reason and provide high-quality explanations and paraphrases.
   - Cons: High latency and cost; often overkill for simple classification; risk of hallucination and unpredictable outputs; explainability and calibration are harder to control.
   - Use-case: research experiments, building advanced moderation assistants, not ideal for deterministic automated interventions.

5. Ensemble Approaches
   - Pros: Combining a transformer with classical models or multiple transformer checkpoints improves robustness and recall on edge cases.
   - Cons: Increases inference cost and system complexity; requires careful calibration/aggregation.

---

## Why the repository's choices are a practical best-fit

- Balanced trade-offs: the default and recommended models (`unitary/toxic-bert`, `roberta-base`) give strong contextual performance without the operational costs of very large models.
- Explainability-first: the pipeline integrates per-label attributions and an ontology, making interventions auditable and defensible.
- Safety & cost control: the system supports CPU-first deployment, ONNX export, quantization, and optional distillation — letting teams optimize based on available infra.

---


This rationale balances accuracy, context understanding, explainability, and operational practicality — matching the project promise of context-aware, severity-based, explainable cyberbullying detection with actionable interventions.
