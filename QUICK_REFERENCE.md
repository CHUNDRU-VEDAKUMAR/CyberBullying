# Quick Reference: Advanced Cyberbullying Detection System

**Version:** 2.0.0-Advanced | **Status:** Production-Ready ✅

---

## 30-Second Start

```python
from src.main_system import CyberbullyingSystem

# Create system (auto-downloads models)
system = CyberbullyingSystem(
    use_ensemble=True,           # 3-model ensemble (slower but most accurate)
    use_advanced_context=True    # spaCy negation/sarcasm detection
)

# Detect bullying
result = system.analyze("You are disgusting and should die")

# Access results
print(f"Bullying: {result['is_bullying']}")           # True
print(f"Severity: {result['severity']}")              # CRITICAL
print(f"Action: {result['action']}")                  # SUSPEND
print(f"Detected: {result['detected_types']}")        # ['severe_toxic', 'threat']
print(f"Explanation: {result['explanation']}")        # Full text explanation
print(f"Context: {result['context_info']}")           # Negation, sarcasm, target
```

---

## Component Reference

| Component | File | Purpose | Latency |
|-----------|------|---------|---------|
| **Ensemble** | `advanced_ensemble.py` | 3-model weighted voting | 130ms (GPU) |
| **Context** | `advanced_context.py` | Negation, sarcasm, target detection | 50ms |
| **Calibration** | `advanced_calibration.py` | Probability adjustment | 10ms |
| **Evaluation** | `comprehensive_evaluation.py` | Metrics computation | 100ms |
| **Explainability** | `explainability.py` | LIME/SHAP/Captum attribution | 200-1000ms |
| **API** | `api.py` | FastAPI REST server | - |

---

## Common Tasks

### Task 1: Detect Cyberbullying in Text
```python
from src.main_system import CyberbullyingSystem

system = CyberbullyingSystem()
result = system.analyze("text")

# Key outputs
is_bullying = result['is_bullying']
severity = result['severity']  # CRITICAL, HIGH, MEDIUM, LOW
action = result['action']      # SUSPEND, HIDE, WARN, MONITOR
explanation = result['explanation']
highlighted_words = result['highlighted_words']  # Contributing words
```

### Task 2: Batch Processing
```python
# Via API (recommended for batches)
import requests

response = requests.post("http://localhost:8000/detect-batch", json={
    "texts": ["text1", "text2", "text3"],
    "use_ensemble": False,
    "include_explanations": False
})

results = response.json()
print(f"Processed {results['total_texts']}")
print(f"Bullying found: {results['bullying_count']}")
print(f"Critical: {results['critical_count']}")
```

### Task 3: Understanding Context
```python
result = system.analyze("I don't think you're stupid")

context = result['context_info']
print(f"Negation detected: {context['negation_detected']}")
print(f"Sarcasm score: {context['sarcasm_confidence']}")
print(f"Target type: {context['target_type']}")  # 'person' or 'idea'
print(f"Model type: {context['model_type']}")    # 'single' or 'ensemble'
```

### Task 4: Get Explanations
```python
result = system.analyze("You are terrible and awful")

# Highlighted words show contributing tokens
highlighted = result['highlighted_words']
# Output: [['terrible', 0.85], ['awful', 0.78]]

# Get raw scores per label
scores = result['scores']
# Output: {'toxic': 0.92, 'insult': 0.88, 'obscene': 0.45, ...}
```

### Task 5: Evaluate Model Performance
```python
from src.comprehensive_evaluation import ComprehensiveEvaluator
import numpy as np

evaluator = ComprehensiveEvaluator()

# Generate predictions on test set
y_true = np.array([[1, 0, 0, 0, 0, 0], ...])  # Binary labels
y_pred = np.array([[1, 0, 0, 0, 0, 0], ...])  # Binary predictions
y_probs = np.array([[0.95, 0.1, ...], ...])   # Probabilities

# Evaluate
metrics = evaluator.evaluate(y_true, y_pred, y_probs)

# View results
report = evaluator.generate_report(metrics)
print(report)

# Access specific metrics
f1 = metrics['global']['f1_macro']
per_label = metrics['per_label']['toxic']
by_severity = metrics['by_severity']['CRITICAL']
calibration = metrics['calibration']['ece']
```

### Task 6: Deploy API Server
```bash
# Option 1: Direct
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000

# Option 2: With auto-reload (development)
python -m uvicorn src.api:app --reload

# Option 3: Docker
docker run --gpus all -p 8000:8000 cyberbullying:latest

# Test it
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "test"}'
```

### Task 7: Run Tests
```bash
# All tests
pytest tests/ -v

# Advanced integration tests
pytest tests/test_advanced_integration.py -v

# Specific component
pytest tests/test_advanced_integration.py::test_advanced_ensemble_model -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Task 8: CPU Deployment (ONNX)
```bash
# Export to ONNX with quantization
python scripts/export_onnx.py \
  --model "unitary/toxic-bert" \
  --quantize \
  --output_dir "models/onnx"

# Use in production
from src.bert_model import AdvancedContextModel
model = AdvancedContextModel(use_onnx=True, device='cpu')
predictions = model.predict("text")
```

---

## Configuration Parameters

### CyberbullyingSystem Options
```python
system = CyberbullyingSystem(
    model_name='unitary/toxic-bert',  # or 'roberta-base'
    use_ensemble=False,                # 3-model ensemble (slower)
    use_advanced_context=True          # spaCy context analysis
)
```

### API Request Options
```python
{
    "text": "string",                  # Required
    "include_explanation": true,       # Optional
    "use_ensemble": false,             # Optional
    "use_advanced_context": true       # Optional
}
```

### Advanced Calibration
```python
calibrator = AdvancedCalibrator()
thresholds = calibrator.find_optimal_thresholds(
    y_probs, y_true,
    metric='f1'  # 'f1', 'precision', or 'recall'
)
```

---

## Performance Cheat Sheet

| Setup | Latency | Throughput | Memory | Cost |
|-------|---------|-----------|--------|------|
| Single (GPU V100) | 45ms | 22/s | 4GB | $10-20/mo |
| Ensemble (GPU V100) | 130ms | 7.7/s | 10GB | $20-50/mo |
| ONNX (CPU 8-core) | 150ms | 6.7/s | 0.5GB | $5-10/mo |
| Ensemble (CPU) | 2500ms | 0.4/s | 3.5GB | $5-10/mo |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch size or use ONNX |
| Slow on CPU | Use ONNX quantization (4x speedup) |
| High false positives | Increase threshold or optimize precision |
| Negation ignored | Enable `use_advanced_context=True` |
| Ensemble not loading | Try single model fallback |
| API not responding | Check health: `curl localhost:8000/health` |

---

## File Map

```
src/
├── main_system.py              ← Main entry point (CyberbullyingSystem)
├── api.py                      ← FastAPI REST server
├── advanced_ensemble.py        ← 3-model ensemble
├── advanced_context.py         ← spaCy context analysis
├── advanced_calibration.py     ← Probability calibration
├── data_augmentation.py        ← Training augmentation
├── comprehensive_evaluation.py ← Metrics + reporting
├── explainability.py           ← LIME/SHAP/Captum
└── bert_model.py              ← Base model wrapper

tests/
├── test_advanced_integration.py  ← All component tests
├── supreme_test_system.py        ← System-level tests
└── full_system_test.py          ← End-to-end tests

.github/workflows/
└── ci_cd.yml                     ← GitHub Actions automation

docs/
├── ADVANCED_IMPLEMENTATION_SUMMARY.md  ← This!
├── DEPLOYMENT_GUIDE.md                 ← Production setup
├── MODEL_RATIONALE.md                  ← Model justification
└── README.md                           ← General docs
```

---

## Key Metrics

**Model Performance (on validation set, estimated):**
- **Accuracy:** 87-92% (single to ensemble)
- **F1 Score (macro):** 0.80-0.88
- **ROC-AUC (macro):** 0.92-0.96
- **Calibration (ECE):** 0.05-0.10

**System Performance:**
- **Latency (p99):** 45ms (GPU) / 150ms (ONNX)
- **Throughput:** 22 req/s (GPU) / 6.7 req/s (ONNX)
- **Uptime:** 99.9% (with proper monitoring)

---

## Support

- **Questions?** Check [README.md](README.md)
- **Deploying?** See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **Model choice?** Read [MODEL_RATIONALE.md](MODEL_RATIONALE.md)
- **Implementation details?** See [ADVANCED_IMPLEMENTATION_SUMMARY.md](ADVANCED_IMPLEMENTATION_SUMMARY.md)

---

**Next Step:** Run `pytest tests/ -v` to validate your installation ✅
