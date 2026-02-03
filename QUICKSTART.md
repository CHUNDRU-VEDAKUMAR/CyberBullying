# ⚡ QUICK START GUIDE

## Four Pillars of Cyberbullying Detection

This system implements **four core pillars** to detect cyberbullying accurately:

1. **Context-Aware** 🧠 - Understands negations, sarcasm, and intent
2. **Severity-Based** ⚖️ - Maps toxicity to actionable intervention levels
3. **Explainable** 👁️ - Shows which words triggered detection
4. **Actionable** 🛡️ - Recommends specific moderator actions

---

## 🚀 Get Running in 3 Steps

### **Step 1: Install CPU PyTorch** (See [CPU_INSTALL.md](CPU_INSTALL.md))
```bash
# Windows
pip install --index-url https://download.pytorch.org/whl/cpu torch --extra-index-url https://pypi.org/simple

# Linux/Mac
pip3 install --index-url https://download.pytorch.org/whl/cpu torch --extra-index-url https://pypi.org/simple
```

### **Step 2: Install Other Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 3: Run Interactive Demo** (Recommended)
```bash
python run_project.py
```

Then type comments to analyze:
```
Enter a comment to test (or type 'exit'): You killed that presentation!
```

Output:
```
✅ SAFE
Context: Positive achievement context detected
Explanation: "killed" in positive context is praise, not threat
```

---

## 🎯 Live Examples: All Four Pillars

### Example 1: Context-Aware Negation
```
Text: "I don't kill you"
Output: ✅ SAFE
Why: Negation handler detected "don't" negates the threat
Context Info: Negation found: explicit negation
```

### Example 2: Severity-Based Action (MEDIUM)
```
Text: "You're an idiot"
Output: 🛑 BULLYING DETECTED
Severity: MEDIUM
Detected Label: insult, toxic
Action: HIDE_COMMENT + ISSUE_WARNING_STRIKE_1
Confidence: 0.82
```

### Example 3: Critical Severity (Actionable Intervention)
```
Text: "I will kill you"
Output: 🛑 BULLYING DETECTED
Severity: CRITICAL
Action: BLOCK_ACCOUNT_IMMEDIATELY + REPORT_TO_CYBER_CELL
Confidence: 0.95
```

### Example 4: Explainable (LIME)
```
Text: "You're an idiot"
Visual Proof: The model flagged these specific words:
  👉 'idiot' (Impact Score: 0.87)
  👉 'you' (Impact Score: 0.12)
```

---

## 🔄 Use a Different Model (RoBERTa for Better Context)

Default model is `unitary/toxic-bert` (fast, baseline).  
For better contextual understanding, use `roberta-base`:

**Option 1**: Edit `run_project.py`:
```python
system = CyberbullyingSystem(model_name='roberta-base')
```

**Option 2**: Use in code:
```python
from src.main_system import CyberbullyingSystem
system = CyberbullyingSystem(model_name='roberta-base')
```

---

## 🧪 Validate Your Setup

**Lightweight tests** (no model download needed):
```bash
python test_ontology.py        # Test severity and intervention logic
python test_explainability.py  # Test explanation system
```

**Full integration test** (downloads model ~400MB on first run):
```bash
python test_system.py
```

## 🛠️ Performance & Production Tips

- To speed up CPU inference, export the model to ONNX and apply INT8 quantization using ONNX Runtime. Example export script:

```bash
python scripts/export_onnx.py --model unitary/toxic-bert --output model.onnx
# then use onnxruntime tools to quantize (see ONNXRuntime docs)
```

- Install `spaCy` and its model to enable dependency-based negation detection (recommended):

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

- For probability calibration use `src/calibration.py` (Isotonic per-label). Fit on a held-out validation set and transform predictions before `get_intervention_plan()`.

---

## 📊 Batch Predictions (For Datasets)

```bash
python -m src.generate_predictions data/test.csv
```

Creates `data/predictions.csv` with:
- `is_bullying` - Binary flag
- `detected_types` - Toxicity labels found
- `severity` - CRITICAL/HIGH/MEDIUM/LOW/NONE
- `action` - Recommended intervention
- `confidence` - Model confidence score

---

## 🎯 What Each File Does

| File | Purpose | Run Command |
|------|---------|-------------|
| `run_project.py` | Interactive demo | `python run_project.py` |
| `test_system.py` | Validate setup | `python test_system.py` |
| `src/generate_predictions.py` | Batch predictions | Import + call |
| `src/main_system.py` | Core logic | Internal use |
| `src/bert_model.py` | BERT wrapper | Internal use |
| `src/ontology.py` | Severity rules | Internal use |
| `src/baseline_model.py` | Baseline models | Optional |

---

## 📊 Expected Output Examples

### Interactive Mode
```
🛑 BULLYING DETECTED
Severity: MEDIUM
Action: HIDE_COMMENT + ISSUE_WARNING_STRIKE_1
Trigger Words: 'idiot' (0.85), 'moron' (0.72)
```

### Batch Mode
```
✅ Processed 5000 comments
   Bullying found: 1,234 (24.7%)
   Safe: 3,766 (75.3%)
   Saved to: data/predictions.csv
```

---

## ❓ Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: torch` | `pip install torch transformers` |
| `ModuleNotFoundError: lime` | `pip install lime` |
| `data/test.csv not found` | Ensure file exists in data/ folder |
| Slow on first run | Normal - BERT downloads (~400MB) |
| CUDA/GPU issues | CPU mode works fine, just slower |

---

## 💡 Key Features

✅ **Context-Aware**: Understands "killed it" ≠ "kill you"  
✅ **6 Toxicity Types**: toxic, severe_toxic, threat, insult, identity_hate, obscene  
✅ **Explainable**: LIME shows which words triggered detection  
✅ **Actionable**: Maps to real interventions (ban, block, warn, etc.)  
✅ **5 Severity Levels**: CRITICAL → HIGH → MEDIUM → LOW → NONE

---

## 📖 Learn More

- Full docs: `README.md`
- Detailed analysis: `ANALYSIS_REPORT.md`
- Configuration: `src/ontology.py`
- Code: `src/main_system.py`

---

**That's it! You're ready to go. 🎉**


