# ⚡ QUICK START GUIDE

## 🚀 Get Running in 3 Steps

### **Step 1: Install** (2-5 minutes)
```bash
pip install -r requirements.txt
```

### **Step 2: Validate** (1-2 minutes)
```bash
python test_system.py
```
✅ All 4 tests should PASS

### **Step 3: Run** (Choose one)

#### **Option A: Interactive Testing** (Recommended for Demo)
```bash
python run_project.py
```
Then type comments:
```
Enter a comment to test (or type 'exit'): you're an idiot
```
See full analysis with LIME explanations.

#### **Option B: Batch Processing** (For Kaggle)
```bash
python -c "from src.generate_predictions import generate_test_predictions; generate_test_predictions('data/test.csv')"
```
Creates `data/predictions.csv`

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
