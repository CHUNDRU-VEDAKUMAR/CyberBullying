# Context-Aware, Severity-Based and Explainable Cyberbullying Detection
“Context-Aware, Severity-Based and Explainable Cyberbullying Detection with Actionable Interventions”


## 📋 Project Overview

This system detects cyberbullying in text using multiple approaches:

### 🎯 **System Architecture**

```
Input Text
    ↓
[1] CONTEXT-AWARE BERT MODEL
    - Pre-trained: unitary/toxic-bert
    - Multi-label classification (6 types)
    - Semantic understanding of context
    ↓
[2] SEVERITY CLASSIFICATION (Ontology)
    - CRITICAL: severe_toxic, threat
    - HIGH: identity_hate
    - MEDIUM: toxic
    - LOW: insult, obscene
    ↓
[3] EXPLAINABILITY (LIME)
    - Highlights which words triggered detection
    - Local Interpretable Model-agnostic Explanations
    ↓
[4] ACTIONABLE INTERVENTIONS
    - Account suspension
    - Content blocking
    - Warnings/timeouts
    - Police alerts (for threats)
```

---

## ⚙️ Installation & Setup

### **Step 1: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 2: Verify Installation**
```bash
python -c "import torch; import transformers; import lime; print('✅ All dependencies installed!')"
```

---

## 🚀 How to Run

### **Option 1: Interactive Mode** (Recommended for Testing)
```bash
python run_project.py
```

Then type comments to analyze:
- Type any comment and press Enter
- View detailed detection report with LIME explanations
- Type `exit` to quit

### **Option 2: Batch Predictions** (For Dataset Analysis)
```bash
python -m src.generate_predictions data/test.csv
```

This will:
- Process all comments in `data/test.csv`
- Save results to `data/predictions.csv`
- Display summary statistics

---

## 📊 Output Format

### **Interactive Mode Output**
```
--- 🛡️  CYBERBULLYING DETECTION REPORT 🛡️  ---
📝 Input Text:     "You're an idiot"
🔍 Verdict:        🛑 BULLYING DETECTED
📊 Types Found:    toxic, insult
🔥 Severity:       MEDIUM
💡 Explanation:    General toxicity. The content is rude, disrespectful, or unreasonable.
👁️  Visual Proof:   The model flagged these specific words:
      👉 'idiot' (Impact Score: 0.85)
      👉 'you' (Impact Score: 0.12)
🛡️  Action:        HIDE_COMMENT + ISSUE_WARNING_STRIKE_1
```

### **Batch Prediction Output**
CSV file with columns:
- `id` - Comment ID
- `is_bullying` - Binary (0/1)
- `detected_types` - Pipe-separated cyberbullying types
- `severity` - CRITICAL/HIGH/MEDIUM/LOW/NONE
- `action` - Recommended intervention
- `max_score` - Highest confidence score

---

## 🔬 Model Details

### **BERT Model: unitary/toxic-bert**
- **Architecture**: BERT-base-uncased fine-tuned on Jigsaw Toxic Comments dataset
- **Labels Detected**: 
  - `toxic` - Rude, disrespectful, unreasonable
  - `severe_toxic` - Extremely offensive language
  - `obscene` - Profanity/vulgarity
  - `threat` - Intent to kill/injure
  - `insult` - Personal disparaging language
  - `identity_hate` - Attacks on protected groups

### **Explainability (LIME)**
- Generates local explanations for each prediction
- Shows word-level contributions to toxicity score
- Helps understand model decisions

---

## 📁 Project Structure

```
.
├── run_project.py              # Main entry point
├── requirements.txt            # Dependencies
├── data/
│   ├── train.csv              # Training data (for baseline)
│   ├── test.csv               # Test data
│   ├── test_labels.csv        # Ground truth labels
│   └── predictions.csv        # Generated predictions (after running)
├── src/
│   ├── main_system.py         # Core CyberbullyingSystem class
│   ├── bert_model.py          # BERT wrapper (AdvancedContextModel)
│   ├── baseline_model.py      # TF-IDF + RandomForest/SVC models
│   ├── ontology.py            # Severity rules & interventions
│   ├── preprocessing.py       # Text cleaning utilities
│   └── generate_predictions.py # Batch prediction pipeline
└── models/                     # Pre-trained models (cached by transformers)
```

---

## 🛠️ Configuration

### **Adjust Detection Threshold**
In `src/main_system.py`, modify:
```python
self.threshold = 0.50  # Change to 0.30 for more sensitivity, 0.70 for stricter
```

### **Customize Intervention Rules**
In `src/ontology.py`, modify the `CYBERBULLYING_ONTOLOGY` dictionary:
```python
"toxic": {
    "severity": "MEDIUM",
    "explanation": "Custom explanation here...",
    "intervention": "CUSTOM_ACTION"
}
```

---

## ⚠️ Troubleshooting

### **Issue: "ModuleNotFoundError: No module named 'torch'"**
```bash
pip install torch transformers
```

### **Issue: "ModuleNotFoundError: No module named 'lime'"**
```bash
pip install lime
```

### **Issue: "data/train.csv not found"**
The baseline tournament will skip if training data is missing. This is optional.

### **Issue: Model Download Takes Long**
BERT model (~400MB) downloads on first run. Subsequent runs use cache.

---

## 📈 Performance Metrics

**To evaluate on test_labels.csv:**
```python
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

# Load predictions and labels
pred_df = pd.read_csv('data/predictions.csv')
labels_df = pd.read_csv('data/test_labels.csv')

# Compare
y_pred = pred_df['is_bullying'].values
y_true = labels_df['toxic'].values  # or whichever column is the main label

f1 = f1_score(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
print(f"F1-Score: {f1:.4f}")
print(f"Accuracy: {acc:.4f}")
```

---

## 🎓 Key Components Explained

### **1. Context-Aware (BERT)**
✅ Understands semantic meaning
✅ Handles negations ("not good" vs "good")
✅ Recognizes context ("kill it" vs "kill you")

### **2. Severity-Based**
✅ 6-level severity hierarchy
✅ Maps to action severity (block vs warn)
✅ Prioritizes critical threats

### **3. Explainable (LIME)**
✅ Word-level attribution scores
✅ Shows which phrases triggered detection
✅ Black-box agnostic (works with any model)

### **4. Actionable Interventions**
✅ Severity-matched responses
✅ Account suspension for repeat offenders
✅ Police alerts for death threats
✅ Content hiding for mild toxicity

---

## 📝 Example Test Cases

```python
from src.main_system import CyberbullyingSystem

system = CyberbullyingSystem()

# Safe comment
result = system.analyze("I really enjoyed this movie!")
# Expected: is_bullying=False

# Toxic comment
result = system.analyze("You're such an idiot")
# Expected: is_bullying=True, types=['toxic', 'insult'], severity=MEDIUM

# Threat
result = system.analyze("I will kill you tomorrow")
# Expected: is_bullying=True, types=['threat'], severity=CRITICAL, action=POLICE_ALERT

# Context-aware example
result = system.analyze("You absolutely killed that presentation!")
# Expected: is_bullying=False (despite "killed")
```

---

## 🔗 References

- **BERT Model**: https://huggingface.co/unitary/toxic-bert
- **LIME**: https://github.com/marcotcr/lime
- **Jigsaw Toxic Comments**: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

---

## 📞 Support

For issues or questions, check:
1. Ensure requirements.txt installed correctly
2. Verify data files exist in data/ folder
3. Check Python version (3.8+ recommended)

---

**Last Updated**: January 29, 2026
