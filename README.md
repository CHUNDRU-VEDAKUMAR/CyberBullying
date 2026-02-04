# Context-Aware, Severity-Based and Explainable Cyberbullying Detection
Context-Aware, Severity-Based and Explainable Cyberbullying Detection with Actionable Interventions”
This system detects cyberbullying in text using a four-pillar approach:

## 🎯 Four Core Pillars

### 1. **Context-Aware** 🧠
The system understands linguistic context, not just keywords:
- **Negation Detection**: "I don't kill you" is flagged as SAFE (not bullying)
- **Positive Achievement**: "You killed that presentation!" is SAFE (sarcasm/praise detection)
- **Target Type Classification**: Distinguishes personal attacks from criticism of ideas
- **Opinion vs Personal**: Separates constructive critique from personal insults
- **Modules**: `src/negation_handler.py`, `src/context_analyzer.py`

### 2. **Severity-Based** ⚖️
Maps detected toxicity types to actionable severity levels:
- **CRITICAL**: Threats, severe toxicity → BLOCK_ACCOUNT + REPORT_TO_CYBER_CELL
- **HIGH**: Identity hate, hate speech → PERMANENT_BAN + HIDE_CONTENT  
- **MEDIUM**: General toxicity → HIDE_COMMENT + ISSUE_WARNING
- **LOW**: Insults, obscenity → FLAG_FOR_REVIEW + USER_TIMEOUT
- **Confidence Calibration**: Adjusts interventions based on model confidence (>50% = strict action, <50% = flag for human review)
- **Module**: `src/ontology.py`

### 3. **Explainable** 👁️
Shows exactly which words triggered detection:
- **LIME Explanations**: Local Interpretable Model-agnostic Explanations per label
- **Fallback Explainer**: Leave-one-out perturbation when LIME is unavailable (CPU-friendly)
- **Per-Label Attribution**: Shows word-level impact for each toxicity type detected
- **Normalized Scores**: Outputs both raw and normalized importance weights
- **Module**: `src/explainability.py`

### 4. **Actionable Interventions** 🛡️
Recommends specific, contextual actions for moderators:
- **Severity-Driven**: Intervention choice depends on severity + confidence
- **Human Review Option**: Low-confidence detections flag for human moderators instead of auto-action
- **Transparency**: Shows reasoning (detected type, severity, confidence, triggering words)
- **Module**: `src/ontology.py` → `recommend_intervention()`

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

## 🔬 Supported Models

The system works with any HuggingFace sequence classification model. Pre-configured models:

### **unitary/toxic-bert** (Default)
- **Strengths**: Pre-trained on Jigsaw toxic comments; fast inference on CPU
- **Use when**: Speed is critical, baseline toxicity detection needed
- **Load**: `python run_project.py` (automatic)

### **roberta-base** (Recommended)
- **Strengths**: Better contextual understanding of negations, sarcasm, and nuance
- **Use when**: Context-awareness and accuracy are priorities
- **Load**: Edit `run_project.py` to change `model_name='roberta-base'`, or:
  ```python
  from src.main_system import CyberbullyingSystem
  system = CyberbullyingSystem(model_name='roberta-base')
  ```

### Custom Models
Load any HuggingFace sequence classification model:
```python
system = CyberbullyingSystem(model_name='your-model-name')
```

---

## 🔬 Model Details

The default **BERT model: unitary/toxic-bert** is fine-tuned on Jigsaw Toxic Comments:
- **Architecture**: BERT-base-uncased
- **Labels Detected**: 
  - `toxic` - Rude, disrespectful, unreasonable
  - `severe_toxic` - Extremely offensive language
  - `obscene` - Profanity/vulgarity
  - `threat` - Intent to kill/injure
  - `insult` - Personal disparaging language
  - `identity_hate` - Attacks on protected groups

**Note**: Each model runs on CPU by design (no CUDA required).

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


CyberBullying Detection System
Overview
This project implements an advanced system for detecting cyberbullying in text data. It leverages state-of-the-art NLP models, ensemble techniques, context analysis, and explainability modules to provide robust and interpretable predictions. The system is designed for research, evaluation, and deployment in real-world scenarios.

Project Structure
├── data/                  # Datasets and resources
│   ├── train.csv
│   ├── test.csv
│   ├── test_labels.csv
│   ├── sample_submission.csv
│   └── offensive_tokens.txt
├── scripts/               # Utility and export scripts
│   └── export_onnx.py
├── src/                   # Source code for all modules
│   ├── advanced_calibration.py
│   ├── advanced_context.py
│   ├── advanced_ensemble.py
│   ├── api.py
│   ├── baseline_model.py
│   ├── bert_model.py
│   ├── calibration.py
│   ├── comprehensive_evaluation.py
│   ├── config.py
│   ├── context_analyzer.py
│   ├── data_augmentation.py
│   ├── evaluate.py
│   ├── explainability.py
│   ├── finetune.py
│   ├── generate_predictions.py
│   ├── main_system.py
│   ├── model_manager.py
│   ├── negation_handler.py
│   ├── ontology.py
│   ├── preprocessing.py
│   └── __pycache__/
├── tests/                 # Test suites
│   ├── full_system_test.py
│   └── supreme_test_system.py
├── run_project.py         # Main entry point for running the system
├── STARTUP.py             # Startup script
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation (this file)
├── README_FINAL.txt       # Final delivery readme
├── README_RESEARCH_PACKAGE.md # Research package readme
├── FINAL_DELIVERY_SUMMARY.md  # Final delivery summary
├── IMPLEMENTATION_STATUS.md   # Implementation progress
├── ADVANCED_IMPLEMENTATION_SUMMARY.md # Advanced implementation details
├── MODEL_RATIONALE.md     # Model rationale and design
├── PAPER_PUBLICATION_GUIDE.md # Guide for publishing research
├── QUICK_REFERENCE.md     # Quick reference guide
├── RESEARCH_PAPER.md      # Main research paper
├── START_HERE_RESEARCH_PAPER.md # Start here for research
├── TECHNICAL_APPENDIX.md  # Technical appendix
├── TEST_RESULTS_FINAL.md  # Final test results
├── final_validation.py    # Final validation script
├── validate_final.py      # Validation script
├── verify_pillars.py      # Pillar verification script


Key Features
Advanced NLP Models: Utilizes BERT and ensemble models for high-accuracy detection.
Contextual Analysis: Handles negation, context, and ontology for nuanced understanding.
Explainability: Provides model explanations for predictions.
Calibration & Evaluation: Includes advanced calibration and comprehensive evaluation modules.
Data Augmentation: Supports robust training with data augmentation techniques.
API Support: Ready for integration via API endpoints.
Getting Started
1. Install Dependencies
Ensure you have Python 3.8+ installed. Install required packages:
pip install -r requirements.txt

2. Prepare Data
Place your datasets in the data directory. The expected files are:

train.csv, test.csv, test_labels.csv, sample_submission.csv, offensive_tokens.txt
3. Run the System
To run the main system:
python run_project.py

Or use the startup script:
python STARTUP.py

4. Testing & Validation
Or run validation scripts:

python final_validation.py
python validate_final.py


Main Modules
src/main_system.py: Orchestrates the full pipeline.
src/baseline_model.py, src/bert_model.py, src/advanced_ensemble.py: Core models.
src/context_analyzer.py, src/negation_handler.py, src/ontology.py: Context and language understanding.
src/explainability.py: Model explainability tools.
src/generate_predictions.py: Generates predictions for test data.
src/api.py: API endpoints for integration.
Documentation
See the following files for more details:

README_FINAL.txt: Final delivery instructions
README_RESEARCH_PACKAGE.md: Research package overview
FINAL_DELIVERY_SUMMARY.md: Summary of final delivery
IMPLEMENTATION_STATUS.md: Implementation progress
MODEL_RATIONALE.md: Model design rationale
TECHNICAL_APPENDIX.md: Technical details
RESEARCH_PAPER.md: Main research paper
Deployment
For deployment instructions, see DEPLOYMENT_GUIDE.md.

Citation
If you use this system in your research, please cite the corresponding research paper (see RESEARCH_PAPER.md).