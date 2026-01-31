# 📚 Documentation Index

## Start Here 👇

### For Quick Start ⚡
→ **[QUICKSTART.md](QUICKSTART.md)** - Get running in 3 steps

### For Full Understanding 📖
→ **[README.md](README.md)** - Complete documentation with examples

### For Project Analysis 🔍
→ **[ANALYSIS_REPORT.md](ANALYSIS_REPORT.md)** - Detailed assessment of your codebase

### For What Changed ✅
→ **[CHANGES_MADE.md](CHANGES_MADE.md)** - List of all fixes and improvements

---

## Quick Links

| Need | File | Description |
|------|------|-------------|
| Get running NOW | [QUICKSTART.md](QUICKSTART.md) | 3-step installation + run guide |
| Install packages | [requirements.txt](requirements.txt) | All Python dependencies |
| Understand system | [README.md](README.md) | Architecture, features, examples |
| Run tests | [test_system.py](test_system.py) | Validate your setup works |
| Interactive demo | [run_project.py](run_project.py) | Main program entry point |
| Batch processing | [src/generate_predictions.py](src/generate_predictions.py) | Process entire test.csv |
| Core logic | [src/main_system.py](src/main_system.py) | CyberbullyingSystem class |
| Severity rules | [src/ontology.py](src/ontology.py) | Intervention mappings |

---

## Installation Summary

```bash
# 1. Install dependencies (2-5 min)
pip install -r requirements.txt

# 2. Validate setup (1-2 min)
python test_system.py

# 3. Run project
python run_project.py        # Interactive mode
# OR
python -c "from src.generate_predictions import generate_test_predictions; generate_test_predictions('data/test.csv')"  # Batch mode
```

---


## Key Features

🧠 **Context-Aware** - BERT understands meaning, not just keywords  
🎯 **Severity-Based** - 5-level classification (CRITICAL to NONE)  
💡 **Explainable** - LIME shows which words triggered detection  
⚡ **Actionable** - Maps to real interventions (ban, warn, flag, etc.)  
🎛️ **6 Detection Types** - toxic, severe_toxic, threat, insult, identity_hate, obscene

---

## Troubleshooting

**Q: Where do I start?**  
A: Read [QUICKSTART.md](QUICKSTART.md) - 3 minutes to running

**Q: How do I process a full dataset?**  
A: Use batch mode in [src/generate_predictions.py](src/generate_predictions.py)

**Q: Can I customize the actions?**  
A: Yes! Modify [src/ontology.py](src/ontology.py)

**Q: Is it working correctly?**  
A: Run `python test_system.py` to verify

**Q: What if packages won't install?**  
A: See "Troubleshooting" section in [README.md](README.md)

---

## File Descriptions

### Documentation (Read These First)
- **QUICKSTART.md** - 3-step setup guide (5 min read)
- **README.md** - Full documentation (15 min read)
- **ANALYSIS_REPORT.md** - Technical analysis (10 min read)
- **CHANGES_MADE.md** - What was fixed (5 min read)
- **INDEX.md** - This file

### Code Files (Production)
- **run_project.py** - Main entry point (interactive mode)
- **test_system.py** - Validation tests (run before using)
- **requirements.txt** - All dependencies (install first)

### Source Code
- **src/main_system.py** - Core CyberbullyingSystem class
- **src/bert_model.py** - BERT model wrapper
- **src/ontology.py** - Severity & intervention rules
- **src/preprocessing.py** - Text cleaning utilities
- **src/baseline_model.py** - Baseline ML models (optional)
- **src/generate_predictions.py** - Batch processing pipeline

### Data
- **data/train.csv** - Optional training data
- **data/test.csv** - Input for batch processing
- **data/test_labels.csv** - Optional ground truth
- **data/predictions.csv** - Output (auto-generated)

---

## Timeline to Full Setup

| Time | Step | Action |
|------|------|--------|
| 0-5 min | Install | `pip install -r requirements.txt` |
| 5-7 min | Validate | `python test_system.py` |
| 7-9 min | Learn | Read [QUICKSTART.md](QUICKSTART.md) |
| 9+ min | Run | `python run_project.py` |

**Total: ~10 minutes to fully operational**

---

## Project Structure

```
📦 Cyberbullying_Project
├── 📄 README.md ............................ Complete documentation
├── 📄 QUICKSTART.md ....................... 3-step setup guide
├── 📄 ANALYSIS_REPORT.md .................. Technical analysis
├── 📄 CHANGES_MADE.md ..................... All fixes made
├── 📄 INDEX.md ............................ This file
├── 📄 requirements.txt .................... Python dependencies
├── 🐍 run_project.py ....................... Interactive demo
├── 🧪 test_system.py ...................... Validation tests
├── 📁 data/
│   ├── train.csv
│   ├── test.csv
│   ├── test_labels.csv
│   └── predictions.csv (auto-generated)
├── 📁 src/
│   ├── main_system.py ..................... Core system
│   ├── bert_model.py ...................... BERT wrapper
│   ├── ontology.py ........................ Rules & actions
│   ├── preprocessing.py ................... Text cleaning
│   ├── baseline_model.py .................. Baseline models
│   └── generate_predictions.py ............ Batch processor
└── 📁 models/
    └── (BERT model - auto-cached)
```

---

## Questions?

1. **How to get started?** → [QUICKSTART.md](QUICKSTART.md)
2. **How does it work?** → [README.md](README.md)  
3. **What changed?** → [CHANGES_MADE.md](CHANGES_MADE.md)
4. **Is it working?** → Run `python test_system.py`
5. **In-depth analysis?** → [ANALYSIS_REPORT.md](ANALYSIS_REPORT.md)

---

