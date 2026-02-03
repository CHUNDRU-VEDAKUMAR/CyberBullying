# 🎯 Advanced Configuration & Edge Cases

## Advanced Configuration Options

### 1. Tuning Negation Strength

Edit `src/negation_handler.py`:

```python
def _calculate_negation_strength(self, has_negation, negation_type, has_sarcasm, has_safe_context):
    """Customize these values to make system stricter/looser"""
    factor = 1.0
    
    # ADJUST THESE:
    if negation_type == 'strong':
        factor *= 0.15  # ← Change: 0.05-0.30 (lower = more reduction)
    elif negation_type == 'weak':
        factor *= 0.40  # ← Change: 0.20-0.60
    
    if has_sarcasm:
        factor *= 0.30  # ← Change: 0.10-0.50
    
    if has_safe_context:
        factor *= 0.25  # ← Change: 0.10-0.40
    
    return max(0.0, min(1.0, factor))
```

**Examples**:
- Make stricter: `strong: 0.05, weak: 0.20` (more reduction)
- Make looser: `strong: 0.25, weak: 0.60` (less reduction)

---

### 2. Customizing Context Scores

Edit `src/context_analyzer.py`:

```python
def analyze_context(self, text):
    # ADJUST THESE SCORES:
    if is_positive:
        context_score = 0.05   # ← Range: 0.01-0.20 (positive achievement)
    elif target == 'thing':
        context_score = 0.30   # ← Range: 0.10-0.50 (thing criticism)
    elif is_opinion:
        context_score = 0.50   # ← Range: 0.30-0.70 (opinion statements)
    elif is_affection:
        context_score = 0.10   # ← Range: 0.01-0.30 (affection/care)
```

**Lower score** = More reduction in toxicity  
**Higher score** = Less reduction in toxicity

---

### 3. Dynamic Threshold Adjustment

Edit `src/main_system.py`:

```python
def analyze(self, user_text):
    # Current logic:
    adjusted_threshold = self.base_threshold
    if negation_context['has_negation']:
        adjusted_threshold = self.base_threshold * 0.7  # ← Change multiplier
    adjusted_threshold = self.context_analyzer.adjust_threshold(adjusted_threshold, ...)
    
    # To make STRICTER (fewer false positives):
    # adjusted_threshold = self.base_threshold * 0.8  # Increase multiplier
    
    # To make LOOSER (fewer false negatives):
    # adjusted_threshold = self.base_threshold * 0.5  # Decrease multiplier
```

---

## Adding Custom Negation Words

Edit `src/negation_handler.py`:

```python
def __init__(self):
    self.negation_words = {
        # Existing words...
        "don't", "won't", "not", "never",
        
        # ADD YOUR CUSTOM WORDS HERE:
        "inainte",  # Romanian for "not"
        "no",       # Spanish
        # etc...
    }
```

---

## Adding Custom Context Patterns

### For Positive Achievement

Edit `src/context_analyzer.py`:

```python
def __init__(self):
    self.positive_patterns = [
        # Existing patterns...
        r"you\s+(?:absolutely\s+)?(?:killed|crushed|smashed|nailed|destroyed)",
        
        # ADD CUSTOM PATTERNS:
        r"that\s+was\s+(?:fire|lit|dope|awesome|sick)",  # Slang
        r"you\s+(?:owned|pwned|fragged)",  # Gaming
    ]
```

### For Opinion Indicators

Edit `src/context_analyzer.py`:

```python
self.opinion_words = {
    # Existing...
    "i think", "i believe", "in my opinion",
    
    # ADD:
    "imo",      # Internet slang
    "imho",     # In my humble opinion
    "from my perspective",
    "it seems to me",
}
```

---

## Edge Cases Handled

### 1. Multiple Negations
```
"I don't think you're NOT smart"
→ Detects both negations
→ Complex context, score reduced significantly
```

### 2. Negation Far from Word
```
"I don't, and I never will kill you"
→ Checks 5-word window, catches both
→ Strong negation = ~15% of original score
```

### 3. False Negatives (Negation Not Actually Negating)
```
"Not killing you is hard" 
→ Detects negation + positive context interpretation
→ Reduced but not eliminated (context-dependent)
```

### 4. Sarcasm
```
"Yeah right, you're so smart" (sarcasm)
→ Detects sarcasm patterns (yeah right, sure, lol)
→ Further reduces score
```

### 5. Mixed Contexts
```
"You're NOT terrible at killing it!"
→ Detects: negation + positive achievement
→ Combines both reductions: score → ~3% of original
→ Result: SAFE
```

---

## Troubleshooting

### Issue: Too Many False Positives (Marking safe as bullying)
**Solution**: Increase threshold or reduce negation factors
```python
# In src/main_system.py
self.base_threshold = 0.60  # Increase from 0.50

# Or in src/negation_handler.py
strong_negation_factor = 0.25  # Increase from 0.15
```

### Issue: Too Many False Negatives (Marking bullying as safe)
**Solution**: Decrease threshold or increase negation factors
```python
# In src/main_system.py
self.base_threshold = 0.40  # Decrease from 0.50

# Or in src/negation_handler.py
strong_negation_factor = 0.05  # Decrease from 0.15
```

### Issue: Certain Words Always Flagged
**Add to opinion/thing context**:
```python
# In src/context_analyzer.py
self.safe_targets.add("my_safe_word")
```

### Issue: Certain Words Never Flagged
**Check if negated or in positive context**, or:
```python
# Lower the threshold in src/main_system.py
self.base_threshold = 0.30  # Make stricter
```

---

## Testing Custom Configurations

Create test file `test_custom.py`:

```python
from src.main_system import CyberbullyingSystem

# Create system with YOUR data
system = CyberbullyingSystem()

test_cases = [
    ("Your custom test", False),  # Expected result
    ("Another test", True),
]

for text, expected_bullying in test_cases:
    result = system.analyze(text)
    is_bullying = result['is_bullying']
    
    status = "✅" if is_bullying == expected_bullying else "❌"
    print(f"{status} '{text}' → {is_bullying} (expected {expected_bullying})")
    
    if is_bullying != expected_bullying:
        print(f"   Context: {result['context_info']['context_reason']}")
        print(f"   Scores: {result['scores']}")
```

Run it:
```bash
python test_custom.py
```

---

## Performance Tuning

### Reduce Context Analysis (Speed Up)
Comment out some checks in `src/context_analyzer.py`:

```python
def analyze_context(self, text):
    # Skip some expensive checks if needed:
    # is_opinion = False  # Skip opinion check
    # is_positive = False  # Skip achievement check
    # result = skip some complex analysis
```

### Reduce LIME Explanation (Speed Up)
In `src/main_system.py`:

```python
exp = self.explainer.explain_instance(
    user_text, 
    self._predict_proba_for_lime, 
    num_features=3  # Reduce from 5 (fewer features = faster)
)
```

---

## Multilingual Support

### Adding Language-Specific Negations

Create `src/negation_handler_multi.py`:

```python
class MultilingualNegationHandler:
    def __init__(self, language='en'):
        self.language = language
        
        self.negation_words = {
            'en': {'not', 'don\'t', 'won\'t', 'never'},
            'es': {'no', 'nunca', 'nada'},  # Spanish
            'fr': {'non', 'jamais', 'ne'},  # French
            'de': {'nicht', 'nein', 'nie'},  # German
            'pt': {'não', 'nunca', 'nada'},  # Portuguese
        }
        
        self.current_negations = self.negation_words.get(language, {})
```

Use it:
```python
from src.negation_handler_multi import MultilingualNegationHandler

handler = MultilingualNegationHandler(language='es')  # Spanish
context = handler.detect_negation_context(spanish_text)
```

---

## Domain-Specific Tuning

### For Gaming Community (More Positive Terms)
```python
# In src/context_analyzer.py
self.positive_verbs.update({
    "pwned", "fragged", "owned", "ganked", "ninja-looted"
})
```

### For Academic Context (More Opinions Allowed)
```python
# In src/context_analyzer.py
self.opinion_words.update({
    "arguably", "one could say", "questionable", 
    "controversial", "debatable"
})
```

### For Legal/Formal Context (Stricter)
```python
# In src/main_system.py
self.base_threshold = 0.40  # Lower threshold for stricter detection
```

---

## Batch Configuration

For `src/generate_predictions.py`, customize behavior:

```python
def generate_test_predictions(
    test_csv_path, 
    output_path='data/predictions.csv',
    threshold=0.50,  # Custom threshold
    language='en'     # Language
):
    system = CyberbullyingSystem()
    system.base_threshold = threshold
    
    # ... rest of code
```

Usage:
```python
generate_test_predictions(
    'data/test.csv',
    threshold=0.45,  # Stricter
    language='en'
)
```

---

## Logging & Debugging

Add to `src/main_system.py` for debugging:

```python
def analyze(self, user_text, debug=False):
    if debug:
        print(f"[DEBUG] Input: {user_text}")
        print(f"[DEBUG] Negation context: {negation_context}")
        print(f"[DEBUG] Linguistic context: {linguistic_context}")
        print(f"[DEBUG] Raw predictions: {predictions}")
        print(f"[DEBUG] Adjusted threshold: {adjusted_threshold}")
        print(f"[DEBUG] Filtered labels: {active_labels}")
    
    # ... rest of code
```

Usage:
```python
result = system.analyze("I don't kill you", debug=True)
```

---

## Validation Checklist for Custom Config

Before deploying, verify:

- ✅ Run `test_enhanced.py` - all tests pass
- ✅ Run custom tests on your data
- ✅ Test edge cases (negations, sarcasm, opinions)
- ✅ Check for false positives/negatives
- ✅ Measure performance (speed, accuracy)
- ✅ Document your changes

---

## Summary

**Three Ways to Customize**:

1. **Easy**: Adjust thresholds/factors (5 min)
2. **Medium**: Add custom words/patterns (10 min)
3. **Advanced**: Multilingual support, domain-specific rules (30+ min)

**Remember**: Always test after changes!

---

**Questions?** See main documentation in:
- `CONTEXT_AWARENESS_GUIDE.md` - Technical details
- `QUICK_REFERENCE.md` - Quick start
- `IMPLEMENTATION_SUMMARY.md` - Overview
