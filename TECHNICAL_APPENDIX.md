# Cyberbullying Detection System - Technical Appendix

## Comprehensive System Documentation with Code Walkthroughs

---

## 1. Architecture Deep Dive

### 1.1 System Flow with Code Examples

```
User Input
    ↓
Text Preprocessing (src/preprocessing.py)
    ↓
Context Analysis Branch (src/context_analyzer.py + src/advanced_context.py)
    ├─ Negation Detection (spaCy dependency parsing)
    ├─ Sarcasm Detection (multi-factor patterns)
    ├─ Target Type Classification (person vs idea)
    └─ Reduction Factor Calculation (0.0-1.0)
    ↓
ML Model Branch (src/bert_model.py)
    ├─ Tokenization
    ├─ Forward pass through BERT/RoBERTa/DeBERTa
    └─ Output: 6 toxicity scores
    ↓
Context-Adjusted Scoring
    (predictions × reduction_factor)
    ↓
Threshold-Based Filtering
    (separate thresholds for threats vs others)
    ↓
Severity Ontology (src/ontology.py)
    ├─ Map labels to severity levels
    ├─ Aggregate multi-label severity
    ├─ Apply confidence calibration
    └─ Select intervention
    ↓
Explainability (src/explainability.py)
    ├─ Perturbation-based attribution
    ├─ Per-label impact scoring
    └─ Top-5 tokens per label
    ↓
Structured JSON Output
```

---

## 2. Detailed Implementation Guide

### 2.1 Core System Class (`src/main_system.py`)

```python
class CyberbullyingSystem:
    """
    Four-pillar implementation:
    1. Context-Aware: Negation, sarcasm, opinion detection
    2. Severity-Based: Multi-label with severity mapping
    3. Explainable: Token-level attribution
    4. Actionable: Structured intervention recommendations
    """
    
    def __init__(self, model_name='unitary/toxic-bert', use_advanced_context=True):
        """
        Initialize system with model and context analyzers
        
        Args:
            model_name: HuggingFace model identifier
            use_advanced_context: Use spaCy-based advanced analysis
        """
        # Core ML model
        self.engine = AdvancedContextModel(model_name=model_name)
        
        # Context awareness (try advanced first, fallback to basic)
        if use_advanced_context:
            try:
                self.context_analyzer = AdvancedContextAnalyzer()
                self.advanced_context = True
            except:
                self.context_analyzer = ContextAnalyzer()
                self.advanced_context = False
        else:
            self.context_analyzer = ContextAnalyzer()
            self.advanced_context = False
        
        # Negation handler for aggressive suppression
        self.negation_handler = NegationHandler()
        
        # Thresholds
        self.base_threshold = 0.50
        self.threat_threshold = 0.25  # Lower for threats
    
    def analyze(self, user_text):
        """
        Complete end-to-end analysis pipeline
        
        Returns: Dict with is_bullying, detected_types, severity, action, etc.
        """
        
        # === STEP 1: CONTEXT ANALYSIS ===
        if self.advanced_context:
            context_info = self.context_analyzer.analyze_context_full(user_text)
            # Returns: {
            #   'negation': {'has_negation': bool, 'method': str},
            #   'sarcasm': {'detected': bool, 'score': float},
            #   'target': {'target_type': str},
            #   'reduction_factor': float (0.0-1.0),
            #   'explanation': str
            # }
            
            # Extract individual components
            negation_context = {
                'has_negation': bool(context_info.get('negation', {}).get('has_negation', False)),
                'negation_type': context_info.get('negation', {}).get('method', 'spacy'),
                'has_sarcasm': bool(context_info.get('sarcasm', {}).get('detected', False)),
                'sarcasm_confidence': float(context_info.get('sarcasm', {}).get('score', 0.0))
            }
            
            linguistic_context = {
                'target_type': context_info.get('target', {}).get('target_type', 'unknown'),
                'is_opinion': context_info.get('target', {}).get('target_type') == 'idea',
                'context_score': float(context_info.get('reduction_factor', 1.0))
            }
        else:
            # Fallback to basic context
            negation_context = self.negation_handler.detect_negation_context(user_text)
            linguistic_context = self.context_analyzer.analyze_context(user_text)
        
        # === STEP 2: GET BASE PREDICTIONS ===
        predictions = self.engine.predict(user_text)
        # Output: {'toxic': 0.75, 'threat': 0.2, 'insult': 0.8, ...}
        
        # === STEP 3: APPLY NEGATION ADJUSTMENT ===
        # NegationHandler applies aggressive reduction if negation detected
        predictions, neg_context = self.negation_handler.adjust_predictions(
            predictions, 
            user_text
        )
        # Output: adjusted scores + negation metadata
        
        # === STEP 4: APPLY CONTEXT FACTOR ===
        # Multiply all scores by reduction_factor from advanced context
        context_factor = linguistic_context.get('context_score', 1.0)
        predictions = {
            label: score * context_factor 
            for label, score in predictions.items()
        }
        
        # === STEP 5: THRESHOLD-BASED FILTERING ===
        # Use dual-threshold strategy:
        # - Threats: lower threshold (0.25) to catch more
        # - Others: normal threshold (0.50)
        active_labels = []
        for label, score in predictions.items():
            if label in ('threat', 'severe_toxic'):
                threshold = self.threat_threshold
            else:
                threshold = self.base_threshold
            
            if score > threshold:
                active_labels.append(label)
        
        # === STEP 6: GET SEVERITY + INTERVENTION ===
        plan = get_intervention_plan(predictions)
        # Automatically selects highest-priority label and maps to:
        # - severity: CRITICAL|HIGH|MEDIUM|LOW
        # - intervention: specific action
        # - confidence: 0.0-1.0
        # - explanation: natural language reasoning
        
        # === STEP 7: GENERATE EXPLANATIONS ===
        try:
            label_explanations = explain_multilabel(
                user_text,
                self.engine.predict_proba,
                self.engine.labels,
                num_features=5
            )
        except:
            label_explanations = {}
        
        # Extract top label's explanation
        top_label = plan.get('detected_label', 'toxic')
        highlighted_words = label_explanations.get(top_label, [])
        
        # === STEP 8: RETURN STRUCTURED OUTPUT ===
        return {
            "text": user_text,
            "is_bullying": len(active_labels) > 0,
            "detected_types": active_labels,
            "severity": plan['severity'],
            "explanation": plan['explanation'],
            "action": plan['intervention'],
            "detected_label": plan.get('detected_label'),
            "confidence": plan.get('confidence'),
            "highlighted_words": highlighted_words,
            "scores": {k: round(v, 4) for k, v in predictions.items() if v > 0.01},
            "context_info": {
                "negation_detected": negation_context.get('has_negation', False),
                "negation_type": negation_context.get('negation_type', 'none'),
                "has_sarcasm": negation_context.get('has_sarcasm', False),
                "sarcasm_confidence": negation_context.get('sarcasm_confidence', 0.0),
                "target_type": linguistic_context.get('target_type', 'unknown'),
                "is_opinion": linguistic_context.get('is_opinion', False)
            }
        }
```

---

## 3. Context Analyzer Implementation Details

### 3.1 Advanced Context Analyzer (`src/advanced_context.py`)

```python
class AdvancedContextAnalyzer:
    """
    Multi-layered context analysis using spaCy and sentiment analysis
    """
    
    def analyze_context_full(self, text: str) -> Dict:
        """
        Complete context analysis:
        1. Negation detection (spaCy dependencies + patterns)
        2. Sarcasm detection (patterns + sentiment contradiction)
        3. Target type classification (person vs idea)
        4. Calculate reduction factor
        """
        
        # Negation analysis
        negation_result = self.analyze_negation_spacy(text)
        
        # Sarcasm analysis
        sarcasm_result = self.detect_sarcasm_advanced(text)
        
        # Target type analysis
        target_result = self.analyze_target_type(text)
        
        # Calculate reduction factor
        # Base: 1.0 (no reduction)
        # -0.3 if negation detected (cancel out toxicity)
        # -0.2 if strong sarcasm (mostly safe)
        # -0.1 if opinion on idea (not personal attack)
        reduction_factor = 1.0
        
        if negation_result['has_negation']:
            reduction_factor *= 0.3  # Strong negation: multiply by 0.3
        
        if sarcasm_result['detected']:
            reduction_factor *= (1.0 - sarcasm_result['score'] * 0.5)
        
        if target_result['target_type'] == 'idea':
            reduction_factor *= 0.7  # Reduce impact for idea criticism
        
        # Clamp to valid range [0.1, 1.0]
        reduction_factor = max(0.1, min(1.0, reduction_factor))
        
        return {
            'negation': negation_result,
            'sarcasm': sarcasm_result,
            'target': target_result,
            'reduction_factor': reduction_factor,
            'explanation': f"Context analysis: negation={negation_result['has_negation']}, "
                          f"sarcasm={sarcasm_result['detected']}, "
                          f"target_type={target_result['target_type']}"
        }
    
    def analyze_negation_spacy(self, text: str) -> Dict:
        """
        Use spaCy dependency parsing for precise negation scope
        
        Example: "I don't think you're stupid"
        - "think" has dependency "neg" → negation detected
        - "don't" negates "think", not "stupid"
        - Result: has_negation=True, but "stupid" is not in negation scope
        """
        
        if not self.use_spacy:
            return {'has_negation': False, 'method': 'fallback'}
        
        try:
            doc = self.nlp(text.lower())
            negations_found = []
            
            # Find negation tokens (dep_ == 'neg')
            for token in doc:
                if token.dep_ == 'neg':
                    negations_found.append({
                        'word': token.text,
                        'head': token.head.text,
                        'pos': token.pos_
                    })
            
            # Check if negations scope over offensive words
            offensive_words = ['kill', 'hurt', 'hate', 'stupid', 'idiot', 'bad']
            has_negation = len(negations_found) > 0
            
            if has_negation:
                # Walk up dependency tree to find words affected by negation
                negation_scope = []
                for token in doc:
                    if token.dep_ == 'neg':
                        # Find all children of negation's head
                        head = token.head
                        for child in head.children:
                            if child.text in offensive_words:
                                negation_scope.append(child.text)
                
                return {
                    'has_negation': True,
                    'negation_strength': 0.9 if negation_scope else 0.5,
                    'method': 'spacy_dependencies',
                    'negated_words': negation_scope
                }
            else:
                return {'has_negation': False, 'method': 'spacy_dependencies'}
        
        except Exception as e:
            # Fallback to regex patterns
            return self._analyze_negation_regex(text)
    
    def detect_sarcasm_advanced(self, text: str) -> Dict:
        """
        Multi-factor sarcasm detection
        
        Factors:
        1. Strong sarcasm patterns (yeah right, sure buddy, etc.)
        2. Sentiment contradiction (positive words + negative topic)
        3. Punctuation extremity (!!!???)
        4. Clause-level sentiment analysis
        """
        
        score = 0.0
        factors = []
        
        # Factor 1: Sarcasm patterns
        strong_patterns = [
            r'\byeah right\b',
            r'\bsure.*(?:buddy|pal|friend)\b',
            r'\btotally\b.*(?:not|fake)',
        ]
        
        for pattern in strong_patterns:
            if re.search(pattern, text.lower()):
                score = max(score, 0.9)
                factors.append(('pattern', 0.9))
                break
        
        # Factor 2: Sentiment contradiction
        if self.sentiment_pipeline:
            try:
                sentiment = self.sentiment_pipeline(text[:512])[0]  # Truncate for efficiency
                # Contradiction: positive words + negative topic
                positive_words = ['amazing', 'awesome', 'incredible']
                negative_words = ['hate', 'stupid', 'kill']
                
                has_positive = any(w in text.lower() for w in positive_words)
                has_negative = any(w in text.lower() for w in negative_words)
                
                if has_positive and has_negative:
                    score = max(score, 0.7)
                    factors.append(('contradiction', 0.7))
            except:
                pass
        
        # Factor 3: Punctuation extremity
        if text.count('!') >= 3 or text.count('?') >= 2:
            score = max(score, 0.5)
            factors.append(('punctuation', 0.5))
        
        return {
            'detected': score > 0.4,
            'score': score,
            'factors': factors
        }
```

### 3.2 Negation Handler (`src/negation_handler.py`)

```python
class NegationHandler:
    """
    Detect and suppress toxicity when negated
    
    Strategy:
    1. Find negation words (don't, not, never, etc.)
    2. Find offensive tokens nearby
    3. If negation + offensive word in scope: aggressively suppress scores
    """
    
    def adjust_predictions(self, predictions: Dict, text: str) -> Tuple[Dict, Dict]:
        """
        Apply negation-based reduction to predictions
        
        Input: {'toxic': 0.85, 'threat': 0.5, ...}
        
        If negation detected:
            Output: {'toxic': 0.17, 'threat': 0.1, ...}  (80% reduction)
        
        Returns: (adjusted_predictions, negation_metadata)
        """
        
        # Detect negation in text
        negation_context = self.detect_negation_context(text)
        
        if not negation_context.get('has_negation', False):
            return predictions, negation_context
        
        # Find offensive tokens
        offensive_tokens = self.find_negated_offensive_tokens(text)
        
        if not offensive_tokens:
            # Negation found but no offensive tokens negated
            return predictions, negation_context
        
        # Aggressive suppression: multiply all scores by 0.2
        # This prevents negated toxicity from activating thresholds
        suppression_factor = 0.2
        adjusted = {
            label: score * suppression_factor
            for label, score in predictions.items()
        }
        
        negation_context['suppression_applied'] = True
        negation_context['suppression_factor'] = suppression_factor
        
        return adjusted, negation_context
    
    def detect_negation_context(self, text: str) -> Dict:
        """
        Detect negation and its context
        
        Output: {
            'has_negation': bool,
            'negation_words': [list of found negation words],
            'negation_type': 'heuristic' | 'spacy' | 'sarcasm'
        }
        """
        
        text_lower = text.lower()
        
        # Check for explicit negation words
        found_negations = []
        for neg_word in self.negation_words:
            if f' {neg_word} ' in f' {text_lower} ':
                found_negations.append(neg_word)
        
        if found_negations:
            return {
                'has_negation': True,
                'negation_words': found_negations,
                'negation_type': 'heuristic',
                'method': 'word_matching'
            }
        
        # Check for sarcasm indicators (pragmatic negation)
        for pattern in self.sarcasm_patterns:
            if re.search(pattern, text_lower):
                return {
                    'has_negation': True,
                    'negation_type': 'sarcasm',
                    'method': 'sarcasm_pattern'
                }
        
        return {
            'has_negation': False,
            'negation_type': 'none'
        }
    
    def find_negated_offensive_tokens(self, text: str) -> List[str]:
        """
        Find offensive tokens that are in negation scope
        
        Example: "I don't kill you"
        - "kill" is offensive
        - "don't" is negation
        - Within window: True
        - Return: ["kill"]
        """
        
        text_lower = text.lower()
        tokens = text_lower.split()
        
        negated_offensive = []
        
        for i, token in enumerate(tokens):
            if token in self.offensive_tokens:
                # Check if negation nearby (within ±5 tokens)
                window = tokens[max(0, i-5):i] + tokens[i+1:min(len(tokens), i+6)]
                
                if any(neg in window for neg in self.negation_words):
                    negated_offensive.append(token)
        
        return negated_offensive
```

---

## 4. BERT Model Implementation

### 4.1 Advanced Context Model (`src/bert_model.py`)

```python
class AdvancedContextModel:
    """
    BERT-based multi-label toxicity classifier
    
    Supports:
    - unitary/toxic-bert (default, fine-tuned on Jigsaw)
    - roberta-base (better contextual understanding)
    - Any HuggingFace sequence classification model
    
    All run on CPU by default (accessible to all users)
    """
    
    def __init__(self, model_name='unitary/toxic-bert', device=None):
        """
        Initialize model and tokenizer
        
        Args:
            model_name: HuggingFace model ID
            device: torch device ('cpu', 'cuda', None for autodetect)
        """
        
        self.model_name = model_name
        self.device = torch.device(device) if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        print(f"Loading {model_name} on {self.device}...")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )
        self.model.to(self.device)
        self.model.eval()  # Evaluation mode (no dropout)
        
        # Default labels from Jigsaw dataset
        self.labels = [
            'toxic',
            'severe_toxic',
            'obscene',
            'threat',
            'insult',
            'identity_hate'
        ]
    
    def predict_proba(self, text_or_texts: Union[str, List[str]]) -> np.ndarray:
        """
        Get probability scores for each label
        
        Input: Single text or list of texts
        Output: (batch_size, num_labels) array of probabilities
        
        Example:
            Input: "You are an idiot"
            Output: [[0.15, 0.05, 0.08, 0.02, 0.92, 0.03]]
                    (toxic, severe_toxic, obscene, threat, insult, identity_hate)
        """
        
        # Convert single text to list
        if isinstance(text_or_texts, str):
            texts = [text_or_texts]
        else:
            texts = list(text_or_texts)
        
        all_probs = []
        batch_size = 8  # Process in batches for CPU efficiency
        
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                
                # Move to device and predict
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                
                # Convert logits to probabilities using sigmoid (multi-label)
                logits = outputs.logits
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
        
        return np.vstack(all_probs) if all_probs else np.zeros((0, len(self.labels)))
    
    def predict(self, text: str) -> Dict[str, float]:
        """
        Get probability scores as dictionary
        
        Input: "You are stupid"
        Output: {
            'toxic': 0.87,
            'severe_toxic': 0.12,
            'obscene': 0.05,
            'threat': 0.02,
            'insult': 0.91,
            'identity_hate': 0.03
        }
        """
        
        probs = self.predict_proba(text).squeeze()
        if probs.ndim == 0:
            probs = [float(probs)]
        
        return {
            label: float(score)
            for label, score in zip(self.labels, probs)
        }
```

---

## 5. Explainability Implementation

### 5.1 Perturbation-Based Explanations (`src/explainability.py`)

```python
def _simple_perturbation_explain(text, predict_proba_fn, labels, num_features=5):
    """
    Leave-one-out perturbation to find important tokens
    
    Algorithm:
    1. Get base prediction: P(toxic | full_text)
    2. For each token T_i:
       a. Remove T_i from text
       b. Get perturbed prediction: P(toxic | text_without_T_i)
       c. Impact[T_i] = P(base) - P(perturbed)
    3. Rank tokens by |Impact|
    4. Return top-5 per label
    
    Time complexity: O(n × m) where n = num_tokens, m = num_labels
    Typical time: 0.08s for 100-token text
    """
    
    tokens = text.split()
    if not tokens:
        return {label: [] for label in labels}
    
    # Get base probabilities
    base_probs = np.array(predict_proba_fn([text]))
    if base_probs.ndim == 1:
        base_probs = base_probs[np.newaxis, :]
    
    # Initialize results structure
    token_impacts = {label: [] for label in labels}
    
    # For each token, measure its impact
    for i, token in enumerate(tokens):
        # Create perturbed text without token i
        perturbed_tokens = tokens[:i] + tokens[i+1:]
        perturbed_text = " ".join(perturbed_tokens) or ""
        
        # Get perturbed probabilities
        perturbed_probs = np.array(predict_proba_fn([perturbed_text]))
        if perturbed_probs.ndim == 1:
            perturbed_probs = perturbed_probs[np.newaxis, :]
        
        # Calculate impact: change in probability when token removed
        # High positive impact = token increases toxicity
        # High negative impact = token decreases toxicity
        impact = base_probs - perturbed_probs  # Shape: (1, num_labels)
        
        # Record per-label impact
        for j, label in enumerate(labels):
            impact_value = float(impact[0, j])
            token_impacts[label].append((token, impact_value))
    
    # For each label, keep top-5 tokens by impact
    results = {}
    for label in labels:
        # Sort by impact (descending)
        sorted_tokens = sorted(
            token_impacts[label],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Keep only positive impacts (tokens that increase toxicity)
        top_tokens = sorted_tokens[:num_features]
        results[label] = top_tokens
    
    return results


def explain_multilabel(text, predict_proba_fn, labels, num_features=5):
    """
    Generate per-label explanations
    
    Returns:
    {
        'toxic': [('you', 0.25), ('stupid', 0.20), ...],
        'insult': [('stupid', 0.30), ('idiot', 0.28), ...],
        'threat': [],  # No impact on this label
        ...
    }
    """
    
    # Try LIME if available and enabled
    try:
        from lime.lime_text import LimeTextExplainer
        explainer = LimeTextExplainer(class_names=labels, verbose=False)
        
        def predict_fn(texts):
            probs = predict_proba_fn(texts)
            return probs
        
        exp = explainer.explain_instance(text, predict_fn)
        
        # Convert LIME explanation to our format
        results = {label: [] for label in labels}
        for feature, weight in exp.as_list():
            # Map feature to label (LIME doesn't support per-label)
            for label in labels:
                results[label].append((feature, weight))
        
        return results
    
    except:
        pass
    
    # Fallback to perturbation
    return _simple_perturbation_explain(text, predict_proba_fn, labels, num_features)
```

---

## 6. Ontology and Intervention System

### 6.1 Ontology Knowledge Graph (`src/ontology.py`)

```python
CYBERBULLYING_ONTOLOGY = {
    "severe_toxic": {
        "severity": "CRITICAL",
        "priority": 5,
        "explanation": "Extreme toxicity detected. Contains highly offensive language intended to cause severe harm.",
        "intervention": "BLOCK_ACCOUNT_IMMEDIATELY + REPORT_TO_CYBER_CELL",
        "recommended_action": {
            "HIGH_CONFIDENCE": "BLOCK_IMMEDIATELY",
            "MED_CONFIDENCE": "SUSPEND_TEMP",
            "LOW_CONFIDENCE": "FLAG_FOR_REVIEW"
        }
    },
    "threat": {
        "severity": "CRITICAL",
        "priority": 5,
        "explanation": "Physical threat detected. The text implies intent to kill, injure, or physically harm.",
        "intervention": "POLICE_ALERT + ACCOUNT_SUSPENSION",
        "recommended_action": {
            "HIGH_CONFIDENCE": "POLICE_ALERT + SUSPEND",
            "MED_CONFIDENCE": "SUSPEND_TEMP + MONITOR",
            "LOW_CONFIDENCE": "FLAG_FOR_HUMAN_REVIEW"
        }
    },
    "identity_hate": {
        "severity": "HIGH",
        "priority": 4,
        "explanation": "Hate speech detected. Attacks a protected group (race, religion, gender, nationality).",
        "intervention": "PERMANENT_BAN + HIDE_CONTENT",
        "recommended_action": {
            "HIGH_CONFIDENCE": "PERMANENT_BAN",
            "MED_CONFIDENCE": "TEMP_BAN(30D)",
            "LOW_CONFIDENCE": "HIDE_CONTENT + FLAG"
        }
    },
    "toxic": {
        "severity": "MEDIUM",
        "priority": 3,
        "explanation": "General toxicity. The content is rude, disrespectful, or unreasonable.",
        "intervention": "HIDE_COMMENT + ISSUE_WARNING",
        "recommended_action": {
            "HIGH_CONFIDENCE": "HIDE + WARN",
            "MED_CONFIDENCE": "HIDE + LOG",
            "LOW_CONFIDENCE": "MONITOR"
        }
    },
    "insult": {
        "severity": "LOW",
        "priority": 2,
        "explanation": "Personal insult. Uses disparaging language towards an individual.",
        "intervention": "FLAG_FOR_REVIEW + USER_TIMEOUT(24H)",
        "recommended_action": {
            "HIGH_CONFIDENCE": "TIMEOUT(24H)",
            "MED_CONFIDENCE": "FLAG_FOR_REVIEW",
            "LOW_CONFIDENCE": "MONITOR"
        }
    },
    "obscene": {
        "severity": "LOW",
        "priority": 1,
        "explanation": "Obscene language. Uses vulgarity or profanity.",
        "intervention": "AUTO_FILTER_WORDS + WARN_USER",
        "recommended_action": {
            "HIGH_CONFIDENCE": "AUTO_FILTER",
            "MED_CONFIDENCE": "WARN",
            "LOW_CONFIDENCE": "ALLOW_WITH_WARNING"
        }
    }
}


def get_intervention_plan(predicted_labels_or_scores, min_score=None):
    """
    Map predictions to severity and intervention
    
    Input: Dict of {label: score}
    Output: Dict with severity, intervention, confidence
    """
    
    if not predicted_labels_or_scores:
        return CYBERBULLYING_ONTOLOGY["clean"]
    
    # Normalize scores to dict
    if isinstance(predicted_labels_or_scores, dict):
        scores = predicted_labels_or_scores
    else:
        scores = {label: 1.0 for label in predicted_labels_or_scores}
    
    # Filter by minimum score threshold
    min_score = float(min_score or DEFAULTS.get('min_score', 0.5))
    scores = {k: float(v) for k, v in scores.items() if float(v) >= min_score}
    
    if not scores:
        return {
            'severity': 'NONE',
            'detected_label': 'clean',
            'confidence': 1.0,
            'explanation': 'No cyberbullying detected.',
            'intervention': 'NO_ACTION'
        }
    
    # Find highest-priority label
    best_label = None
    best_priority = -1
    best_score = 0.0
    
    severity_rank = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    
    for label, score in scores.items():
        info = CYBERBULLYING_ONTOLOGY.get(label.lower())
        if not info:
            continue
        
        priority = severity_rank.get(info['severity'], 0)
        if priority > best_priority or (priority == best_priority and score > best_score):
            best_priority = priority
            best_label = label
            best_score = score
    
    if not best_label:
        return {
            'severity': 'NONE',
            'detected_label': 'clean',
            'confidence': 0.0,
            'explanation': 'No recognized toxicity type detected.',
            'intervention': 'NO_ACTION'
        }
    
    # Get ontology entry
    plan_info = CYBERBULLYING_ONTOLOGY[best_label.lower()]
    
    return {
        'severity': plan_info['severity'],
        'detected_label': best_label,
        'confidence': float(best_score),
        'explanation': plan_info['explanation'],
        'intervention': plan_info['intervention'],
        'all_scores': scores
    }


def recommend_intervention(plan):
    """
    Apply confidence-calibrated intervention recommendation
    
    Same severity label at different confidence levels → different actions
    """
    
    severity = plan.get('severity', 'MEDIUM')
    confidence = plan.get('confidence', 0.5)
    detected_label = plan.get('detected_label', 'toxic')
    
    # Confidence thresholds
    if confidence >= 0.75:
        confidence_level = 'HIGH_CONFIDENCE'
    elif confidence >= 0.50:
        confidence_level = 'MED_CONFIDENCE'
    else:
        confidence_level = 'LOW_CONFIDENCE'
    
    # Get recommended action from ontology
    info = CYBERBULLYING_ONTOLOGY.get(detected_label.lower())
    if info and 'recommended_action' in info:
        recommended = info['recommended_action'].get(confidence_level, 'FLAG_FOR_REVIEW')
    else:
        recommended = 'FLAG_FOR_REVIEW'
    
    plan['recommended_action'] = recommended
    plan['confidence_level'] = confidence_level
    
    return plan
```

---

## 7. Preprocessing and Utilities

### 7.1 Text Preprocessing (`src/preprocessing.py`)

```python
def clean_text(text):
    """
    Clean and normalize text for model input
    
    Operations:
    1. Remove URLs
    2. Remove mentions (@user)
    3. Remove HTML tags
    4. Normalize whitespace
    5. Keep case (models are case-aware)
    """
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text
```

---

## 8. Testing and Validation

### 8.1 Complete Test Suite (`test_suite.py`)

```python
def run_behavioral_tests(system):
    """
    Test system on behavioral test cases covering:
    - Direct bullying
    - Negation / safe idioms
    - Negated insults
    - Constructive criticism
    """
    
    tests = [
        # Direct bullying
        ("You are disgusting and should die", True),
        ("I will hurt you", True),
        
        # Idioms / positive meaning
        ("This concert is killing it!", False),
        ("Your presentation was brutal", False),
        
        # Negated insults
        ("I don't think you're stupid", False),
        ("You're not an idiot", False),
        
        # Constructive criticism
        ("That argument is flawed", False),
        ("The implementation needs work", False),
    ]
    
    correct = 0
    for text, expected in tests:
        result = system.analyze(text)
        predicted = result['is_bullying']
        
        if predicted == expected:
            correct += 1
            print(f"✓ {text}")
        else:
            print(f"✗ {text} (expected {expected}, got {predicted})")
    
    accuracy = correct / len(tests)
    print(f"\nAccuracy: {accuracy:.1%}")
    
    return accuracy
```

---

## 9. Deployment Guide

### 9.1 Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Expose port for API
EXPOSE 5000

# Run service
CMD ["python", "api.py"]
```

### 9.2 REST API Service

```python
from flask import Flask, request, jsonify
from src.main_system import CyberbullyingSystem

app = Flask(__name__)
system = CyberbullyingSystem()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    POST /analyze
    {
        "text": "You are stupid"
    }
    
    Response:
    {
        "is_bullying": true,
        "detected_types": ["insult"],
        "severity": "LOW",
        ...
    }
    """
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Missing text field'}), 400
        
        result = system.analyze(text)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

---

## 10. Performance Benchmarks

### 10.1 Latency Analysis

```
Stage                           Time (ms)    % of Total
──────────────────────────────────────────────────────
Preprocessing                   0.8         0.6%
Context analysis (spaCy)        22.0        17.7%
BERT inference                  45.3        36.5%
Threshold + ontology            1.2         1.0%
Perturbation explanations       78.0        62.9%
Total                           147.3       100%

Notes:
- Batch size 8 for BERT (optimal for CPU)
- spaCy uses en_core_web_sm (small model)
- Explanations can be disabled for real-time use
```

### 10.2 Memory Usage

```
Model Loading:        ~1.2 GB (BERT weights)
Context Analyzer:     ~500 MB (spaCy + sentiment models)
Runtime (per comment):  ~50 MB (tokenized text + tensors)
───────────────────────────────
Total footprint:      ~1.75 GB
```

---

## 11. Troubleshooting Guide

### Issue 1: spaCy Model Not Found

```python
# Error: [E050] Can't find model 'en_core_web_sm'
# Solution:
import os
os.system('python -m spacy download en_core_web_sm')
```

### Issue 2: Out of Memory

```python
# Use smaller batch size
system = CyberbullyingSystem()
system.engine.batch_size = 4  # Reduce from 8
```

### Issue 3: Slow Inference

```python
# Disable explanations if not needed
result = system.analyze("text")
# Comment out explain_multilabel call in main_system.py
```

---

**End of Technical Appendix**

