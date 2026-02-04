import numpy as np
from src.bert_model import AdvancedContextModel
from src.ontology import get_intervention_plan
from src.negation_handler import NegationHandler
from src.context_analyzer import ContextAnalyzer
from src.advanced_context import AdvancedContextAnalyzer
from src.explainability import explain_multilabel

class CyberbullyingSystem:
    """Complete cyberbullying detection system.
    
    Implements four core pillars:
    1. CONTEXT-AWARE: Detects negations, sarcasm, opinion vs personal attack, positive achievements
    2. SEVERITY-BASED: Maps detected labels to CRITICAL/HIGH/MEDIUM/LOW severity and actionable interventions
    3. EXPLAINABLE: Uses LIME (or perturbation fallback) to show which words triggered detection
    4. ACTIONABLE: Recommends specific interventions (suspend, hide, warn, etc.) based on severity + confidence
    
    Supports multiple models:
    - 'unitary/toxic-bert' (default): Jigsaw-fine-tuned BERT
    - 'roberta-base': RoBERTa for better context understanding
    - 'advanced-ensemble' (NEW): 3-model weighted ensemble (DeBERTa v3 + RoBERTa-large + DistilBERT)
    """
    def __init__(self, model_name='unitary/toxic-bert', use_advanced_context=True, use_ensemble=False):
        """
        Args:
            model_name: 'unitary/toxic-bert', 'roberta-base', or 'advanced-ensemble'
            use_advanced_context: If True, use spaCy-based AdvancedContextAnalyzer
            use_ensemble: If True, use AdvancedEnsembleModel (overrides model_name)
        """
        # Try to use ensemble if requested
        self.use_ensemble = use_ensemble
        if use_ensemble:
            try:
                from src.advanced_ensemble import AdvancedEnsembleModel
                self.engine = AdvancedEnsembleModel()
                self.ensemble_mode = True
            except Exception as e:
                print(f"Warning: Failed to load ensemble model, falling back to {model_name}: {e}")
                self.engine = AdvancedContextModel(model_name=model_name)
                self.ensemble_mode = False
        else:
            self.engine = AdvancedContextModel(model_name=model_name)
            self.ensemble_mode = False
        
        self.base_threshold = 0.50
        self.threshold = 0.50
        
        # Initialize context awareness modules
        self.negation_handler = NegationHandler()
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
        
        # LIME will be invoked per-label via explain_multilabel when needed
        self.explainer = None

    def _predict_proba_for_lime(self, texts):
        """
        LIME needs a function that takes a list of texts and returns probabilities 
        specifically formatted as [prob_neutral, prob_toxic].
        We map our complex BERT output to this simple format for visualization.
        """
        # Handle single string input if necessary
        if isinstance(texts, str):
            texts = [texts]
            
        results = []
        for text in texts:
            preds = self.engine.predict(text)
            # We take the maximum 'bad' score as the 'toxic' probability
            toxic_score = max(preds.values()) 
            neutral_score = 1.0 - toxic_score
            results.append([neutral_score, toxic_score])
            
        return np.array(results)

    def analyze(self, user_text):
        print(f"\nProcessing: '{user_text}'...")
        
        # 0. CONTEXT ANALYSIS (Advanced or Basic)
        if self.advanced_context:
            try:
                # Use spaCy-based advanced context analyzer
                context_info = self.context_analyzer.analyze_context_full(user_text)
                # `analyze_context_full` returns nested dicts: 'negation', 'sarcasm', 'target'
                neg = context_info.get('negation', {})
                sarcasm = context_info.get('sarcasm', {})
                target = context_info.get('target', {})

                negation_context = {
                    'has_negation': bool(neg.get('has_negation', False)),
                    'negation_type': neg.get('method', 'advanced_spacy'),
                    'has_sarcasm': bool(sarcasm.get('detected', False)),
                    'sarcasm_confidence': float(sarcasm.get('score', 0.0)),
                    'target_type': target.get('target_type', 'unknown')
                }
                # `reduction_factor` from advanced analyzer is already the multiplier
                linguistic_context = {
                    'target_type': target.get('target_type', 'unknown'),
                    'is_opinion': target.get('target_type', 'unknown') == 'idea',
                    'is_positive_achievement': False,  # Let model decide
                    'reason': context_info.get('explanation', ''),
                    'context_score': float(context_info.get('reduction_factor', 1.0))
                }
            except Exception as e:
                # Fallback to basic if advanced fails
                print(f"Note: Advanced context unavailable ({e}), using basic context analysis")
                negation_context = self.negation_handler.detect_negation_context(user_text)
                linguistic_context = self.context_analyzer.analyze_context(user_text)
        else:
            # Use basic context analyzer
            negation_context = self.negation_handler.detect_negation_context(user_text)
            linguistic_context = self.context_analyzer.analyze_context(user_text)
        
        # Adjust threshold based on context
        adjusted_threshold = self.base_threshold
        
        # Don't increase threshold based on simple negation detection
        # The negation_handler.adjust_predictions() will suppress scores if needed
        # Only increase if we have strong sarcasm indicators
        if self.advanced_context and negation_context.get('has_sarcasm', False):
            adjusted_threshold *= 1.2
            
        adjusted_threshold = self.context_analyzer.adjust_threshold(adjusted_threshold, linguistic_context)
        
        # NEW: Apply lower threshold for threat/severe language detection
        # to catch false negatives like "You're subhuman, not worth living"
        threat_threshold = self.base_threshold * 0.50  # Even lower for threat detection
        
        # 1. Get Context-Aware Predictions (BERT or Ensemble)
        predictions = self.engine.predict(user_text)
        
        # 1.5 ADJUST FOR NEGATIONS (applies aggressive reduction)
        predictions, neg_context = self.negation_handler.adjust_predictions(predictions, user_text)
        
        # 1.6 ADJUST FOR LINGUISTIC CONTEXT
        context_factor = linguistic_context['context_score']
        predictions = {label: score * context_factor for label, score in predictions.items()}
        
        # 2. Filter Active Labels with DUAL threshold strategy:
        #    - Use regular threshold for general toxicity
        #    - Use lower threat threshold for threat/severe_toxic labels
        active_labels = []
        for label, score in predictions.items():
            # Threat-related labels use lower threshold
            if label in ('threat', 'severe_toxic'):
                if score > threat_threshold:
                    active_labels.append(label)
            # All other labels use adjusted threshold
            else:
                if score > adjusted_threshold:
                    active_labels.append(label)
        
        # 3. Get Ontology Logic (Actionable Plan)
        # Pass full score dict to ontology to compute severity + confidence
        plan = get_intervention_plan(predictions)
        
        # 4. Generate per-label LIME explanations (Visual Proof)
        try:
            # engine.predict_proba accepts list input
            labels = list(self.engine.labels)
            label_explanations = explain_multilabel(user_text, self.engine.predict_proba, labels, num_features=5)
        except Exception:
            label_explanations = {}
        # For backward compatibility, produce a flattened highlighted_words list from top label
        highlighted_words = []
        # pick highest-scoring label
        try:
            top_label = max(predictions.items(), key=lambda x: x[1])[0]
            highlighted_words = label_explanations.get(top_label, [])
        except Exception:
            highlighted_words = []
        
        # 5. Final Report
        return {
            "text": user_text,
            "is_bullying": len(active_labels) > 0,
            "detected_types": active_labels,
            "severity": plan['severity'],
            "explanation": plan['explanation'],
            "action": plan['intervention'],
            # Include ontology-selected label and confidence for display
            "detected_label": plan.get('detected_label'),
            "confidence": plan.get('confidence'),
            "highlighted_words": highlighted_words,
            "scores": {k: round(v, 4) for k, v in predictions.items() if v > 0.01},
            
            # NEW: Context information for transparency
            "context_info": {
                "negation_detected": negation_context.get('has_negation', False),
                "negation_type": negation_context.get('negation_type', 'none'),
                "has_sarcasm": negation_context.get('has_sarcasm', False),
                "sarcasm_confidence": negation_context.get('sarcasm_confidence', 0.0),
                "target_type": linguistic_context.get('target_type', 'unknown'),
                "is_opinion": linguistic_context.get('is_opinion', False),
                "is_positive_achievement": linguistic_context.get('is_positive_achievement', False),
                "context_reason": linguistic_context.get('reason', ''),
                "adjusted_threshold": round(adjusted_threshold, 3),
                "threat_threshold": round(threat_threshold, 3),
                "model_type": "ensemble" if self.ensemble_mode else "single",
                "advanced_context": self.advanced_context
            }
        }