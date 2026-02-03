import numpy as np
from src.bert_model import AdvancedContextModel
from src.ontology import get_intervention_plan
from src.negation_handler import NegationHandler
from src.context_analyzer import ContextAnalyzer
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
    """
    def __init__(self, model_name='unitary/toxic-bert'):
        self.engine = AdvancedContextModel(model_name=model_name)
        self.base_threshold = 0.50
        self.threshold = 0.50
        
        # Initialize context awareness modules
        self.negation_handler = NegationHandler()
        self.context_analyzer = ContextAnalyzer()
        
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
        
        # 0. CONTEXT ANALYSIS (NEW - Negations and Linguistic Context)
        negation_context = self.negation_handler.detect_negation_context(user_text)
        linguistic_context = self.context_analyzer.analyze_context(user_text)
        
        # Adjust threshold based on context
        adjusted_threshold = self.base_threshold
        if negation_context['has_negation']:
            # When negation is present we should make it harder to trigger bullying
            # by increasing the threshold (was mistakenly lowering it).
            adjusted_threshold = self.base_threshold * 1.5
        adjusted_threshold = self.context_analyzer.adjust_threshold(adjusted_threshold, linguistic_context)
        
        # 1. Get Context-Aware Predictions (BERT)
        predictions = self.engine.predict(user_text)
        
        # 1.5 ADJUST FOR NEGATIONS
        predictions, neg_context = self.negation_handler.adjust_predictions(predictions, user_text)
        
        # 1.6 ADJUST FOR LINGUISTIC CONTEXT
        context_factor = linguistic_context['context_score']
        predictions = {label: score * context_factor for label, score in predictions.items()}
        
        # 2. Filter Active Labels (Severity Check with adjusted threshold)
        active_labels = [label for label, score in predictions.items() if score > adjusted_threshold]
        
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
                "negation_detected": negation_context['has_negation'],
                "negation_type": negation_context['negation_type'],
                "has_sarcasm": negation_context['has_sarcasm'],
                "target_type": linguistic_context['target_type'],
                "is_opinion": linguistic_context['is_opinion'],
                "is_positive_achievement": linguistic_context['is_positive_achievement'],
                "context_reason": linguistic_context['reason'],
                "adjusted_threshold": round(adjusted_threshold, 3)
            }
        }