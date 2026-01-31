import numpy as np
from lime.lime_text import LimeTextExplainer
from src.bert_model import AdvancedContextModel
from src.ontology import get_intervention_plan

class CyberbullyingSystem:
    def __init__(self):
        self.engine = AdvancedContextModel()
        self.threshold = 0.50 
        
        # Initialize LIME Explainer (The Advanced XAI Component)
        self.explainer = LimeTextExplainer(class_names=['neutral', 'toxic'])

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
        
        # 1. Get Context-Aware Predictions (BERT)
        predictions = self.engine.predict(user_text)
        
        # 2. Filter Active Labels (Severity Check)
        active_labels = [label for label, score in predictions.items() if score > self.threshold]
        
        # 3. Get Ontology Logic (Actionable Plan)
        plan = get_intervention_plan(active_labels)
        
        # 4. Generate LIME Explanation (Visual Proof)
        # This highlights which words triggered the decision
        exp = self.explainer.explain_instance(
            user_text, 
            self._predict_proba_for_lime, 
            num_features=5
        )
        # Extract the top contributing words (e.g., [('idiot', 0.85), ('shut', 0.12)])
        highlighted_words = exp.as_list()
        
        # 5. Final Report
        return {
            "text": user_text,
            "is_bullying": len(active_labels) > 0,
            "detected_types": active_labels,
            "severity": plan['severity'],
            "explanation": plan['explanation'],
            "action": plan['intervention'],
            "highlighted_words": highlighted_words, # <--- NEW ADVANCED FEATURE
            "scores": {k: round(v, 4) for k, v in predictions.items() if v > 0.01}
        }