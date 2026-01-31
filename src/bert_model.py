# src/bert_model.py
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from src.preprocessing import clean_text

class AdvancedContextModel:
    def __init__(self):
        print("\n[CONTEXT-AWARE] Loading BERT Model (unitary/toxic-bert)...")
        # This model is pre-trained specifically on the Jigsaw Toxic Comment dataset
        # It understands context like "I will kill you" vs "You killed it!"
        self.model_name = "unitary/toxic-bert" 
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval() # Set to inference mode (faster)

    def predict(self, text):
        """
        Returns dictionary of labels and their probabilities.
        Minimal preprocessing to preserve BERT's contextual understanding.
        """
        # Use minimal preprocessing to preserve context
        text = clean_text(text)
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Convert logits to probabilities (Sigmoid for multi-label)
        probs = torch.sigmoid(outputs.logits).squeeze().tolist()
        
        # Ensure probs is a list (if single input)
        if isinstance(probs, float): 
            probs = [probs]
            
        labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        # Map labels to scores
        results = {label: score for label, score in zip(labels, probs)}
        return results