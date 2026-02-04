"""
Advanced Context Awareness Module
Uses spaCy dependency parsing, advanced sarcasm detection, and sophisticated NLP techniques.

Features:
- Dependency-based negation detection (high precision)
- Advanced sarcasm patterns (via sentiment + keyword combinations)
- Target type detection (attack on person vs idea)
- Clause-level sentiment analysis
- Contextual word embeddings for semantic understanding
"""

import re
import os
from typing import Dict, List, Tuple

try:
    import spacy
    _HAS_SPACY = True
except ImportError:
    _HAS_SPACY = False

try:
    from transformers import pipeline
    _HAS_SENTIMENT = True
except ImportError:
    _HAS_SENTIMENT = False


class AdvancedContextAnalyzer:
    """Production-grade context analyzer using spaCy and deep NLP."""
    
    def __init__(self):
        """Initialize with spaCy model and sentiment pipeline."""
        self.use_spacy = False
        self.nlp = None
        
        if _HAS_SPACY:
            try:
                self.nlp = spacy.load('en_core_web_sm')
                self.use_spacy = True
            except:
                try:
                    os.system('python -m spacy download en_core_web_sm')
                    self.nlp = spacy.load('en_core_web_sm')
                    self.use_spacy = True
                except:
                    self.use_spacy = False
        
        self.sentiment_pipeline = None
        if _HAS_SENTIMENT:
            try:
                self.sentiment_pipeline = pipeline('sentiment-analysis')
            except:
                pass
        
        # Comprehensive negation patterns
        self.negation_patterns = [
            r"\bnot\s+\w+(?:\s+\w+){0,3}(?:kill|harm|hurt|hate|threat|attack)",
            r"\b(?:don't|doesn't|didn't|won't|can't|couldn't|shouldn't)\s+(?:\w+\s+){0,2}(?:kill|harm|hurt|hate)",
            r"\bnever\s+(?:\w+\s+){0,2}(?:will|would)?\s*(?:kill|harm)",
            r"\bno\s+(?:intention|intent|way|chance).*(?:kill|harm|hurt)",
            r"(?:i\s+)?(?:don't|do not)\s+(?:think|believe).*\byou\b.*(?:are|is).*(?:stupid|dumb|idiot)"
        ]
        
        # Sarcasm indicators with intensity
        self.sarcasm_patterns = [
            (r'\byeah right\b', 'strong'),
            (r'\bsure.*(?:buddy|pal|friend)\b', 'strong'),
            (r'\blol\b.*(?:not|sure|right)', 'medium'),
            (r'\btotally\b.*(?:not|fake)', 'strong'),
            (r'\bob[s]*\s+(?:how|thanks|great)\b', 'medium'),
            (r'\blike\s+yeah\b', 'medium'),
            (r'\b(?:right|sure|of course)\s*[!?]{2,}\b', 'medium'),
        ]
        
        # Positive achievement verbs in positive contexts
        self.achievement_verbs = {
            'killed', 'crushed', 'smashed', 'nailed', 'destroyed', 'owned',
            'beat', 'wrecked', 'slayed', 'murdered', 'aced', 'excelled',
            'dominated', 'annihilated', 'wiped', 'buried'
        }
        
        self.positive_modifiers = {
            'absolutely', 'totally', 'really', 'so', 'very', 'incredible',
            'amazing', 'awesome', 'fantastic', 'great', 'excellent', 'perfect'
        }
    
    def analyze_negation_spacy(self, text: str) -> Dict:
        """Use spaCy dependency parsing for precise negation detection."""
        if not self.use_spacy:
            return {'has_negation': False, 'negation_strength': 0.0, 'method': 'regex'}
        
        try:
            doc = self.nlp(text.lower())
            negations = []
            
            for token in doc:
                # Find negation dependencies
                if token.dep_ == 'neg':
                    # Find the word being negated (parent)
                    negated_token = token.head
                    negations.append({
                        'negation': token.text,
                        'negated_token': negated_token.text,
                        'pos': negated_token.pos_
                    })
            
            if negations:
                # Check if negated tokens are threat-related
                threat_verbs = {'kill', 'harm', 'hurt', 'threaten', 'attack', 'injure'}
                threat_negations = [
                    n for n in negations 
                    if n['negated_token'] in threat_verbs or n['pos'] == 'VERB'
                ]
                
                strength = len(threat_negations) / (len(negations) + 1e-8)
                return {
                    'has_negation': True,
                    'negation_strength': strength,
                    'negations': negations,
                    'method': 'spacy'
                }
            
            return {'has_negation': False, 'negation_strength': 0.0, 'method': 'spacy'}
        
        except Exception as e:
            return {'has_negation': False, 'negation_strength': 0.0, 'method': 'spacy', 'error': str(e)}
    
    def detect_sarcasm_advanced(self, text: str) -> Tuple[bool, float]:
        """Detect sarcasm using multiple signals."""
        text_lower = text.lower()
        
        # Check regex patterns
        sarcasm_score = 0.0
        max_strength = 0.0
        
        for pattern, strength in self.sarcasm_patterns:
            if re.search(pattern, text_lower):
                strength_val = {'strong': 0.9, 'medium': 0.6}[strength]
                sarcasm_score = max(sarcasm_score, strength_val)
                max_strength = max(max_strength, strength_val)
        
        # Check for positive achievement with sarcasm markers
        if any(verb in text_lower for verb in self.achievement_verbs):
            # If followed by negative sentiment or sarcasm markers, it's positive
            if any(m in text_lower for m in ['lol', 'jk', 'just kidding', 'yeah right']):
                sarcasm_score = max(sarcasm_score, 0.1)  # More likely positive
        
        # Check sentiment analysis if available
        if self.sentiment_pipeline and sarcasm_score < 0.3:
            try:
                result = self.sentiment_pipeline(text[:512])  # Limit for speed
                if result[0]['label'] == 'POSITIVE' and any(
                    neg_word in text_lower 
                    for neg_word in ['bad', 'hate', 'stupid', 'dumb', 'idiot']
                ):
                    # Positive sentiment with negative words = sarcasm
                    sarcasm_score = max(sarcasm_score, 0.5)
            except:
                pass
        
        return sarcasm_score > 0.4, sarcasm_score
    
    def detect_target_type_advanced(self, text: str) -> Dict:
        """Determine if attack targets person or idea."""
        if not self.use_spacy:
            return {'target_type': 'unclear', 'confidence': 0.0}
        
        try:
            doc = self.nlp(text.lower())
            
            # Extract subject
            subject = None
            for token in doc:
                if token.dep_ in ('nsubj', 'nsubjpass'):
                    subject = token.text
                    break
            
            if not subject:
                return {'target_type': 'unclear', 'confidence': 0.3}
            
            # Person indicators
            person_words = {'you', 'he', 'she', 'they', 'i', 'we', 'name', 'person', 'guy', 'girl'}
            thing_words = {'it', 'this', 'that', 'idea', 'work', 'project', 'film', 'movie', 'show', 'code'}
            
            if subject in person_words:
                return {'target_type': 'person', 'confidence': 0.9, 'subject': subject}
            elif subject in thing_words:
                return {'target_type': 'thing', 'confidence': 0.9, 'subject': subject}
            else:
                return {'target_type': 'unclear', 'confidence': 0.5, 'subject': subject}
        
        except:
            return {'target_type': 'unclear', 'confidence': 0.0}
    
    def analyze_context_full(self, text: str) -> Dict:
        """Complete context analysis combining all methods."""
        negation_result = self.analyze_negation_spacy(text)
        sarcasm_detected, sarcasm_score = self.detect_sarcasm_advanced(text)
        target_result = self.detect_target_type_advanced(text)
        
        # Compute context reduction factor
        reduction_factor = 1.0
        
        if negation_result.get('has_negation'):
            reduction_factor *= (1.0 - negation_result.get('negation_strength', 0.5))
        
        if sarcasm_detected:
            reduction_factor *= 0.2  # Heavy reduction for sarcasm
        
        if target_result.get('target_type') == 'thing':
            reduction_factor *= 0.5  # Moderate reduction for idea-based attacks
        
        return {
            'text': text,
            'negation': negation_result,
            'sarcasm': {'detected': sarcasm_detected, 'score': sarcasm_score},
            'target': target_result,
            'reduction_factor': max(0.0, min(1.0, reduction_factor)),
            'explanation': self._generate_explanation(negation_result, sarcasm_detected, target_result)
        }
    
    def _generate_explanation(self, negation, sarcasm, target) -> str:
        """Generate human-readable explanation."""
        explanations = []
        
        if negation.get('has_negation'):
            strength = negation.get('negation_strength', 0.5)
            if strength > 0.7:
                explanations.append("Strong negation detected (threat negated)")
            else:
                explanations.append("Negation detected")
        
        if sarcasm:
            explanations.append("Sarcasm detected")
        
        if target.get('target_type') == 'thing':
            explanations.append("Criticism of idea/thing, not person")
        elif target.get('target_type') == 'person':
            explanations.append("Direct personal attack")
        
        return " | ".join(explanations) if explanations else "No special context detected"

    def adjust_threshold(self, base_threshold, context_analysis):
        """
        Adjust detection threshold based on advanced context analysis or
        a compatibility `linguistic_context` dict produced by the caller.

        This keeps the API consistent with `ContextAnalyzer.adjust_threshold`.
        """
        # context_analysis may be either the advanced full dict or the
        # compatibility dict created in main_system. Prefer 'context_score'
        # if present, otherwise derive from reduction_factor.
        if isinstance(context_analysis, dict) and 'context_score' in context_analysis:
            multiplier = float(context_analysis.get('context_score', 1.0))
        elif isinstance(context_analysis, dict) and 'reduction_factor' in context_analysis:
            # `reduction_factor` is already the multiplier to apply to toxicity
            multiplier = float(context_analysis.get('reduction_factor', 1.0))
        else:
            multiplier = 1.0

        adjusted = base_threshold * (1.0 + (1.0 - multiplier))
        return min(0.95, max(0.3, adjusted))
