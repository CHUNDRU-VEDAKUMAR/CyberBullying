"""
Negation Handler Module
Detects negations and context reversals that flip the meaning of toxicity.
Handles cases like: "I don't kill you", "I will NOT harm you", "You are NOT an idiot"
"""

import re
import os

# Optional spaCy integration for precise negation scope (best-effort)
try:
    import spacy
    _HAS_SPACY = True
except Exception:
    _HAS_SPACY = False

class NegationHandler:
    def __init__(self):
        # Words that negate/reverse toxicity
        self.negation_words = {
            "don't", "dont", "do not", "doesn't", "does not", "didn't", "did not",
            "won't", "wont", "will not", "wouldn't", "would not", "can't", "cant",
            "cannot", "couldn't", "could not", "shouldn't", "should not", "isn't",
            "is not", "isn't", "aren't", "are not", "wasn't", "was not", "weren't",
            "were not", "not", "no", "never", "hardly", "barely", "scarcely", "neither"
        }
        
        # Intensifiers that INCREASE toxicity
        self.intensifiers = {
            "very", "extremely", "absolutely", "definitely", "certainly", "absolutely",
            "utterly", "completely", "totally", "really", "seriously", "damn", "fucking",
            "shit", "asshole", "bastard"
        }
        
        # Sarcasm indicators (context reversers)
        self.sarcasm_patterns = [
            r"yeah right",
            r"sure.*(?:not|buddy|pal)",
            r"oh please",
            r"lol(?:\s+.*)?(?:not|right|sure)",
            r"right.*and.*i'm",
            r"(?:and\s+)?(you|we|i).{0,10}believe(?:\s+that)?",
        ]
        
        # Context patterns that reduce threat perception
        self.safe_context = {
            "jk", "just kidding", "joking", "joke", "sarcasm", "kidding", "lol",
            "haha", "hehe", "rofl", "for fun", "in jest", "hypothetically"
        }

        # Common offensive tokens to check scope against (non-exhaustive)
        # If a data file exists at data/offensive_tokens.txt we will load from it
        default_offensive = {
            'idiot', 'bitch', 'asshole', 'bastard', 'whore', 'dumb', 'stupid', 'fag', 'kill', 'murder', 'slut',
            'terrible', 'disgusting', 'worthless', 'trash', 'loser', 'ugly', 'hate', 'hurt', 'die', 'threat',
            'evil', 'wrong', 'bad', 'fool', 'jerk', 'cruel', 'mean', 'despise', 'abhor', 'vile'
        }
        self.offensive_tokens = set()
        tokens_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'offensive_tokens.txt')
        if os.path.exists(tokens_path):
            try:
                with open(tokens_path, 'r', encoding='utf-8') as fh:
                    for line in fh:
                        t = line.strip()
                        if t:
                            self.offensive_tokens.add(t.lower())
            except Exception:
                self.offensive_tokens = default_offensive
        else:
            self.offensive_tokens = default_offensive

        # spaCy model instance (loaded lazily if available)
        self.use_spacy = False
        self._nlp = None
        if _HAS_SPACY:
            try:
                # Try to load small English model if installed
                self._nlp = spacy.load('en_core_web_sm')
                self.use_spacy = True
            except Exception:
                # spaCy is available but model not installed; fallback to heuristic
                self._nlp = None
                self.use_spacy = False

        # Threat verbs vs insult nouns
        self.threat_verbs = {'kill', 'murder', 'stab', 'shoot', 'hurt', 'attack'}

    def has_negation_nearby(self, text, toxic_word, window=5):
        """
        Check if a toxic word is negated within a word window.
        Example: "I don't kill you" -> "kill" is negated by "don't"
        
        Args:
            text: full text
            toxic_word: the word detected as toxic
            window: words before/after to check
        
        Returns:
            bool: True if negation found near toxic word
        """
        words = text.lower().split()
        
        try:
            idx = words.index(toxic_word.lower())
        except ValueError:
            # Handle cases where exact word not found (due to preprocessing)
            return self._check_fuzzy_negation(text, toxic_word)
        
        # Check words before and after
        start = max(0, idx - window)
        end = min(len(words), idx + window + 1)
        context_words = words[start:end]
        
        return any(word in self.negation_words for word in context_words)

    def _expand_contractions(self, text):
        # Simple contraction expansion for n't forms to improve negation detection
        # common explicit mappings
        text = re.sub(r"\bwon't\b", "will not", text, flags=re.IGNORECASE)
        text = re.sub(r"\bcan't\b", "can not", text, flags=re.IGNORECASE)
        text = re.sub(r"\bain't\b", "is not", text, flags=re.IGNORECASE)

        # Generic n't handling: replace n't at end of contractions with ' not'
        # e.g. weren't -> were not, didn't -> did not
        text = re.sub(r"n['’]t\b", " not", text, flags=re.IGNORECASE)

        # Expand some common short forms (optional)
        text = re.sub(r"\bI'm\b", "I am", text, flags=re.IGNORECASE)
        text = re.sub(r"\byou're\b", "you are", text, flags=re.IGNORECASE)
        text = re.sub(r"\bhe's\b", "he is", text, flags=re.IGNORECASE)
        text = re.sub(r"\bshe's\b", "she is", text, flags=re.IGNORECASE)

        return text

    def find_negated_offensive_tokens(self, text, window=4):
        """Return list of offensive tokens that are directly negated in the text.

        Looks for patterns like: negation ... [determinant] offensive_token
        Example: "you are not a bitch" -> detects 'bitch' as negated
        Also detects: "I don't think you're stupid" -> negation applies through sentence structure
        """
        text_proc = self._expand_contractions(text.lower())
        words = text_proc.split()
        negated = []

        for i, w in enumerate(words):
            # strip punctuation
            token = re.sub(r"[^a-z0-9]", "", w)
            if token in self.offensive_tokens:
                # search window before token for negation words
                start = max(0, i - window)
                context = words[start:i]
                if any(re.sub(r"[^a-z0-9]", "", cw) in self.negation_words for cw in context):
                    negated.append(token)
        return negated

    def _check_fuzzy_negation(self, text, toxic_word):
        """Fuzzy negation check using regex"""
        # Look for negation + any words + toxic_word pattern
        pattern = rf"(?:don't|do not|won't|will not|not|no|never)\s+(?:\w+\s+){{0,3}}.*{toxic_word}"
        return bool(re.search(pattern, text.lower()))

    def _find_negated_tokens_spacy(self, text):
        """Use spaCy dependency parse to identify directly negated offensive tokens.

        Returns list of token texts that are negated.
        """
        if not self.use_spacy or not self._nlp:
            return []
        negated = []
        try:
            doc = self._nlp(text)
            for token in doc:
                tok_text = token.text.lower()
                norm = re.sub(r"[^a-z0-9]", "", tok_text)
                if norm in self.offensive_tokens:
                    # Check for negation dependency in token's children
                    if any(child.dep_ == 'neg' for child in token.children):
                        negated.append(norm)
                        continue
                    # Check ancestors for negation markers
                    for anc in token.ancestors:
                        if any(child.dep_ == 'neg' for child in anc.children):
                            negated.append(norm)
                            break
        except Exception:
            return []
        return negated

    def detect_negation_context(self, text):
        """
        Full analysis: detect negations, intensifiers, sarcasm
        
        Returns:
            dict: {
                'has_negation': bool,
                'negation_type': 'strong' | 'weak' | 'none',
                'has_intensifier': bool,
                'has_sarcasm': bool,
                'confidence': float (0-1)
            }
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        # Check for negations
        has_negation = any(word in self.negation_words for word in words)
        negation_type = self._classify_negation(text_lower)
        
        # Check for intensifiers
        has_intensifier = any(word in self.intensifiers for word in words)
        
        # Check for sarcasm patterns
        has_sarcasm = any(re.search(pattern, text_lower) for pattern in self.sarcasm_patterns)
        
        # Safety context indicators
        has_safe_context = any(phrase in text_lower for phrase in self.safe_context)
        
        return {
            'has_negation': has_negation,
            'negation_type': negation_type,
            'has_intensifier': has_intensifier,
            'has_sarcasm': has_sarcasm,
            'has_safe_context': has_safe_context,
            'confidence': self._calculate_negation_strength(
                has_negation, negation_type, has_sarcasm, has_safe_context
            )
        }

    def _classify_negation(self, text_lower):
        """Classify strength of negation"""
        strong_negations = ["never", "will not", "won't", "absolutely not"]
        
        for negation in strong_negations:
            if negation in text_lower:
                return 'strong'
        
        if any(word in text_lower for word in self.negation_words):
            return 'weak'
        
        return 'none'

    def _calculate_negation_strength(self, has_negation, negation_type, has_sarcasm, has_safe_context):
        """
        Calculate how much negation reduces toxicity score.
        Returns factor to multiply toxicity by (0.0 = completely negated, 1.0 = no negation)
        """
        if not has_negation:
            return 1.0
        
        factor = 1.0
        
        # Strong negations reduce score significantly
        if negation_type == 'strong':
            factor *= 0.15  # Reduce to 15% of original
        elif negation_type == 'weak':
            factor *= 0.40  # Reduce to 40% of original
        
        # Sarcasm further reduces it
        if has_sarcasm:
            factor *= 0.30
        
        # Safe context keywords
        if has_safe_context:
            factor *= 0.25
        
        return max(0.0, min(1.0, factor))

    def adjust_predictions(self, predictions, text):
        """
        Adjust toxicity scores based on negation context.
        Uses sophisticated detection to identify when insults are negated vs implied.
        
        Args:
            predictions: dict of label -> score
            text: original text
        
        Returns:
            adjusted_predictions: dict with reduced scores for negated toxic content
        """
        context = self.detect_negation_context(text)
        text_lower = text.lower()
        
        # STRATEGY: Look for strong negation markers that typically indicate the speaker
        # is NOT insulting the target, but rather expressing doubt about an insult
        # Examples:
        #  "I don't think you're stupid" → speaker doubts you're stupid
        #  "I wouldn't call you disgusting" → speaker wouldn't use that label
        #  "You're not an idiot" → speaker denies the idiot label
        
        # Pattern 1: "I don't/won't/wouldn't [think/believe/call/say]... you are/you're [INSULT]"
        # This pattern is very reliable for negation
        strong_negation_markers = [
            r"(?:i|we)\s+(?:don't|do\s+not|won't|would\s+not|wouldn't)\s+(?:think|believe|say|call|consider)",
            r"you're?\s+(?:not|never|aren't|isn't)",
            r"(?:is\s+|are\s+)?not\s+(?:a|an)?\s+(?:very\s+)?(?:that\s+)?",
        ]
        
        has_strong_negation = any(re.search(pattern, text_lower) for pattern in strong_negation_markers)
        
        # Pattern 2: Check if the text contains BOTH a negation word AND an offensive token
        has_negation = any(word in text_lower for word in self.negation_words)
        has_offensive = any(token in text_lower for token in self.offensive_tokens)
        
        # If strong negation found AND offensive words present → high confidence in negation
        if has_strong_negation and has_offensive:
            # HEAVILY suppress all toxicity when negation is detected
            adjusted = {label: score * 0.001 for label, score in predictions.items()}
            context['confidence'] = 0.001
            context['negated_tokens'] = list(self.offensive_tokens.intersection(set(text_lower.split())))
            return adjusted, context
        
        # Pattern 3: General negation detection (fallback)
        # Apply negation factor if weak negation is present
        if has_negation and context['confidence'] < 0.6:
            negation_factor = context['confidence']
            adjusted = {label: score * negation_factor for label, score in predictions.items()}
            return adjusted, context

        return predictions, context
