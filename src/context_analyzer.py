"""
Context Analyzer Module
Analyzes linguistic context to improve cyberbullying detection accuracy.
Handles: positive intent, neutral statements, opinion vs personal attack, etc.
"""

import re

class ContextAnalyzer:
    def __init__(self):
        # Words that indicate SAFE context (statements about things, not people)
        self.safe_targets = {
            "it", "this", "that", "movie", "song", "game", "show", "book", "code",
            "project", "work", "food", "weather", "performance", "presentation",
            "idea", "plan", "approach", "method", "result", "outcome", "decision",
            "thing", "stuff", "action", "behavior", "attempt", "job", "task"
        }
        
        # Words indicating OPINION (neutral, not personal attack)
        self.opinion_words = {
            "i think", "i believe", "in my opinion", "seems", "appears",
            "looks like", "i feel", "to me", "arguably", "arguably", "some say",
            "it seems", "it appears"
        }
        
        # Positive achievement words that can flip meaning
        self.positive_verbs = {
            "killed", "crushed", "smashed", "nailed", "destroyed", "owned",
            "beat", "wrecked", "slayed", "murdered", "aced", "excelled",
            "dominated", "annihilated", "wiped", "buried"
        }
        
        # Words indicating AFFECTION/CARE (opposite of harm)
        self.affection_words = {
            "love", "care", "appreciate", "admire", "respect", "honor",
            "cherish", "adore", "like", "enjoy", "favor", "prefer"
        }
        
        # Context patterns for positive use of harsh words
        self.positive_patterns = [
            r"you\s+(?:absolutely\s+)?(?:killed|crushed|smashed|nailed|destroyed|owned|beat)",
            r"that\s+(?:was\s+)?(?:amazing|awesome|incredible|fantastic|great|excellent)",
            r"(?:great|awesome|cool|nice)\s+(?:job|work|effort|try|attempt)",
            r"you\s+(?:did\s+)?(?:great|well|amazing|awesome|fantastic|incredible)",
        ]

    def detect_target_type(self, text):
        """
        Determine who/what is being targeted: person or thing?
        BULLYING = attack on PERSON
        OPINION = criticism of THING/IDEA
        
        Returns:
            'person' | 'thing' | 'unclear'
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        # Check who comes after accusation words
        # "You are stupid" -> person | "This is stupid" -> thing
        person_indicators = ["you", "he", "she", "they", "them", "him", "her", "i"]
        thing_indicators = ["it", "this", "that", "the"] + list(self.safe_targets)
        
        # Look at context around negative words
        negative_patterns = [
            r"(?:are|is|seems|looks)\s+\w+\s*(?:and\s+)?(\w+)",  # "you are stupid"
            r"(?:stupid|idiot|dumb|hate|terrible|bad)\s+(\w+)",   # "stupid idiot"
        ]
        
        for pattern in negative_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                target = match.group(1) if match.groups() else None
                if target:
                    if target in person_indicators:
                        return 'person'
                    elif target in thing_indicators or any(t in target for t in self.safe_targets):
                        return 'thing'
        
        # Check sentence subject
        if text_lower.startswith("you "):
            return 'person'
        elif text_lower.startswith("this ") or text_lower.startswith("that ") or text_lower.startswith("it "):
            return 'thing'
        
        return 'unclear'

    def is_positive_achievement(self, text):
        """
        Check if seemingly negative words are used positively.
        Examples: "You killed it!", "That was awesome!"
        
        Returns:
            bool: True if context is positive achievement
        """
        text_lower = text.lower()
        
        # Check positive patterns
        for pattern in self.positive_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Check for achievement context
        achievement_words = ["great", "awesome", "excellent", "amazing", "incredible", "fantastic"]
        positive_verb_used = any(verb in text_lower for verb in self.positive_verbs)
        achievement_nearby = any(word in text_lower for word in achievement_words)
        
        if positive_verb_used and achievement_nearby:
            return True
        
        return False

    def analyze_context(self, text):
        """
        Comprehensive context analysis
        
        Returns:
            dict: {
                'target_type': 'person' | 'thing' | 'unclear',
                'is_opinion': bool,
                'is_positive_achievement': bool,
                'is_affection': bool,
                'context_score': float (0-1, how much to reduce toxicity),
                'reason': str
            }
        """
        text_lower = text.lower()
        
        # Detect target
        target = self.detect_target_type(text)
        
        # Check if it's opinion-based
        is_opinion = any(phrase in text_lower for phrase in self.opinion_words)
        
        # Check for positive achievement context
        is_positive = self.is_positive_achievement(text)
        
        # Check for affection words (opposite of harm)
        is_affection = any(word in text_lower for word in self.affection_words)
        
        # Calculate context score (0 = reduce toxicity, 1 = keep toxicity)
        context_score = 1.0
        reason = ""
        
        if is_positive:
            context_score = 0.05  # Positive achievement context, nearly negate
            reason = "Positive achievement language (e.g., 'you killed it')"
        elif target == 'thing':
            context_score = 0.30  # Criticism of thing, not person
            reason = "Opinion about thing/idea, not personal attack"
        elif is_opinion:
            context_score = 0.50  # Opinion-based, not aggressive
            reason = "Opinion-based statement"
        elif is_affection:
            context_score = 0.10  # Affection/care indicated
            reason = "Indicates care/affection"
        
        return {
            'target_type': target,
            'is_opinion': is_opinion,
            'is_positive_achievement': is_positive,
            'is_affection': is_affection,
            'context_score': context_score,
            'reason': reason if reason else "No special context detected"
        }

    def adjust_threshold(self, base_threshold, context_analysis):
        """
        Adjust detection threshold based on context.
        Higher threshold = harder to trigger alarm
        
        Example: "you are dumb" (thing) -> raise threshold from 0.5 to 0.65
        """
        multiplier = context_analysis['context_score']
        
        # Adjust threshold upward (make it harder to detect as bullying)
        adjusted = base_threshold * (1.0 + (1.0 - multiplier))
        
        return min(0.95, max(0.3, adjusted))  # Cap between 0.3 and 0.95
