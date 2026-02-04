"""
Data Augmentation Pipeline for Cyberbullying Detection
Implements multiple augmentation strategies to increase dataset diversity and robustness.

Techniques:
1. Easy Data Augmentation (EDA): synonym replacement, random insertion, swap, deletion
2. Back-translation: translate to intermediate language and back
3. Paraphrasing via T5
4. Contextual word embeddings swaps
"""

import random
import numpy as np
from typing import List, Tuple

try:
    from nlpaug.augmenter.word import SynonymAug, RandomWordAug
    from nlpaug.augmenter.sentence import BackTranslationAug
    _HAS_NLPAUG = True
except ImportError:
    _HAS_NLPAUG = False

try:
    from transformers import pipeline
    _HAS_T5 = True
except ImportError:
    _HAS_T5 = False


class EDAugmenter:
    """Easy Data Augmentation (EDA) implementation.
    
    Original paper: Wei & Zou (2019)
    https://arxiv.org/abs/1901.11196
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        random.seed(random_state)
    
    def synonym_replacement(self, text: str, n: int = 2) -> str:
        """Replace n random words with synonyms.
        
        Uses a simple approach: WordNet-based synonym lookup
        """
        if not _HAS_NLPAUG:
            return text
        
        try:
            aug = SynonymAug(aug_p=min(n / len(text.split()), 0.3))
            return aug.augment(text)
        except:
            return text
    
    def random_insertion(self, text: str, n: int = 2) -> str:
        """Insert n random words at random positions."""
        words = text.split()
        if len(words) < 2:
            return text
        
        for _ in range(n):
            random_word = random.choice(words)
            random_idx = random.randint(0, len(words))
            words.insert(random_idx, random_word)
        
        return ' '.join(words)
    
    def random_swap(self, text: str, n: int = 2) -> str:
        """Randomly swap n pairs of words."""
        words = text.split()
        if len(words) < 2:
            return text
        
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete words with probability p."""
        if len(text.split()) == 1:
            return text
        
        words = [w for w in text.split() if random.uniform(0, 1) > p]
        return ' '.join(words) if words else text
    
    def augment(self, text: str, num_aug: int = 4) -> List[str]:
        """Generate num_aug augmented versions of text."""
        augmented = [text]  # Original
        
        for _ in range(num_aug):
            aug_method = random.choice(['syn_rep', 'insert', 'swap', 'delete'])
            if aug_method == 'syn_rep':
                augmented.append(self.synonym_replacement(text, n=2))
            elif aug_method == 'insert':
                augmented.append(self.random_insertion(text, n=2))
            elif aug_method == 'swap':
                augmented.append(self.random_swap(text, n=2))
            else:
                augmented.append(self.random_deletion(text, p=0.1))
        
        return augmented


class BackTranslationAugmenter:
    """Back-translation augmentation via neutral languages.
    
    Translates text to intermediate language and back to create natural paraphrases.
    """
    
    def __init__(self):
        self.available = _HAS_NLPAUG
    
    def augment(self, text: str, intermediate_lang: str = 'de') -> List[str]:
        """Generate back-translated versions.
        
        Args:
            text: input text
            intermediate_lang: intermediate language ('de', 'fr', 'es', etc.)
        
        Returns:
            list of augmented texts
        """
        if not self.available:
            return [text]
        
        try:
            aug = BackTranslationAug(
                from_model_name='Helsinki-NLP/opus-mt-en-' + intermediate_lang,
                to_model_name='Helsinki-NLP/opus-mt-' + intermediate_lang + '-en'
            )
            augmented = [text]
            for _ in range(2):  # Generate 2 back-translations
                augmented.append(aug.augment(text))
            return augmented
        except:
            return [text]


class ParaphraseAugmenter:
    """Paraphrase via T5-based models."""
    
    def __init__(self):
        self.available = _HAS_T5
        if self.available:
            try:
                self.paraphraser = pipeline(
                    "text2text-generation",
                    model="t5-small",
                    device=0 if __import__('torch').cuda.is_available() else -1
                )
            except:
                self.available = False
    
    def augment(self, text: str, num_paraphrases: int = 2) -> List[str]:
        """Generate paraphrases using T5."""
        if not self.available:
            return [text]
        
        try:
            augmented = [text]
            prompt = f"paraphrase: {text} </s>"
            outputs = self.paraphraser(
                prompt,
                max_length=len(text.split()) + 10,
                num_beams=4,
                num_return_sequences=num_paraphrases,
                temperature=1.5
            )
            augmented.extend([out['generated_text'] for out in outputs])
            return augmented
        except:
            return [text]


class SmartAugmentationPipeline:
    """Complete data augmentation pipeline for training data.
    
    Applies multiple augmentation strategies with label preservation.
    """
    
    def __init__(self, augmentation_strategy='balanced'):
        """
        Args:
            augmentation_strategy: 'balanced' (mix all), 'eda' (lightweight), 'heavy' (all methods)
        """
        self.strategy = augmentation_strategy
        self.eda = EDAugmenter()
        self.back_translator = BackTranslationAugmenter()
        self.paraphraser = ParaphraseAugmenter()
    
    def augment_dataset(self, texts: List[str], labels: List, factor: int = 2) -> Tuple[List[str], List]:
        """Augment entire dataset.
        
        Args:
            texts: list of input texts
            labels: corresponding labels (preserved through augmentation)
            factor: augmentation factor (2 = double dataset size)
        
        Returns:
            (augmented_texts, augmented_labels)
        """
        augmented_texts = texts.copy()
        augmented_labels = labels.copy() if isinstance(labels, list) else list(labels)
        
        for text, label in zip(texts, labels):
            if self.strategy == 'eda':
                augmented = self.eda.augment(text, num_aug=factor)
            elif self.strategy == 'balanced':
                # Mix EDA + back-translation
                eda_aug = self.eda.augment(text, num_aug=factor // 2)
                bt_aug = self.back_translator.augment(text, intermediate_lang='de')
                augmented = eda_aug + bt_aug[:factor // 2]
            else:  # heavy
                eda_aug = self.eda.augment(text, num_aug=factor)
                bt_aug = self.back_translator.augment(text)
                para_aug = self.paraphraser.augment(text, num_paraphrases=factor)
                augmented = eda_aug + bt_aug + para_aug
            
            # Add augmentations (skip original which is first)
            for aug_text in augmented[1:]:
                augmented_texts.append(aug_text)
                augmented_labels.append(label)
        
        return augmented_texts, augmented_labels
    
    @staticmethod
    def augment_rare_labels(texts: List[str], labels: List, rare_threshold: int = 100) -> Tuple[List[str], List]:
        """Oversample rare labels via augmentation.
        
        Args:
            texts: input texts
            labels: labels (supports multi-label as list of lists)
            rare_threshold: labels with < this count get augmented
        
        Returns:
            (augmented_texts, augmented_labels)
        """
        pipeline = SmartAugmentationPipeline(augmentation_strategy='balanced')
        
        # Count label frequencies
        label_counts = {}
        for label in labels:
            if isinstance(label, list):
                for l in label:
                    label_counts[l] = label_counts.get(l, 0) + 1
            else:
                label_counts[label] = label_counts.get(label, 0) + 1
        
        # Find rare labels
        rare_labels = {l for l, count in label_counts.items() if count < rare_threshold}
        
        # Augment rare label samples
        augmented_texts = texts.copy()
        augmented_labels = labels.copy() if isinstance(labels, list) else list(labels)
        
        for text, label in zip(texts, labels):
            # Check if sample contains rare label
            label_set = set(label) if isinstance(label, list) else {label}
            if label_set & rare_labels:
                eda = EDAugmenter()
                for aug_text in eda.augment(text, num_aug=3)[1:]:
                    augmented_texts.append(aug_text)
                    augmented_labels.append(label)
        
        return augmented_texts, augmented_labels
