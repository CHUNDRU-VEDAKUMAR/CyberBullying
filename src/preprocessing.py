# src/preprocessing.py
import re

def clean_text(text):
    """
    Minimal text cleaning for Jigsaw Dataset.
    - BERT models work best with original text (grammar, punctuation, casing)
    - Only remove URLs, IPs, and usernames to protect privacy
    """
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove IP addresses
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '', text)
    
    # Remove Usernames
    text = re.sub(r'\@\w+', '', text)
    
    # Remove Newlines and excessive whitespace
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def clean_text_aggressive(text):
    """
    Aggressive preprocessing for Baseline models (TF-IDF + SVM/RF).
    Removes all punctuation for TF-IDF vectorizer.
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '', text)
    text = re.sub(r'\@\w+', '', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()