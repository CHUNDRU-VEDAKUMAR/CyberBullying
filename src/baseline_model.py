# src/baseline_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from src.preprocessing import clean_text_aggressive

def run_baseline_tournament(data_path):
    print("\n[BASELINE] Loading Dataset for Tournament...")
    try:
        # Load only a sample for speed if dataset is huge, or full dataset
        df = pd.read_csv(data_path).sample(20000, random_state=42) 
    except FileNotFoundError:
        print("Error: File not found. Check path.")
        return None

    # Create Binary Label (Bullying vs Not)
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df['label'] = df[label_cols].max(axis=1)

    print("[BASELINE] Preprocessing & Vectorizing...")
    df['clean_text'] = df['comment_text'].apply(clean_text_aggressive)
    
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['clean_text'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Linear SVC": LinearSVC(dual=False)
    }

    best_model = None
    best_score = 0

    print("[BASELINE] Training Models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = f1_score(y_test, pred)
        print(f"   > {name} F1 Score: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model = model

    print(f"[BASELINE] Winner: {best_model} (F1: {best_score:.4f})")
    return best_model