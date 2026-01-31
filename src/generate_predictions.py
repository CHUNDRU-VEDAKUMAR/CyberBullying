# src/generate_predictions.py
import pandas as pd
from src.main_system import CyberbullyingSystem

def generate_test_predictions(test_csv_path, output_path='data/predictions.csv'):
    """
    Generate predictions for test.csv and save as CSV.
    Useful for Kaggle submissions or further analysis.
    """
    print(f"\n[PREDICTIONS] Loading test data from {test_csv_path}...")
    try:
        test_df = pd.read_csv(test_csv_path)
    except FileNotFoundError:
        print(f"Error: {test_csv_path} not found.")
        return None
    
    print(f"[PREDICTIONS] Initializing system...")
    system = CyberbullyingSystem()
    
    predictions = []
    
    for idx, row in test_df.iterrows():
        text = row['comment_text']
        
        # Analyze each comment
        result = system.analyze(text)
        
        predictions.append({
            'id': row.get('id', idx),
            'is_bullying': 1 if result['is_bullying'] else 0,
            'detected_types': '|'.join(result['detected_types']),
            'severity': result['severity'],
            'action': result['action'],
            'max_score': max(result['scores'].values()) if result['scores'] else 0.0
        })
        
        if (idx + 1) % 100 == 0:
            print(f"   [Progress] {idx + 1} / {len(test_df)} processed...")
    
    # Save predictions
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(output_path, index=False)
    print(f"\n[PREDICTIONS] ✅ Saved to {output_path}")
    
    # Summary
    bullying_count = pred_df['is_bullying'].sum()
    print(f"\n[SUMMARY]")
    print(f"  Total comments: {len(pred_df)}")
    print(f"  Bullying detected: {bullying_count} ({100*bullying_count/len(pred_df):.1f}%)")
    print(f"  Safe comments: {len(pred_df) - bullying_count}")
    
    return pred_df

if __name__ == "__main__":
    generate_test_predictions('data/test.csv')
