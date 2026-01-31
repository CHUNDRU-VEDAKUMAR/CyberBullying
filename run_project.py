# run_project.py
import sys
import os

# Ensure src is visible
sys.path.append(os.path.abspath("."))

from src.baseline_model import run_baseline_tournament
from src.main_system import CyberbullyingSystem

def main():
    print("=============================================================")
    print(" PROJECT: Context-Aware & Explainable Cyberbullying Detection")
    print("=============================================================")

    # STAGE 1: Baseline Comparison (Proof of Concept)
    # NOTE: Ensure 'data/train.csv' exists, or comment this out if you just want the final system
    data_path = 'data/train.csv'
    if os.path.exists(data_path):
        print("\n--- STAGE 1: Baseline Model Selection ---")
        run_baseline_tournament(data_path)
    else:
        print("\n[INFO] 'data/train.csv' not found. Skipping Baseline Tournament.")

    # STAGE 2: The Advanced System (BERT + Ontology)
    print("\n--- STAGE 2: Initializing Advanced System ---")
    system = CyberbullyingSystem()

    # Interactive Loop
    while True:
        print("\n" + "="*60)
        user_input = input("Enter a comment to test (or type 'exit'): ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting system. Stay safe!")
            break
            
        # Run the full analysis (BERT + Ontology + LIME)
        result = system.analyze(user_input)
        
        # --- FINAL REPORT DISPLAY ---
        print("\n--- 🛡️  CYBERBULLYING DETECTION REPORT 🛡️  ---")
        print(f"📝 Input Text:    {result['text']}")
        print(f"🔍 Verdict:       {'🛑 BULLYING DETECTED' if result['is_bullying'] else '✅ SAFE'}")
        
        if result['is_bullying']:
            print(f"📊 Types Found:   {', '.join(result['detected_types'])}")
            print(f"🔥 Severity:      {result['severity']}")
            print(f"💡 Explanation:   {result['explanation']}")
            
            # --- THE NEW "ADVANCED" LIME SECTION ---
            print(f"👁️  Visual Proof:  The model flagged these specific words:")
            # LIME returns a list of tuples like [('idiot', 0.85), ('the', -0.01)]
            # We only want to show positive weights (words that increased toxicity)
            found_triggers = False
            for word, weight in result['highlighted_words']:
                if weight > 0: 
                    found_triggers = True
                    # weight represents "Impact". 0.85 means it contributed 85% to the decision
                    print(f"      👉 '{word}' (Impact Score: {weight:.2f})")
            
            if not found_triggers:
                print("      (Complex context detected; no single trigger word dominated.)")

            print(f"🛡️  Action:       {result['action']}")
        else:
            print(f"💡 Explanation:   Clean content. No intervention needed.")
            
        print("="*60)

if __name__ == "__main__":
    main()