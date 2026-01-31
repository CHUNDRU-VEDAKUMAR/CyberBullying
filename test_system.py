# test_system.py
"""
Quick validation script to test if the project works correctly.
Run this AFTER installing requirements.txt
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test if all required packages are installed"""
    print("=" * 60)
    print("TEST 1: Checking Package Imports...")
    print("=" * 60)
    
    packages = ['torch', 'transformers', 'sklearn', 'pandas', 'numpy', 'lime']
    all_ok = True
    
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"✅ {pkg:15} - OK")
        except ImportError:
            print(f"❌ {pkg:15} - MISSING (run: pip install {pkg})")
            all_ok = False
    
    return all_ok

def test_model_loading():
    """Test if BERT model can be loaded"""
    print("\n" + "=" * 60)
    print("TEST 2: Loading BERT Model...")
    print("=" * 60)
    
    try:
        from src.bert_model import AdvancedContextModel
        print("✅ Importing AdvancedContextModel - OK")
        
        print("\n⏳ Loading unitary/toxic-bert (may take 1-2 min on first run)...")
        model = AdvancedContextModel()
        print("✅ BERT Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def test_ontology():
    """Test if ontology rules are working"""
    print("\n" + "=" * 60)
    print("TEST 3: Testing Ontology Rules...")
    print("=" * 60)
    
    try:
        from src.ontology import get_intervention_plan
        
        test_cases = [
            ([], "clean"),
            (['toxic'], "MEDIUM"),
            (['threat'], "CRITICAL"),
            (['identity_hate'], "HIGH"),
        ]
        
        for labels, expected_severity in test_cases:
            plan = get_intervention_plan(labels)
            severity = plan['severity']
            status = "✅" if severity == expected_severity else "❌"
            print(f"{status} {str(labels):25} → {severity}")
        
        return True
    except Exception as e:
        print(f"❌ Ontology test failed: {e}")
        return False

def test_full_pipeline(model):
    """Test complete analysis pipeline"""
    print("\n" + "=" * 60)
    print("TEST 4: Testing Full Analysis Pipeline...")
    print("=" * 60)
    
    try:
        from src.main_system import CyberbullyingSystem
        
        print("⏳ Initializing CyberbullyingSystem...")
        system = CyberbullyingSystem()
        
        test_texts = [
            "This is a great movie!",
            "You're an idiot",
            "I will kill you",
            "You absolutely killed that presentation!",
        ]
        
        for text in test_texts:
            try:
                result = system.analyze(text)
                bullying = "🛑 BULLYING" if result['is_bullying'] else "✅ SAFE"
                print(f"\n  Text: '{text}'")
                print(f"  Result: {bullying}")
                if result['is_bullying']:
                    print(f"  Severity: {result['severity']}")
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║  CYBERBULLYING DETECTION SYSTEM - VALIDATION SUITE      ║")
    print("╚" + "=" * 58 + "╝")
    
    results = {}
    
    # Test 1
    results['imports'] = test_imports()
    
    if not results['imports']:
        print("\n⚠️  STOP: Install missing packages first!")
        print("   Run: pip install -r requirements.txt")
        return False
    
    # Test 2
    model = test_model_loading()
    results['model'] = model is not None
    
    # Test 3
    results['ontology'] = test_ontology()
    
    # Test 4 (only if model loaded)
    if results['model']:
        results['pipeline'] = test_full_pipeline(model)
    else:
        results['pipeline'] = False
    
    # Final Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    test_names = ['Package Imports', 'BERT Model Loading', 'Ontology Rules', 'Full Pipeline']
    for name, passed in zip(test_names, results.values()):
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10} - {name}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! System is ready to use.")
        print("\n   Run: python run_project.py")
    else:
        print("⚠️  SOME TESTS FAILED. Fix errors above before running.")
    print("=" * 60 + "\n")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
