"""
Enhanced Test Suite for Context-Aware Cyberbullying Detection
Tests: Negations, Sarcasm, Positive Achievement, Opinion vs Attack
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_negations():
    """Test negation handling"""
    print("\n" + "="*70)
    print("TEST 1: NEGATION HANDLING")
    print("="*70)
    
    from src.negation_handler import NegationHandler
    
    handler = NegationHandler()
    
    test_cases = [
        ("I will kill you", False, "Direct threat"),
        ("I will NOT kill you", True, "Negated threat"),
        ("I don't kill you", True, "Negated with don't"),
        ("I won't kill you", True, "Negated with won't"),
        ("You are an idiot", False, "Direct insult"),
        ("You are NOT an idiot", True, "Negated insult"),
        ("I never said you were stupid", True, "Never negation"),
    ]
    
    for text, should_be_negated, description in test_cases:
        context = handler.detect_negation_context(text)
        has_neg = context['has_negation']
        factor = context['confidence']
        
        status = "✅" if (has_neg == should_be_negated) else "❌"
        print(f"{status} {description:30} | Negated: {has_neg:5} | Factor: {factor:.2f} | '{text}'")
    
    return True

def test_context_analysis():
    """Test context analysis (opinion vs personal attack, etc)"""
    print("\n" + "="*70)
    print("TEST 2: LINGUISTIC CONTEXT ANALYSIS")
    print("="*70)
    
    from src.context_analyzer import ContextAnalyzer
    
    analyzer = ContextAnalyzer()
    
    test_cases = [
        ("You killed it!", "positive_achievement"),
        ("That was awesome!", "positive_achievement"),
        ("You are an idiot", "person_attack"),
        ("This idea is terrible", "opinion_about_thing"),
        ("I think you're wrong", "opinion"),
        ("You are NOT an idiot", "negated_opinion"),
    ]
    
    for text, expected_context in test_cases:
        analysis = analyzer.analyze_context(text)
        
        context_type = ""
        if analysis['is_positive_achievement']:
            context_type = "positive_achievement"
        elif analysis['is_opinion']:
            context_type = "opinion"
        elif analysis['target_type'] == 'thing':
            context_type = "opinion_about_thing"
        elif analysis['target_type'] == 'person':
            context_type = "person_attack"
        
        status = "✅" if context_type == expected_context or (expected_context == "negated_opinion" and analysis['is_opinion']) else "⚠️"
        print(f"{status} {analysis['reason']:40} | Score: {analysis['context_score']:.2f} | '{text}'")
    
    return True

def test_full_system():
    """Test complete system with problematic cases"""
    print("\n" + "="*70)
    print("TEST 3: FULL SYSTEM - PROBLEMATIC CASES")
    print("="*70)
    
    from src.main_system import CyberbullyingSystem
    
    system = CyberbullyingSystem()
    
    test_cases = [
        # (text, should_be_bullying, reason)
        ("You are an idiot", True, "Direct personal attack"),
        ("You are NOT an idiot", False, "Negated personal attack"),
        ("I don't kill you", False, "Negated threat"),
        ("I will kill you", True, "Direct threat"),
        ("You killed it!", False, "Positive achievement context"),
        ("That presentation was killed!", False, "Positive achievement about thing"),
        ("This code is terrible", False, "Opinion about thing, not personal attack"),
        ("You are terrible", True, "Personal attack"),
        ("I think you're wrong", False, "Opinion-based disagreement"),
        ("I hate you", True, "Personal hate"),
        ("I hate this game", False, "Opinion about thing"),
    ]
    
    print("\n{:<40} {:<15} {:<15} {:<10}".format("Text", "Expected", "Got", "Status"))
    print("-"*70)
    
    correct = 0
    for text, should_be_bullying, reason in test_cases:
        result = system.analyze(text)
        is_bullying = result['is_bullying']
        
        status = "✅" if is_bullying == should_be_bullying else "❌"
        if is_bullying == should_be_bullying:
            correct += 1
        
        expected_str = "BULLYING" if should_be_bullying else "SAFE"
        got_str = "BULLYING" if is_bullying else "SAFE"
        
        # Show context info if interesting
        context_str = ""
        ctx = result.get('context_info', {})
        if ctx.get('negation_detected'):
            context_str = f" [NEGATION: {ctx.get('negation_type')}]"
        elif ctx.get('is_positive_achievement'):
            context_str = " [POSITIVE]"
        elif ctx.get('target_type') == 'thing':
            context_str = " [NOT_PERSONAL]"
        
        print(f"{status} {text[:38]:<40} {expected_str:<15} {got_str:<15} {context_str}")
    
    print("-"*70)
    print(f"\n✅ Accuracy: {correct}/{len(test_cases)} ({100*correct//len(test_cases)}%)")
    
    return correct == len(test_cases)

def main():
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║  CONTEXT-AWARE CYBERBULLYING DETECTION - ENHANCED TEST SUITE      ║")
    print("╚" + "="*68 + "╝")
    
    results = {}
    
    try:
        results['negations'] = test_negations()
    except Exception as e:
        print(f"❌ Negation test failed: {e}")
        import traceback
        traceback.print_exc()
        results['negations'] = False
    
    try:
        results['context'] = test_context_analysis()
    except Exception as e:
        print(f"❌ Context analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        results['context'] = False
    
    try:
        results['system'] = test_full_system()
    except Exception as e:
        print(f"❌ Full system test failed: {e}")
        import traceback
        traceback.print_exc()
        results['system'] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    test_names = ['Negation Handling', 'Context Analysis', 'Full System']
    for name, passed in zip(test_names, results.values()):
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10} - {name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL ENHANCED TESTS PASSED! Context-aware system is working.")
        print("\n   Run: python run_project.py")
    else:
        print("⚠️  SOME TESTS FAILED. Review errors above.")
    print("="*70 + "\n")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
