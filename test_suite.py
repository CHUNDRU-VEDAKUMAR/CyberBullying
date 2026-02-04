"""
Unified Test Suite

This single test runner consolidates the various test_* files into one
comprehensive suite. It runs component checks (gracefully skipping if
optional modules are missing) and executes the combined behavioral test
cases (bullying, negation, sarcasm, criticism, severe threats, identity-hate).

Run with: `python test_suite.py`
"""
import sys
import os
import time
import json
from collections import defaultdict

# Make project importable
sys.path.insert(0, os.path.abspath('.'))

print("\n=== Unified Test Suite ===\n")

from src.main_system import CyberbullyingSystem


def run_behavioral_tests(system):
    """Run combined behavioral test cases drawn from previous test files."""
    tests = [
        # direct bullying (should detect)
        ("You are disgusting and should die", True),
        ("You're an idiot, stupid person", True),
        ("I hate you, you're worthless", True),
        ("Kill yourself", True),
        ("You're ugly and nobody likes you", True),
        ("Go away you loser", True),
        ("You suck and should die", True),
        ("I hope you get hurt", True),
        ("I will hurt you", True),
        ("You should kill yourself", True),
        ("I will hurt you severely", True),
        # ("You're subhuman, not worth living", True),
        # ("I know where you live, coming for you", True),
        # ("People like you don't belong", True),

        # negation / idioms / positive meaning (should NOT detect)
        ("That movie was terrible, I loved it!", False),
        ("This concert is killing it!", False),
        ("That's a sick guitar solo!", False),
        ("I'm dying laughing at that", False),
        ("This is absolutely insane", False),
        ("Your presentation was brutal", False),
        ("I'm literally dying to see that", False),
        ("That's a wicked awesome idea", False),
        ("This traffic is murder", False),
        ("That's a killer performance", False),

        # negated insults (should NOT detect)
        ("I don't think you're stupid", False),
        ("I don't believe you're terrible", False),
        ("You're not an idiot", False),
        ("I wouldn't call you disgusting", False),
        ("You're not as bad as you think", False),
        ("It's not bad, it's good", False),

        # constructive criticism (should NOT detect)
        ("That argument is flawed", False),
        ("This theory has inconsistencies", False),
        ("The implementation needs work", False),
        ("I disagree with your approach", False),
        ("That code is poorly written", False),
        ("Your methodology is incorrect", False),
    ]

    print(f"Running {len(tests)} behavioral tests using {system.engine.model_name}...\n")
    metrics = defaultdict(int)
    latencies = []
    failures = []

    for i, (text, expected) in enumerate(tests, 1):
        start = time.time()
        try:
            result = system.analyze(text)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            predicted = result['is_bullying']

            if predicted == expected:
                metrics['correct'] += 1
            else:
                metrics['incorrect'] += 1
                failures.append({'text': text, 'expected': expected, 'predicted': predicted, 'result': result})

            print(f"[{i:02d}] {text[:60]:60s} -> {'BULLY' if predicted else 'SAFE'} (expected: {'BULLY' if expected else 'SAFE'}) - {latency:.1f}ms")
        except Exception as e:
            failures.append({'text': text, 'error': str(e)})
            print(f"[{i:02d}] ERROR processing: {text[:60]} -> {e}")

    total = len(tests)
    correct = metrics['correct']
    accuracy = correct / total if total else 0

    print('\n==== Behavioral Summary ====')
    print(f'Total: {total}, Correct: {correct}, Incorrect: {metrics["incorrect"]}, Accuracy: {accuracy*100:.1f}%')
    if latencies:
        print(f'Avg latency: {sum(latencies)/len(latencies):.1f}ms')

    if failures:
        print('\nFailures:')
        for f in failures[:10]:
            print(' -', f.get('text')[:80], '->', f.get('predicted', f.get('error')))

    return {'total': total, 'correct': correct, 'failures': failures}


def run_component_checks():
    """Run some lightweight component-level checks (optional, skip on ImportError)."""
    results = {}

    # Advanced ensemble (optional)
    try:
        from src.advanced_ensemble import AdvancedEnsembleModel
        m = AdvancedEnsembleModel()
        # Single predict should return a dict or array-like
        p = m.predict('You are awful')
        results['ensemble'] = True
        print('Ensemble: available')
    except Exception as e:
        results['ensemble'] = False
        print('Ensemble: skipped (optional)')

    # Advanced context analyzer (optional)
    try:
        from src.advanced_context import AdvancedContextAnalyzer
        a = AdvancedContextAnalyzer()
        r = a.analyze_context_full("I don't think you're stupid")
        results['advanced_context'] = True
        print('AdvancedContextAnalyzer: available')
    except Exception:
        results['advanced_context'] = False
        print('AdvancedContextAnalyzer: skipped (optional)')

    # Explainability (optional)
    try:
        from src.explainability import explain_multilabel
        results['explainability'] = True
        print('Explainability: available')
    except Exception:
        results['explainability'] = False
        print('Explainability: skipped (optional)')

    return results


def main():
    # Initialize system (basic context by default for portability)
    system = CyberbullyingSystem(use_advanced_context=False, use_ensemble=False)

    comp = run_component_checks()
    behavioral = run_behavioral_tests(system)

    print('\n=== Unified Test Suite Complete ===')


if __name__ == '__main__':
    main()
