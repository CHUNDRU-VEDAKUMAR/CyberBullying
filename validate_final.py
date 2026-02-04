"""Final validation - Test key scenarios"""

from src.main_system import CyberbullyingSystem

system = CyberbullyingSystem(
    model_name='unitary/toxic-bert',
    use_advanced_context=False,
    use_ensemble=False
)

test_cases = [
    # Fixed cases (were failing, now working)
    ("I don't think you're stupid", False, "Negated insult - FIXED"),
    ("I don't believe you're terrible", False, "Negated insult - FIXED"),
    ("You're not an idiot", False, "Negated insult - FIXED"),
    ("I wouldn't call you disgusting", False, "Negated insult - FIXED"),
    
    # Sarcasm/idioms (should NOT trigger false positives)
    ("That movie was terrible, I loved it!", False, "Sarcasm - NO FALSE POS"),
    ("This concert is killing it!", False, "Slang - NO FALSE POS"),
    ("I'm dying laughing", False, "Idiom - NO FALSE POS"),
    
    # Actual bullying (should be detected)
    ("You're an idiot", True, "Direct insult - DETECTED"),
    ("I hate you", True, "Hate speech - DETECTED"),
    ("You're ugly", True, "Direct insult - DETECTED"),
]

print("FINAL VALIDATION TEST\n" + "="*70 + "\n")

passed = 0
failed = 0

for text, expected_bullying, description in test_cases:
    result = system.analyze(text)
    predicted_bullying = result['is_bullying']
    is_correct = (predicted_bullying == expected_bullying)
    
    status = "[PASS]" if is_correct else "[FAIL]"
    expected_str = "BULLYING" if expected_bullying else "SAFE"
    predicted_str = "BULLYING" if predicted_bullying else "SAFE"
    
    print(f"{status} {description:30s} | Expected: {expected_str:8s} Got: {predicted_str:8s}")
    
    if is_correct:
        passed += 1
    else:
        failed += 1

print("\n" + "="*70)
print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
print(f"Validation Score: {100*passed/len(test_cases):.1f}%")

if failed == 0:
    print("\n✓ ALL VALIDATION TESTS PASSED")
    print("✓ System is ready for deployment")
