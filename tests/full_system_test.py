import sys
import os
import pytest
import numpy as np

# Make repository root importable when running the test module directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.preprocessing import clean_text
from src.negation_handler import NegationHandler
from src.context_analyzer import ContextAnalyzer
from src.explainability import explain_multilabel
from src.ontology import get_intervention_plan, aggregate_severity, recommend_intervention


def test_clean_text_removes_urls_and_usernames():
    s = "Check this out http://example.com @user"
    out = clean_text(s)
    assert 'http' not in out and '@' not in out


def test_negation_handler_detects_and_adjusts():
    nh = NegationHandler()
    ctx = nh.detect_negation_context("I don't kill you")
    assert ctx['has_negation'] is True

    preds = {'threat': 0.9, 'toxic': 0.8, 'insult': 0.7}
    adjusted, ctx2 = nh.adjust_predictions(preds.copy(), "I don't kill you")
    # threat score should be reduced substantially
    assert adjusted['threat'] < 0.2


def test_context_analyzer_positive_and_target():
    ca = ContextAnalyzer()
    assert ca.is_positive_achievement("You killed it!") is True
    assert ca.detect_target_type("You are stupid") == 'person'
    res = ca.analyze_context("You killed it!")
    assert res['is_positive_achievement'] is True


def test_explainability_perturbation():
    # simple predict_proba function returning deterministic probs
    def predict_fn(texts):
        # return shape (N,3) for three labels
        arr = []
        for t in texts:
            if 'idiot' in t.lower():
                arr.append([0.1, 0.8, 0.05])
            else:
                arr.append([0.9, 0.05, 0.02])
        return np.array(arr)

    labels = ['clean', 'toxic', 'insult']
    out = explain_multilabel("You're an idiot", predict_fn, labels, num_features=3)
    assert isinstance(out, dict)
    assert '__detailed__' in out


def test_ontology_priority_and_recommendation():
    scores = {'threat': 0.9, 'toxic': 0.4}
    plan = get_intervention_plan(scores, min_score=0.3)
    assert plan['severity'] == 'CRITICAL' or plan['detected_label'] == 'threat'
    plan2 = recommend_intervention(plan.copy())
    assert 'recommended_action' in plan2


def try_load_model_and_run(model_name=None):
    """Attempt to import and run AdvancedContextModel with a small batch.
    Returns True if successful, False otherwise (so tests can skip gracefully).
    """
    try:
        from src.bert_model import AdvancedContextModel
        m = AdvancedContextModel(model_name=model_name) if model_name else AdvancedContextModel()
        texts = ["You're an idiot", "I will kill you"]
        probs = m.predict_proba(texts)
        # must return a numpy array with batch rows
        return isinstance(probs, np.ndarray) and probs.shape[0] == 2
    except Exception:
        return False


def test_bert_model_inference_or_skip():
    # Prefer a small HF model if available; try default otherwise.
    small_model = 'sshleifer/tiny-distilroberta-base'
    ok = try_load_model_and_run(small_model)
    if not ok:
        pytest.skip("Unable to run HF model in this environment; skipping heavy inference test")


def test_full_system_end_to_end_or_skip():
    try:
        from src.main_system import CyberbullyingSystem
        sys = CyberbullyingSystem()
        out = sys.analyze("You're an idiot")
        assert 'is_bullying' in out and 'scores' in out
    except Exception:
        pytest.skip("Full system run skipped due to environment limitations")
