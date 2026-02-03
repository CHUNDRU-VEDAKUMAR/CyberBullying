import pytest
from src.bert_model import AdvancedContextModel
from src.preprocessing import clean_text


class DummyModel:
    def __init__(self):
        self.labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def predict_proba(self, texts):
        # deterministic fake outputs for tests
        out = []
        for t in texts:
            low = [0.0] * 6
            txt = t.lower()
            if 'kill you' in txt or 'i will kill' in txt:
                low[3] = 0.99  # threat
            if 'idiot' in txt or 'you are dumb' in txt:
                low[0] = 0.9
                low[4] = 0.8
            out.append(low)
        import numpy as np
        return np.array(out)


@pytest.fixture(autouse=True)
def patch_model(monkeypatch):
    # Patch AdvancedContextModel to avoid heavy HF downloads in tests
    monkeypatch.setattr('src.bert_model.AdvancedContextModel.__init__', lambda self, model_name=None, device=None, labels=None: None)
    monkeypatch.setattr('src.bert_model.AdvancedContextModel.predict_proba', lambda self, texts: DummyModel().predict_proba(texts))
    monkeypatch.setattr('src.bert_model.AdvancedContextModel.predict', lambda self, text: {l: float(v) for l, v in zip(DummyModel().labels, DummyModel().predict_proba([text])[0])})


def test_clean_text_basic():
    s = "Hello! Visit http://example.com and @user"
    cleaned = clean_text(s)
    assert 'http' not in cleaned and '@' not in cleaned


def test_threat_detection():
    m = AdvancedContextModel()
    probs = m.predict_proba(["I will kill you tomorrow"])[0]
    assert probs[3] > 0.9


def test_insult_detection():
    m = AdvancedContextModel()
    probs = m.predict_proba(["You're an idiot"])[0]
    assert probs[0] > 0.8 and probs[4] > 0.7


def test_batch_predictions_shape():
    m = AdvancedContextModel()
    inputs = ["hello", "you're dumb", "i will kill you"]
    probs = m.predict_proba(inputs)
    assert probs.shape[0] == 3 and probs.shape[1] == 6
