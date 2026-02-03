from src.explainability import explain_multilabel


def mock_predict_proba(texts):
    # simple deterministic predictor for testing
    out = []
    for t in texts:
        t = t.lower()
        if 'idiot' in t:
            out.append([0.9, 0.1])
        elif 'great' in t:
            out.append([0.1, 0.9])
        else:
            out.append([0.4, 0.6])
    return out


def test_explainability_basic():
    labels = ['toxic', 'praise']
    text = "You are an idiot"
    res = explain_multilabel(text, mock_predict_proba, labels, num_features=3)
    assert isinstance(res, dict)
    assert 'toxic' in res and 'praise' in res
    # toxic should show 'idiot' as important
    toks = [w for w, _ in res['toxic']]
    assert any('idiot' in t for t in toks)


if __name__ == '__main__':
    print('Running explainability test...')
    test_explainability_basic()
    print('Explainability test passed.')
