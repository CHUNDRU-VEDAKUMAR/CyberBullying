from src.negation_handler import NegationHandler


def test_negation_simple():
    nh = NegationHandler()
    preds = {'toxic': 0.9, 'insult': 0.8, 'obscene': 0.7, 'severe_toxic': 0.6, 'threat': 0.2}
    text = "you are not a bitch"
    adjusted, ctx = nh.adjust_predictions(preds, text)
    # Expect insult/toxic/obscene to be greatly reduced
    assert adjusted['insult'] < 0.1
    assert adjusted['toxic'] < 0.1
    assert adjusted['obscene'] < 0.1
    print('test_negation_simple OK')


def test_negation_contraction():
    nh = NegationHandler()
    preds = {'toxic': 0.9, 'insult': 0.8, 'obscene': 0.7, 'severe_toxic': 0.6, 'threat': 0.2}
    text = "you weren't a bitch"
    adjusted, ctx = nh.adjust_predictions(preds, text)
    assert adjusted['insult'] < 0.1
    assert adjusted['toxic'] < 0.1
    print('test_negation_contraction OK')


if __name__ == '__main__':
    test_negation_simple()
    test_negation_contraction()
    print('All negation tests passed')
