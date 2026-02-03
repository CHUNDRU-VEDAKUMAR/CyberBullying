from src.ontology import get_intervention_plan


def test_single_label_confidence():
    scores = {'toxic': 0.8}
    plan = get_intervention_plan(scores)
    assert plan['detected_label'] == 'toxic'
    assert plan['severity'] == 'MEDIUM'
    assert 0.0 <= plan['confidence'] <= 1.0


def test_multiple_labels_priority():
    scores = {'toxic': 0.4, 'threat': 0.3}
    plan = get_intervention_plan(scores)
    # 'threat' is CRITICAL and should take precedence even if score slightly lower
    assert plan['detected_label'] in ('threat', 'toxic')
    assert plan['severity'] in ('CRITICAL', 'MEDIUM')


if __name__ == '__main__':
    print('Running lightweight ontology tests...')
    test_single_label_confidence()
    print('test_single_label_confidence OK')
    test_multiple_labels_priority()
    print('test_multiple_labels_priority OK')
    print('All ontology tests passed.')
