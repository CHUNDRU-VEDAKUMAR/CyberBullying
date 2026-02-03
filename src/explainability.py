import numpy as np

try:
    from lime.lime_text import LimeTextExplainer
    _HAS_LIME = True
except Exception:
    _HAS_LIME = False

def _simple_perturbation_explain(text, predict_proba_fn, labels, num_features=5):
    """Lightweight fallback explainer: leave-one-out perturbation per word.

    Computes per-label impact by removing each token and measuring change
    in the predicted probability for that label.
    """
    tokens = text.split()
    if not tokens:
        return {label: [] for label in labels}

    # base probabilities for the full text
    base = np.array(predict_proba_fn([text]))
    if base.ndim == 1:
        base = base[np.newaxis, :]

    results = {label: [] for label in labels}

    for i, token in enumerate(tokens):
        perturbed = " ".join(tokens[:i] + tokens[i+1:]) or ""
        probs = np.array(predict_proba_fn([perturbed]))
        if probs.ndim == 1:
            probs = probs[np.newaxis, :]

        # impact per label = base_prob - perturbed_prob
        impact = (base - probs).squeeze()
        for j, label in enumerate(labels):
            results[label].append((token, float(impact[j])))

    # keep top `num_features` tokens per label by positive impact
    for label in labels:
        lst = results[label]
        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        # merge duplicates (same token may appear multiple times) by summing impact
        merged = {}
        for tok, val in lst:
            merged[tok] = merged.get(tok, 0.0) + val
        final = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:num_features]
        results[label] = final

    return results


def explain_multilabel(text, predict_proba_fn, labels, num_features=5, use_lime=False):
    """Run per-label explanations using LIME when available, otherwise a perturbation fallback.

    Returns dict: {label: [(word, weight), ...]}
    """
    # By default prefer the lightweight perturbation explainer unless use_lime=True
    if use_lime and _HAS_LIME:
        try:
            explainer = LimeTextExplainer(class_names=labels)

            def prob_fn(texts):
                probs = predict_proba_fn(texts)
                return np.array(probs)

            explanation = explainer.explain_instance(text, prob_fn, num_features=num_features, labels=tuple(range(len(labels))))
            results = {}
            for i, label in enumerate(labels):
                try:
                    results[label] = explanation.as_list(label=i)
                except Exception:
                    results[label] = []
            # Also provide a detailed normalized form for downstream consumers
            detailed = {}
            for i, label in enumerate(labels):
                try:
                    raw = explanation.as_list(label=i)
                    total = sum(abs(w) for _, w in raw) + 1e-8
                    detailed[label] = [{'token': w, 'impact': float(v), 'score_norm': float(v) / total} for w, v in raw]
                except Exception:
                    detailed[label] = []
            results['__detailed__'] = detailed
            return results
        except Exception:
            # fall through to simple perturbation
            pass

    # LIME not available or failed: use cheap perturbation explainer
    simple = _simple_perturbation_explain(text, predict_proba_fn, labels, num_features=num_features)
    # produce detailed normalized view as well
    detailed = {}
    for label, items in simple.items():
        total = sum(abs(v) for _, v in items) + 1e-8
        detailed[label] = [{'token': w, 'impact': float(v), 'score_norm': float(v) / total} for w, v in items]
    simple['__detailed__'] = detailed
    return simple
