import numpy as np

try:
    from lime.lime_text import LimeTextExplainer
    _HAS_LIME = True
except Exception:
    _HAS_LIME = False

try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

try:
    from captum.attr import IntegratedGradients, LayerIntegratedGradients
    import torch
    _HAS_CAPTUM = True
except Exception:
    _HAS_CAPTUM = False

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


def explain_multilabel(text, predict_proba_fn, labels, num_features=5, use_lime=False, use_shap=False, use_captum=False, model=None):
    """Run per-label explanations using LIME / SHAP / Captum / perturbation fallback.

    Args:
        text: input text to explain
        predict_proba_fn: function that takes list of texts and returns (N, num_labels) probs
        labels: list of label names
        num_features: number of top features to return per label
        use_lime: prefer LIME if available
        use_shap: prefer SHAP if available
        use_captum: prefer Captum (requires model)
        model: optional HF model for Captum/SHAP integration

    Returns:
        dict: {label: [(word, weight), ...], '__detailed__': {...}}
    """
    
    # Try Captum first if use_captum=True and model available
    if use_captum and model is not None and _HAS_CAPTUM:
        try:
            result = _explain_with_captum(text, model, labels, num_features)
            if result:
                return result
        except Exception:
            pass

    # Try SHAP if use_shap=True
    if use_shap and _HAS_SHAP:
        try:
            result = _explain_with_shap(text, predict_proba_fn, labels, num_features)
            if result:
                return result
        except Exception:
            pass

    # Try LIME if use_lime=True
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

    # LIME/SHAP/Captum not available or failed: use cheap perturbation explainer
    simple = _simple_perturbation_explain(text, predict_proba_fn, labels, num_features=num_features)
    # produce detailed normalized view as well
    detailed = {}
    for label, items in simple.items():
        total = sum(abs(v) for _, v in items) + 1e-8
        detailed[label] = [{'token': w, 'impact': float(v), 'score_norm': float(v) / total} for w, v in items]
    simple['__detailed__'] = detailed
    return simple


def _explain_with_captum(text, model, labels, num_features=5):
    """Explain using Captum's Integrated Gradients (requires differentiable model and embeddings)."""
    if not _HAS_CAPTUM:
        return None

    try:
        from transformers import AutoTokenizer
        
        # Get model's tokenizer and embeddings
        if hasattr(model, 'tokenizer'):
            tokenizer = model.tokenizer
        else:
            # Assume model has a .get_tokenizer() or similar; fallback to generic
            return None
        
        # Tokenize
        tokens = tokenizer.tokenize(text)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # Setup attribution
        ig = IntegratedGradients(model.model)
        
        # For each label, compute attributions
        results = {}
        for label_idx, label in enumerate(labels):
            try:
                # Create input tensors
                input_ids_tensor = torch.tensor([input_ids]).to(model.device)
                
                # Define target function (specific label logit)
                def target_fn(embeddings):
                    outputs = model.model(inputs_embeds=embeddings, attention_mask=torch.ones_like(input_ids_tensor))
                    logits = outputs.logits[:, label_idx]
                    return logits
                
                # Compute attributions (simplified; full impl would require embedding integration)
                # For now, fallback since full integration is complex
                return None
            except Exception:
                return None
        
        return None
    except Exception:
        return None


def _explain_with_shap(text, predict_proba_fn, labels, num_features=5):
    """Explain using SHAP's GradientExplainer or KernelExplainer."""
    if not _HAS_SHAP:
        return None

    try:
        # Tokenize into words
        tokens = text.split()
        
        # Create a masking function: given a binary mask, return predictions
        def mask_predict_fn(mask):
            # mask is (n_samples, n_tokens); shape (N, len(tokens))
            masked_texts = []
            for row in mask:
                masked = ' '.join([t for t, m in zip(tokens, row) if m])
                masked_texts.append(masked)
            probs = predict_proba_fn(masked_texts)
            return np.array(probs)
        
        # Use SHAP KernelExplainer (model-agnostic)
        explainer = shap.KernelExplainer(mask_predict_fn, np.ones((1, len(tokens))))
        
        # Get SHAP values
        shap_values = explainer.shap_values(np.ones((1, len(tokens))))
        
        # Parse results per label
        results = {}
        if isinstance(shap_values, list):
            # multi-output
            for label_idx, label in enumerate(labels):
                sv = shap_values[label_idx][0]
                items = list(zip(tokens, sv))
                items = sorted(items, key=lambda x: abs(x[1]), reverse=True)[:num_features]
                results[label] = items
        else:
            # single output
            sv = shap_values[0]
            items = list(zip(tokens, sv))
            items = sorted(items, key=lambda x: abs(x[1]), reverse=True)[:num_features]
            results[labels[0]] = items
        
        # Normalize
        detailed = {}
        for label, items in results.items():
            total = sum(abs(v) for _, v in items) + 1e-8
            detailed[label] = [{'token': w, 'impact': float(v), 'score_norm': float(v) / total} for w, v in items]
        results['__detailed__'] = detailed
        return results
    except Exception:
        return None
