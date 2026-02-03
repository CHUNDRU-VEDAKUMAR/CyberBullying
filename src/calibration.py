from sklearn.isotonic import IsotonicRegression
import numpy as np

class PerLabelIsotonicCalibrator:
    """Per-label isotonic regression calibrator for multi-label probabilities.

    Usage:
        cal = PerLabelIsotonicCalibrator()
        cal.fit(probs, y_true)
        probs_cal = cal.transform(probs)

    `probs` should be shape (n_samples, n_labels), y_true is binary matrix same shape.
    """
    def __init__(self):
        self.models = None

    def fit(self, probs, y_true):
        probs = np.asarray(probs)
        y_true = np.asarray(y_true)
        n_labels = probs.shape[1]
        self.models = []
        for j in range(n_labels):
            p = probs[:, j]
            y = y_true[:, j]
            # fit isotonic only if there is variability
            try:
                ir = IsotonicRegression(out_of_bounds='clip')
                ir.fit(p, y)
                self.models.append(ir)
            except Exception:
                self.models.append(None)

    def transform(self, probs):
        probs = np.asarray(probs)
        if self.models is None:
            return probs
        n_labels = probs.shape[1]
        out = np.zeros_like(probs)
        for j in range(n_labels):
            model = self.models[j]
            p = probs[:, j]
            if model is None:
                out[:, j] = p
            else:
                out[:, j] = model.transform(p)
        return out

    def fit_transform(self, probs, y_true):
        self.fit(probs, y_true)
        return self.transform(probs)
