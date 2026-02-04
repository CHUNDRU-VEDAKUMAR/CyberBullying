"""
Advanced Threshold Optimization and Calibration
Optimizes per-label decision thresholds and calibrates probabilities for reliable interventions.

Methods:
1. Per-label threshold optimization (F1, precision-recall optimization)
2. Temperature scaling (single parameter calibration)
3. Isotonic regression (non-parametric, flexible)
4. Expected Calibration Error (ECE) computation
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, precision_recall_curve, auc
from typing import Tuple, Dict


class TemperatureScaler:
    """Temperature scaling for probability calibration.
    
    Single parameter calibration: divides logits by temperature T before sigmoid.
    Simple but effective for neural networks.
    
    Reference: Guo et al. (2017) "On Calibration of Modern Neural Networks"
    """
    
    def __init__(self):
        self.temperature = 1.0
        self.is_fitted = False
    
    def fit(self, logits: np.ndarray, y_true: np.ndarray):
        """Fit temperature on validation set using NLL (negative log-likelihood).
        
        Args:
            logits: raw model outputs (before sigmoid), shape (N, num_labels)
            y_true: binary labels, shape (N, num_labels)
        """
        from scipy.optimize import minimize
        
        def nll(T):
            # Compute negative log likelihood at temperature T
            scaled_logits = logits / T
            probs = 1.0 / (1.0 + np.exp(-scaled_logits))  # sigmoid
            nll_val = -np.mean(y_true * np.log(probs + 1e-8) + (1 - y_true) * np.log(1 - probs + 1e-8))
            return nll_val
        
        result = minimize(nll, x0=1.0, bounds=[(0.1, 5.0)], method='L-BFGS-B')
        self.temperature = float(result.x)
        self.is_fitted = True
    
    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits."""
        if not self.is_fitted:
            return 1.0 / (1.0 + np.exp(-logits))  # Basic sigmoid
        
        scaled_logits = logits / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled_logits))


class PerLabelThresholdOptimizer:
    """Optimize detection threshold per label for maximum F1 or custom metric.
    
    Uses precision-recall curves to find optimal operating point.
    """
    
    def __init__(self, metric='f1'):
        """
        Args:
            metric: 'f1' (default), 'precision', 'recall', or custom callable
        """
        self.metric = metric
        self.thresholds = {}
        self.is_fitted = False
    
    def fit(self, probs: np.ndarray, y_true: np.ndarray, labels: list = None):
        """Find optimal thresholds for each label.
        
        Args:
            probs: predicted probabilities, shape (N, num_labels)
            y_true: binary labels, shape (N, num_labels)
            labels: label names (optional)
        """
        num_labels = probs.shape[1]
        if labels is None:
            labels = [f'label_{i}' for i in range(num_labels)]
        
        for label_idx, label_name in enumerate(labels):
            y_label = y_true[:, label_idx]
            probs_label = probs[:, label_idx]
            
            # Compute precision-recall curve
            precisions, recalls, threshs = precision_recall_curve(y_label, probs_label)
            
            # Find optimal threshold
            if self.metric == 'f1':
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
                best_idx = np.argmax(f1_scores)
                best_threshold = threshs[best_idx] if best_idx < len(threshs) else 0.5
            elif self.metric == 'precision':
                # Maximize precision while keeping recall > 0.8
                valid_idx = recalls >= 0.8
                if np.any(valid_idx):
                    best_idx = np.argmax(precisions[valid_idx])
                    best_threshold = threshs[best_idx] if best_idx < len(threshs) else 0.5
                else:
                    best_threshold = 0.5
            elif self.metric == 'recall':
                # Maximize recall while keeping precision > 0.7
                valid_idx = precisions >= 0.7
                if np.any(valid_idx):
                    best_idx = np.argmax(recalls[valid_idx])
                    best_threshold = threshs[best_idx] if best_idx < len(threshs) else 0.5
                else:
                    best_threshold = 0.5
            else:
                best_threshold = 0.5
            
            self.thresholds[label_name] = float(best_threshold)
        
        self.is_fitted = True
    
    def transform(self, probs: np.ndarray, labels: list = None) -> np.ndarray:
        """Apply optimal thresholds to probabilities."""
        if not self.is_fitted:
            return (probs > 0.5).astype(int)
        
        num_labels = probs.shape[1]
        if labels is None:
            labels = [f'label_{i}' for i in range(num_labels)]
        
        preds = np.zeros_like(probs, dtype=int)
        for label_idx, label_name in enumerate(labels):
            threshold = self.thresholds.get(label_name, 0.5)
            preds[:, label_idx] = (probs[:, label_idx] > threshold).astype(int)
        
        return preds


class AdvancedCalibrator:
    """Complete calibration pipeline combining temperature scaling + isotonic regression."""
    
    def __init__(self):
        self.temperature_scaler = TemperatureScaler()
        self.isotonic_scalers = []
        self.is_fitted = False
    
    def fit(self, logits: np.ndarray, probs: np.ndarray, y_true: np.ndarray):
        """Two-stage calibration:
        1. Temperature scaling on logits
        2. Per-label isotonic regression on probabilities
        
        Args:
            logits: raw model outputs, shape (N, num_labels)
            probs: sigmoid(logits), shape (N, num_labels)
            y_true: binary labels, shape (N, num_labels)
        """
        # Stage 1: Temperature scaling
        self.temperature_scaler.fit(logits, y_true)
        probs_temp = self.temperature_scaler.transform(logits)
        
        # Stage 2: Per-label isotonic regression
        num_labels = probs.shape[1]
        for label_idx in range(num_labels):
            p_label = probs_temp[:, label_idx]
            y_label = y_true[:, label_idx]
            
            try:
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(p_label, y_label)
                self.isotonic_scalers.append(iso)
            except:
                self.isotonic_scalers.append(None)
        
        self.is_fitted = True
    
    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply full calibration pipeline."""
        if not self.is_fitted:
            return 1.0 / (1.0 + np.exp(-logits))
        
        # Stage 1: Temperature
        probs = self.temperature_scaler.transform(logits)
        
        # Stage 2: Isotonic
        num_labels = probs.shape[1]
        for label_idx in range(min(num_labels, len(self.isotonic_scalers))):
            iso = self.isotonic_scalers[label_idx]
            if iso is not None:
                probs[:, label_idx] = iso.transform(probs[:, label_idx])
        
        return probs


def compute_expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, num_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) - measures calibration quality.
    
    ECE = average of |accuracy - confidence| over bins
    Lower ECE = better calibrated predictions
    
    Args:
        probs: predicted probabilities, shape (N,) or (N, num_labels)
        y_true: binary labels, same shape as probs
        num_bins: number of confidence bins
    
    Returns:
        ECE score (0-1, lower is better)
    """
    if probs.ndim == 2:
        # Multi-label: compute per-label ECE and average
        eces = []
        for label_idx in range(probs.shape[1]):
            ece = compute_expected_calibration_error(
                probs[:, label_idx], y_true[:, label_idx], num_bins=num_bins
            )
            eces.append(ece)
        return np.mean(eces)
    
    # Single label
    probs = np.clip(probs, 0, 1)
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    
    ece = 0.0
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Samples in this bin
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        if not np.any(in_bin):
            continue
        
        # Accuracy in bin
        accuracy = np.mean(y_true[in_bin] == (probs[in_bin] > 0.5).astype(int))
        
        # Average confidence in bin
        confidence = np.mean(probs[in_bin])
        
        # Weight by bin size
        bin_size = np.sum(in_bin)
        ece += (bin_size / len(probs)) * np.abs(accuracy - confidence)
    
    return ece
