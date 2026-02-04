"""
Comprehensive Evaluation Suite for Cyberbullying Detection
Tracks all critical metrics: F1, precision, recall, ROC-AUC, calibration, per-severity analysis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, auc
)
from sklearn.preprocessing import label_binarize
from typing import Dict, Tuple, List


class ComprehensiveEvaluator:
    """Production-grade evaluation with all metrics."""
    
    def __init__(self, labels: List[str] = None):
        """
        Args:
            labels: label names
        """
        self.labels = labels or ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray = None) -> Dict:
        """Compute all evaluation metrics.
        
        Args:
            y_true: binary labels, shape (N, num_labels)
            y_pred: binary predictions, shape (N, num_labels)
            y_probs: predicted probabilities, shape (N, num_labels) (optional)
        
        Returns:
            dict with comprehensive metrics
        """
        results = {}
        
        # 1. Global metrics
        results['global'] = self._compute_global_metrics(y_true, y_pred, y_probs)
        
        # 2. Per-label metrics
        results['per_label'] = self._compute_per_label_metrics(y_true, y_pred, y_probs)
        
        # 3. Severity-stratified analysis
        results['by_severity'] = self._analyze_by_severity(y_true, y_pred)
        
        # 4. Calibration analysis
        if y_probs is not None:
            results['calibration'] = self._analyze_calibration(y_true, y_probs)
        
        # 5. Error analysis
        results['error_analysis'] = self._error_analysis(y_true, y_pred, y_probs)
        
        return results
    
    def _compute_global_metrics(self, y_true, y_pred, y_probs) -> Dict:
        """Global accuracy, F1, precision, recall."""
        metrics = {}
        
        # Micro and macro averages
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        # ROC-AUC if probabilities available
        if y_probs is not None:
            try:
                metrics['roc_auc_micro'] = roc_auc_score(y_true, y_probs, average='micro')
                metrics['roc_auc_macro'] = roc_auc_score(y_true, y_probs, average='macro')
            except:
                metrics['roc_auc_micro'] = None
                metrics['roc_auc_macro'] = None
        
        return metrics
    
    def _compute_per_label_metrics(self, y_true, y_pred, y_probs) -> Dict:
        """Per-label F1, precision, recall, support."""
        metrics = {}
        
        for label_idx, label_name in enumerate(self.labels):
            y_true_label = y_true[:, label_idx]
            y_pred_label = y_pred[:, label_idx]
            
            metrics[label_name] = {
                'f1': f1_score(y_true_label, y_pred_label, zero_division=0),
                'precision': precision_score(y_true_label, y_pred_label, zero_division=0),
                'recall': recall_score(y_true_label, y_pred_label, zero_division=0),
                'support': int(np.sum(y_true_label)),
                'tn': int(np.sum((y_true_label == 0) & (y_pred_label == 0))),
                'fp': int(np.sum((y_true_label == 0) & (y_pred_label == 1))),
                'fn': int(np.sum((y_true_label == 1) & (y_pred_label == 0))),
                'tp': int(np.sum((y_true_label == 1) & (y_pred_label == 1)))
            }
            
            # ROC-AUC per label
            if y_probs is not None:
                try:
                    metrics[label_name]['roc_auc'] = roc_auc_score(y_true_label, y_probs[:, label_idx])
                except:
                    metrics[label_name]['roc_auc'] = None
        
        return metrics
    
    def _analyze_by_severity(self, y_true, y_pred) -> Dict:
        """Analyze performance grouped by severity."""
        severity_map = {
            'CRITICAL': ['severe_toxic', 'threat'],
            'HIGH': ['identity_hate'],
            'MEDIUM': ['toxic'],
            'LOW': ['insult', 'obscene']
        }
        
        results = {}
        for severity, label_list in severity_map.items():
            # Find samples with these labels
            severity_mask = np.zeros(len(y_true), dtype=bool)
            for label_name in label_list:
                if label_name in self.labels:
                    label_idx = self.labels.index(label_name)
                    severity_mask |= (y_true[:, label_idx] == 1)
            
            if np.sum(severity_mask) == 0:
                continue
            
            y_true_sev = y_true[severity_mask]
            y_pred_sev = y_pred[severity_mask]
            
            results[severity] = {
                'f1': f1_score(y_true_sev, y_pred_sev, average='micro', zero_division=0),
                'precision': precision_score(y_true_sev, y_pred_sev, average='micro', zero_division=0),
                'recall': recall_score(y_true_sev, y_pred_sev, average='micro', zero_division=0),
                'samples': int(np.sum(severity_mask))
            }
        
        return results
    
    def _analyze_calibration(self, y_true, y_probs) -> Dict:
        """Compute calibration metrics."""
        from src.advanced_calibration import compute_expected_calibration_error
        
        ece = compute_expected_calibration_error(y_probs, y_true, num_bins=10)
        
        # Confidence vs accuracy
        confidence_bins = np.linspace(0, 1, 11)
        accuracies = []
        
        for i in range(len(confidence_bins) - 1):
            mask = (np.max(y_probs, axis=1) >= confidence_bins[i]) & \
                   (np.max(y_probs, axis=1) < confidence_bins[i + 1])
            if np.sum(mask) > 0:
                acc = np.mean(np.argmax(y_probs[mask], axis=1) == np.argmax(y_true[mask], axis=1))
                accuracies.append(acc)
        
        return {
            'ece': float(ece),
            'confidence_accuracies': accuracies,
            'comment': 'ECE < 0.1 is well-calibrated'
        }
    
    def _error_analysis(self, y_true, y_pred, y_probs) -> Dict:
        """Analyze false positives and false negatives."""
        results = {}
        
        for label_idx, label_name in enumerate(self.labels):
            y_true_label = y_true[:, label_idx]
            y_pred_label = y_pred[:, label_idx]
            
            fp_indices = np.where((y_true_label == 0) & (y_pred_label == 1))[0]
            fn_indices = np.where((y_true_label == 1) & (y_pred_label == 0))[0]
            
            results[label_name] = {
                'false_positives': int(len(fp_indices)),
                'false_negatives': int(len(fn_indices)),
                'fp_rate': float(len(fp_indices) / (np.sum(y_true_label == 0) + 1e-8)),
                'fn_rate': float(len(fn_indices) / (np.sum(y_true_label == 1) + 1e-8))
            }
            
            # False positive severity analysis
            if y_probs is not None and len(fp_indices) > 0:
                fp_probs = y_probs[fp_indices, label_idx]
                results[label_name]['avg_fp_confidence'] = float(np.mean(fp_probs))
                results[label_name]['high_confidence_fps'] = int(np.sum(fp_probs > 0.8))
        
        return results
    
    def generate_report(self, metrics: Dict) -> str:
        """Generate human-readable evaluation report."""
        report = []
        report.append("=" * 80)
        report.append("CYBERBULLYING DETECTION - COMPREHENSIVE EVALUATION REPORT")
        report.append("=" * 80)
        
        # Global metrics
        report.append("\n📊 GLOBAL METRICS")
        report.append("-" * 40)
        global_m = metrics.get('global', {})
        report.append(f"Accuracy:          {global_m.get('accuracy', 0):.4f}")
        report.append(f"F1 (macro):        {global_m.get('f1_macro', 0):.4f}")
        report.append(f"F1 (micro):        {global_m.get('f1_micro', 0):.4f}")
        report.append(f"Precision (macro): {global_m.get('precision_macro', 0):.4f}")
        report.append(f"Recall (macro):    {global_m.get('recall_macro', 0):.4f}")
        if global_m.get('roc_auc_macro'):
            report.append(f"ROC-AUC (macro):   {global_m.get('roc_auc_macro'):.4f}")
        
        # Per-label metrics
        report.append("\n🏷️  PER-LABEL METRICS")
        report.append("-" * 40)
        per_label = metrics.get('per_label', {})
        for label, m in per_label.items():
            report.append(f"\n{label.upper()}:")
            report.append(f"  F1={m.get('f1', 0):.3f} | P={m.get('precision', 0):.3f} | R={m.get('recall', 0):.3f} | Support={m.get('support', 0)}")
            report.append(f"  TP={m.get('tp', 0)} FP={m.get('fp', 0)} FN={m.get('fn', 0)}")
        
        # Severity analysis
        report.append("\n⚖️  SEVERITY-BASED PERFORMANCE")
        report.append("-" * 40)
        by_sev = metrics.get('by_severity', {})
        for sev, m in by_sev.items():
            report.append(f"{sev}: F1={m.get('f1', 0):.3f} | Samples={m.get('samples', 0)}")
        
        # Calibration
        report.append("\n🎯 CALIBRATION ANALYSIS")
        report.append("-" * 40)
        cal = metrics.get('calibration', {})
        ece = cal.get('ece', 0)
        report.append(f"Expected Calibration Error: {ece:.4f} {'✓ Well-calibrated' if ece < 0.1 else '⚠ Needs improvement'}")
        
        return "\n".join(report)
