"""
evaluate.py

Statistical model evaluation framework to compute metrics, confidence intervals,
and perform significance testing before promoting models.
"""

import os
import json
import dataclasses
from dataclasses import dataclass
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, log_loss, brier_score_loss,
    roc_curve, precision_recall_curve, auc
)
from sklearn.calibration import calibration_curve
from sklearn.utils import resample

import mlflow

@dataclass
class EvaluationReport:
    metrics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]

class Evaluator:
    @staticmethod
    def _calculate_ece(y_true, y_prob, n_bins=10):
        """Calculates Expected Calibration Error (ECE)"""
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
        bins = np.linspace(0., 1., n_bins + 1)
        binids = np.digitize(y_prob, bins) - 1
        
        ece = 0.0
        for b in range(n_bins):
            mask = binids == b
            if np.any(mask):
                bin_acc = y_true[mask].mean()
                bin_conf = y_prob[mask].mean()
                bin_weight = np.sum(mask) / len(y_prob)
                ece += bin_weight * np.abs(bin_acc - bin_conf)
        return float(ece)

    def _compute_metrics(self, y_true, y_pred, y_prob) -> Dict[str, float]:
        """Calculates all 9 base metrics"""
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'f1_weighted': float(f1_score(y_true, y_pred, average='weighted')),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
            'precision_weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall_weighted': float(recall_score(y_true, y_pred, average='weighted'))
        }
        
        if len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
            metrics['log_loss'] = float(log_loss(y_true, y_prob))
            metrics['brier_score'] = float(brier_score_loss(y_true, y_prob))
            metrics['ece'] = self._calculate_ece(y_true, y_prob)
        else:
            metrics['roc_auc'] = np.nan
            metrics['log_loss'] = np.nan
            metrics['brier_score'] = np.nan
            metrics['ece'] = np.nan
            
        return metrics

    def evaluate(self, model, X_test, y_test) -> EvaluationReport:
        """
        Calculates 9 metrics alongside 95% confidence intervals generated via 
        bootstrap resampling (1000 samples). Logs automatically if MLflow active run exists.
        """
        X_test_np = np.asarray(X_test)
        y_test_np = np.asarray(y_test)
        
        y_pred = model.predict(X_test_np)
        y_prob = model.predict_proba(X_test_np)[:, 1]
        
        base_metrics = self._compute_metrics(y_test_np, y_pred, y_prob)
        
        n_bootstraps = 1000
        boot_metrics = {k: [] for k in base_metrics.keys()}
        
        for _ in range(n_bootstraps):
            indices = resample(np.arange(len(y_test_np)))
            y_true_boot = y_test_np[indices]
            y_pred_boot = y_pred[indices]
            y_prob_boot = y_prob[indices]
            
            if len(np.unique(y_true_boot)) < 2:
                continue
                
            m = self._compute_metrics(y_true_boot, y_pred_boot, y_prob_boot)
            for k, v in m.items():
                boot_metrics[k].append(v)
                
        cis = {}
        for k in base_metrics.keys():
            if boot_metrics[k]:
                lower = float(np.percentile(boot_metrics[k], 2.5))
                upper = float(np.percentile(boot_metrics[k], 97.5))
                cis[k] = (lower, upper)
            else:
                cis[k] = (np.nan, np.nan)
                
        report = EvaluationReport(metrics=base_metrics, confidence_intervals=cis)
        
        if mlflow.active_run():
            mlflow.log_metrics(base_metrics)
            ci_metrics = {}
            for k, (l, u) in cis.items():
                ci_metrics[f"{k}_ci_lower"] = l
                ci_metrics[f"{k}_ci_upper"] = u
            mlflow.log_metrics(ci_metrics)
            
        return report

    def compare_models(self, model_a, model_b, X_test, y_test) -> Dict[str, Any]:
        """Runs McNemar's significance test to determine model dominance statistically."""
        X_test_np = np.asarray(X_test)
        y_test_np = np.asarray(y_test)
        
        pred_a = model_a.predict(X_test_np)
        pred_b = model_b.predict(X_test_np)
        
        a_correct = (pred_a == y_test_np)
        b_correct = (pred_b == y_test_np)
        
        yn = np.sum(a_correct & ~b_correct)
        ny = np.sum(~a_correct & b_correct)
        
        # McNemar's test with continuity correction
        if yn + ny > 0:
            statistic = (abs(yn - ny) - 1.0)**2 / (yn + ny)
            p_value = float(chi2.sf(statistic, df=1))
        else:
            p_value = 1.0
            
        is_significant = p_value < 0.05
        
        f1_a = f1_score(y_test_np, pred_a, average='weighted')
        f1_b = f1_score(y_test_np, pred_b, average='weighted')
        f1_delta = float(f1_b - f1_a)
        
        if not is_significant:
            better_model = 'same'
        elif f1_delta > 0:
            better_model = 'B'
        else:
            better_model = 'A'
            
        return {
            'p_value': p_value,
            'is_significant': is_significant,
            'better_model': better_model,
            'f1_delta': f1_delta
        }

    def plot_calibration_curve(self, model, X_test, y_test, save_path='calibration_curve.png'):
        """Plots how well the predicted probabilities match actual outcomes."""
        X_test_np = np.asarray(X_test)
        y_prob = model.predict_proba(X_test_np)[:, 1]
        
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        
        fig, ax = plt.subplots()
        ax.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Model')
        ax.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly calibrated')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title('Calibration Curve')
        ax.legend()
        
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        
        if mlflow.active_run():
            mlflow.log_artifact(save_path)

    def plot_roc_pr_curves(self, model, X_test, y_test, save_path='roc_pr_curves.png'):
        """Plots both ROC and Precision-Recall curves."""
        X_test_np = np.asarray(X_test)
        y_prob = model.predict_proba(X_test_np)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc_val = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc_val = auc(recall, precision)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_val:.2f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend(loc="lower right")
        
        # PR
        ax2.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc_val:.2f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc="lower left")
        
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        
        if mlflow.active_run():
            mlflow.log_artifact(save_path)

    def generate_report(self, evaluation_report: EvaluationReport, save_path='evaluation_report.json'):
        """Saves JSON metrics and generates a human-readable Markdown summary."""
        report_dict = dataclasses.asdict(evaluation_report)
        
        with open(save_path, 'w') as f:
            json.dump(report_dict, f, indent=4)
            
        md_path = save_path.replace('.json', '.md')
        with open(md_path, 'w') as f:
            f.write("# Model Evaluation Report\n\n")
            f.write("| Metric | Value | 95% Confidence Interval |\n")
            f.write("| --- | --- | --- |\n")
            for metric, value in report_dict['metrics'].items():
                ci = report_dict['confidence_intervals'].get(metric, (np.nan, np.nan))
                f.write(f"| **{metric}** | {value:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |\n")

        if mlflow.active_run():
            mlflow.log_artifact(save_path)
            mlflow.log_artifact(md_path)
