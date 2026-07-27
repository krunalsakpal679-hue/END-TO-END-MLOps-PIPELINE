import json
import logging
import time
import numpy as np
from scipy.stats import chi2_contingency
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class ModelChallenger:
    
    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        accuracy = float(np.mean(y_true == y_pred))
        
        labels = np.unique(np.concatenate([y_true, y_pred]))
        f1s, precisions, recalls, supports = [], [], [], []
        
        for label in labels:
            tp = np.sum((y_pred == label) & (y_true == label))
            fp = np.sum((y_pred == label) & (y_true != label))
            fn = np.sum((y_pred != label) & (y_true == label))
            sup = np.sum(y_true == label)
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
            supports.append(sup)
            
        total_sup = np.sum(supports)
        if total_sup > 0:
            prec_w = sum(p * s for p, s in zip(precisions, supports)) / total_sup
            rec_w = sum(r * s for r, s in zip(recalls, supports)) / total_sup
            f1_w = sum(f * s for f, s in zip(f1s, supports)) / total_sup
        else:
            prec_w, rec_w, f1_w = 0.0, 0.0, 0.0

        # Log Loss
        eps = 1e-15
        y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
        log_loss_val = -np.mean(y_true * np.log(y_prob_clipped) + (1 - y_true) * np.log(1 - y_prob_clipped))
        
        # ROC AUC
        desc_score_indices = np.argsort(y_prob)[::-1]
        y_true_sorted = y_true[desc_score_indices]
        tps = np.cumsum(y_true_sorted)
        fps = np.cumsum(1 - y_true_sorted)
        if tps[-1] > 0 and fps[-1] > 0:
            tpr = tps / tps[-1]
            fpr = fps / fps[-1]
            tpr = np.concatenate(([0.0], tpr))
            fpr = np.concatenate(([0.0], fpr))
            try:
                roc_auc_val = np.trapezoid(tpr, fpr)
            except AttributeError:
                roc_auc_val = np.trapz(tpr, fpr)
        else:
            roc_auc_val = 0.5
            
        return {
            "accuracy": float(accuracy),
            "precision": float(prec_w),
            "recall": float(rec_w),
            "f1_weighted": float(f1_w),
            "roc_auc": float(roc_auc_val),
            "log_loss": float(log_loss_val)
        }

    @staticmethod
    def _bootstrap_f1(y_true: np.ndarray, y_pred: np.ndarray, n_bootstrap: int = 1000) -> Tuple[float, float]:
        n = len(y_true)
        f1_list = []
        for _ in range(n_bootstrap):
            idx = np.random.randint(0, n, n)
            sample_true = y_true[idx]
            sample_pred = y_pred[idx]
            
            # Inline f1_weighted for performance in loop
            labels = np.unique(np.concatenate([sample_true, sample_pred]))
            f1s, supports = [], []
            for label in labels:
                tp = np.sum((sample_pred == label) & (sample_true == label))
                fp = np.sum((sample_pred == label) & (sample_true != label))
                fn = np.sum((sample_pred != label) & (sample_true == label))
                sup = np.sum(sample_true == label)
                
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                f1s.append(f1)
                supports.append(sup)
                
            total = np.sum(supports)
            f1_w = sum(f * s for f, s in zip(f1s, supports)) / total if total > 0 else 0.0
            f1_list.append(f1_w)
            
        return float(np.percentile(f1_list, 2.5)), float(np.percentile(f1_list, 97.5))

    @staticmethod
    def _measure_latency(model, X) -> float:
        latencies = []
        sample = X[0:1] if len(X.shape) > 1 else X[0]
        for _ in range(100):
            start = time.perf_counter()
            model.predict(sample)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        return float(np.percentile(latencies, 95))

    def compare(self, champion_model, challenger_model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)
        
        # 1. Predictions
        champ_pred = champion_model.predict(X_test)
        chall_pred = challenger_model.predict(X_test)
        
        # Probabilities handling
        def get_prob(model):
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)
                return probs[:, 1] if len(probs.shape) == 2 and probs.shape[1] == 2 else probs
            return model.predict(X_test)
            
        champ_prob = get_prob(champion_model)
        chall_prob = get_prob(challenger_model)
        
        # 2. Metric comparison
        champ_metrics = self._compute_metrics(y_test, champ_pred, champ_prob)
        chall_metrics = self._compute_metrics(y_test, chall_pred, chall_prob)
        
        # 3. Metric deltas
        deltas = {}
        for k in champ_metrics.keys():
            # For log_loss, lower is better. 
            # Reversing delta logic for log_loss ensures positive delta remains "better" universally
            if k == "log_loss":
                deltas[k] = champ_metrics[k] - chall_metrics[k]
            else:
                deltas[k] = chall_metrics[k] - champ_metrics[k]
                
        # 4. McNemar's Test
        champ_correct = (y_test == champ_pred)
        chall_correct = (y_test == chall_pred)
        
        yy = int(np.sum(champ_correct & chall_correct))
        yn = int(np.sum(champ_correct & ~chall_correct))
        ny = int(np.sum(~champ_correct & chall_correct))
        nn = int(np.sum(~champ_correct & ~chall_correct))
        
        table = [[yy, yn], [ny, nn]]
        chi2, p_value, dof, expected = chi2_contingency(table)
        
        # 5. Bootstrap CI
        champ_ci = self._bootstrap_f1(y_test, champ_pred)
        chall_ci = self._bootstrap_f1(y_test, chall_pred)
        
        # 6. Latency Comparison
        champ_lat = self._measure_latency(champion_model, X_test)
        chall_lat = self._measure_latency(challenger_model, X_test)
        
        return {
            "champion_metrics": champ_metrics,
            "challenger_metrics": chall_metrics,
            "metric_deltas": deltas,
            "p_value": float(p_value),
            "is_significant": bool(p_value < 0.05),
            "champion_f1_ci": champ_ci,
            "challenger_f1_ci": chall_ci,
            "champion_p95_latency_ms": champ_lat,
            "challenger_p95_latency_ms": chall_lat
        }

    def should_promote(self, comparison_result: dict) -> bool:
        champ_f1 = comparison_result["champion_metrics"]["f1_weighted"]
        chall_f1 = comparison_result["challenger_metrics"]["f1_weighted"]
        p_value = comparison_result["p_value"]
        
        cond_a = chall_f1 > (champ_f1 + 0.01)
        cond_b = p_value < 0.05
        
        if cond_a and cond_b:
            return True
            
        if not cond_a:
            logger.info(f"Promotion failed: Challenger F1 ({chall_f1:.4f}) is not > Champion F1 ({champ_f1:.4f}) + 0.01.")
        if not cond_b:
            logger.info(f"Promotion failed: McNemar p-value ({p_value:.4f}) is not < 0.05 (Not statistically significant).")
            
        return False

    def generate_comparison_report(self, comparison_result: dict, output_path: str = 'comparison_report.json') -> None:
        with open(output_path, 'w') as f:
            json.dump(comparison_result, f, indent=4)
            
        print("="*65)
        print("MODEL COMPARISON SUMMARY")
        print("="*65)
        metrics = ["f1_weighted", "roc_auc", "accuracy", "precision", "recall", "log_loss"]
        print(f"{'Metric':<15} | {'Champion':<12} | {'Challenger':<12} | {'Delta':<12}")
        print("-" * 65)
        for m in metrics:
            champ_val = comparison_result["champion_metrics"][m]
            chall_val = comparison_result["challenger_metrics"][m]
            delta = comparison_result["metric_deltas"][m]
            print(f"{m:<15} | {champ_val:<12.4f} | {chall_val:<12.4f} | {delta:<12.4f}")
            
        print("-" * 65)
        print(f"{'McNemar p-value':<20}: {comparison_result['p_value']:.4f}")
        print(f"{'Significant?':<20}: {comparison_result['is_significant']}")
        
        c1, c2 = comparison_result['champion_f1_ci']
        print(f"{'Champion 95% CI':<20}: [{c1:.4f}, {c2:.4f}]")
        
        c1, c2 = comparison_result['challenger_f1_ci']
        print(f"{'Challenger 95% CI':<20}: [{c1:.4f}, {c2:.4f}]")
        
        print(f"{'Champ p95 lat':<20}: {comparison_result['champion_p95_latency_ms']:.2f} ms")
        print(f"{'Chall p95 lat':<20}: {comparison_result['challenger_p95_latency_ms']:.2f} ms")
        print("="*65)
