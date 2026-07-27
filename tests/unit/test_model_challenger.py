import os
import json
import numpy as np
import pytest
from unittest.mock import MagicMock
from src.model_challenger import ModelChallenger

@pytest.fixture
def test_models():
    class MockModel:
        def __init__(self, prob_output, pred_output):
            self.prob_output = prob_output
            self.pred_output = pred_output
            self.predict = MagicMock(return_value=self.pred_output)
            self.predict_proba = MagicMock(return_value=self.prob_output)
            
    # Synthetic data sizes
    n = 100
    y_test = np.random.choice([0, 1], size=n)
    X_test = np.random.rand(n, 5)
    
    # Champion output (slightly worse)
    champ_pred = y_test.copy()
    champ_pred[:20] = 1 - champ_pred[:20]  # 80% acc
    champ_prob = np.where(champ_pred == 1, 0.8, 0.2)
    
    # Challenger output (better)
    chall_pred = y_test.copy()
    chall_pred[:5] = 1 - chall_pred[:5]  # 95% acc
    chall_prob = np.where(chall_pred == 1, 0.9, 0.1)
    
    champion = MockModel(champ_prob, champ_pred)
    challenger = MockModel(chall_prob, chall_pred)
    
    return champion, challenger, X_test, y_test

def test_compare_returns_all_keys(test_models):
    champion, challenger, X_test, y_test = test_models
    challenger_instance = ModelChallenger()
    result = challenger_instance.compare(champion, challenger, X_test, y_test)
    
    expected_keys = [
        "champion_metrics", "challenger_metrics", "metric_deltas", 
        "p_value", "is_significant", "champion_f1_ci", "challenger_f1_ci"
    ]
    for key in expected_keys:
        assert key in result

def test_should_promote_thresholds():
    challenger_instance = ModelChallenger()
    
    # F1 delta > 0.01 AND p_value < 0.05 -> True
    res_true = {
        "champion_metrics": {"f1_weighted": 0.85},
        "challenger_metrics": {"f1_weighted": 0.87},
        "p_value": 0.01
    }
    assert challenger_instance.should_promote(res_true) is True
    
    # F1 delta is 0.005 (below 0.01 threshold) -> False
    res_false_f1 = {
        "champion_metrics": {"f1_weighted": 0.85},
        "challenger_metrics": {"f1_weighted": 0.855},
        "p_value": 0.01
    }
    assert challenger_instance.should_promote(res_false_f1) is False
    
    # p_value >= 0.05 -> False
    res_false_pval = {
        "champion_metrics": {"f1_weighted": 0.85},
        "challenger_metrics": {"f1_weighted": 0.87},
        "p_value": 0.10
    }
    assert challenger_instance.should_promote(res_false_pval) is False

def test_confidence_intervals(test_models):
    champion, challenger, X_test, y_test = test_models
    challenger_instance = ModelChallenger()
    result = challenger_instance.compare(champion, challenger, X_test, y_test)
    
    c_lower, c_upper = result["champion_f1_ci"]
    assert c_lower < c_upper
    
    ch_lower, ch_upper = result["challenger_f1_ci"]
    assert ch_lower < ch_upper

def test_generate_comparison_report(test_models, tmp_path):
    champion, challenger, X_test, y_test = test_models
    challenger_instance = ModelChallenger()
    result = challenger_instance.compare(champion, challenger, X_test, y_test)
    
    out_file = tmp_path / "comparison_report.json"
    challenger_instance.generate_comparison_report(result, str(out_file))
    
    assert out_file.exists()
    with open(out_file, "r") as f:
        data = json.load(f)
        assert "p_value" in data

def test_latency_comparison(test_models):
    champion, challenger, X_test, y_test = test_models
    challenger_instance = ModelChallenger()
    result = challenger_instance.compare(champion, challenger, X_test, y_test)
    
    # predict() should have been called 1 time initially to get predictions,
    # plus 100 times precisely in the latency loop.
    assert champion.predict.call_count == 101
    assert challenger.predict.call_count == 101

def test_metric_deltas_correctness(test_models):
    champion, challenger, X_test, y_test = test_models
    challenger_instance = ModelChallenger()
    result = challenger_instance.compare(champion, challenger, X_test, y_test)
    
    deltas = result["metric_deltas"]
    expected_metrics = ["f1_weighted", "roc_auc", "accuracy", "precision", "recall", "log_loss"]
    for m in expected_metrics:
        assert m in deltas
        if m == "log_loss":
            # For log_loss, delta is champ - chall
            expected = result["champion_metrics"][m] - result["challenger_metrics"][m]
        else:
            expected = result["challenger_metrics"][m] - result["champion_metrics"][m]
        assert np.isclose(deltas[m], expected)
