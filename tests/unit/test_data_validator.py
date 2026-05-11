import pytest
import pandas as pd
import numpy as np
from src.data_validator import DataValidator, DataValidationError

def generate_perfect_data(n=5000):
    np.random.seed(42)
    data = {
        f"feat_{i:02d}": np.random.normal(0, 1, n) for i in range(1, 11)
    }
    data["label"] = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
    return pd.DataFrame(data)

def test_clean_data_passes():
    df = generate_perfect_data()
    validator = DataValidator(min_samples=5000)
    report = validator.validate(df)
    
    assert report.passed is True
    assert len(report.checks) == 8
    
    expected_names = [
        "Schema check", "Label check", "Null check", "Class balance check",
        "Minimum size check", "Duplicate check", "Outlier check", "Constant feature check"
    ]
    actual_names = [c.check_name for c in report.checks]
    assert set(expected_names) == set(actual_names)
    
    assert report.n_rows == 5000
    assert report.n_features == 10
    
def test_data_hash_consistency():
    df1 = generate_perfect_data()
    df2 = generate_perfect_data()  # Identical seed logic ensures identical data
    
    validator = DataValidator(min_samples=5000)
    report1 = validator.validate(df1)
    report2 = validator.validate(df2)
    
    assert report1.data_hash == report2.data_hash

def test_null_check_fails():
    df = generate_perfect_data()
    # Add exactly > 5% nulls to feat_01 (10% nulls)
    df.loc[:500, "feat_01"] = np.nan
    validator = DataValidator()
    
    with pytest.raises(DataValidationError) as exc:
        validator.validate(df)
        
    assert "Null check" in str(exc.value)

def test_label_check_fails():
    df = generate_perfect_data()
    df.loc[0, "label"] = 2  # Inject invalid class
    validator = DataValidator()
    
    with pytest.raises(DataValidationError) as exc:
        validator.validate(df)
        
    assert "Label check" in str(exc.value)

def test_class_imbalance_fails():
    df = generate_perfect_data()
    # Heavily imbalanced label (95% to 5%)
    df["label"] = np.random.choice([0, 1], size=5000, p=[0.95, 0.05])
    validator = DataValidator()
    
    with pytest.raises(DataValidationError) as exc:
        validator.validate(df)
        
    assert "Class balance check" in str(exc.value)

def test_all_checks_run_independently():
    df = generate_perfect_data(n=1000)  # Size check fails (1000 < 5000)
    df.loc[:150, "feat_02"] = np.nan     # Null check fails (>5% nulls on 1000 rows)
    df["label"] = 3                      # Label check fails (3 not 0,1) and Balance fails (minority = 0.0)
    
    validator = DataValidator(min_samples=5000)
    
    with pytest.raises(DataValidationError) as exc:
        validator.validate(df)
        
    err_msg = str(exc.value)
    # The error payload should contain every distinct check that failed
    assert "Minimum size check" in err_msg
    assert "Null check" in err_msg
    assert "Label check" in err_msg
    assert "Class balance check" in err_msg
