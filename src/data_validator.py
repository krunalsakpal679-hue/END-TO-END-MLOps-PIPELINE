import pandas as pd
import numpy as np
import hashlib
from dataclasses import dataclass
from typing import List

@dataclass
class CheckResult:
    check_name: str
    passed: bool
    message: str

@dataclass
class ValidationReport:
    passed: bool
    checks: List[CheckResult]
    n_rows: int
    n_features: int
    data_hash: str

class DataValidationError(Exception):
    pass

class DataValidator:
    def __init__(self, min_samples: int = 5000):
        self.min_samples = min_samples

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        checks = []
        n_rows = len(df)
        
        def add_check(name: str, passed: bool, msg: str):
            checks.append(CheckResult(check_name=name, passed=passed, message=msg))

        # Check 1 - Schema check
        expected_features = [f"feat_{i:02d}" for i in range(1, 11)]
        schema_passed = True
        schema_msgs = []
        for feat in expected_features:
            if feat not in df.columns:
                schema_passed = False
                schema_msgs.append(f"Missing {feat}")
            elif not pd.api.types.is_numeric_dtype(df[feat]):
                schema_passed = False
                schema_msgs.append(f"Non-numeric {feat}")
        
        msg = "All expected features present and numeric." if schema_passed else "Schema issues: " + ", ".join(schema_msgs)
        add_check("Schema check", schema_passed, msg)

        # Check 2 - Label check
        label_passed = False
        label_msg = ""
        if "label" not in df.columns:
            label_msg = "Missing 'label' column."
        else:
            unique_labels = df["label"].dropna().unique()
            valid = set(unique_labels).issubset({0, 1})
            if valid:
                label_passed = True
                label_msg = "Label column contains only 0 and 1."
            else:
                label_msg = f"Label contains invalid values: {unique_labels}"
        add_check("Label check", label_passed, label_msg)

        present_features = [f for f in expected_features if f in df.columns]

        # Check 3 - Null check
        null_passed = True
        null_msgs = []
        for feat in present_features:
            null_pct = df[feat].isnull().mean()
            if null_pct > 0.05:
                null_passed = False
                null_msgs.append(f"{feat} has {null_pct:.1%} nulls")
        msg = "No feature exceeds 5% nulls." if null_passed else "Null threshold exceeded: " + ", ".join(null_msgs)
        add_check("Null check", null_passed, msg)

        # Check 4 - Class balance check
        balance_passed = False
        balance_msg = "Cannot check balance (label missing or empty)"
        if "label" in df.columns and len(df["label"].dropna()) > 0:
            value_counts = df["label"].value_counts(normalize=True)
            if len(value_counts) > 0:
                minority_pct = value_counts.min() if len(value_counts) > 1 else 0.0
                if minority_pct >= 0.2:
                    balance_passed = True
                    balance_msg = f"Minority class represents {minority_pct:.1%} (>= 20%)."
                else:
                    balance_msg = f"Minority class is too small: {minority_pct:.1%} (< 20%)."
        add_check("Class balance check", balance_passed, balance_msg)

        # Check 5 - Minimum size check
        size_passed = n_rows >= self.min_samples
        size_msg = f"Row count {n_rows} >= {self.min_samples}." if size_passed else f"Row count {n_rows} < {self.min_samples}."
        add_check("Minimum size check", size_passed, size_msg)

        # Check 6 - Duplicate check
        dup_pct = df.duplicated().mean() if n_rows > 0 else 0
        dup_passed = dup_pct < 0.01
        dup_msg = f"Duplicates: {dup_pct:.2%} (< 1%)." if dup_passed else f"Duplicates: {dup_pct:.2%} (>= 1%)."
        add_check("Duplicate check", dup_passed, dup_msg)

        # Check 7 - Outlier check
        outlier_passed = True
        outlier_msgs = []
        for feat in present_features:
            col_data = df[feat].dropna()
            if len(col_data) > 0:
                mean = col_data.mean()
                std = col_data.std()
                if std > 0:
                    outliers = np.abs(col_data - mean) > 5 * std
                    outlier_pct = outliers.mean()
                    if outlier_pct >= 0.1:
                        outlier_passed = False
                        outlier_msgs.append(f"{feat} has {outlier_pct:.1%} outliers")
        msg = "No feature has >= 10% outliers." if outlier_passed else "Outliers exceeded: " + ", ".join(outlier_msgs)
        add_check("Outlier check", outlier_passed, msg)

        # Check 8 - Constant feature check
        constant_passed = True
        constant_msgs = []
        for feat in present_features:
            col_data = df[feat].dropna()
            if len(col_data) > 0:
                variance = col_data.var()
                if pd.isna(variance) or variance == 0 or len(col_data.unique()) <= 1:
                    constant_passed = False
                    constant_msgs.append(f"{feat} has zero variance")
        msg = "No constant features found." if constant_passed else "Constant features found: " + ", ".join(constant_msgs)
        add_check("Constant feature check", constant_passed, msg)

        # Finalize report
        all_passed = all(c.passed for c in checks)
        
        # Calculate data MD5 hash via reliable serialization
        data_hash = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
        
        report = ValidationReport(
            passed=all_passed,
            checks=checks,
            n_rows=n_rows,
            n_features=len(expected_features),
            data_hash=data_hash
        )

        if not all_passed:
            failed = [c for c in checks if not c.passed]
            failed_msgs = [f"- {c.check_name}: {c.message}" for c in failed]
            error_text = "Data validation failed!\n" + "\n".join(failed_msgs)
            raise DataValidationError(error_text)

        return report
