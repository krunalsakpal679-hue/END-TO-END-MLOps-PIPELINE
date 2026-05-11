import numpy as np

class DriftSimulator:
    @staticmethod
    def generate_reference(n_samples=5000, n_features=10, seed=42):
        """Creates synthetic reference data using numpy random."""
        np.random.seed(seed)
        X = np.random.normal(0, 1, size=(n_samples, n_features))
        y = np.random.choice([0, 1], size=n_samples)
        return X, y

    @staticmethod
    def inject_mean_shift(X: np.ndarray, feature_idx: int, shift_amount: float) -> np.ndarray:
        """Returns a copy of X where feature at feature_idx has shift_amount added to every value."""
        X_shifted = X.copy()
        X_shifted[:, feature_idx] += shift_amount
        return X_shifted

    @staticmethod
    def inject_variance_change(X: np.ndarray, feature_idx: int, scale_factor: float) -> np.ndarray:
        """Returns a copy of X where feature at feature_idx has its values multiplied by scale_factor."""
        X_scaled = X.copy()
        # Scale relative to the mean of the feature
        mean_val = np.mean(X_scaled[:, feature_idx])
        X_scaled[:, feature_idx] = mean_val + (X_scaled[:, feature_idx] - mean_val) * scale_factor
        return X_scaled

    @staticmethod
    def inject_label_flip(y: np.ndarray, flip_fraction: float) -> np.ndarray:
        """Returns a copy of y where flip_fraction of labels are randomly flipped (0->1 or 1->0)."""
        y_flipped = y.copy()
        n_flip = int(len(y) * flip_fraction)
        flip_indices = np.random.choice(len(y), size=n_flip, replace=False)
        y_flipped[flip_indices] = 1 - y_flipped[flip_indices]
        return y_flipped
