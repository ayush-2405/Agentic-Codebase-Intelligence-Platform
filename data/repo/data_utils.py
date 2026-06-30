"""
data_utils.py — Dataset loading and preprocessing utilities.
"""
import numpy as np
from typing import Tuple


class DataLoader:
    """Loads datasets from various sources."""

    def __init__(self, path: str, batch_size: int = 32):
        self.path = path
        self.batch_size = batch_size
        self._data = None

    def load_csv(self) -> np.ndarray:
        """Load CSV file into a numpy array."""
        import csv
        rows = []
        with open(self.path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append([float(v) for v in row.values()])
        self._data = np.array(rows)
        return self._data

    def batches(self):
        """Yield mini-batches of data."""
        if self._data is None:
            raise RuntimeError("Call load_csv() first.")
        n = len(self._data)
        for i in range(0, n, self.batch_size):
            yield self._data[i : i + self.batch_size]


class StandardScaler:
    """Zero-mean, unit-variance normalisation."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Fit before transforming.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into random train/test subsets."""
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    split = int(n * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def generate_synthetic_data(n_samples: int = 1000, n_features: int = 5, seed: int = 0) -> Tuple:
    """Generate a synthetic regression dataset."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    true_weights = rng.standard_normal(n_features)
    y = X @ true_weights + rng.normal(0, 0.1, n_samples)
    return X, y
