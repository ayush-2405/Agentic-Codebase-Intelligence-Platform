"""
evaluator.py — Metrics computation for trained models.
"""
import numpy as np


class ModelEvaluator:
    """Computes regression and classification metrics."""

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Return a dict of standard regression metrics."""
        return {
            "mse": self._mse(y_true, y_pred),
            "rmse": self._mse(y_true, y_pred) ** 0.5,
            "mae": self._mae(y_true, y_pred),
            "r2": self._r2(y_true, y_pred),
        }

    @staticmethod
    def _mse(y_true, y_pred) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def _mae(y_true, y_pred) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def _r2(y_true, y_pred) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-10))
