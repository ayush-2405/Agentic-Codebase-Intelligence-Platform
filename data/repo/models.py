"""
models.py — Data models for a hypothetical ML training pipeline.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for a machine learning model."""
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    hidden_dims: list = field(default_factory=lambda: [128, 64])
    dropout: float = 0.3
    optimizer: str = "adam"


class BaseModel:
    """Abstract base class for all models."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._trained = False
        self._weights = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def is_trained(self) -> bool:
        return self._trained


class LinearRegression(BaseModel):
    """Ordinary least-squares linear regression via gradient descent."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._weights: Optional[np.ndarray] = None
        self._bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        self._weights = np.zeros(n_features)
        self._bias = 0.0

        for epoch in range(self.config.epochs):
            y_pred = self._forward(X)
            loss = self._mse_loss(y, y_pred)
            dw, db = self._gradients(X, y, y_pred)
            self._weights -= self.config.learning_rate * dw
            self._bias -= self.config.learning_rate * db

        self._trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Model must be fit before predicting.")
        return self._forward(X)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        return X @ self._weights + self._bias

    @staticmethod
    def _mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def _gradients(X, y_true, y_pred):
        n = len(y_true)
        error = y_pred - y_true
        dw = (2 / n) * X.T @ error
        db = (2 / n) * np.sum(error)
        return dw, db
