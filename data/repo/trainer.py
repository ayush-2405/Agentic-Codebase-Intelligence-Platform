"""
trainer.py — Training loop and experiment management.
"""
import time
import numpy as np
from models import BaseModel, ModelConfig, LinearRegression
from data_utils import DataLoader, StandardScaler
from evaluator import ModelEvaluator


class Trainer:
    """Manages the full training pipeline for a model."""

    def __init__(self, model: BaseModel, config: ModelConfig):
        self.model = model
        self.config = config
        self.evaluator = ModelEvaluator()
        self._history: list[dict] = []

    def train(self, X_train, y_train, X_val=None, y_val=None) -> dict:
        """
        Execute the training loop.
        Returns a metrics dictionary.
        """
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        if X_val is not None:
            X_val = scaler.transform(X_val)

        t0 = time.time()
        self.model.fit(X_train, y_train)
        elapsed = time.time() - t0

        metrics = {"train_time_s": elapsed}
        if X_val is not None and y_val is not None:
            val_preds = self.model.predict(X_val)
            metrics.update(self.evaluator.compute_metrics(y_val, val_preds))

        self._history.append(metrics)
        return metrics

    def get_history(self) -> list[dict]:
        return self._history


def run_experiment(config: ModelConfig, X, y) -> dict:
    """
    Convenience function to run a full training experiment.
    Splits data, trains model, and evaluates.
    """
    from data_utils import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression(config)
    trainer = Trainer(model, config)
    return trainer.train(X_train, y_train, X_test, y_test)
