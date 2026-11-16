"""Two-stage MPDR model: stage 1 OLS on persistence variables, stage 2 a learner on its residual."""
import numpy as np
from sklearn.linear_model import LinearRegression

from .config import PERSISTENCE


class PersistenceStage:
    """OLS of MPDR on last year's performance and the benchmark regressors."""

    def fit(self, D, y):
        self.model_ = LinearRegression().fit(D[PERSISTENCE], y)
        return self

    def predict(self, D):
        return self.model_.predict(D[PERSISTENCE])

    def residual(self, D, y):
        return np.asarray(y, dtype=float) - self.predict(D)
