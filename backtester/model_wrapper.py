from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

class ModelWrapper:
    """Wrap Logistic Regression (default) or RandomForest with scaling and helpers."""
    def __init__(self, name: str = "lr"):
        self.name = name
        if name == "rf":
            self.model = Pipeline([
                ("scaler", StandardScaler()),
                ("est", RandomForestClassifier(
                    criterion="gini", max_depth=25, n_estimators=100, random_state=42
                ))
            ])
        else:
            self.model = Pipeline([
                ("scaler", StandardScaler()),
                ("est", LogisticRegression(
                    random_state=42, multi_class='multinomial', solver='sag',
                    C=1.0, penalty='l2', max_iter=2000
                ))
            ])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def coefficients(self, feature_names: List[str]) -> Optional[List[float]]:
        """Return LR coefficients for positive class if multinomial; None for RF."""
        if self.name != "lr":
            return None
        est = self.model.named_steps["est"]
        coefs = est.coef_
        if coefs.shape[0] > 1:
            try:
                row = list(est.classes_).index(1)
            except ValueError:
                row = 0
        else:
            row = 0
        return coefs[row].tolist()