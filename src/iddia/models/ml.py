"""Özellik tabanlı 1X2 sınıflandırıcıları."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from iddia.metrics import OUTCOMES


def make(kind: str):
    if kind == "lr":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    if kind == "gb":
        return HistGradientBoostingClassifier(learning_rate=0.03, max_iter=300, max_leaf_nodes=15,
                                              min_samples_leaf=80, l2_regularization=1.0, random_state=0)
    raise ValueError(kind)


class Model:
    def __init__(self, kind: str, features: list[str]):
        self.kind, self.features = kind, features
        self.est = make(kind)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Model":
        self.est.fit(X[self.features], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p = self.est.predict_proba(X[self.features])
        order = [list(self.est.classes_).index(o) for o in OUTCOMES]
        return p[:, order]
