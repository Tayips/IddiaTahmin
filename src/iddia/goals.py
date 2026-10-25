"""Gol pazarları: Üst/Alt 2.5 ve Karşılıklı Gol (KG) olasılıkları.

`python -m iddia.goals` son 5 tam sezonda modelleri piyasa ve taban oranla kıyaslar.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from iddia.compare import prepare
from iddia.features import FEATURES_NO_MARKET, GOAL_FEATURES

TARGETS = {
    "ust25": lambda df: (df["fthg"] + df["ftag"] > 2.5).astype(int),
    "kg": lambda df: ((df["fthg"] > 0) & (df["ftag"] > 0)).astype(int),
}


class BinaryModel:
    def __init__(self, features: list[str]):
        self.features = features
        self.est = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 LogisticRegression(C=0.1, max_iter=2000))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BinaryModel":
        self.est.fit(X[self.features], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.est.predict_proba(X[self.features])[:, 1]


def scores(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n": len(y),
        "isabet": float(((p >= 0.5) == y).mean()),
        "brier": float(((p - y) ** 2).mean()),
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        **{f"isabet_{t:.0%}+": float(((p >= 0.5) == y)[np.maximum(p, 1 - p) >= t].mean())
           if (np.maximum(p, 1 - p) >= t).any() else np.nan for t in (0.6, 0.65)},
        **{f"mac_{t:.0%}+": int((np.maximum(p, 1 - p) >= t).sum()) for t in (0.6, 0.65)},
    }


def main() -> None:
    df, X = prepare()
    seasons = sorted(df["season"].unique())
    test_seasons = seasons[-6:-1]
    rows = []
    for target, fn in TARGETS.items():
        y = fn(df)
        preds = {k: np.full(len(df), np.nan) for k in ["taban", "form", "form_piyasa"]}
        for season in test_seasons:
            tr, te = (df["season"] < season).to_numpy(), (df["season"] == season).to_numpy()
            preds["taban"][te] = y[tr].mean()
            preds["form"][te] = BinaryModel(FEATURES_NO_MARKET).fit(X[tr], y[tr]).predict(X[te])
            has = tr & X["mk_o25"].notna().to_numpy()
            preds["form_piyasa"][te] = BinaryModel(GOAL_FEATURES).fit(X[has], y[has]).predict(X[te])
        if target == "ust25":
            preds["piyasa"] = X["mk_o25"].to_numpy()
        ok = df["season"].isin(test_seasons).to_numpy() & X["mk_o25"].notna().to_numpy()
        for name, p in preds.items():
            rows.append({"pazar": target, "model": name, **scores(p[ok], y.to_numpy()[ok])})
    t = pd.DataFrame(rows)
    t.to_csv("reports/gol_pazarlari.csv", index=False)
    pd.set_option("display.width", 160)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
