import numpy as np
import pandas as pd

from iddia.features import build_features, team_form
from iddia.models.elo import Elo
from iddia.models.ml import Model
from tests.test_models import synthetic


def with_stats(df):
    df = df.copy()
    for c in ("hs", "as_", "hst", "ast"):
        df[c] = 5.0
    df["pre_h"], df["pre_d"], df["pre_a"] = 2.0, 3.4, 3.8
    return df


def test_form_uses_only_earlier_matches():
    df = with_stats(synthetic(n_seasons=1))
    base = team_form(df)
    changed = df.copy()
    changed.loc[10, ["fthg", "ftag"]] = [9, 0]
    after = team_form(changed)
    # Maçın kendi özellikleri değişmez, sonraki maçlarınki değişir.
    pd.testing.assert_series_equal(base.loc[10], after.loc[10])
    later = df.index[(df.index > 10) & ((df["home"] == df.loc[10, "home"]) | (df["away"] == df.loc[10, "home"]))][0]
    assert not base.loc[later].equals(after.loc[later])


def test_future_match_without_score_does_not_change_history():
    df = with_stats(synthetic(n_seasons=1))
    future = df.iloc[[0]].assign(date=df["date"].max() + pd.Timedelta(days=3), fthg=np.nan, ftag=np.nan, ftr=None)
    combined = pd.concat([df, future], ignore_index=True)
    a, b = team_form(df), team_form(combined)
    pd.testing.assert_frame_equal(a, b.iloc[:-1])
    assert np.isfinite(b.iloc[-1]["h_gf"])


def test_model_returns_probabilities_in_h_d_a_order():
    df = with_stats(synthetic(n_seasons=3))
    X = build_features(df, Elo().run(df))
    m = Model("lr", ["elo_diff", "h_gf", "a_gf"]).fit(X, df["ftr"])
    p = m.predict(X)
    assert np.allclose(p.sum(axis=1), 1)
    strong_home = (df["home"] == "A") & (df["away"] == "F")
    assert p[strong_home.to_numpy(), 0].mean() > p[strong_home.to_numpy(), 2].mean()
