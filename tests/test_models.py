import numpy as np
import pandas as pd
import pytest

from iddia import metrics, odds
from iddia.evaluate import drop_forfeits
from iddia.models.dixon_coles import DixonColes
from iddia.models.elo import Elo, OrderedLogit


def synthetic(n_seasons=3, seed=0):
    """Gücü bilinen 6 takımlı yapay lig: A en güçlü, F en zayıf."""
    rng = np.random.default_rng(seed)
    strength = dict(zip("ABCDEF", [0.5, 0.3, 0.1, -0.1, -0.3, -0.5]))
    rows, day = [], pd.Timestamp("2020-08-01")
    for s in range(n_seasons):
        for h in strength:
            for a in strength:
                if h == a:
                    continue
                hg = rng.poisson(np.exp(0.3 + strength[h] - strength[a] / 2))
                ag = rng.poisson(np.exp(strength[a] - strength[h] / 2))
                ftr = "H" if hg > ag else "A" if hg < ag else "D"
                rows.append((day, "X", f"{s:02d}{s + 1:02d}", h, a, hg, ag, ftr))
                day += pd.Timedelta(days=1)
    return pd.DataFrame(rows, columns=["date", "league", "season", "home", "away", "fthg", "ftag", "ftr"])


def test_rps_perfect_and_worst():
    assert metrics.rps(np.array([[1.0, 0, 0]]), ["H"])[0] == 0
    assert metrics.rps(np.array([[1.0, 0, 0]]), ["A"])[0] == 1
    # Beraberlik, deplasman galibiyetinden daha az cezalandırılır.
    p = np.array([[0.7, 0.2, 0.1]])
    assert metrics.rps(p, ["D"])[0] < metrics.rps(p, ["A"])[0]


def test_margin_removal_sums_to_one_and_shin_shrinks_longshot():
    o = np.array([[1.85, 3.10, 3.25], [1.20, 7.0, 12.0]])
    assert odds.margin(o)[0] == pytest.approx(0.1701, abs=1e-3)
    for fn in (odds.proportional, odds.shin):
        assert fn(o).sum(axis=1) == pytest.approx([1, 1])
    assert odds.shin(o)[1, 2] < odds.proportional(o)[1, 2]


def test_elo_ranks_strong_team_higher_and_uses_only_past():
    df = synthetic()
    elo = Elo()
    diffs = elo.run(df)
    assert diffs.iloc[0] == elo.home_adv   # ilk maçta iki takım da eşit başlar
    table = elo.ratings["X"]
    assert table["A"] > table["F"]


def test_ordered_logit_probabilities_valid_and_monotone():
    df = synthetic()
    df["d"] = Elo().run(df)
    p = OrderedLogit().fit(df["d"].to_numpy(), df["ftr"]).predict(np.array([-200.0, 0.0, 200.0]))
    assert p.sum(axis=1) == pytest.approx([1, 1, 1])
    assert (p > 0).all()
    assert p[0, 0] < p[1, 0] < p[2, 0]


def test_dixon_coles_learns_strength_and_handles_unknown_team():
    df = synthetic(n_seasons=4)
    model = DixonColes().fit(df, as_of=df["date"].max() + pd.Timedelta(days=1))
    assert model.attack["A"] > model.attack["F"]
    strong, weak = model.predict("A", "F"), model.predict("F", "A")
    assert strong.sum() == pytest.approx(1)
    assert strong[0] > weak[0]
    assert model.predict("YeniTakim", "A")[0] < model.predict("A", "YeniTakim")[0]


def test_drop_forfeits_only_removes_unpriced_3_0():
    df = pd.DataFrame({
        "fthg": [3, 3, 0], "ftag": [0, 0, 3],
        "odds_h": [np.nan, 1.5, np.nan], "odds_d": [np.nan, 4.0, np.nan], "odds_a": [np.nan, 6.0, np.nan],
    })
    assert len(drop_forfeits(df)) == 1
