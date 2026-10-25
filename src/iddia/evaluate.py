"""Faz 1 değerlendirmesi: modelleri zamanda ileriye yürüyerek test eder.

Her test sezonu için modeller yalnızca o sezondan önceki maçlarla kurulur.
Dixon-Coles her hafta, o haftadan önceki maçlarla yeniden eğitilir.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from iddia import metrics
from iddia.models import baseline
from iddia.models.dixon_coles import DixonColes
from iddia.models.elo import Elo, OrderedLogit

MATCHES = Path("data/processed/matches.csv")
REPORT_DIR = Path("reports")
DC_WINDOW_DAYS = 3 * 365


def load(path: Path = MATCHES) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], dtype={"season": str})
    return df.sort_values(["date", "league", "home"]).reset_index(drop=True)


def drop_forfeits(df: pd.DataFrame) -> pd.DataFrame:
    """Hükmen sonuçlar (oran yok, skor 3-0 veya 0-3) oynanmamış maçlardır; modele girmez."""
    no_odds = df[["odds_h", "odds_d", "odds_a"]].isna().any(axis=1)
    score = ((df["fthg"] == 3) & (df["ftag"] == 0)) | ((df["fthg"] == 0) & (df["ftag"] == 3))
    return df[~(no_odds & score)].reset_index(drop=True)


def predict_dixon_coles(df: pd.DataFrame, test_mask: pd.Series) -> np.ndarray:
    out = np.full((len(df), 3), np.nan)
    for league, games in df.groupby("league"):
        model = DixonColes()
        test = games[test_mask.loc[games.index]]
        for week, chunk in test.groupby(test["date"].dt.to_period("W")):
            start = week.start_time
            train = games[(games["date"] < start) & (games["date"] >= start - pd.Timedelta(days=DC_WINDOW_DAYS))]
            model.fit(train, as_of=start)
            out[chunk.index] = np.vstack([model.predict(h, a) for h, a in zip(chunk["home"], chunk["away"])])
    return out


def run(df: pd.DataFrame, test_seasons: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["elo_diff"] = Elo().run(df)
    test_mask = df["season"].isin(test_seasons)
    preds = {name: np.full((len(df), 3), np.nan) for name in ["frekans", "elo", "oran_kapanis"]}

    for season in test_seasons:
        for league in df["league"].unique():
            train = df[(df["season"] < season) & (df["league"] == league)]
            test = df[(df["season"] == season) & (df["league"] == league)]
            if test.empty:
                continue
            preds["frekans"][test.index] = baseline.frequencies(train, test)
            logit = OrderedLogit().fit(train["elo_diff"].to_numpy(), train["ftr"])
            preds["elo"][test.index] = logit.predict(test["elo_diff"].to_numpy())
    preds["oran_kapanis"][test_mask.to_numpy()] = baseline.closing_odds(df[test_mask])
    preds["dixon_coles"] = predict_dixon_coles(df, test_mask)

    for name, p in preds.items():
        cols = pd.DataFrame(p, columns=[f"{name}_H", f"{name}_D", f"{name}_A"], index=df.index)
        df = df.join(cols)
    return df[test_mask].reset_index(drop=True)


def score_table(pred: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    groups = [("Hepsi", pred)] + list(pred.groupby("league"))
    for group, part in groups:
        for m in models:
            p = part[[f"{m}_H", f"{m}_D", f"{m}_A"]].to_numpy()
            s = metrics.summary(p, part["ftr"])
            s["ece"] = metrics.ece(p, part["ftr"])
            rows.append({"lig": group, "model": m, **s})
    return pd.DataFrame(rows)


def confidence_table(pred: pd.DataFrame, models: list[str], thresholds=(0.5, 0.6, 0.7)) -> pd.DataFrame:
    """En olası sonucun olasılığı eşiği geçtiğinde isabet ve kalan maç sayısı (DG-05)."""
    rows = []
    for m in models:
        p = pred[[f"{m}_H", f"{m}_D", f"{m}_A"]].to_numpy()
        top = p.max(axis=1)
        hit = metrics.accuracy(p, pred["ftr"])
        for t in thresholds:
            sel = top >= t
            rows.append({"model": m, "esik": t, "mac": int(sel.sum()), "isabet": hit[sel].mean() if sel.any() else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-seasons", type=int, default=5, help="son kaç sezon test edilsin")
    args = parser.parse_args()

    df = drop_forfeits(load())
    seasons = sorted(df["season"].unique())
    test_seasons = seasons[-args.test_seasons:]
    pred = run(df, test_seasons)
    models = ["frekans", "elo", "dixon_coles", "oran_kapanis"]
    table = score_table(pred, models)

    REPORT_DIR.mkdir(exist_ok=True)
    pred.to_csv(REPORT_DIR / "faz1_tahminler.csv", index=False)
    table.to_csv(REPORT_DIR / "faz1_skorlar.csv", index=False)
    conf = confidence_table(pred, models)
    conf.to_csv(REPORT_DIR / "faz1_guven.csv", index=False)
    pd.set_option("display.width", 120)
    print(f"Test sezonları: {', '.join(test_seasons)}")
    print(table.round(4).to_string(index=False))
    print()
    print(conf.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
