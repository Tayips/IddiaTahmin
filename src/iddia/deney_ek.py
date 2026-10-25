"""Ek deneyler: daha fazla ligle eğitim ve puan durumu / motivasyon özellikleri.

Test her zaman yalnızca Premier Lig ve Süper Lig maçlarındadır (son 5 tam sezon),
modeller her test sezonu için yalnızca önceki sezonlarla eğitilir. Dört
birleşim karşılaştırılır:

  A  şu anki model: E0 + T1 ile eğitim, form + piyasa
  B  tüm liglerle eğitim, form + piyasa
  C  E0 + T1 ile eğitim, form + piyasa + puan durumu
  D  tüm liglerle eğitim, form + piyasa + puan durumu

Ek ligler football-data.co.uk'ten data/raw_ek altına indirilir.
Çalıştırmak için: python -m iddia.deney_ek [--indir]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from iddia import download, metrics
from iddia.build import build
from iddia.evaluate import drop_forfeits
from iddia.features import FEATURES_NO_MARKET, GOAL_FEATURES, MARKET_FEATURES, build_features
from iddia.goals import TARGETS, BinaryModel, scores
from iddia.models.elo import Elo
from iddia.models.ml import Model

RAW_EK = Path("data/raw_ek")
EXTRA = ["E1", "D1", "I1", "SP1", "F1", "N1", "P1", "B1", "SC0", "G1"]
TARGET = ["E0", "T1"]
BASE = FEATURES_NO_MARKET + MARKET_FEATURES
RELEGATED = 3
TABLE = ["h_progress", "h_safe", "a_safe", "h_title", "a_title", "h_ppg", "a_ppg",
         "late_h_safe", "late_a_safe", "late_h_title", "late_a_title"]


def fetch_extra(first: int = 2014) -> None:
    for league in EXTRA:
        for year in range(first, download.current_season_start() + 1):
            try:
                download.download(league, year, RAW_EK)
            except requests.HTTPError as e:
                print(f"{league} {year}: {e}")


def standings(df: pd.DataFrame) -> pd.DataFrame:
    """Her maç için maç ÖNCESİ puan durumu özellikleri (aynı gün oynanan maçlar birbirini görmez).

    safe:  küme düşme hattının (sondan 4. sıra) ne kadar üstünde, kalan maçlarda alınabilecek puana oranla
    title: lidere puan farkı, kalan maçlarda alınabilecek puana oranla
    progress: sezonun oynanan kısmı
    """
    out = pd.DataFrame(index=df.index, columns=TABLE, dtype=float)
    for (_, _), g in df.groupby(["league", "season"]):
        teams = sorted(set(g["home"]) | set(g["away"]))
        n = len(teams)
        total = 2 * (n - 1)
        pts = dict.fromkeys(teams, 0)
        played = dict.fromkeys(teams, 0)
        for _, day in g.groupby("date", sort=True):
            ranked = sorted(pts.values(), reverse=True)
            leader, safe_line = ranked[0], ranked[min(n - RELEGATED - 1, n - 1)]
            for i, r in day.iterrows():
                vals = {}
                for side, team in (("h", r["home"]), ("a", r["away"])):
                    left = max(total - played[team], 1) * 3
                    vals[f"{side}_safe"] = np.clip((pts[team] - safe_line) / left, -1, 1)
                    vals[f"{side}_title"] = np.clip((leader - pts[team]) / left, 0, 1)
                    vals[f"{side}_ppg"] = pts[team] / played[team] if played[team] else np.nan
                prog = (played[r["home"]] + played[r["away"]]) / (2 * total)
                vals["h_progress"] = prog
                for k in ("h_safe", "a_safe", "h_title", "a_title"):
                    vals[f"late_{k}"] = vals[k] * prog
                out.loc[i, list(vals)] = list(vals.values())
            for r in day.itertuples():
                if pd.isna(r.ftr):
                    continue
                played[r.home] += 1
                played[r.away] += 1
                pts[r.home] += {"H": 3, "D": 1, "A": 0}[r.ftr]
                pts[r.away] += {"H": 0, "D": 1, "A": 3}[r.ftr]
    return out.astype(float)


def prepare(raw_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.concat([build(d) for d in raw_dirs if d.exists() and any(d.glob("*/*.csv"))], ignore_index=True)
    df = drop_forfeits(df.sort_values(["date", "league", "home"]).reset_index(drop=True))
    df = df[df["ftr"].notna()].reset_index(drop=True)
    # Ek liglerde bozuk oran satırları var (marj negatif ya da çok yüksek); marj ayrıştırması bunlarda çözümsüz
    for cols in (["pre_h", "pre_d", "pre_a"], ["odds_h", "odds_d", "odds_a"], ["pre_o25", "pre_u25"]):
        o = df[cols].astype(float)
        book = (1 / o).sum(axis=1, min_count=len(cols))
        bad = (o < 1.01).any(axis=1) | (book < 1.0) | (book > 1.3)
        df.loc[bad, cols] = np.nan
    X = build_features(df, Elo().run(df)).join(standings(df))
    return df, X


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", action="store_true", help="ek ligleri indir")
    parser.add_argument("--test-seasons", type=int, default=5)
    args = parser.parse_args()
    if args.indir:
        fetch_extra()
    df, X = prepare([download.RAW_DIR, RAW_EK])
    print("Lig başına maç:", df["league"].value_counts().to_dict())
    target = df["league"].isin(TARGET)
    seasons = sorted(df.loc[target, "season"].unique())[:-1][-args.test_seasons:]
    has_mk = X["mk_h"].notna()
    variants = {"A": (True, BASE), "B": (False, BASE), "C": (True, BASE + TABLE), "D": (False, BASE + TABLE)}
    preds = {k: np.full((len(df), 3), np.nan) for k in variants}
    for season in seasons:
        test = (target & (df["season"] == season)).to_numpy()
        for name, (only_target, feats) in variants.items():
            train = (df["season"] < season) & has_mk & (target if only_target else True)
            preds[name][test] = Model("lr", feats).fit(X[train], df.loc[train, "ftr"]).predict(X[test])
    mask = target & df["season"].isin(seasons) & np.isfinite(preds["A"]).all(axis=1)
    rows = []
    for name, p in preds.items():
        for lg, m in [("Hepsi", mask)] + [(lg, mask & (df["league"] == lg)) for lg in TARGET]:
            m = m.to_numpy()
            s = metrics.summary(p[m], df.loc[m, "ftr"])
            rows.append({"model": name, "lig": lg, **{k: round(float(v), 4) for k, v in s.items()}})
    table = pd.DataFrame(rows)
    per_season = pd.DataFrame({
        name: [float(metrics.rps(p[(mask & (df["season"] == s)).to_numpy()],
                                 df.loc[mask & (df["season"] == s), "ftr"]).mean()) for s in seasons]
        for name, p in preds.items()}, index=seasons)
    pd.set_option("display.width", 160)
    print(f"Test sezonları: {', '.join(seasons)} ({int(mask.sum())} maç)")
    print(table.to_string(index=False))
    print("\nSezonluk RPS (düşük daha iyi):\n" + per_season.round(4).to_string())
    print("\nA'ya göre fark (negatif = daha iyi):\n" + per_season.sub(per_season["A"], axis=0).round(4).to_string())
    goal_markets(df, X, target, seasons, mask)


def goal_markets(df, X, target, seasons, mask) -> None:
    """Üst/Alt 2.5 ve KG için aynı dört birleşim (günlük sistemdeki gibi tüm maçlarla eğitim)."""
    variants = {"A": (True, GOAL_FEATURES), "B": (False, GOAL_FEATURES),
                "C": (True, GOAL_FEATURES + TABLE), "D": (False, GOAL_FEATURES + TABLE)}
    m = mask.to_numpy()
    rows = []
    for tname, fn in TARGETS.items():
        y = fn(df).to_numpy()
        for name, (only_target, feats) in variants.items():
            p = np.full(len(df), np.nan)
            for season in seasons:
                test = (target & (df["season"] == season)).to_numpy()
                train = ((df["season"] < season) & (target if only_target else True)).to_numpy()
                p[test] = BinaryModel(feats).fit(X[train], y[train]).predict(X[test])
            s = scores(p[m], y[m])
            rows.append({"pazar": tname, "model": name, **{k: round(float(v), 4) for k, v in s.items()}})
    print("\nGol pazarları:\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
