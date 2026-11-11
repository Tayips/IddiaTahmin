"""Faz 2 karşılaştırması: özellik tabanlı modelleri Elo ve piyasa ile kıyaslar.

Her test sezonu için modeller yalnızca önceki sezonlarla eğitilir.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from iddia import metrics
from iddia.build import build
from iddia.evaluate import drop_forfeits
from iddia.features import FEATURES_NO_MARKET, INJURY_FEATURES, MARKET_FEATURES, XG_FEATURES, build_features
from iddia.models.baseline import closing_odds
from iddia.models.elo import Elo, OrderedLogit
from iddia.models.ml import Model

REPORT_DIR = Path("reports")

CANDIDATES = {
    "lr_form": ("lr", FEATURES_NO_MARKET),
    "gb_form": ("gb", FEATURES_NO_MARKET),
    "lr_form_piyasa": ("lr", FEATURES_NO_MARKET + MARKET_FEATURES),
    "gb_form_piyasa": ("gb", FEATURES_NO_MARKET + MARKET_FEATURES),
}


def prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = drop_forfeits(build())
    df = df[df["ftr"].notna()].reset_index(drop=True)
    elo_diff = Elo().run(df)
    return df, build_features(df, elo_diff)


def run(df: pd.DataFrame, X: pd.DataFrame, test_seasons: list[str]) -> dict[str, np.ndarray]:
    preds = {name: np.full((len(df), 3), np.nan) for name in ["elo", "piyasa_mac_oncesi", *CANDIDATES]}
    for season in test_seasons:
        train = df["season"] < season
        test = df["season"] == season
        for league in df["league"].unique():
            tr, te = train & (df["league"] == league), test & (df["league"] == league)
            logit = OrderedLogit().fit(X.loc[tr, "elo_diff"].to_numpy() * 100, df.loc[tr, "ftr"])
            preds["elo"][te.to_numpy()] = logit.predict(X.loc[te, "elo_diff"].to_numpy() * 100)
        has_mk = X["mk_h"].notna()
        for name, (kind, feats) in CANDIDATES.items():
            tr = train & (has_mk if "mk_h" in feats else True)
            model = Model(kind, feats).fit(X[tr], df.loc[tr, "ftr"])
            preds[name][test.to_numpy()] = model.predict(X[test])
    preds["piyasa_mac_oncesi"] = X[["mk_h", "mk_d", "mk_a"]].to_numpy()
    preds["piyasa_kapanis"] = closing_odds(df)
    return preds


def ablation(df: pd.DataFrame, X: pd.DataFrame, test_seasons: list[str], rows: pd.Series,
             base: list[str], extra: list[str]) -> dict[str, np.ndarray]:
    """Aynı maçlarda (rows) eğitilip test edilen iki modeli kıyaslar: base ve base + extra.
    Böylece fark yalnızca eklenen özellikten gelir."""
    preds = {name: np.full((len(df), 3), np.nan) for name in ("temel", "ek")}
    for season in test_seasons:
        tr, te = rows & (df["season"] < season), rows & (df["season"] == season)
        if not tr.any() or not te.any():
            continue
        for name, feats in (("temel", base), ("ek", base + extra)):
            preds[name][te.to_numpy()] = Model("lr", feats).fit(X[tr], df.loc[tr, "ftr"]).predict(X[te])
    return preds


def ablation_table(df, preds, mask) -> pd.DataFrame:
    mask = mask & np.isfinite(preds["temel"]).all(axis=1) & np.isfinite(preds["ek"]).all(axis=1)
    out = []
    for name, p in preds.items():
        s = metrics.summary(p[mask.to_numpy()], df.loc[mask, "ftr"])
        out.append({"model": name, **s})
    return pd.DataFrame(out)


def extras(df: pd.DataFrame, X: pd.DataFrame, test_seasons: list[str]) -> pd.DataFrame:
    """xG (Premier Lig) ve sakat/cezalı oyuncu katkısı, piyasalı ve piyasasız."""
    has_mk = X["mk_h"].notna()
    tables = []
    xg_rows = (df["league"] == "E0") & X["h_xgf"].notna() & X["a_xgf"].notna()
    inj_rows = df["out_h"].notna()
    for label, rows, extra in (("xG", xg_rows, XG_FEATURES), ("sakat/cezalı", inj_rows, INJURY_FEATURES)):
        if not rows.any():
            continue
        seasons = [s for s in test_seasons if (rows & (df["season"] < s)).any()]
        for market_label, base in (("form", FEATURES_NO_MARKET), ("form + piyasa", FEATURES_NO_MARKET + MARKET_FEATURES)):
            r = rows & has_mk if "mk_h" in base else rows
            t = ablation_table(df, ablation(df, X, seasons, r, base, extra), r & df["season"].isin(seasons))
            t.insert(0, "temel_model", market_label)
            t.insert(0, "ozellik", label)
            tables.append(t)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def table(df, preds, mask) -> pd.DataFrame:
    rows = []
    for group, gmask in [("Hepsi", mask)] + [(lg, mask & (df["league"] == lg)) for lg in sorted(df["league"].unique())]:
        for name, p in preds.items():
            s = metrics.summary(p[gmask.to_numpy()], df.loc[gmask, "ftr"])
            s["ece"] = metrics.ece(p[gmask.to_numpy()], df.loc[gmask, "ftr"])
            rows.append({"lig": group, "model": name, **s})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-seasons", type=int, default=5)
    args = parser.parse_args()
    df, X = prepare()
    seasons = sorted(df["season"].unique())
    test_seasons = seasons[-args.test_seasons - 1:-1]   # devam eden sezon hariç son N tam sezon
    preds = run(df, X, test_seasons)
    # Tüm modellerin tahmini ve iki oran da olan maçlarda kıyas
    ok = df["season"].isin(test_seasons)
    for p in preds.values():
        ok &= np.isfinite(p).all(axis=1)
    t = table(df, preds, ok)
    REPORT_DIR.mkdir(exist_ok=True)
    t.to_csv(REPORT_DIR / "faz2_skorlar.csv", index=False)
    pd.set_option("display.width", 140)
    print(f"Test sezonları: {', '.join(test_seasons)}  ({int(ok.sum())} maç)")
    print(t.round(4).to_string(index=False))
    e = extras(df, X, test_seasons)
    if not e.empty:
        e.to_csv(REPORT_DIR / "faz3_ek_veri.csv", index=False)
        print("\nEk veri katkısı (aynı maçlarda, temel vs temel + ek):")
        print(e.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
