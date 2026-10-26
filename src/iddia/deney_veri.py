"""Deney: football-data.co.uk dosyalarında olup henüz kullanmadığımız veriler katkı sağlıyor mu?

  AH       Asya handikapı çizgisi ve oranı (maç öncesi)
  SAHA     iç saha / deplasman formu ayrı ayrı (ev sahibinin evdeki, deplasmanın dışarıdaki maçları)
  KART     kırmızı kart formu (takımın gördüğü ve rakibine gösterilen)
  IY       ilk yarı gol formu
  KORNER   korner formu

Test son 5 tam sezonun Premier Lig ve Süper Lig maçlarıdır; modeller her test sezonu
için yalnızca önceki sezonlarla eğitilir. Çalıştırmak için: python -m iddia.deney_veri
"""
from pathlib import Path

import numpy as np
import pandas as pd

from iddia import features, metrics
from iddia.build import RAW_DIR, parse_dates, read_csv, team_name
from iddia.deney_ek import BASE, prepare
from iddia.features import GOAL_FEATURES, HALF_LIFE
from iddia.goals import TARGETS, BinaryModel, scores
from iddia.models.ml import Model

AH_CANDIDATES = [("AHh", "AvgAHH", "AvgAHA"), ("BbAHh", "BbAvAHH", "BbAvAHA"), ("AHh", "B365AHH", "B365AHA")]
EXTRA = ["hthg", "htag", "hr", "ar", "ah_line", "ah_oh", "ah_oa"]

GROUPS = {
    "AH": ["ah_line", "ah_p"],
    "SAHA": ["hv_gf", "hv_ga", "hv_stf", "hv_sta", "av_gf", "av_ga", "av_stf", "av_sta"],
    "KART": ["h_rcf", "h_rca", "a_rcf", "a_rca"],
    "IY": ["h_htgf", "h_htga", "a_htgf", "a_htga"],
    "KORNER": ["h_cf", "h_ca", "a_cf", "a_ca"],
}


def load_extra(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    frames = []
    for p in sorted(raw_dir.glob("*/*.csv")):
        d = read_csv(p).dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        out = pd.DataFrame({"date": parse_dates(d["Date"]), "league": p.parent.name, "season": p.stem,
                            "home": d["HomeTeam"].map(team_name), "away": d["AwayTeam"].map(team_name)})
        for src, dst in {"HTHG": "hthg", "HTAG": "htag", "HR": "hr", "AR": "ar"}.items():
            out[dst] = pd.to_numeric(d[src], errors="coerce") if src in d else np.nan
        out[["ah_line", "ah_oh", "ah_oa"]] = np.nan
        for cols in AH_CANDIDATES:
            if not set(cols) <= set(d.columns):
                continue
            part = d[list(cols)].apply(pd.to_numeric, errors="coerce").to_numpy()
            miss = out["ah_line"].isna().to_numpy() & np.isfinite(part).all(axis=1)
            out.loc[miss, ["ah_line", "ah_oh", "ah_oa"]] = part[miss]
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def form(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """features.team_form ile aynı hesap, başka istatistiklerle."""
    old = features.TEAM_STATS
    features.TEAM_STATS = stats
    try:
        out = features.team_form(df)
    finally:
        features.TEAM_STATS = old
    return out[[c for c in out.columns if not c.endswith("_rest")]]


def venue_form(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    stats = {"gf": ("fthg", "ftag"), "ga": ("ftag", "fthg"), "stf": ("hst", "ast"), "sta": ("ast", "hst")}
    order = df.sort_values("date").index
    for side, col in (("hv", "home"), ("av", "away")):
        d = df.loc[order]
        g = d.groupby(["league", col], sort=False)
        for stat, (hcol, acol) in stats.items():
            src = hcol if side == "hv" else acol
            out.loc[order, f"{side}_{stat}"] = g[src].transform(
                lambda s: s.shift(1).ewm(halflife=HALF_LIFE, ignore_na=True).mean()).to_numpy()
    return out


def extra_features(df: pd.DataFrame) -> pd.DataFrame:
    out = venue_form(df)
    out = out.join(form(df, {"rcf": ("hr", "ar"), "rca": ("ar", "hr")}))
    out = out.join(form(df, {"htgf": ("hthg", "htag"), "htga": ("htag", "hthg")}))
    out = out.join(form(df, {"cf": ("hc", "ac"), "ca": ("ac", "hc")}))
    out["ah_line"] = df["ah_line"]
    inv_h, inv_a = 1 / df["ah_oh"], 1 / df["ah_oa"]
    out["ah_p"] = inv_h / (inv_h + inv_a)
    return out


def main() -> None:
    df, X = prepare([RAW_DIR])
    df = df.merge(load_extra(), on=["league", "season", "date", "home", "away"], how="left").set_axis(df.index)
    print("Ek veri dolu oranı:", df[EXTRA].notna().mean().round(3).to_dict())
    X = X.join(extra_features(df))
    seasons = sorted(df["season"].unique())[:-1][-5:]
    has_mk = X["mk_h"].notna()
    variants = {"Şu anki": BASE, **{f"+{k}": BASE + v for k, v in GROUPS.items()},
                "+Hepsi": BASE + sum(GROUPS.values(), [])}
    preds = {k: np.full((len(df), 3), np.nan) for k in variants}
    for season in seasons:
        test = (df["season"] == season).to_numpy()
        train = (df["season"] < season) & has_mk
        for name, feats in variants.items():
            preds[name][test] = Model("lr", feats).fit(X[train], df.loc[train, "ftr"]).predict(X[test])
    mask = (df["season"].isin(seasons) & np.isfinite(preds["Şu anki"]).all(axis=1)).to_numpy()
    rows = []
    for name, p in preds.items():
        s = metrics.summary(p[mask], df.loc[mask, "ftr"])
        per = [float(metrics.rps(p[mask & (df["season"] == s_).to_numpy()],
                                 df.loc[mask & (df["season"] == s_).to_numpy(), "ftr"]).mean()) for s_ in seasons]
        rows.append({"model": name, **{k: round(float(v), 4) for k, v in s.items()}, "sezon_rps": per})
    base_per = np.array(rows[0]["sezon_rps"])
    for r in rows:
        r["daha_iyi_sezon"] = int((np.array(r.pop("sezon_rps")) < base_per - 1e-6).sum())
    pd.set_option("display.width", 180)
    print(f"Test sezonları: {', '.join(seasons)} ({int(mask.sum())} maç)")
    print("1-X-2:\n" + pd.DataFrame(rows).to_string(index=False))

    grows = []
    for tname, fn in TARGETS.items():
        y = fn(df).to_numpy()
        gv = {"Şu anki": GOAL_FEATURES, **{f"+{k}": GOAL_FEATURES + v for k, v in GROUPS.items()}}
        for name, feats in gv.items():
            p = np.full(len(df), np.nan)
            for season in seasons:
                test = (df["season"] == season).to_numpy()
                train = (df["season"] < season).to_numpy()
                p[test] = BinaryModel(feats).fit(X[train], y[train]).predict(X[test])
            grows.append({"pazar": tname, "model": name,
                          **{k: round(float(v), 4) for k, v in scores(p[mask], y[mask]).items()}})
    print("\nGol pazarları:\n" + pd.DataFrame(grows).to_string(index=False))


if __name__ == "__main__":
    main()
