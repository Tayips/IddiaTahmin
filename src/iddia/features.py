"""Maç öncesi özellikler. Her satırın özellikleri yalnızca o maçtan önceki maçlardan hesaplanır.

Sonucu henüz belli olmayan (gelecek) maçlar da aynı tabloya eklenebilir: skor
sütunları boş olduğundan kimsenin formunu etkilemezler.
"""
import numpy as np
import pandas as pd

from iddia import odds as odds_mod

HALF_LIFE = 6          # form ortalamalarında yarı ömür (maç)
REST_CAP = 14          # dinlenme günü bu değerde kesilir (yaz arası vb.)

# Takım bakışından istatistik: (ev sahibi sütunu, deplasman sütunu)
TEAM_STATS = {
    "gf": ("fthg", "ftag"),
    "ga": ("ftag", "fthg"),
    "sf": ("hs", "as_"),
    "sa": ("as_", "hs"),
    "stf": ("hst", "ast"),
    "sta": ("ast", "hst"),
}

FORM_FEATURES = [f"{side}_{stat}" for side in ("h", "a") for stat in TEAM_STATS]
FEATURES_NO_MARKET = ["elo_diff", "is_t1", *FORM_FEATURES, "h_rest", "a_rest",
                      "gd_form_diff", "sot_form_diff"]
MARKET_FEATURES = ["mk_h", "mk_a", "mk_logit"]
GOAL_FEATURES = FEATURES_NO_MARKET + MARKET_FEATURES + ["mk_o25", "goal_form_sum"]


def team_form(df: pd.DataFrame) -> pd.DataFrame:
    """Her maç için iki takımın maç öncesi üstel ağırlıklı ortalamaları ve dinlenme günü."""
    rows = []
    for side, other in (("h", "a"), ("a", "h")):
        team = df["home"] if side == "h" else df["away"]
        part = pd.DataFrame({"idx": df.index, "league": df["league"], "team": team, "date": df["date"], "side": side})
        for stat, (hcol, acol) in TEAM_STATS.items():
            part[stat] = df[hcol] if side == "h" else df[acol]
        rows.append(part)
    long = pd.concat(rows).sort_values(["date", "idx"]).reset_index(drop=True)
    g = long.groupby(["league", "team"], sort=False)
    out = pd.DataFrame({"idx": long["idx"], "side": long["side"]})
    for stat in TEAM_STATS:
        # shift: maçın kendisi hariç; ewm boş değerleri (oynanmamış maç, eksik istatistik) atlar
        out[stat] = g[stat].transform(lambda s: s.shift(1).ewm(halflife=HALF_LIFE, ignore_na=True).mean())
    prev = g["date"].shift(1)
    out["rest"] = (long["date"] - prev).dt.days.clip(upper=REST_CAP)
    wide = out.pivot(index="idx", columns="side")
    wide.columns = [f"{side}_{stat}" for stat, side in wide.columns]
    return wide.reindex(df.index)


def market(df: pd.DataFrame, cols=("pre_h", "pre_d", "pre_a")) -> pd.DataFrame:
    """Maç öncesi oranlardan marjı temizlenmiş olasılıklar."""
    o = df[list(cols)].to_numpy(dtype=float)
    p = np.full(o.shape, np.nan)
    ok = np.isfinite(o).all(axis=1) & (o > 1).all(axis=1)
    if ok.any():
        p[ok] = odds_mod.shin(o[ok])
    out = pd.DataFrame({"mk_h": p[:, 0], "mk_d": p[:, 1], "mk_a": p[:, 2]}, index=df.index)
    out["mk_logit"] = np.log(out["mk_h"] / out["mk_a"])
    return out


def market_ou(df: pd.DataFrame) -> pd.Series:
    """Üst 2.5 için marjı orantılı temizlenmiş olasılık."""
    if "pre_o25" not in df:
        return pd.Series(np.nan, index=df.index, name="mk_o25")
    inv_o, inv_u = 1 / df["pre_o25"], 1 / df["pre_u25"]
    return (inv_o / (inv_o + inv_u)).rename("mk_o25")


def build_features(df: pd.DataFrame, elo_diff: pd.Series) -> pd.DataFrame:
    feats = team_form(df)
    feats["elo_diff"] = elo_diff / 100.0
    feats["is_t1"] = (df["league"] == "T1").astype(float)
    feats["gd_form_diff"] = (feats["h_gf"] - feats["h_ga"]) - (feats["a_gf"] - feats["a_ga"])
    feats["sot_form_diff"] = (feats["h_stf"] - feats["h_sta"]) - (feats["a_stf"] - feats["a_sta"])
    feats["goal_form_sum"] = feats["h_gf"] + feats["h_ga"] + feats["a_gf"] + feats["a_ga"]
    return feats.join(market(df)).join(market_ou(df))
