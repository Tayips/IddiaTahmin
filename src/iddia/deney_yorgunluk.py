"""Deney: Avrupa kupası yorgunluğu tahmine katkı sağlıyor mu?

Model son lig maçından bu yana geçen günü (h_rest, a_rest) zaten kullanıyor; burada
Avrupa maçları eklenir. Özellikler (her takım için, maç ÖNCESİ bilinenler):
  eu_prev  son Avrupa kupası maçından bu yana geçen gün (en fazla 10; yoksa 10)
  eu_next  sonraki Avrupa kupası maçına kalan gün (en fazla 10; yoksa 10; fikstür önceden bilinir)
  rest_all son maçından (lig ya da Avrupa) bu yana geçen gün (en fazla 14)

Avrupa maçları football-data.org'dan alınır (FOOTBALL_DATA_ORG_KEY). Test her zaman
Premier Lig ve Süper Lig'in son 5 tam sezonudur; modeller her sezon için yalnızca
önceki sezonlarla eğitilir. Çalıştırmak için: python -m iddia.deney_yorgunluk [--indir]
"""
import argparse
import os
import re
import time

import numpy as np
import pandas as pd
import requests

from iddia import deney_ek, download, metrics
from iddia.goals import TARGETS, BinaryModel, scores
from iddia.features import GOAL_FEATURES
from iddia.models.ml import Model
from iddia.oddsapi import similarity

ORG = "https://api.football-data.org/v4/competitions/{code}/matches"
CUPS = ["CL", "EL", "ECL"]
TARGET = deney_ek.TARGET
EU = ["h_eu_prev", "a_eu_prev", "h_eu_next", "a_eu_next", "h_rest_all", "a_rest_all"]


def fetch_cups(first: int, last: int) -> pd.DataFrame:
    key = os.environ.get("FOOTBALL_DATA_ORG_KEY")
    rows = []
    if not key:
        print("FOOTBALL_DATA_ORG_KEY yok, Avrupa maçları alınamadı")
        return pd.DataFrame({"team": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]"),
                             "cup": pd.Series(dtype=str)})
    for code in CUPS:
        for year in range(first, last + 1):
            try:
                r = requests.get(ORG.format(code=code), params={"season": year},
                                 headers={"X-Auth-Token": key}, timeout=30)
                status = r.status_code
                if status == 200:
                    ms = r.json().get("matches", [])
                    for m in ms:
                        d = pd.Timestamp(m["utcDate"]).tz_convert(None).normalize()
                        for side in ("homeTeam", "awayTeam"):
                            if (m.get(side) or {}).get("name"):
                                rows.append({"team": m[side]["name"], "date": d, "cup": code})
                    print(f"{code} {year}: {len(ms)} maç")
                else:
                    print(f"{code} {year}: {status}")
            except (requests.RequestException, ValueError, KeyError) as e:
                print(f"{code} {year}: {type(e).__name__}")
            time.sleep(6.5)                     # ücretsiz plan: dakikada 10 istek
    return pd.DataFrame(rows, columns=["team", "date", "cup"]).astype({"date": "datetime64[ns]"})


def map_names(cups: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """Avrupa maçlarındaki adları lig adlarına çevirir; yalnızca E0 ve T1 takımları kalır."""
    mapping = {}
    for name in cups["team"].unique():
        clean = re.sub(r"\s+(A\.Ş\.|AŞ)$", "", name)
        best = max(teams, key=lambda t: similarity(clean, t))
        if similarity(clean, best) >= 0.85:
            mapping[name] = best
    print("Eşleşen Avrupa takımları:", dict(sorted(mapping.items(), key=lambda x: x[1])))
    out = cups.assign(team=cups["team"].map(mapping)).dropna(subset=["team"])
    return out.drop_duplicates(["team", "date"])


def days_since(dates: pd.Series, events: np.ndarray, before: bool) -> np.ndarray:
    """Her tarih için en yakın önceki (before) ya da sonraki olaya gün farkı; yoksa NaN."""
    d = dates.to_numpy(dtype="datetime64[D]")
    e = np.sort(events.astype("datetime64[D]"))
    if len(e) == 0:
        return np.full(len(d), np.nan)
    if before:
        i = np.searchsorted(e, d, side="left") - 1
        ok = i >= 0
        out = np.full(len(d), np.nan)
        out[ok] = (d[ok] - e[i[ok]]).astype(float)
    else:
        i = np.searchsorted(e, d, side="right")
        ok = i < len(e)
        out = np.full(len(d), np.nan)
        out[ok] = (e[i[ok]] - d[ok]).astype(float)
    return out


def fatigue(df: pd.DataFrame, cups: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index, columns=EU, dtype=float)
    games = pd.concat([df[["date", "home"]].rename(columns={"home": "team"}),
                       df[["date", "away"]].rename(columns={"away": "team"})])
    by_team = games.groupby("team")["date"].apply(lambda s: s.to_numpy())
    eu_by_team = cups.groupby("team")["date"].apply(lambda s: s.to_numpy())
    has_eu = df["league"].isin(TARGET)
    for side in ("h", "a"):
        col = "home" if side == "h" else "away"
        for team, idx in df.groupby(col).groups.items():
            dates = df.loc[idx, "date"]
            eu = eu_by_team.get(team, np.array([], dtype="datetime64[ns]"))
            prev = np.nan_to_num(np.minimum(days_since(dates, eu, True), 10), nan=10)
            nxt = np.nan_to_num(np.minimum(days_since(dates, eu, False), 10), nan=10)
            allg = np.concatenate([by_team[team], eu]) if len(eu) else by_team[team]
            out.loc[idx, f"{side}_eu_prev"] = prev
            out.loc[idx, f"{side}_eu_next"] = nxt
            out.loc[idx, f"{side}_rest_all"] = np.minimum(days_since(dates, allg, True), 14)
    # Ek liglerde Avrupa verisi yok: bilinmiyor say (eğitimde medyanla doldurulur)
    out.loc[~has_eu, EU] = np.nan
    return out.astype(float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", action="store_true", help="ek ligleri indir")
    args = parser.parse_args()
    if args.indir:
        deney_ek.fetch_extra()
    df, X = deney_ek.prepare([download.RAW_DIR, deney_ek.RAW_EK])
    target = df["league"].isin(TARGET)
    seasons = sorted(df.loc[target, "season"].unique())[:-1][-5:]
    first = int("20" + seasons[0][:2]) - 2
    cups = fetch_cups(first, download.current_season_start())
    teams = sorted(set(df.loc[target, "home"]) | set(df.loc[target, "away"]))
    cups = map_names(cups, teams)
    print("Avrupa maçı olan takım-gün:", len(cups), cups.groupby(cups["date"].dt.year).size().to_dict())
    X = X.join(fatigue(df, cups))
    for c in EU:
        print(f"{c}: ortalama {X.loc[target, c].mean():.2f}")

    base = deney_ek.BASE
    has_mk = X["mk_h"].notna()
    variants = {"A": (True, base), "A+Avrupa": (True, base + EU), "B": (False, base), "B+Avrupa": (False, base + EU)}
    preds = {k: np.full((len(df), 3), np.nan) for k in variants}
    for season in seasons:
        test = (target & (df["season"] == season)).to_numpy()
        for name, (only_target, feats) in variants.items():
            train = (df["season"] < season) & has_mk & (target if only_target else True)
            preds[name][test] = Model("lr", feats).fit(X[train], df.loc[train, "ftr"]).predict(X[test])
    mask = target & df["season"].isin(seasons) & np.isfinite(preds["A"]).all(axis=1)
    # Avrupa'da oynayan takımların maçları ayrıca: etkinin görülmesi gereken yer
    eu_game = mask & ((X["h_eu_prev"] < 10) | (X["a_eu_prev"] < 10) | (X["h_eu_next"] < 10) | (X["a_eu_next"] < 10))
    rows = []
    for name, p in preds.items():
        for lg, m in [("Hepsi", mask), ("Avrupa haftası", eu_game)] + [(lg, mask & (df["league"] == lg)) for lg in TARGET]:
            m = m.to_numpy()
            s = metrics.summary(p[m], df.loc[m, "ftr"])
            rows.append({"model": name, "grup": lg, **{k: round(float(v), 4) for k, v in s.items()}})
    pd.set_option("display.width", 180)
    print(f"Test sezonları: {', '.join(seasons)} ({int(mask.sum())} maç, {int(eu_game.sum())} Avrupa haftası)")
    print(pd.DataFrame(rows).to_string(index=False))
    per_season = pd.DataFrame({
        name: [float(metrics.rps(p[(mask & (df["season"] == s)).to_numpy()],
                                 df.loc[mask & (df["season"] == s), "ftr"]).mean()) for s in seasons]
        for name, p in preds.items()}, index=seasons)
    print("\nB'ye göre sezonluk RPS farkı (negatif = daha iyi):\n"
          + per_season.sub(per_season["B"], axis=0).round(4).to_string())

    gv = {"B": GOAL_FEATURES, "B+Avrupa": GOAL_FEATURES + EU}
    m = mask.to_numpy()
    grows = []
    for tname, fn in TARGETS.items():
        y = fn(df).to_numpy()
        for name, feats in gv.items():
            p = np.full(len(df), np.nan)
            for season in seasons:
                test = (target & (df["season"] == season)).to_numpy()
                train = (df["season"] < season).to_numpy()
                p[test] = BinaryModel(feats).fit(X[train], y[train]).predict(X[test])
            grows.append({"pazar": tname, "model": name, **{k: round(float(v), 4) for k, v in scores(p[m], y[m]).items()}})
    print("\nGol pazarları:\n" + pd.DataFrame(grows).to_string(index=False))


if __name__ == "__main__":
    main()
