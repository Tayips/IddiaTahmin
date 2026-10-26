"""football-data.org: Premier Lig maç saatleri, ertelemeler ve hızlı sonuçlar.

football-data.co.uk sonuçları birkaç günde bir günceller ve fikstür saatlerini
sonradan değişse de düzeltmeyebilir. Bu modül sezonun tüm Premier Lig maçlarını
tek istekle çeker. Anahtar FOOTBALL_DATA_ORG_KEY ortam değişkeninden okunur;
yoksa ya da istek başarısız olursa sistem bu kaynak olmadan çalışmaya devam eder.
"""
import os
import re
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from iddia.build import team_name
from iddia.log import warn

URL = "https://api.football-data.org/v4/competitions/PL/matches"
ENV_KEY = "FOOTBALL_DATA_ORG_KEY"
UK = ZoneInfo("Europe/London")
STOPPED = {"POSTPONED", "CANCELLED", "SUSPENDED"}

# football-data.org adı (FC/AFC ekleri atıldıktan sonra) → football-data.co.uk adı
NAMES = {
    "Brighton & Hove Albion": "Brighton",
    "Cardiff City": "Cardiff",
    "Huddersfield Town": "Huddersfield",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Luton Town": "Luton",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Norwich City": "Norwich",
    "Nottingham Forest": "Nott'm Forest",
    "Queens Park Rangers": "QPR",
    "Stoke City": "Stoke",
    "Swansea City": "Swansea",
    "Tottenham Hotspur": "Tottenham",
    "West Bromwich Albion": "West Brom",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Coventry City": "Coventry",
    "Sheffield Wednesday": "Sheffield Weds",
}


def to_local_name(name: str) -> str:
    name = re.sub(r"^AFC\s+|\s+A?FC$", "", name.strip())
    return team_name(NAMES.get(name, name))


def parse(payload: dict) -> pd.DataFrame:
    rows = []
    for m in payload.get("matches", []):
        full = (m.get("score") or {}).get("fullTime") or {}
        rows.append({
            "home": to_local_name(m["homeTeam"]["name"]),
            "away": to_local_name(m["awayTeam"]["name"]),
            "utc": m["utcDate"],
            "org_status": m["status"],
            "org_fthg": full.get("home"),
            "org_ftag": full.get("away"),
        })
    df = pd.DataFrame(rows, columns=["home", "away", "utc", "org_status", "org_fthg", "org_ftag"])
    utc = pd.to_datetime(df["utc"], utc=True)
    df["league"] = "E0"
    df["date"] = utc.dt.tz_convert(UK).dt.strftime("%Y-%m-%d")
    df["kickoff"] = utc.dt.strftime("%Y-%m-%dT%H:%MZ")
    # Bir sezonda her ev sahibi–deplasman eşleşmesi bir kez oynanır.
    return df.drop(columns="utc").drop_duplicates(["home", "away"], keep="last").reset_index(drop=True)


def fetch(season_start: int, key: str | None = None) -> pd.DataFrame | None:
    key = key or os.environ.get(ENV_KEY)
    if not key:
        warn("football-data.org", f"{ENV_KEY} yok, atlandı")
        return None
    try:
        r = requests.get(URL, params={"season": season_start}, headers={"X-Auth-Token": key}, timeout=30)
        r.raise_for_status()
        df = parse(r.json())
    except (requests.RequestException, ValueError, KeyError) as e:
        warn("football-data.org", f"okunamadı, atlandı: {type(e).__name__}")
        return None
    print(f"football-data.org: {len(df)} Premier Lig maçı")
    return df


def apply_to_fixtures(fixtures: pd.DataFrame, org: pd.DataFrame | None) -> pd.DataFrame:
    """Premier Lig fikstüründe saat ve tarihi günceller, ertelenen maçları çıkarır."""
    if org is None or org.empty:
        return fixtures
    df = fixtures.merge(org[["home", "away", "date", "kickoff", "org_status"]].assign(league="E0"),
                        on=["league", "home", "away"], how="left", suffixes=("", "_org"))
    known = df["org_status"].notna() & ~df["org_status"].isin(STOPPED)
    df.loc[known, "date"] = df.loc[known, "date_org"]
    df.loc[known, "kickoff"] = df.loc[known, "kickoff_org"]
    df = df[~df["org_status"].isin(STOPPED)]
    return df[fixtures.columns].reset_index(drop=True)


def fill_results(df: pd.DataFrame, org: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.Series]:
    """Henüz sonucu gelmemiş Premier Lig maçlarına sonucu ekler, ertelenenleri işaretler.

    Sonuç yalnızca tarih tuttuğunda alınır; tarihi değişmiş ya da durdurulmuş bir
    maçın eski tahmini ertelendi sayılır.
    """
    if org is None or org.empty:
        return df, pd.Series(False, index=df.index)
    o = org.rename(columns={"date": "org_date"})[["home", "away", "org_date", "org_status", "org_fthg", "org_ftag"]]
    m = df.merge(o.assign(league="E0"), on=["league", "home", "away"], how="left").set_axis(df.index)
    open_ = m["ftr"].isna() & m["org_status"].notna()
    same_day = m["org_date"] == m["date"]
    finished = open_ & same_day & (m["org_status"] == "FINISHED") & m["org_fthg"].notna()
    h, a = m.loc[finished, "org_fthg"], m.loc[finished, "org_ftag"]
    m.loc[finished, "fthg"] = h
    m.loc[finished, "ftag"] = a
    m["ftr"] = m["ftr"].astype(object)
    m.loc[finished, "ftr"] = np.select([h > a, h < a], ["H", "A"], default="D")
    moved = open_ & ~finished & (~same_day | m["org_status"].isin(STOPPED))
    return m[df.columns], moved
