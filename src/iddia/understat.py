"""Understat'tan Premier Lig maç bazında xG verisi.

Understat Süper Lig'i kapsamıyor; bu veri yalnızca E0 maçlarına eklenir.
Sezon dosyaları data/raw/understat altında saklanır. Biten sezonlar bir kez
indirilir, devam eden sezon her çalıştırmada yenilenir.
"""
import json
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from iddia import download

RAW_DIR = Path("data/raw/understat")
URL = "https://understat.com/getLeagueData/{league}/{year}"
LEAGUES = {"E0": "EPL"}
FIRST_YEAR = 2014

# Understat adı -> football-data adı (aynı yazılanlar listede yok)
TEAM_NAMES = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Queens Park Rangers": "QPR",
    "West Bromwich Albion": "West Brom",
    "Wolverhampton Wanderers": "Wolves",
}

COLUMNS = ["league", "date", "home", "away", "xg_h", "xg_a", "npxg_h", "npxg_a",
           "deep_h", "deep_a", "ppda_h", "ppda_a"]


def fetch(league: str, year: int) -> dict:
    headers = {"X-Requested-With": "XMLHttpRequest", "Referer": f"https://understat.com/league/{league}/{year}"}
    for attempt in range(4):
        try:
            r = requests.get(URL.format(league=league, year=year), headers=headers, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))


def download_all(today: date | None = None, raw_dir: Path = RAW_DIR) -> None:
    current = download.current_season_start(today or date.today())
    for code, league in LEAGUES.items():
        for year in range(FIRST_YEAR, current + 1):
            path = raw_dir / league / f"{year}.json"
            if path.exists() and year < current:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(fetch(league, year)))
            time.sleep(1)


def ppda(d: dict) -> float:
    """Rakibin pas sayısı / savunma aksiyonu: düşük değer yoğun pres demek."""
    return d["att"] / d["def"] if d["def"] else float("nan")


def load_season(path: Path, league: str) -> pd.DataFrame:
    data = json.loads(path.read_text())
    # Takım bazlı geçmiş: npxG, derin pas, PPDA (takım adı + başlama zamanı ile eşlenir)
    extra = {}
    for team in data["teams"].values():
        for h in team["history"]:
            extra[(team["title"], h["date"])] = (float(h["npxG"]), h["deep"], ppda(h["ppda"]))
    rows = []
    for m in data["dates"]:
        if not m["isResult"]:
            continue
        home, away = m["h"]["title"], m["a"]["title"]
        nh = extra.get((home, m["datetime"]), (float("nan"),) * 3)
        na = extra.get((away, m["datetime"]), (float("nan"),) * 3)
        rows.append({
            "league": league,
            "date": m["datetime"][:10],
            "home": TEAM_NAMES.get(home, home),
            "away": TEAM_NAMES.get(away, away),
            "xg_h": float(m["xG"]["h"]), "xg_a": float(m["xG"]["a"]),
            "npxg_h": nh[0], "npxg_a": na[0],
            "deep_h": nh[1], "deep_a": na[1],
            "ppda_h": nh[2], "ppda_a": na[2],
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def load(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    frames = [load_season(p, code) for code, league in LEAGUES.items()
              for p in sorted((raw_dir / league).glob("*.json"))]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def add_xg(matches: pd.DataFrame, xg: pd.DataFrame) -> pd.DataFrame:
    """Understat sütunlarını maç tablosuna ekler. Understat xG'si varsa football-data'nınkinin
    yerine geçer, böylece tüm Premier Lig geçmişi tek kaynaktan gelir.

    Eşleme lig + takım çifti ile, tarih farkı en fazla bir gün olacak şekilde yapılır
    (saat dilimi ve ertelenen maç kayıtları yüzünden)."""
    out = matches.copy()
    for c in COLUMNS[4:]:
        if c not in out:
            out[c] = float("nan")
    if xg.empty:
        return out
    key = ["league", "home", "away"]
    left = out.reset_index()[["index", "date", *key]]
    cand = left.merge(xg, on=key, suffixes=("", "_us"))
    cand = cand[(cand["date"] - cand["date_us"]).abs() <= pd.Timedelta(days=1)]
    cand = cand.drop_duplicates("index").set_index("index")
    out.loc[cand.index, COLUMNS[4:]] = cand[COLUMNS[4:]]
    return out


def main() -> None:
    download_all()
    df = load()
    print(f"{len(df)} maç xG verisi -> {RAW_DIR}")


if __name__ == "__main__":
    main()
