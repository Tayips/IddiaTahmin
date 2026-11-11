"""API-Football'dan sakat ve cezalı oyuncu listesi.

Anahtar API_FOOTBALL_KEY ortam değişkeninden okunur; koda veya depoya yazılmaz.
Ücretsiz plan günde 100 istek verir, bu yüzden her yanıt data/raw/apifootball
altında saklanır. Biten sezonlar bir kez indirilir.

Sakatlık verisi her maç için "o maçta eksik" (Missing Fixture) ve "şüpheli"
(Questionable) oyuncuları listeler. Kadro (ilk 11) maçtan yaklaşık bir saat önce
açıklandığı için günlük tahmin saatlerinde henüz yoktur; modelde kullanılmaz.
"""
import argparse
import json
import os
import time
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from iddia import download

BASE = "https://v3.football.api-sports.io"
RAW_DIR = Path("data/raw/apifootball")
LEAGUES = {"T1": 203, "E0": 39}
COLUMNS = ["league", "date", "home", "away", "out_h", "out_a", "doubt_h", "doubt_a"]


class NoKey(RuntimeError):
    pass


def get(endpoint: str, params: dict) -> dict:
    key = os.environ.get("API_FOOTBALL_KEY")
    if not key:
        raise NoKey("API_FOOTBALL_KEY ortam değişkeni tanımlı değil")
    for attempt in range(4):
        r = requests.get(f"{BASE}/{endpoint}", params=params, headers={"x-apisports-key": key}, timeout=60)
        if r.status_code == 429 and attempt < 3:
            time.sleep(2 ** (attempt + 3))
            continue
        r.raise_for_status()
        data = r.json()
        if data.get("errors"):
            raise RuntimeError(f"API-Football {endpoint} {params}: {data['errors']}")
        return data
    raise RuntimeError("API-Football istek sınırı")


def cached(endpoint: str, params: dict, refresh: bool, raw_dir: Path = RAW_DIR) -> list:
    name = "_".join(f"{k}{v}" for k, v in sorted(params.items()))
    path = raw_dir / endpoint / f"{name}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    data = get(endpoint, params)
    rows = data["response"]
    # Sayfalı uç noktalar (ör. /injuries büyük sezonlarda)
    page, total = data.get("paging", {}).get("current", 1), data.get("paging", {}).get("total", 1)
    while page < total:
        page += 1
        rows += get(endpoint, {**params, "page": page})["response"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
    return rows


def download_all(seasons: list[int], today: date | None = None, raw_dir: Path = RAW_DIR) -> None:
    """Planın açmadığı sezonlar (ücretsiz planda eski sezonlar) atlanır ve yazdırılır."""
    current = download.current_season_start(today or date.today())
    for league_id in LEAGUES.values():
        for season in seasons:
            try:
                for endpoint in ("fixtures", "injuries"):
                    rows = cached(endpoint, {"league": league_id, "season": season}, season >= current, raw_dir)
                    print(f"lig {league_id} sezon {season} {endpoint}: {len(rows)} kayıt")
            except RuntimeError as e:
                if isinstance(e, NoKey):
                    raise
                print("ATLANDI:", e)


def fixtures_frame(rows: list, league: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "fixture": r["fixture"]["id"],
        "league": league,
        "date": pd.Timestamp(r["fixture"]["date"]).tz_convert("Europe/London").tz_localize(None).normalize(),
        "home_api": r["teams"]["home"]["name"],
        "away_api": r["teams"]["away"]["name"],
        "home_id": r["teams"]["home"]["id"],
        "away_id": r["teams"]["away"]["id"],
        "fthg": r["goals"]["home"],
        "ftag": r["goals"]["away"],
    } for r in rows])


def team_map(api: pd.DataFrame, matches: pd.DataFrame) -> dict[str, str]:
    """API takım adlarını football-data adlarına eşler: aynı gün aynı skorla oynanan
    maçlarda en sık karşılaşılan ev sahibi/deplasman adı."""
    m = api.dropna(subset=["fthg"]).merge(
        matches[["league", "date", "home", "away", "fthg", "ftag"]], on=["league", "date", "fthg", "ftag"])
    votes: dict[str, Counter] = {}
    for a, f in [*zip(m["home_api"], m["home"]), *zip(m["away_api"], m["away"])]:
        votes.setdefault(a, Counter())[f] += 1
    return {a: c.most_common(1)[0][0] for a, c in votes.items()}


def load(matches: pd.DataFrame, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Her maç için iki takımın eksik (out) ve şüpheli (doubt) oyuncu sayısı."""
    frames = []
    for code, league_id in LEAGUES.items():
        for path in sorted((raw_dir / "fixtures").glob(f"league{league_id}_season*.json")):
            fx = fixtures_frame(json.loads(path.read_text()), code)
            inj_path = raw_dir / "injuries" / path.name
            if fx.empty or not inj_path.exists():
                continue
            counts = Counter()
            for r in json.loads(inj_path.read_text()):
                kind = "out" if r["player"]["type"] == "Missing Fixture" else "doubt"
                counts[(r["fixture"]["id"], r["team"]["id"], kind)] += 1
            for side in ("h", "a"):
                team = fx["home_id"] if side == "h" else fx["away_id"]
                for kind in ("out", "doubt"):
                    fx[f"{kind}_{side}"] = [counts[(f, t, kind)] for f, t in zip(fx["fixture"], team)]
            frames.append(fx)
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    fx = pd.concat(frames, ignore_index=True)
    names = team_map(fx, matches)
    fx["home"], fx["away"] = fx["home_api"].map(names), fx["away_api"].map(names)
    return fx.dropna(subset=["home", "away"])[COLUMNS]


def add_injuries(matches: pd.DataFrame, inj: pd.DataFrame) -> pd.DataFrame:
    """Sakatlık sayılarını maç tablosuna ekler (lig + takım çifti, tarih farkı en fazla 1 gün)."""
    out = matches.copy()
    cols = COLUMNS[4:]
    for c in cols:
        out[c] = float("nan")
    if inj.empty:
        return out
    key = ["league", "home", "away"]
    left = out.reset_index()[["index", "date", *key]]
    cand = left.merge(inj, on=key, suffixes=("", "_api"))
    cand = cand[(cand["date"] - cand["date_api"]).abs() <= pd.Timedelta(days=1)]
    cand = cand.drop_duplicates("index").set_index("index")
    out.loc[cand.index, cols] = cand[cols].astype(float)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", required=True, help="sezon başlangıç yılları, ör. 2022 2023")
    args = parser.parse_args()
    download_all(args.seasons)
    print("Kaydedildi:", RAW_DIR)


if __name__ == "__main__":
    main()
