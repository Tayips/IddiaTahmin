"""The Odds API: maça yakın, güncel bahis oranları.

football-data.co.uk fikstür oranlarını haftada bir iki kez günceller; oysa
oranlar maç saatine yaklaştıkça daha isabetli olur (kapanış oranları bu
projedeki en iyi tahmin). Bu modül her çalıştırmada yaklaşan maçların güncel
oranlarını çeker ve fikstürdeki oranların yerine koyar.

Anahtar ODDS_API_KEY ortam değişkeninden okunur; yoksa ya da istek başarısız
olursa sistem fikstür oranlarıyla çalışmaya devam eder. Ücretsiz plan ayda 500
kredi verir; bir lig için bir istek (1X2 + Üst/Alt, tek bölge) 2 kredidir.
Bu yüzden yalnızca önümüzdeki WINDOW saatte maçı olan ligler istenir: tahmin
maçtan önceki son çalıştırmada donduğu için oran asıl o çalıştırmalarda işe yarar
(son 5 sezonda erken oranların RPS'i 0.1939, kapanış oranlarınınki 0.1923).
"""
import os
import re
import unicodedata
from datetime import datetime, timedelta
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import requests

URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
ENV_KEY = "ODDS_API_KEY"
SPORTS = {"E0": "soccer_epl", "T1": "soccer_turkey_super_league"}
ODDS = ["odds_h", "odds_d", "odds_a", "odds_o25", "odds_u25"]
WINDOW = timedelta(hours=8)
MAX_KICKOFF_GAP = timedelta(hours=3)

# Normalize edilmiş ad → fikstürdeki ad (benzerliğin yetmediği adlar)
ALIASES = {
    "istanbul basaksehir": "buyuksehyr", "basaksehir": "buyuksehyr",
    "manchester united": "man united", "manchester city": "man city",
    "nottingham forest": "nott'm forest", "wolverhampton wanderers": "wolves",
    "brighton and hove albion": "brighton", "tottenham hotspur": "tottenham",
    "sheffield wednesday": "sheffield weds", "west bromwich albion": "west brom",
}
NOISE = re.compile(r"\b(fc|afc|sk|jk|fk|as|kulubu|a\.s\.|cf)\b")


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name.replace("ı", "i").replace("İ", "I"))
    s = s.encode("ascii", "ignore").decode().lower().replace("&", "and")
    s = re.sub(r"\s+", " ", NOISE.sub(" ", s)).strip()
    return ALIASES.get(s, s)


def similarity(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    if a == b or a.startswith(b) or b.startswith(a):
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def parse(league: str, events: list[dict]) -> pd.DataFrame:
    """Her maç için bürolar arasındaki ortalama oran (football-data.co.uk'teki Avg sütunları gibi)."""
    rows = []
    for e in events:
        home, away = e["home_team"], e["away_team"]
        h2h, over, under = [], [], []
        for b in e.get("bookmakers", []):
            for m in b.get("markets", []):
                prices = {o["name"]: o for o in m.get("outcomes", [])}
                if m["key"] == "h2h" and {home, away, "Draw"} <= prices.keys():
                    h2h.append([prices[home]["price"], prices["Draw"]["price"], prices[away]["price"]])
                elif m["key"] == "totals" and prices.get("Over", {}).get("point") == 2.5 and "Under" in prices:
                    over.append(prices["Over"]["price"])
                    under.append(prices["Under"]["price"])
        if not h2h:
            continue
        avg = np.mean(h2h, axis=0)
        rows.append({"league": league, "home": home, "away": away,
                     "kickoff": pd.Timestamp(e["commence_time"]).strftime("%Y-%m-%dT%H:%MZ"),
                     "odds_h": avg[0], "odds_d": avg[1], "odds_a": avg[2],
                     "odds_o25": np.mean(over) if over else np.nan, "odds_u25": np.mean(under) if under else np.nan})
    return pd.DataFrame(rows, columns=["league", "home", "away", "kickoff", *ODDS]).round({c: 2 for c in ODDS})


def fetch(fixtures: pd.DataFrame, now: datetime, key: str | None = None) -> pd.DataFrame | None:
    key = key or os.environ.get(ENV_KEY)
    if not key:
        print(f"{ENV_KEY} yok, canlı oranlar atlandı")
        return None
    stamp, until = now.strftime("%Y-%m-%dT%H:%MZ"), (now + WINDOW).strftime("%Y-%m-%dT%H:%MZ")
    soon = fixtures[(fixtures["kickoff"] > stamp) & (fixtures["kickoff"] <= until)]
    frames = []
    for league in sorted(set(soon["league"]) & SPORTS.keys()):
        try:
            r = requests.get(URL.format(sport=SPORTS[league]), timeout=30, params={
                "apiKey": key, "regions": "eu", "markets": "h2h,totals", "oddsFormat": "decimal"})
            r.raise_for_status()
            frames.append(parse(league, r.json()))
        except (requests.RequestException, ValueError, KeyError) as e:
            # Hata mesajı istek adresini (anahtarla birlikte) içerebilir; yalnızca türünü yaz
            print(f"The Odds API okunamadı ({league}), atlandı: {type(e).__name__}")
            continue
        print(f"The Odds API {league}: {len(frames[-1])} maç, kalan kredi {r.headers.get('x-requests-remaining', '?')}")
    return pd.concat(frames, ignore_index=True) if frames else None


def match_events(fixtures: pd.DataFrame, live: pd.DataFrame) -> dict[int, int]:
    """Fikstür satırı → canlı oran satırı. Aynı ligde, başlama saati yakın ve takım adları benzer olmalı."""
    pairs = []
    fk = pd.to_datetime(fixtures["kickoff"], utc=True, errors="coerce")
    lk = pd.to_datetime(live["kickoff"], utc=True)
    for i, f in fixtures.iterrows():
        for j, e in live.iterrows():
            if f["league"] != e["league"] or pd.isna(fk[i]) or abs(fk[i] - lk[j]) > MAX_KICKOFF_GAP:
                continue
            sh, sa = similarity(f["home"], e["home"]), similarity(f["away"], e["away"])
            if min(sh, sa) >= 0.35 and sh + sa >= 1.2:
                pairs.append((sh + sa, i, j))
    out, used = {}, set()
    for _, i, j in sorted(pairs, reverse=True):
        if i not in out and j not in used:
            out[i] = j
            used.add(j)
    return out


def apply(fixtures: pd.DataFrame, live: pd.DataFrame | None) -> pd.DataFrame:
    """Eşleşen maçların oranlarını güncel oranlarla değiştirir; odds_src sütunu kaynağı gösterir."""
    df = fixtures.copy()
    df["odds_src"] = np.where(df["odds_h"].notna(), "fikstur", None)
    if live is None or live.empty:
        return df
    matched = match_events(df, live)
    for i, j in matched.items():
        new = live.loc[j, ODDS]
        cols = [c for c in ODDS if pd.notna(new[c])]
        df.loc[i, cols] = new[cols].to_numpy(dtype=float)
        df.loc[i, "odds_src"] = "canli"
    print(f"Canlı oran: {len(matched)}/{len(live)} maç eşleşti")
    return df
