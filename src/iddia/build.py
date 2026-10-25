"""Ham CSV'leri tek ve temiz bir maç tablosuna dönüştürür."""
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/matches.csv")

# Kapanış oranı sütunları, tercih sırasıyla: Pinnacle kapanış, piyasa ortalaması kapanış, eski ortalama.
ODDS_CANDIDATES = [
    ("PSCH", "PSCD", "PSCA"),
    ("AvgCH", "AvgCD", "AvgCA"),
    ("PSH", "PSD", "PSA"),
    ("AvgH", "AvgD", "AvgA"),
    ("BbAvH", "BbAvD", "BbAvA"),
    ("B365H", "B365D", "B365A"),
]

COLUMNS = ["date", "league", "season", "home", "away", "fthg", "ftag", "ftr",
           "odds_h", "odds_d", "odds_a"]


def pick_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Her satır için mevcut ilk oran üçlüsünü seçer."""
    odds = pd.DataFrame(index=df.index, columns=["odds_h", "odds_d", "odds_a"], dtype=float)
    for cols in ODDS_CANDIDATES:
        if not set(cols) <= set(df.columns):
            continue
        part = df[list(cols)].apply(pd.to_numeric, errors="coerce")
        part.columns = odds.columns
        missing = odds.isna().any(axis=1) & part.notna().all(axis=1)
        odds.loc[missing] = part.loc[missing]
    return odds


def load_season(path: Path, league: str, season: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    out = pd.DataFrame({
        "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
        "league": league,
        "season": season,
        "home": df["HomeTeam"].str.strip(),
        "away": df["AwayTeam"].str.strip(),
        "fthg": df["FTHG"].astype(int),
        "ftag": df["FTAG"].astype(int),
        "ftr": df["FTR"],
    })
    out = out.join(pick_odds(df))
    return out[COLUMNS]


def check(df: pd.DataFrame) -> list[str]:
    """Veri kalitesi sorunlarını listeler (VR-11)."""
    problems = []
    if df["date"].isna().any():
        problems.append(f"{df['date'].isna().sum()} satırda tarih okunamadı")
    dups = df.duplicated(subset=["league", "date", "home", "away"]).sum()
    if dups:
        problems.append(f"{dups} çift kayıt")
    expected = pd.Series("D", index=df.index)
    expected[df["fthg"] > df["ftag"]] = "H"
    expected[df["fthg"] < df["ftag"]] = "A"
    bad = (expected != df["ftr"]).sum()
    if bad:
        problems.append(f"{bad} satırda FTR skorla uyuşmuyor")
    missing_odds = df[["odds_h", "odds_d", "odds_a"]].isna().any(axis=1).sum()
    if missing_odds:
        problems.append(f"{missing_odds} maçta oran yok")
    return problems


def build(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    frames = [load_season(p, p.parent.name, p.stem) for p in sorted(raw_dir.glob("*/*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["date", "league", "home"]).reset_index(drop=True)


def main() -> None:
    df = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"{len(df)} maç -> {OUT_PATH}")
    for problem in check(df):
        print("UYARI:", problem)


if __name__ == "__main__":
    main()
