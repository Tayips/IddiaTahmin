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

# Aynı kulübün yıllar içindeki farklı yazımları; değer, güncel adıdır.
TEAM_ALIASES = {
    "Erzurum BB": "Erzurumspor",
}

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


def read_csv(path: Path) -> pd.DataFrame:
    """Yeni dosyalar BOM'lu UTF-8, eskileri latin-1."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")


def parse_dates(col: pd.Series) -> pd.Series:
    """Eski sezonlarda 'gg/aa/yy', yenilerde 'gg/aa/yyyy'."""
    long = pd.to_datetime(col, format="%d/%m/%Y", errors="coerce")
    short = pd.to_datetime(col, format="%d/%m/%y", errors="coerce")
    return long.fillna(short)


def team_name(name: str) -> str:
    name = name.strip()
    return TEAM_ALIASES.get(name, name)


def load_season(path: Path, league: str, season: str) -> pd.DataFrame:
    df = read_csv(path)
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    out = pd.DataFrame({
        "date": parse_dates(df["Date"]),
        "league": league,
        "season": season,
        "home": df["HomeTeam"].map(team_name),
        "away": df["AwayTeam"].map(team_name),
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
