from pathlib import Path

from iddia.build import check, load_season
from iddia.download import season_code

FIXTURE = Path(__file__).parent / "fixtures" / "T1_sample.csv"


def test_season_code():
    assert season_code(2024) == "2425"
    assert season_code(1999) == "9900"


def test_load_season_drops_empty_rows_and_parses_dates():
    df = load_season(FIXTURE, "T1", "2425")
    assert len(df) == 3
    assert df["date"].iloc[0].strftime("%Y-%m-%d") == "2024-08-09"


def test_pinnacle_closing_preferred_with_fallback():
    df = load_season(FIXTURE, "T1", "2425")
    assert df.loc[0, "odds_h"] == 1.22          # Pinnacle kapanış
    assert df.loc[1, "odds_h"] == 1.15          # Pinnacle yok, Bet365'e düşer


def test_check_flags_result_mismatch():
    df = load_season(FIXTURE, "T1", "2425")
    assert check(df) == []
    df.loc[0, "ftr"] = "A"
    assert any("FTR" in p for p in check(df))
