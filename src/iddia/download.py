"""football-data.co.uk sezon CSV'lerini indirir."""
import argparse
from datetime import date
from pathlib import Path

import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
RAW_DIR = Path("data/raw")
FIXTURES_PATH = RAW_DIR / "fixtures.csv"


def current_season_start(today: date | None = None) -> int:
    """Avrupa sezonu Temmuz'da başlar: 2026-10-25 -> 2026, 2027-03-01 -> 2026."""
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def season_code(start_year: int) -> str:
    """2024 -> '2425'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def download(league: str, start_year: int, out_dir: Path = RAW_DIR) -> Path:
    code = season_code(start_year)
    out = out_dir / league / f"{code}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(BASE_URL.format(season=code, league=league), timeout=30)
    resp.raise_for_status()
    out.write_bytes(resp.content)
    return out


def download_fixtures(out: Path = FIXTURES_PATH) -> Path:
    """Önümüzdeki günlerin maçları ve güncel oranları (tüm ligler tek dosyada)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(FIXTURES_URL, timeout=30)
    resp.raise_for_status()
    out.write_bytes(resp.content)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leagues", nargs="+", default=["T1", "E0"])
    parser.add_argument("--from", dest="first", type=int, default=2014)
    parser.add_argument("--to", dest="last", type=int, default=current_season_start())
    parser.add_argument("--fixtures", action="store_true", help="yaklaşan maçlar dosyasını da indir")
    args = parser.parse_args()
    for league in args.leagues:
        for year in range(args.first, args.last + 1):
            try:
                print(download(league, year))
            except requests.HTTPError as err:
                print(f"{league} {year}: {err}")
    if args.fixtures:
        print(download_fixtures())


if __name__ == "__main__":
    main()
