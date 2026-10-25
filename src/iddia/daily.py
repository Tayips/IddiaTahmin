"""Günlük iş: yaklaşan maçlara tahmin üretir, biten maçları puanlar, sayfayı yazar.

Tahminler data/predictions.csv dosyasında birikir. Bir maçın tahmini başlama
saatine kadar her çalıştırmada yenilenir, başlama saatinden sonra bir daha
değişmez. Böylece başarı oranı yalnızca maçtan önce yapılmış tahminleri ölçer.
"""
import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from iddia import download, metrics, site
from iddia.build import build, read_csv, team_name
from iddia.evaluate import drop_forfeits
from iddia.models.elo import Elo, OrderedLogit

LEAGUES = {"T1": "Süper Lig", "E0": "Premier Lig"}
MODEL = "elo-v1"
PREDICTIONS = Path("data/predictions.csv")
SITE_DIR = Path("docs")
UK = ZoneInfo("Europe/London")          # football-data saatleri İngiltere saatidir
KEY = ["league", "date", "home", "away"]
PRED_COLUMNS = KEY + ["season", "kickoff", "p_h", "p_d", "p_a", "pick",
                      "odds_h", "odds_d", "odds_a", "model", "predicted_at"]
BACKFILL = "geriye-donuk"              # predicted_at değeri: sezon başından kurulum gününe kadarki maçlar
RESULT_GRACE_DAYS = 4                   # bu kadar gün sonuç gelmezse maç ertelenmiş sayılır


def season_of(d: date) -> str:
    return download.season_code(download.current_season_start(d))


def load_fixtures(path: Path = download.FIXTURES_PATH) -> pd.DataFrame:
    df = read_csv(path)
    df = df[df["Div"].isin(LEAGUES)].copy()
    local = pd.to_datetime(df["Date"] + " " + df["Time"].fillna("00:00"), dayfirst=True)
    kickoff = local.dt.tz_localize(UK, ambiguous=True, nonexistent="shift_forward").dt.tz_convert("UTC")
    out = pd.DataFrame({
        "league": df["Div"],
        "date": local.dt.strftime("%Y-%m-%d"),
        "kickoff": kickoff.dt.strftime("%Y-%m-%dT%H:%MZ"),
        "home": df["HomeTeam"].map(team_name),
        "away": df["AwayTeam"].map(team_name),
        "odds_h": pd.to_numeric(df.get("AvgH"), errors="coerce"),
        "odds_d": pd.to_numeric(df.get("AvgD"), errors="coerce"),
        "odds_a": pd.to_numeric(df.get("AvgA"), errors="coerce"),
    })
    out["season"] = local.dt.date.map(season_of)
    return out.reset_index(drop=True)


def fit(matches: pd.DataFrame) -> tuple[Elo, dict[str, OrderedLogit]]:
    elo = Elo()
    diffs = elo.run(matches)
    logits = {
        league: OrderedLogit().fit(diffs[games.index].to_numpy(), games["ftr"])
        for league, games in matches.groupby("league")
    }
    return elo, logits


def predict(fixtures: pd.DataFrame, elo: Elo, logits: dict[str, OrderedLogit], now: datetime) -> pd.DataFrame:
    if fixtures.empty:
        return pd.DataFrame(columns=PRED_COLUMNS)
    diffs = np.array([elo.diff(r.league, r.home, r.away, r.season) for r in fixtures.itertuples()])
    probs = np.vstack([logits[lg].predict(np.array([d]))[0] for lg, d in zip(fixtures["league"], diffs)])
    out = fixtures.copy()
    out[["p_h", "p_d", "p_a"]] = probs.round(4)
    out["pick"] = np.array(metrics.OUTCOMES)[probs.argmax(axis=1)]
    out["model"] = MODEL
    out["predicted_at"] = now.strftime("%Y-%m-%dT%H:%MZ")
    return out[PRED_COLUMNS]


def backfill(matches: pd.DataFrame, season: str) -> pd.DataFrame:
    """Bu sezonun oynanmış maçları için geriye dönük tahmin.

    Elo farkı her maçtan önceki duruma göre hesaplanır, sıralı lojistik yalnızca
    önceki sezonlarla kurulur; yani hiçbir tahmin kendi sonucunu görmez.
    """
    diffs = Elo().run(matches)
    rows = []
    for league, games in matches.groupby("league"):
        past = games[games["season"] < season]
        current = games[games["season"] == season]
        if current.empty or past.empty:
            continue
        probs = OrderedLogit().fit(diffs[past.index].to_numpy(), past["ftr"]).predict(diffs[current.index].to_numpy())
        out = pd.DataFrame({
            "league": league,
            "date": current["date"].dt.strftime("%Y-%m-%d"),
            "home": current["home"],
            "away": current["away"],
            "season": season,
            "kickoff": current["date"].dt.strftime("%Y-%m-%dT00:00Z"),
            "odds_h": current["odds_h"], "odds_d": current["odds_d"], "odds_a": current["odds_a"],
        })
        out[["p_h", "p_d", "p_a"]] = probs.round(4)
        out["pick"] = np.array(metrics.OUTCOMES)[probs.argmax(axis=1)]
        rows.append(out)
    if not rows:
        return pd.DataFrame(columns=PRED_COLUMNS)
    df = pd.concat(rows, ignore_index=True)
    df["model"] = MODEL
    df["predicted_at"] = BACKFILL
    return df[PRED_COLUMNS]


def merge_predictions(old: pd.DataFrame, new: pd.DataFrame, now: datetime, first_run: bool = False) -> pd.DataFrame:
    """Başlamamış maçların tahminini günceller; başlamış maçlarınkini korur.

    Başlamış bir maça sonradan tahmin eklenmez. Tek istisna ilk kurulumdur: o gün
    fikstürde olup sonucu henüz veride olmayan maçlar geriye dönük olarak eklenir.
    """
    stamp = now.strftime("%Y-%m-%dT%H:%MZ")
    started = new["kickoff"] <= stamp
    if first_run:
        new = new.copy()
        new.loc[started, "predicted_at"] = BACKFILL
        new = new[~new.set_index(KEY).index.isin(old.set_index(KEY).index)]
        return pd.concat([old, new], ignore_index=True).sort_values(["kickoff", "league", "home"]).reset_index(drop=True)[PRED_COLUMNS]
    new = new[~started]
    frozen = old[old["kickoff"] <= stamp]
    pending = old[old["kickoff"] > stamp]
    pending = pending[~pending.set_index(KEY).index.isin(new.set_index(KEY).index)]
    merged = pd.concat([frozen, pending, new], ignore_index=True)
    return merged.sort_values(["kickoff", "league", "home"]).reset_index(drop=True)[PRED_COLUMNS]


def score(preds: pd.DataFrame, matches: pd.DataFrame, now: datetime) -> pd.DataFrame:
    results = matches.assign(date=matches["date"].dt.strftime("%Y-%m-%d"))[KEY + ["fthg", "ftag", "ftr"]]
    df = preds.merge(results, on=KEY, how="left")
    stamp = now.strftime("%Y-%m-%dT%H:%MZ")
    late = pd.to_datetime(df["kickoff"]) + pd.Timedelta(days=RESULT_GRACE_DAYS) < pd.Timestamp(now)
    df["status"] = np.select(
        [df["ftr"].notna(), df["kickoff"] > stamp, late],
        ["bitti", "yaklasan", "ertelendi"],
        default="sonuc_bekleniyor",
    )
    df["correct"] = np.where(df["status"] == "bitti", df["pick"] == df["ftr"], np.nan)
    odds = df[["odds_h", "odds_d", "odds_a"]].to_numpy(dtype=float)
    has_odds = np.isfinite(odds).all(axis=1)
    fav = np.where(has_odds, np.array(metrics.OUTCOMES)[np.nan_to_num(odds, nan=np.inf).argmin(axis=1)], None)
    df["odds_pick"] = fav
    df["odds_correct"] = np.where((df["status"] == "bitti") & has_odds, df["odds_pick"] == df["ftr"], np.nan)
    return df


def _summary(done: pd.DataFrame) -> dict:
    if done.empty:
        return {"n": 0, "correct": 0, "accuracy": None, "rps": None, "odds_accuracy": None}
    p = done[["p_h", "p_d", "p_a"]].to_numpy(dtype=float)
    with_odds = done["odds_correct"].dropna()
    return {
        "n": int(len(done)),
        "correct": int(done["correct"].sum()),
        "accuracy": round(float(done["correct"].mean()), 4),
        "rps": round(float(metrics.rps(p, done["ftr"]).mean()), 4),
        "odds_accuracy": round(float(with_odds.mean()), 4) if len(with_odds) else None,
    }


def stats(scored: pd.DataFrame, now: datetime) -> dict:
    done = scored[scored["status"] == "bitti"]
    top = done[["p_h", "p_d", "p_a"]].max(axis=1)
    recent = done[pd.to_datetime(done["kickoff"]) >= pd.Timestamp(now) - pd.Timedelta(days=30)]
    live = done[done["predicted_at"] != BACKFILL]
    return {
        "genel": _summary(done),
        "canli": _summary(live),
        "lig": {lg: _summary(done[done["league"] == lg]) for lg in LEAGUES},
        "son_30_gun": _summary(recent),
        "guven": {f"{t:.0%}+": _summary(done[top >= t]) for t in (0.5, 0.6, 0.7)},
    }


def to_records(df: pd.DataFrame) -> list[dict]:
    cols = PRED_COLUMNS + ["fthg", "ftag", "ftr", "status", "correct", "odds_pick", "odds_correct"]
    out = df[cols].astype(object).where(df[cols].notna(), None)
    return out.to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true", help="data/raw içindeki dosyaları kullan")
    args = parser.parse_args()
    now = datetime.now(timezone.utc)

    if not args.skip_download:
        for league in LEAGUES:
            for year in range(2014, download.current_season_start() + 1):
                download.download(league, year)
        download.download_fixtures()

    matches = drop_forfeits(build())
    elo, logits = fit(matches)
    fixtures = load_fixtures()
    first_run = not PREDICTIONS.exists()
    old = backfill(matches, season_of(now.date())) if first_run else pd.read_csv(PREDICTIONS)
    preds = merge_predictions(old, predict(fixtures, elo, logits, now), now, first_run)
    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(PREDICTIONS, index=False)

    scored = score(preds, matches, now)
    data = {
        "updated_at": now.strftime("%Y-%m-%dT%H:%MZ"),
        "model": MODEL,
        "leagues": LEAGUES,
        "stats": stats(scored, now),
        "matches": to_records(scored),
    }
    SITE_DIR.mkdir(exist_ok=True)
    (SITE_DIR / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    (SITE_DIR / "index.html").write_text(site.render(data))
    g = data["stats"]["genel"]
    print(f"Fikstürde {len(fixtures)} maç, {len(preds)} kayıtlı tahmin, {g['n']} biten maç, isabet {g['accuracy']}")


if __name__ == "__main__":
    main()
