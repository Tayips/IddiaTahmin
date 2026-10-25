"""Günlük iş: yaklaşan maçlara tahmin üretir, biten maçları puanlar, sayfayı yazar.

Tahminler data/predictions.csv dosyasında birikir. Bir maçın tahmini başlama
saatine kadar her çalıştırmada yenilenir, başlama saatinden sonra bir daha
değişmez. Böylece başarı oranı yalnızca maçtan önce yapılmış tahminleri ölçer.
"""
import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from iddia import coupons, download, fdorg, gemini, haberler, metrics, oddsapi, site
from iddia.build import build, read_csv, team_name
from iddia.evaluate import drop_forfeits
from iddia.features import FEATURES_NO_MARKET, GOAL_FEATURES, MARKET_FEATURES, build_features, market
from iddia.goals import TARGETS, BinaryModel
from iddia.models.elo import Elo, OrderedLogit
from iddia.models.ml import Model

LEAGUES = {"T1": "Süper Lig", "E0": "Premier Lig"}
MODEL = "form-piyasa-v2"
MODEL_NO_MARKET = "form-v2"
PREDICTIONS = Path("data/predictions.csv")
SITE_DIR = Path("docs")
SIMILAR = Path("reports/benzer_durumlar.json")   # python -m iddia.benzer ile üretilir
UK = ZoneInfo("Europe/London")          # football-data saatleri İngiltere saatidir
KEY = ["league", "date", "home", "away"]
# Tahminin dayandığı özellikler, tahminle birlikte saklanır (analiz ekranı için):
# kayıt adı -> özellik tablosundaki sütun
EXPLAIN_SOURCE = {"f_elo": "elo_diff", "f_h_gf": "h_gf", "f_h_ga": "h_ga", "f_a_gf": "a_gf", "f_a_ga": "a_ga",
                  "f_h_sot": "h_stf", "f_a_sot": "a_stf", "f_h_rest": "h_rest", "f_a_rest": "a_rest"}
EXPLAIN = list(EXPLAIN_SOURCE)
TEAM_RECENT = 6                         # analiz ekranında gösterilen son maç sayısı
PRED_COLUMNS = KEY + ["season", "kickoff", "p_h", "p_d", "p_a", "pick",
                      "odds_h", "odds_d", "odds_a", "model", "predicted_at",
                      "p_o25", "p_kg", "odds_o25", "odds_u25", "odds_src", "data_through", *EXPLAIN]
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
        "odds_o25": pd.to_numeric(df.get("Avg>2.5"), errors="coerce"),
        "odds_u25": pd.to_numeric(df.get("Avg<2.5"), errors="coerce"),
    })
    out["season"] = local.dt.date.map(season_of)
    return out.reset_index(drop=True)


@dataclass
class Predictor:
    """Piyasa oranı olan maçlarda form + piyasa modeli, olmayanlarda yalnızca form modeli."""
    elo: Elo
    diffs: pd.Series
    with_market: Model
    form_only: Model
    goals: dict[str, BinaryModel]


def fit(matches: pd.DataFrame) -> Predictor:
    elo = Elo()
    diffs = elo.run(matches)
    X = build_features(matches, diffs)
    y = matches["ftr"]
    has_mk = X["mk_h"].notna()
    return Predictor(
        elo=elo,
        diffs=diffs,
        with_market=Model("lr", FEATURES_NO_MARKET + MARKET_FEATURES).fit(X[has_mk], y[has_mk]),
        form_only=Model("lr", FEATURES_NO_MARKET).fit(X, y),
        goals={name: BinaryModel(GOAL_FEATURES).fit(X, fn(matches)) for name, fn in TARGETS.items()},
    )


def fixture_features(fixtures: pd.DataFrame, matches: pd.DataFrame, model: Predictor) -> pd.DataFrame:
    """Gelecek maçları geçmişin sonuna ekleyip özelliklerini hesaplar."""
    future = pd.DataFrame({
        "date": pd.to_datetime(fixtures["date"]),
        "league": fixtures["league"].to_numpy(),
        "season": fixtures["season"].to_numpy(),
        "home": fixtures["home"].to_numpy(),
        "away": fixtures["away"].to_numpy(),
        "pre_h": fixtures["odds_h"].to_numpy(),
        "pre_d": fixtures["odds_d"].to_numpy(),
        "pre_a": fixtures["odds_a"].to_numpy(),
        "pre_o25": fixtures["odds_o25"].to_numpy(),
        "pre_u25": fixtures["odds_u25"].to_numpy(),
    })
    combined = pd.concat([matches, future], ignore_index=True)
    future_idx = combined.index[len(matches):]
    future_diffs = pd.Series(
        [model.elo.diff(r.league, r.home, r.away, r.season) for r in fixtures.itertuples()], index=future_idx)
    diffs = pd.concat([model.diffs.reset_index(drop=True), future_diffs])
    return build_features(combined, diffs).loc[future_idx].reset_index(drop=True)


def predict(fixtures: pd.DataFrame, matches: pd.DataFrame, model: Predictor, now: datetime) -> pd.DataFrame:
    if fixtures.empty:
        return pd.DataFrame(columns=PRED_COLUMNS)
    X = fixture_features(fixtures, matches, model)
    probs = model.form_only.predict(X)
    has_mk = X["mk_h"].notna().to_numpy()
    if has_mk.any():
        probs[has_mk] = model.with_market.predict(X[has_mk])
    out = fixtures.copy()
    if "odds_src" not in out:
        out["odds_src"] = np.where(out["odds_h"].notna(), "fikstur", None)
    out[["p_h", "p_d", "p_a"]] = probs.round(4)
    out["pick"] = np.array(metrics.OUTCOMES)[probs.argmax(axis=1)]
    out["model"] = np.where(has_mk, MODEL, MODEL_NO_MARKET)
    out["p_o25"] = model.goals["ust25"].predict(X).round(4)
    out["p_kg"] = model.goals["kg"].predict(X).round(4)
    out["predicted_at"] = now.strftime("%Y-%m-%dT%H:%MZ")
    out["data_through"] = out["league"].map(data_through(matches))
    for name, col in EXPLAIN_SOURCE.items():
        out[name] = X[col].to_numpy().round(2)
    out["f_elo"] = (out["f_elo"] * 100).round(0)    # özellik tablosunda Elo farkı /100 ölçeğinde
    return out[PRED_COLUMNS]


def data_through(matches: pd.DataFrame) -> dict[str, str]:
    """Her lig için verideki son sonuçlu maçın tarihi: tahmin hangi veriyle yapıldı."""
    return matches.groupby("league")["date"].max().dt.strftime("%Y-%m-%d").to_dict()


def team_profiles(matches: pd.DataFrame, season: str) -> dict[str, dict]:
    """Bu sezonun takımları için son maçlar ve iç saha / deplasman karnesi (analiz ekranı)."""
    rows = []
    for side, other, gf, ga in (("home", "away", "fthg", "ftag"), ("away", "home", "ftag", "fthg")):
        rows.append(pd.DataFrame({
            "league": matches["league"], "season": matches["season"], "date": matches["date"],
            "team": matches[side], "opp": matches[other], "venue": "H" if side == "home" else "A",
            "gf": matches[gf], "ga": matches[ga]}))
    long = pd.concat(rows).sort_values("date")
    long["res"] = np.select([long["gf"] > long["ga"], long["gf"] < long["ga"]], ["G", "M"], default="B")

    def record(part):
        return {"p": len(part), "g": int((part["res"] == "G").sum()), "b": int((part["res"] == "B").sum()),
                "m": int((part["res"] == "M").sum()), "gf": int(part["gf"].sum()), "ga": int(part["ga"].sum())}

    out = {}
    for (league, team), g in long[long["season"] == season].groupby(["league", "team"]):
        # Son maçlar sezon başındaysa geçen sezona uzanabilir
        last = long[(long["league"] == league) & (long["team"] == team)].tail(TEAM_RECENT).iloc[::-1]
        out[f"{league}|{team}"] = {
            "son": [{"date": r.date.strftime("%Y-%m-%d"), "opp": r.opp, "venue": r.venue,
                     "gf": int(r.gf), "ga": int(r.ga), "res": r.res} for r in last.itertuples()],
            "ic": record(g[g["venue"] == "H"]),
            "dis": record(g[g["venue"] == "A"]),
        }
    return out


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
    df["model"] = "elo-v1"
    df["predicted_at"] = BACKFILL
    return df.reindex(columns=PRED_COLUMNS)


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


def score(preds: pd.DataFrame, matches: pd.DataFrame, now: datetime, org: pd.DataFrame | None = None) -> pd.DataFrame:
    results = matches.assign(date=matches["date"].dt.strftime("%Y-%m-%d"))[KEY + ["fthg", "ftag", "ftr"]]
    df = preds.merge(results, on=KEY, how="left")
    df, moved = fdorg.fill_results(df, org)
    stamp = now.strftime("%Y-%m-%dT%H:%MZ")
    late = pd.to_datetime(df["kickoff"]) + pd.Timedelta(days=RESULT_GRACE_DAYS) < pd.Timestamp(now)
    df["status"] = np.select(
        [df["ftr"].notna(), moved, df["kickoff"] > stamp, late],
        ["bitti", "ertelendi", "yaklasan", "ertelendi"],
        default="sonuc_bekleniyor",
    )
    df["correct"] = np.where(df["status"] == "bitti", df["pick"] == df["ftr"], np.nan)
    odds = df[["odds_h", "odds_d", "odds_a"]].to_numpy(dtype=float)
    has_odds = np.isfinite(odds).all(axis=1)
    fav = np.where(has_odds, np.array(metrics.OUTCOMES)[np.nan_to_num(odds, nan=np.inf).argmin(axis=1)], None)
    df["odds_pick"] = fav
    df["odds_correct"] = np.where((df["status"] == "bitti") & has_odds, df["odds_pick"] == df["ftr"], np.nan)
    goals = df["fthg"] + df["ftag"]
    done = (df["status"] == "bitti") & df["p_o25"].notna()
    df["o25_correct"] = np.where(done, (df["p_o25"] >= 0.5) == (goals > 2.5), np.nan)
    df["kg_correct"] = np.where(done, (df["p_kg"] >= 0.5) == ((df["fthg"] > 0) & (df["ftag"] > 0)), np.nan)
    return df


def _summary(done: pd.DataFrame) -> dict:
    if done.empty:
        return {"n": 0, "correct": 0, "accuracy": None, "rps": None, "brier": None, "log_loss": None,
                "odds_accuracy": None}
    p = done[["p_h", "p_d", "p_a"]].to_numpy(dtype=float)
    with_odds = done["odds_correct"].dropna()
    return {
        "n": int(len(done)),
        "correct": int(done["correct"].sum()),
        "accuracy": round(float(done["correct"].mean()), 4),
        "rps": round(float(metrics.rps(p, done["ftr"]).mean()), 4),
        "brier": round(float(metrics.brier(p, done["ftr"]).mean()), 4),
        "log_loss": round(float(metrics.log_loss(p, done["ftr"]).mean()), 4),
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
    mk = market(df.rename(columns={"odds_h": "pre_h", "odds_d": "pre_d", "odds_a": "pre_a"}))
    df = df.assign(mk_h=mk["mk_h"].round(4), mk_d=mk["mk_d"].round(4), mk_a=mk["mk_a"].round(4))
    cols = PRED_COLUMNS + ["fthg", "ftag", "ftr", "status", "correct", "odds_pick", "odds_correct",
                           "o25_correct", "kg_correct", "mk_h", "mk_d", "mk_a"]
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
    model = fit(matches)
    org = fdorg.fetch(download.current_season_start(now.date()))
    fixtures = fdorg.apply_to_fixtures(load_fixtures(), org)
    fixtures = oddsapi.apply(fixtures, oddsapi.fetch(fixtures, now))
    first_run = not PREDICTIONS.exists()
    old = backfill(matches, season_of(now.date())) if first_run else pd.read_csv(PREDICTIONS)
    old = old.reindex(columns=PRED_COLUMNS)   # eski kayıtlarda yeni sütunlar boş kalır
    preds = merge_predictions(old, predict(fixtures, matches, model, now), now, first_run)
    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(PREDICTIONS, index=False)

    scored = score(preds, matches, now, org)
    old_coupons = pd.read_csv(coupons.COUPONS) if coupons.COUPONS.exists() else pd.DataFrame(columns=coupons.LEG_COLUMNS)
    legs = coupons.update(old_coupons, preds, now)
    legs.to_csv(coupons.COUPONS, index=False)
    coupon_records = coupons.to_records(legs, scored)
    news = haberler.load()
    if not args.skip_download:
        news = haberler.update(news, preds, now)
        news = gemini.analyze(news, preds, now)
        haberler.save(news)
    data = {
        "updated_at": now.strftime("%Y-%m-%dT%H:%MZ"),
        "model": MODEL,
        "leagues": LEAGUES,
        "data_through": data_through(matches),
        "stats": stats(scored, now),
        "matches": to_records(scored),
        "teams": team_profiles(matches, season_of(now.date())),
        "benzer": json.loads(SIMILAR.read_text()) if SIMILAR.exists() else None,
        "haberler": haberler.for_site(news, scored),
        "coupon_types": coupons.TYPES,
        "coupon_stats": coupons.summary(coupon_records),
        "coupons": coupons.recent(coupon_records),
    }
    SITE_DIR.mkdir(exist_ok=True)
    (SITE_DIR / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    (SITE_DIR / "index.html").write_text(site.render(data))
    g = data["stats"]["genel"]
    print(f"Fikstürde {len(fixtures)} maç, {len(preds)} kayıtlı tahmin, {g['n']} biten maç, isabet {g['accuracy']}")


if __name__ == "__main__":
    main()
