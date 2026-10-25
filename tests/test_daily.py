from datetime import datetime, timezone

import numpy as np
import pandas as pd

from iddia.daily import BACKFILL, PRED_COLUMNS, load_fixtures, merge_predictions, score

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)


def pred(home, kickoff, p_h=0.5, predicted_at="2026-10-24T05:00Z"):
    row = dict(league="T1", date=kickoff[:10], home=home, away="X", season="2627", kickoff=kickoff,
               p_h=p_h, p_d=0.3, p_a=0.2, pick="H", odds_h=2.0, odds_d=3.4, odds_a=3.8,
               model="elo-v1", predicted_at=predicted_at, p_o25=0.6, p_kg=0.3)
    return pd.DataFrame([row]).reindex(columns=PRED_COLUMNS)


def test_started_match_prediction_is_frozen_and_future_one_updates():
    old = pd.concat([pred("Basladi", "2026-10-25T11:00Z", 0.5), pred("Yarin", "2026-10-26T17:00Z", 0.5)])
    new = pd.concat([pred("Basladi", "2026-10-25T11:00Z", 0.9), pred("Yarin", "2026-10-26T17:00Z", 0.7)])
    merged = merge_predictions(old, new, NOW).set_index("home")
    assert merged.loc["Basladi", "p_h"] == 0.5
    assert merged.loc["Yarin", "p_h"] == 0.7
    assert len(merged) == 2


def test_no_new_prediction_for_started_match_except_first_run():
    empty = pd.DataFrame(columns=PRED_COLUMNS)
    late = pred("Gec", "2026-10-25T11:00Z")
    assert merge_predictions(empty, late, NOW).empty
    first = merge_predictions(empty, late, NOW, first_run=True)
    assert first.loc[0, "predicted_at"] == BACKFILL


def test_score_statuses_and_correctness():
    preds = pd.concat([
        pred("Kazandi", "2026-10-20T17:00Z"),
        pred("Bekliyor", "2026-10-24T17:00Z"),
        pred("Ertelendi", "2026-10-10T17:00Z"),
        pred("Gelecek", "2026-10-27T17:00Z"),
    ], ignore_index=True)
    matches = pd.DataFrame({"league": ["T1"], "date": pd.to_datetime(["2026-10-20"]), "home": ["Kazandi"],
                            "away": ["X"], "fthg": [2], "ftag": [1], "ftr": ["H"]})
    s = score(preds, matches, NOW).set_index("home")
    assert s.loc["Kazandi", "status"] == "bitti" and s.loc["Kazandi", "correct"] == 1
    assert s.loc["Kazandi", "odds_correct"] == 1
    assert s.loc["Kazandi", "o25_correct"] == 1     # 2-1: 3 gol, Üst doğru
    assert s.loc["Kazandi", "kg_correct"] == 0      # KG Yok dedik, iki takım da attı
    assert s.loc["Bekliyor", "status"] == "sonuc_bekleniyor"
    assert s.loc["Ertelendi", "status"] == "ertelendi"
    assert s.loc["Gelecek", "status"] == "yaklasan"
    assert np.isnan(s.loc["Gelecek", "correct"])


def test_fixture_times_converted_from_uk_to_utc(tmp_path):
    f = tmp_path / "fixtures.csv"
    f.write_text("﻿Div,Date,Time,HomeTeam,AwayTeam,AvgH,AvgD,AvgA\n"
                 "T1,24/10/2026,17:00,Samsunspor,Amedspor,3.34,3.46,2.01\n"
                 "T1,26/10/2026,18:30,Galatasaray,Fenerbahce,2.13,3.73,2.85\n"
                 "B1,24/10/2026,15:00,Antwerp,Charleroi,3.9,3.7,1.75\n", encoding="utf-8")
    fx = load_fixtures(f)
    assert list(fx["league"]) == ["T1", "T1"]
    assert fx.loc[0, "kickoff"] == "2026-10-24T16:00Z"   # yaz saati: İngiltere UTC+1
    assert fx.loc[1, "kickoff"] == "2026-10-26T18:30Z"   # kış saati: İngiltere UTC+0
    assert fx.loc[0, "season"] == "2627"


def test_predictions_are_valid_probabilities_and_record_data_cutoff():
    from iddia.daily import fit, predict
    from tests.test_features import with_stats
    from tests.test_models import synthetic

    matches = with_stats(synthetic(n_seasons=3)).assign(pre_o25=1.9, pre_u25=1.9)
    fixtures = pd.DataFrame({
        "league": "X", "date": ["2026-10-26", "2026-10-27"], "kickoff": ["2026-10-26T17:00Z", "2026-10-27T17:00Z"],
        "home": ["A", "F"], "away": ["F", "A"], "season": "0304",
        "odds_h": [1.5, np.nan], "odds_d": [4.0, np.nan], "odds_a": [6.0, np.nan],
        "odds_o25": [1.8, np.nan], "odds_u25": [2.0, np.nan],
    })
    out = predict(fixtures, matches, fit(matches), NOW)
    p = out[["p_h", "p_d", "p_a"]].to_numpy()
    assert ((p >= 0) & (p <= 1)).all()
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-3)
    assert out[["p_o25", "p_kg"]].apply(lambda c: c.between(0, 1)).all().all()
    assert list(out["model"]) == ["form-piyasa-v2", "form-v2"]
    assert (out["data_through"] == matches["date"].max().strftime("%Y-%m-%d")).all()


def test_merge_keeps_one_row_per_match():
    old = pd.concat([pred("A", "2026-10-26T17:00Z"), pred("B", "2026-10-20T17:00Z")])
    new = pd.concat([pred("A", "2026-10-26T17:00Z", 0.6), pred("A", "2026-10-26T17:00Z", 0.6), pred("B", "2026-10-20T17:00Z")])
    merged = merge_predictions(old, new.drop_duplicates(), NOW)
    assert not merged.duplicated(["league", "date", "home", "away"]).any()
