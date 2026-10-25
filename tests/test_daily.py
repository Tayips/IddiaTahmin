from datetime import datetime, timezone

import numpy as np
import pandas as pd

from iddia.daily import BACKFILL, PRED_COLUMNS, load_fixtures, merge_predictions, score

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)


def pred(home, kickoff, p_h=0.5, predicted_at="2026-10-24T05:00Z"):
    row = dict(league="T1", date=kickoff[:10], home=home, away="X", season="2627", kickoff=kickoff,
               p_h=p_h, p_d=0.3, p_a=0.2, pick="H", odds_h=2.0, odds_d=3.4, odds_a=3.8,
               model="elo-v1", predicted_at=predicted_at)
    return pd.DataFrame([row])[PRED_COLUMNS]


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
