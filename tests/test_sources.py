import json

import numpy as np
import pandas as pd

from iddia import apifootball, understat


def understat_season():
    team = lambda name, deep: {"title": name, "history": [
        {"date": "2024-08-17 14:00:00", "npxG": 1.1, "deep": deep, "ppda": {"att": 200, "def": 20}}]}
    return {
        "teams": {"1": team("Manchester United", 7), "2": team("Fulham", 3)},
        "dates": [
            {"isResult": True, "h": {"title": "Manchester United"}, "a": {"title": "Fulham"},
             "xG": {"h": "2.4", "a": "0.4"}, "datetime": "2024-08-17 14:00:00"},
            {"isResult": False, "h": {"title": "Fulham"}, "a": {"title": "Manchester United"},
             "xG": {"h": None, "a": None}, "datetime": "2025-05-01 14:00:00"},
        ],
    }


def test_understat_parses_maps_names_and_joins_within_a_day(tmp_path):
    (tmp_path / "EPL").mkdir()
    (tmp_path / "EPL" / "2024.json").write_text(json.dumps(understat_season()))
    xg = understat.load(tmp_path)
    assert len(xg) == 1                                   # oynanmamış maç atlanır
    assert xg.loc[0, "home"] == "Man United"
    assert xg.loc[0, "ppda_h"] == 10.0 and xg.loc[0, "deep_a"] == 3
    matches = pd.DataFrame({
        "league": ["E0", "E0", "T1"],
        "date": pd.to_datetime(["2024-08-16", "2024-08-24", "2024-08-16"]),   # gece yarısı kayması
        "home": ["Man United", "Man United", "Galatasaray"],
        "away": ["Fulham", "Fulham", "Fenerbahce"],
        "xg_h": [np.nan, np.nan, 1.7],
        "xg_a": [np.nan, np.nan, 0.9],
    })
    out = understat.add_xg(matches, xg)
    assert out.loc[0, "xg_h"] == 2.4
    assert np.isnan(out.loc[1, "xg_h"])                  # bir haftalık fark eşleşmez
    assert out.loc[2, "xg_h"] == 1.7                     # football-data xG'si korunur


def api_fixture(fid, day, home, away, hid, aid, hg, ag):
    return {"fixture": {"id": fid, "date": f"{day}T19:00:00+00:00"},
            "teams": {"home": {"name": home, "id": hid}, "away": {"name": away, "id": aid}},
            "goals": {"home": hg, "away": ag}}


def test_apifootball_maps_team_names_by_result_and_counts_injuries(tmp_path):
    fixtures = [api_fixture(1, "2023-08-11", "Manchester City", "Burnley", 50, 44, 3, 0),
                api_fixture(2, "2023-08-19", "Burnley", "Manchester City", 44, 50, 1, 1)]
    injuries = [
        {"player": {"type": "Missing Fixture"}, "team": {"id": 50}, "fixture": {"id": 1}},
        {"player": {"type": "Missing Fixture"}, "team": {"id": 50}, "fixture": {"id": 1}},
        {"player": {"type": "Questionable"}, "team": {"id": 44}, "fixture": {"id": 1}},
    ]
    for endpoint, rows in (("fixtures", fixtures), ("injuries", injuries)):
        (tmp_path / endpoint).mkdir()
        (tmp_path / endpoint / "league39_season2023.json").write_text(json.dumps(rows))
    matches = pd.DataFrame({
        "league": "E0", "date": pd.to_datetime(["2023-08-11", "2023-08-19"]),
        "home": ["Man City", "Burnley"], "away": ["Burnley", "Man City"],
        "fthg": [3, 1], "ftag": [0, 1],
    })
    inj = apifootball.load(matches, tmp_path)
    out = apifootball.add_injuries(matches, inj)
    assert out.loc[0, ["out_h", "out_a", "doubt_h", "doubt_a"]].tolist() == [2, 0, 0, 1]
    assert out.loc[1, ["out_h", "out_a"]].tolist() == [0, 0]
