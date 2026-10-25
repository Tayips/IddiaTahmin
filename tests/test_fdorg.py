from datetime import datetime, timezone

import pandas as pd

from iddia import fdorg
from iddia.daily import score
from tests.test_daily import pred

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)


def match(home, away, utc, status, h=None, a=None):
    return {"homeTeam": {"name": home}, "awayTeam": {"name": away}, "utcDate": utc, "status": status,
            "score": {"fullTime": {"home": h, "away": a}}}


ORG = fdorg.parse({"matches": [
    match("Manchester United FC", "AFC Bournemouth", "2026-10-24T14:00:00Z", "FINISHED", 1, 1),
    match("Nottingham Forest FC", "Wolverhampton Wanderers FC", "2026-11-01T16:30:00Z", "TIMED"),
    match("Arsenal FC", "Chelsea FC", "2026-10-26T15:00:00Z", "POSTPONED"),
    match("Liverpool FC", "Everton FC", "2026-10-29T20:00:00Z", "SCHEDULED"),
]})


def e0(home, away, kickoff):
    row = pred(home, kickoff)
    row["league"], row["away"] = "E0", away
    return row


def test_team_names_match_football_data_co_uk():
    assert list(ORG["home"]) == ["Man United", "Nott'm Forest", "Arsenal", "Liverpool"]
    assert list(ORG["away"]) == ["Bournemouth", "Wolves", "Chelsea", "Everton"]
    assert ORG.loc[1, "date"] == "2026-11-01" and ORG.loc[1, "kickoff"] == "2026-11-01T16:30Z"


def test_fixtures_get_new_time_and_postponed_ones_are_dropped():
    fixtures = pd.DataFrame({
        "league": ["E0", "E0", "T1"], "date": ["2026-11-01", "2026-10-26", "2026-10-26"],
        "kickoff": ["2026-11-01T15:00Z", "2026-10-26T15:00Z", "2026-10-26T17:00Z"],
        "home": ["Nott'm Forest", "Arsenal", "Arsenal"], "away": ["Wolves", "Chelsea", "Chelsea"],
    })
    out = fdorg.apply_to_fixtures(fixtures, ORG)
    assert list(out["league"]) == ["E0", "T1"]
    assert out.loc[0, "kickoff"] == "2026-11-01T16:30Z"
    assert out.loc[1, "kickoff"] == "2026-10-26T17:00Z"     # başka lig, dokunulmaz


def test_score_uses_fast_results_and_marks_moved_matches():
    preds = pd.concat([
        e0("Man United", "Bournemouth", "2026-10-24T14:00Z"),   # sonuç yalnız football-data.org'da
        e0("Arsenal", "Chelsea", "2026-10-26T15:00Z"),          # ertelendi
        e0("Liverpool", "Everton", "2026-10-26T20:00Z"),        # başka güne alındı
        e0("Nott'm Forest", "Wolves", "2026-11-01T16:30Z"),     # normal
    ], ignore_index=True)
    no_results = pd.DataFrame({"league": [], "date": pd.to_datetime([]), "home": [], "away": [],
                               "fthg": [], "ftag": [], "ftr": []})
    s = score(preds, no_results, NOW, ORG).set_index("home")
    assert s.loc["Man United", "status"] == "bitti" and s.loc["Man United", "ftr"] == "D"
    assert s.loc["Man United", "correct"] == 0
    assert s.loc["Arsenal", "status"] == "ertelendi"
    assert s.loc["Liverpool", "status"] == "ertelendi"
    assert s.loc["Nott'm Forest", "status"] == "yaklasan"


def test_without_key_nothing_changes(monkeypatch):
    monkeypatch.delenv(fdorg.ENV_KEY, raising=False)
    assert fdorg.fetch(2026) is None
    preds = e0("Arsenal", "Chelsea", "2026-10-26T15:00Z")
    empty = pd.DataFrame({"league": [], "date": pd.to_datetime([]), "home": [], "away": [],
                          "fthg": [], "ftag": [], "ftr": []})
    assert score(preds, empty, NOW).loc[0, "status"] == "yaklasan"
