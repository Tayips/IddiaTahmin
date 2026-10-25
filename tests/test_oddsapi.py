from datetime import datetime, timezone

import pandas as pd

from iddia import oddsapi

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)


def event(home, away, utc, books):
    return {"home_team": home, "away_team": away, "commence_time": utc, "bookmakers": [
        {"markets": [
            {"key": "h2h", "outcomes": [{"name": home, "price": h}, {"name": "Draw", "price": d}, {"name": away, "price": a}]},
            {"key": "totals", "outcomes": [{"name": "Over", "price": o, "point": 2.5}, {"name": "Under", "price": u, "point": 2.5}]},
        ]} for h, d, a, o, u in books]}


LIVE = pd.concat([
    oddsapi.parse("T1", [
        event("Galatasaray", "Fenerbahçe", "2026-10-26T17:00:00Z", [(2.0, 3.6, 3.4, 1.7, 2.1), (2.2, 3.4, 3.2, 1.8, 2.0)]),
        event("İstanbul Başakşehir", "Göztepe", "2026-10-26T13:00:00Z", [(1.9, 3.5, 4.0, 1.9, 1.9)]),
    ]),
    oddsapi.parse("E0", [event("Manchester United", "Brighton and Hove Albion", "2026-10-26T14:00:00Z",
                               [(2.1, 3.5, 3.3, 1.6, 2.3)])]),
], ignore_index=True)


def fixtures():
    return pd.DataFrame({
        "league": ["T1", "T1", "E0", "E0"],
        "home": ["Galatasaray", "Buyuksehyr", "Man United", "Arsenal"],
        "away": ["Fenerbahce", "Goztep", "Brighton", "Chelsea"],
        "kickoff": ["2026-10-26T17:00Z", "2026-10-26T13:00Z", "2026-10-26T14:00Z", "2026-10-26T16:30Z"],
        "odds_h": [2.5, None, 2.4, 1.9], "odds_d": [3.3, None, 3.4, 3.5], "odds_a": [2.8, None, 3.0, 4.0],
        "odds_o25": [1.8, None, 1.7, 1.9], "odds_u25": [2.0, None, 2.1, 1.9],
    })


def test_parse_averages_bookmakers():
    gs = LIVE.iloc[0]
    assert (gs["odds_h"], gs["odds_d"], gs["odds_a"], gs["odds_o25"]) == (2.1, 3.5, 3.3, 1.75)
    assert gs["kickoff"] == "2026-10-26T17:00Z"


def test_live_odds_replace_fixture_odds_with_turkish_and_long_names():
    out = oddsapi.apply(fixtures(), LIVE)
    assert list(out["odds_src"]) == ["canli", "canli", "canli", "fikstur"]
    assert out.loc[0, "odds_h"] == 2.1 and out.loc[1, "odds_h"] == 1.9 and out.loc[2, "odds_u25"] == 2.3
    assert out.loc[3, "odds_h"] == 1.9                       # canlı oranı olmayan maç olduğu gibi kalır


def test_no_match_when_kickoff_far_or_names_differ():
    f = fixtures()
    f.loc[0, "kickoff"] = "2026-10-29T17:00Z"                # aynı takımlar, başka gün
    f.loc[2, "home"] = "Liverpool"                           # aynı saat, başka maç
    out = oddsapi.apply(f, LIVE)
    assert out.loc[0, "odds_src"] == "fikstur" and out.loc[0, "odds_h"] == 2.5
    assert out.loc[2, "odds_src"] == "fikstur"


def test_fetch_without_key_or_with_error_is_skipped(monkeypatch):
    monkeypatch.delenv(oddsapi.ENV_KEY, raising=False)
    assert oddsapi.fetch(fixtures(), NOW) is None

    def boom(*a, **k):
        raise oddsapi.requests.ConnectionError("https://api...?apiKey=gizli")
    monkeypatch.setattr(oddsapi.requests, "get", boom)
    assert oddsapi.fetch(fixtures(), NOW, key="x") is None
