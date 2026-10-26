from iddia import log, oddsapi


def test_skipped_source_is_recorded(monkeypatch):
    monkeypatch.setattr(log, "WARNINGS", [])
    monkeypatch.delenv(oddsapi.ENV_KEY, raising=False)
    import pandas as pd
    oddsapi.fetch(pd.DataFrame(columns=["league", "kickoff"]), pd.Timestamp("2026-10-26", tz="UTC"))
    assert log.WARNINGS == [{"kaynak": "The Odds API", "mesaj": "ODDS_API_KEY yok, canlı oranlar atlandı"}]
