from datetime import datetime, timezone

import pandas as pd

from iddia import haberler

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)


def rss(*items):
    body = "".join(f"<item><title>{t} - {s}</title><link>{l}</link><pubDate>{d}</pubDate>"
                   f"<source url=\"https://x\">{s}</source></item>" for t, s, l, d in items)
    return f"<rss><channel>{body}</channel></rss>"


def test_parse_strips_source_flags_absence_and_drops_old_or_unsafe_links():
    xml = rss(
        ("Galatasaray'da Icardi sakatlandı", "Fanatik", "https://news.google.com/a", "Sun, 25 Oct 2026 09:00:00 GMT"),
        ("Galatasaray derbiye hazır", "NTV", "https://news.google.com/b", "Sat, 24 Oct 2026 20:00:00 GMT"),
        ("Eski haber", "NTV", "https://news.google.com/c", "Mon, 19 Oct 2026 20:00:00 GMT"),
        ("Kötü bağlantı", "NTV", "javascript:alert(1)", "Sun, 25 Oct 2026 10:00:00 GMT"),
    )
    items = haberler.parse(xml, NOW)
    assert [x["title"] for x in items] == ["Galatasaray'da Icardi sakatlandı", "Galatasaray derbiye hazır"]
    assert [x["eksik"] for x in items] == [True, False]
    assert items[0]["source"] == "Fanatik" and items[0]["at"] == "2026-10-25T09:00Z"


def preds():
    return pd.DataFrame({
        "league": ["T1", "T1", "T1", "T1"], "date": ["2026-10-24", "2026-10-26", "2026-10-26", "2026-11-05"],
        "home": ["Eski", "Galatasaray", "Besiktas", "Uzak"], "away": ["X", "Fenerbahce", "Galatasaray", "Y"],
        "kickoff": ["2026-10-24T17:00Z", "2026-10-26T17:00Z", "2026-10-26T14:00Z", "2026-11-05T17:00Z"],
    })


def test_update_only_upcoming_keeps_started_and_fetches_each_team_once():
    calls = []

    def fake(team, league, now):
        calls.append(team)
        return None if team == "Besiktas" else [{"title": f"{team} haberi", "eksik": False}]

    old = {"T1|2026-10-24|Eski|X": {"fetched_at": "eski"}}
    news = haberler.update(old, preds(), NOW, fetch=fake)
    assert news["T1|2026-10-24|Eski|X"] == {"fetched_at": "eski"}                 # başlamış maç donmuş
    assert "T1|2026-11-05|Uzak|Y" not in news                                     # 2 günden uzak
    assert news["T1|2026-10-26|Besiktas|Galatasaray"]["home"] == []               # okunamayan taraf boş
    assert news["T1|2026-10-26|Galatasaray|Fenerbahce"]["away"][0]["title"] == "Fenerbahce haberi"
    assert sorted(calls) == ["Besiktas", "Fenerbahce", "Galatasaray"]            # Galatasaray bir kez


def test_failed_fetch_keeps_previous_record():
    key = "T1|2026-10-26|Galatasaray|Fenerbahce"
    old = {key: {"fetched_at": "önceki", "home": [], "away": []}}
    news = haberler.update(old, preds(), NOW, fetch=lambda *a: None)
    assert news[key]["fetched_at"] == "önceki"
