from datetime import datetime, timezone

from iddia import gemini, haberler
from tests.test_haberler import preds

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
KEY = "T1|2026-10-26|Galatasaray|Fenerbahce"
ANSWER = '''```json
{"home": {"eksikler": [{"oyuncu": "Victor Osimhen", "neden": "sakat", "onem": "kilit"},
                       {"oyuncu": "", "neden": "sakat"}, {"oyuncu": "Yedek", "neden": "uydurma", "onem": "x"}],
          "ozet": "Golcü yok."},
 "away": {"eksikler": [], "ozet": ""}, "fazla": 1}
```'''


def test_parse_keeps_only_valid_fields():
    out = gemini.parse(ANSWER)
    assert out["home"]["eksikler"] == [
        {"oyuncu": "Victor Osimhen", "neden": "sakat", "onem": "kilit"},
        {"oyuncu": "Yedek", "neden": "belirsiz", "onem": "rotasyon"}]
    assert out["home"]["ozet"] == "Golcü yok." and out["away"] == {"eksikler": [], "ozet": ""}
    assert gemini.parse("anlamadım") is None and gemini.parse("[1, 2]") is None


def news():
    return {KEY: {"home": [{"title": "Osimhen sakat", "source": "A"}], "away": []},
            "T1|2026-10-24|Eski|X": {"home": [], "away": []}}


def test_analyze_only_upcoming_and_skips_when_headlines_unchanged():
    prompts = []

    def call(prompt):
        prompts.append(prompt)
        return ANSWER

    out = gemini.analyze(news(), preds(), NOW, key="k", call=call)
    assert out[KEY]["ai"]["home"]["eksikler"][0]["oyuncu"] == "Victor Osimhen"
    assert "ai" not in out["T1|2026-10-24|Eski|X"]                     # başlamış maç
    assert len(prompts) == 1 and "Osimhen sakat (A)" in prompts[0]
    gemini.analyze(out, preds(), NOW, key="k", call=call)                # başlıklar aynı: tekrar sorulmaz
    assert len(prompts) == 1
    out[KEY]["home"].append({"title": "Yeni haber", "source": "B"})
    gemini.analyze(out, preds(), NOW, key="k", call=call)
    assert len(prompts) == 2


def test_failed_answer_keeps_old_analysis_and_missing_key_skips(monkeypatch):
    first = gemini.analyze(news(), preds(), NOW, key="k", call=lambda p: ANSWER)
    first[KEY]["home"].append({"title": "Yeni haber", "source": "B"})
    again = gemini.analyze(first, preds(), NOW, key="k", call=lambda p: None)
    assert again[KEY]["ai"]["home"]["eksikler"][0]["oyuncu"] == "Victor Osimhen"
    monkeypatch.delenv(gemini.ENV_KEY, raising=False)
    assert gemini.analyze(news(), preds(), NOW) == news()


def test_news_refresh_keeps_ai_analysis():
    old = gemini.analyze(news(), preds(), NOW, key="k", call=lambda p: ANSWER)
    new = haberler.update(old, preds(), NOW, fetch=lambda *a: [{"title": "x", "source": "y", "eksik": False}])
    assert new[KEY]["ai"] == old[KEY]["ai"]
