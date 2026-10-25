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


class Resp:
    def __init__(self, status, data=None):
        self.status_code, self.data = status, data or {}

    def json(self):
        return self.data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise gemini.requests.HTTPError(str(self.status_code))


def test_pick_models_newest_flash_then_lite(monkeypatch):
    names = ["gemini-2.5-flash", "gemini-3.8-flash", "gemini-3.8-flash-lite", "gemini-3.5-flash",
             "gemini-3.8-pro", "gemini-3.9-flash-preview-09", "gemini-3.0-flash-lite"]
    models = {"models": [{"name": f"models/{n}", "supportedGenerationMethods": ["generateContent"]} for n in names]}
    monkeypatch.delenv(gemini.ENV_MODEL, raising=False)
    monkeypatch.setattr(gemini.requests, "get", lambda *a, **k: Resp(200, models))
    assert gemini.pick_models("k") == ["gemini-3.8-flash", "gemini-3.5-flash",
                                       "gemini-3.8-flash-lite", "gemini-3.0-flash-lite"]


def test_client_moves_past_quota_errors_and_stays_on_working_option(monkeypatch):
    seen = []

    def post(url, json, **k):
        seen.append((url.split("/models/")[1].split(":")[0], "tools" in json))
        if seen[-1] != ("b", False):
            return Resp(429)
        return Resp(200, {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]})

    monkeypatch.setattr(gemini.requests, "post", post)
    client = gemini.Client("k", ["a", "b"])
    assert client("soru") == "{}" and client.model == "b (aramasız)"
    assert seen == [("a", True), ("a", False), ("b", True), ("b", False)]
    client("ikinci soru")
    assert seen[-1] == ("b", False) and len(seen) == 5


def test_client_moves_past_server_errors_and_timeouts(monkeypatch):
    seen = []

    def post(url, json, **k):
        seen.append(url.split("/models/")[1].split(":")[0])
        if len(seen) == 1:
            return Resp(503)
        if len(seen) == 2:
            raise gemini.requests.ReadTimeout("yavaş")
        return Resp(200, {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]})

    monkeypatch.setattr(gemini.requests, "post", post)
    client = gemini.Client("k", ["a", "b"])
    assert client("soru") == "{}" and client.model == "b"
    assert seen == ["a", "a", "b"]
