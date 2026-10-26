"""Gemini ile maç öncesi eksik oyuncu analizi.

Toplanan haber başlıkları ve Google araması (Gemini'nin arama aracı) ile her
maçta iki takımın eksik oyuncularını, eksiklik nedenini ve oyuncunun takımdaki
önemini yapılandırılmış olarak çıkarır. Sonuç haber kaydına "ai" alanı olarak
eklenir; analiz ekranında gösterilir, tahmine dahil edilmez.

Anahtar GEMINI_API_KEY ortam değişkeninden okunur; yoksa ya da istek
başarısız olursa bu adım atlanır. Ücretsiz katmanın dakika sınırına takılmamak
için istekler arasında beklenir ve bir çalıştırmada en fazla MAX_CALLS istek
yapılır. Başlıklar değişmediyse maç yeniden sorulmaz.
"""
import hashlib
import json
import os
import re
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from iddia.haberler import LOOKAHEAD, match_key

API = "https://generativelanguage.googleapis.com/v1beta"
ENV_KEY = "GEMINI_API_KEY"
ENV_MODEL = "GEMINI_MODEL"
FALLBACK_MODEL = "gemini-2.5-flash"
TIMEOUT = 60                            # saniye
MAX_MODELS = 2                          # her türden denenecek en yeni model sayısı
MAX_CALLS = 20
PAUSE = 7.0                             # ücretsiz katman: dakikada ~10 istek
REFRESH = timedelta(hours=6)            # başlıklar değişmese de bu kadar eskiyse yeniden sor
REASONS = {"sakat", "cezalı", "belirsiz", "diğer"}
ROLES = {"kilit", "ilk11", "rotasyon"}

PROMPT = """Futbol maçı: {home} (ev sahibi) - {away} (deplasman), {league}, başlama {kickoff} UTC.
Bugün {today}. Bu maçta iki takımda da forma giyemeyecek ya da durumu belirsiz olan oyuncuları bul.
Google aramasını kullan; yalnızca son günlerin güvenilir kaynaklarına dayan, tahmin uydurma.
Aşağıdaki başlıklar ipucu olabilir ama başka takımlar ya da branşlarla ilgili olanları yok say:
{headlines}

Yalnızca şu biçimde JSON döndür, başka metin yazma:
{{"home": {{"eksikler": [{{"oyuncu": "Ad Soyad", "neden": "sakat|cezalı|belirsiz|diğer", "onem": "kilit|ilk11|rotasyon"}}], "ozet": "en fazla 2 cümle Türkçe"}},
 "away": {{"eksikler": [], "ozet": ""}}}}
onem: kilit = takımın en önemli birkaç oyuncusundan biri, ilk11 = düzenli ilk 11 oyuncusu, rotasyon = yedek ya da az oynayan.
Eksik oyuncu bulamazsan boş liste döndür."""

LEAGUE_NAMES = {"T1": "Türkiye Süper Lig", "E0": "İngiltere Premier Lig"}


def pick_models(key: str) -> list[str]:
    """Denenecek modeller: en yeni kararlı 'flash' modelleri, sonra 'flash-lite' modelleri.

    En yeni model ücretsiz katmanda kapalı olabilir (429); o zaman sıradakine geçilir.
    """
    if os.environ.get(ENV_MODEL):
        return [os.environ[ENV_MODEL]]
    try:
        r = requests.get(f"{API}/models", headers={"x-goog-api-key": key}, params={"pageSize": 200}, timeout=20)
        r.raise_for_status()
        names = [m["name"].removeprefix("models/") for m in r.json().get("models", [])
                 if "generateContent" in m.get("supportedGenerationMethods", [])]
    except (requests.RequestException, ValueError, KeyError):
        return [FALLBACK_MODEL]

    def ranked(suffix):
        found = [(tuple(map(int, m.group(1).split("."))), n) for n in names if (m := re.fullmatch(rf"gemini-(\d+(?:\.\d+)?)-{suffix}", n))]
        return [n for _, n in sorted(found, reverse=True)]

    return (ranked("flash")[:MAX_MODELS] + ranked("flash-lite")[:MAX_MODELS]) or [FALLBACK_MODEL]


class Client:
    """Model ve arama aracı birleşimlerini sırayla dener; çalışan ilk birleşimde kalır.

    Kota (429), erişim, sunucu hatası ya da zaman aşımı alınan birleşim bu
    çalıştırmada bir daha denenmez; aynı soru sıradaki birleşimle sorulur.
    """

    def __init__(self, key: str, models: list[str]):
        self.key = key
        self.options = [(m, tools) for m in models for tools in (True, False)]
        self.i = 0

    @property
    def model(self) -> str:
        if self.i >= len(self.options):
            return "-"
        m, tools = self.options[self.i]
        return m if tools else f"{m} (aramasız)"

    def __call__(self, prompt: str) -> str | None:
        body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.1}}
        while self.i < len(self.options):
            model, tools = self.options[self.i]
            payload = {**body, "tools": [{"google_search": {}}]} if tools else body
            try:
                r = requests.post(f"{API}/models/{model}:generateContent", json=payload, timeout=TIMEOUT,
                                  headers={"x-goog-api-key": self.key})
                r.raise_for_status()
                parts = r.json()["candidates"][0]["content"]["parts"]
                return "".join(p.get("text", "") for p in parts)
            except (requests.RequestException, ValueError, KeyError, IndexError) as e:
                # Anahtar istek başlığında gider, hata metninde yer almaz; yalnızca tür ve durum kodu yazılır
                status = getattr(getattr(e, "response", None), "status_code", "")
                print(f"Gemini {self.model}: {type(e).__name__} {status}, sıradaki deneniyor")
                self.i += 1
        return None


def _clean(side) -> dict:
    """Model çıktısı güvenilmez veridir: yalnızca beklenen alanlar, sınırlı uzunlukta."""
    side = side if isinstance(side, dict) else {}
    out = []
    for p in side.get("eksikler", [])[:8] if isinstance(side.get("eksikler"), list) else []:
        if not isinstance(p, dict) or not str(p.get("oyuncu", "")).strip():
            continue
        reason, role = str(p.get("neden", "")).lower(), str(p.get("onem", "")).lower()
        out.append({"oyuncu": str(p["oyuncu"]).strip()[:60],
                    "neden": reason if reason in REASONS else "belirsiz",
                    "onem": role if role in ROLES else "rotasyon"})
    return {"eksikler": out, "ozet": str(side.get("ozet", "")).strip()[:300]}


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    return {"home": _clean(data.get("home")), "away": _clean(data.get("away"))}


def headline_hash(entry: dict) -> str:
    titles = [x["title"] for side in ("home", "away") for x in entry.get(side, [])]
    return hashlib.sha1("\n".join(titles).encode()).hexdigest()[:12]


def due(entry: dict, now: datetime) -> bool:
    ai = entry.get("ai")
    if not ai:
        return True
    old = datetime.strptime(ai["at"], "%Y-%m-%dT%H:%MZ").replace(tzinfo=now.tzinfo)
    return ai.get("hash") != headline_hash(entry) or now - old > REFRESH


def analyze(news: dict, preds: pd.DataFrame, now: datetime, key: str | None = None, call=None) -> dict:
    """Yaklaşan maçların haber kayıtlarına Gemini analizini ekler; başlamış maçlara dokunmaz."""
    key = key or os.environ.get(ENV_KEY)
    if not key:
        print(f"{ENV_KEY} yok, Gemini analizi atlandı")
        return news
    real = call is None
    if real:
        call = Client(key, pick_models(key))
    stamp, until = now.strftime("%Y-%m-%dT%H:%MZ"), (now + LOOKAHEAD).strftime("%Y-%m-%dT%H:%MZ")
    soon = preds[(preds["kickoff"] > stamp) & (preds["kickoff"] <= until)].sort_values("kickoff")
    out, calls, done = dict(news), 0, 0
    for r in soon.to_dict("records"):
        k = match_key(r)
        if k not in out or not due(out[k], now) or calls >= MAX_CALLS:
            continue
        entry = out[k]
        heads = [f"- {x['title']} ({x['source']})" for side in ("home", "away") for x in entry.get(side, [])]
        prompt = PROMPT.format(home=r["home"], away=r["away"], league=LEAGUE_NAMES.get(r["league"], r["league"]),
                               kickoff=r["kickoff"], today=now.strftime("%Y-%m-%d"),
                               headlines="\n".join(heads) or "(başlık yok)")
        if calls and real:
            time.sleep(PAUSE)
        calls += 1
        result = parse(call(prompt) or "")
        if result is None:
            continue                          # okunamadıysa varsa eski analiz kalır
        done += 1
        model = getattr(call, "model", "test")
        out[k] = {**entry, "ai": {**result, "model": model, "at": stamp, "hash": headline_hash(entry)}}
    print(f"Gemini: {done}/{calls} maç analiz edildi")
    return out
