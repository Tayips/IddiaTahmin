"""Maç öncesi haberler: yaklaşan maçların takımları için son haber başlıkları.

Google Haberler'in ücretsiz RSS aramasından (anahtar gerekmez) her takım için
son 3 günün sakatlık, ceza ve kadro haberleri çekilir. Eksik oyuncuya işaret
eden başlıklar anahtar kelimelerle işaretlenir. Haberler şimdilik yalnızca
analiz ekranında gösterilir, tahmine dahil edilmez; işaretlerin sonuçla
ilişkisi ölçüldükten sonra modele eklenip eklenmeyeceğine karar verilir.

Kayıtlar data/haberler.json'da maç başına tutulur. Maç başlayınca kayıt
donar (tahminler gibi), böylece sonradan yalnızca maç öncesi haberler ölçülür.
"""
import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
import requests

NEWS = Path("data/haberler.json")
URL = "https://news.google.com/rss/search"
LOOKAHEAD = timedelta(days=2)
MAX_AGE = timedelta(days=3)
PER_TEAM = 5
PAUSE = 1.0                             # istekler arası bekleme (saniye)

LANG = {
    "T1": {"hl": "tr", "gl": "TR", "ceid": "TR:tr",
           "terms": '(sakat OR sakatlık OR cezalı OR "kadroda yok" OR "ilk 11" OR kadro) -voleybol -basketbol'},
    "E0": {"hl": "en-GB", "gl": "GB", "ceid": "GB:en",
           "terms": '(injury OR injured OR suspended OR "team news" OR "ruled out" OR doubt)'},
}
# Fikstürdeki kısaltılmış adlar → aramada kullanılan ad
SEARCH_NAMES = {
    "Buyuksehyr": "Başakşehir", "Goztep": "Göztepe", "Genclerbirligi": "Gençlerbirliği",
    "Kasimpasa": "Kasımpaşa", "Besiktas": "Beşiktaş", "Fenerbahce": "Fenerbahçe", "Corum": "Çorum FK",
    "Eyupspor": "Eyüpspor", "Gaziantep": "Gaziantep FK", "Karagumruk": "Karagümrük",
    "Man United": "Manchester United", "Man City": "Manchester City", "Nott'm Forest": "Nottingham Forest",
    "Hull": "Hull City", "Ipswich": "Ipswich Town", "Leeds": "Leeds United", "Coventry": "Coventry City",
    "Newcastle": "Newcastle United", "Wolves": "Wolverhampton", "Sheffield Weds": "Sheffield Wednesday",
}
# Eksik oyuncuya işaret eden kelimeler (başlıkta geçerse haber işaretlenir)
# Kulübün başka branşları ya da rakibin rakibi hakkındaki haberler (aynı kulüp adı geçse de konu dışı)
OFF_TOPIC = re.compile(
    r"voleybol|basketbol|basket|vakıfbank|sultanlar|euroleague|kadın futbol|futsal|u19|u21|"
    r"rakibi|women|wsl|academy|under-21|u21s|basketball", re.IGNORECASE)
ABSENCE = re.compile(
    r"sakat|cezal|kadroda yok|forma giyemeyecek|oynayamayacak|kadro dışı|ameliyat|"
    r"injur|ruled out|suspend|sidelined|miss(es|ing)? |doubt|blow|setback|out for", re.IGNORECASE)


def match_key(r) -> str:
    """Sitedeki matchKey ile aynı: lig|tarih|ev|deplasman."""
    return f"{r['league']}|{r['date']}|{r['home']}|{r['away']}"


def parse(xml: str, now: datetime) -> list[dict]:
    items = []
    for it in ET.fromstring(xml).iter("item"):
        title, link, pub = it.findtext("title") or "", it.findtext("link") or "", it.findtext("pubDate")
        source = it.findtext("source") or ""
        try:
            when = parsedate_to_datetime(pub)
        except (TypeError, ValueError):
            continue
        if now - when > MAX_AGE or not link.startswith("https://") or OFF_TOPIC.search(title):
            continue
        # Google başlığın sonuna " - Kaynak" ekler
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3]
        items.append({"title": title.strip(), "source": source, "link": link,
                      "at": when.strftime("%Y-%m-%dT%H:%MZ"), "eksik": bool(ABSENCE.search(title))})
    items.sort(key=lambda x: x["at"], reverse=True)
    return items[:PER_TEAM]


def fetch_team(team: str, league: str, now: datetime) -> list[dict] | None:
    lang = LANG[league]
    q = f'"{SEARCH_NAMES.get(team, team)}" {lang["terms"]} when:3d'
    try:
        r = requests.get(URL, params={"q": q, "hl": lang["hl"], "gl": lang["gl"], "ceid": lang["ceid"]},
                         timeout=20, headers={"User-Agent": "Mozilla/5.0 (IddiaTahmin)"})
        r.raise_for_status()
        return parse(r.text, now)
    except (requests.RequestException, ET.ParseError) as e:
        print(f"Haberler okunamadı ({team}): {type(e).__name__}")
        return None


def update(old: dict, preds: pd.DataFrame, now: datetime, fetch=fetch_team) -> dict:
    """Önümüzdeki 2 günün başlamamış maçları için haberleri yeniler; başlamış maçların kaydına dokunmaz."""
    stamp, until = now.strftime("%Y-%m-%dT%H:%MZ"), (now + LOOKAHEAD).strftime("%Y-%m-%dT%H:%MZ")
    soon = preds[(preds["kickoff"] > stamp) & (preds["kickoff"] <= until) & preds["league"].isin(LANG.keys())]
    out = dict(old)
    cache: dict[tuple, list | None] = {}
    for r in soon.to_dict("records"):
        sides = {}
        for side in ("home", "away"):
            k = (r["league"], r[side])
            if k not in cache:
                cache[k] = fetch(r[side], r["league"], now)
                time.sleep(PAUSE if fetch is fetch_team else 0)
            sides[side] = cache[k]
        if sides["home"] is None and sides["away"] is None:
            continue                      # ikisi de okunamadıysa eski kaydı koru
        out[match_key(r)] = {"fetched_at": stamp, "kickoff": r["kickoff"],
                             "home": sides["home"] or [], "away": sides["away"] or []}
    print(f"Haberler: {len(soon)} maç için güncellendi")
    return out


def load() -> dict:
    return json.loads(NEWS.read_text()) if NEWS.exists() else {}


def save(news: dict) -> None:
    NEWS.parent.mkdir(parents=True, exist_ok=True)
    NEWS.write_text(json.dumps(news, ensure_ascii=False, indent=1, sort_keys=True))


def for_site(news: dict, preds: pd.DataFrame) -> dict:
    """Sitede yalnızca listedeki maçların haberleri gösterilir."""
    keys = {match_key(r) for r in preds.to_dict("records")}
    return {k: v for k, v in news.items() if k in keys}
