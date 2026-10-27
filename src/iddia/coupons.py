"""Kuponlar: yaklaşan maçlardan her gün üç kombine kupon kurar ve sonuçlarını izler.

Her türün aynı anda en fazla bir açık kuponu olur. Açık kupon, ilk maçı başlayana
kadar her çalıştırmada yeniden kurulur; ilk maç başlayınca donar ve sonuçlanınca
puanlanır. Böylece kupon geçmişi yalnızca önceden kurulmuş kuponları ölçer.

Kupon olasılığı, maçların birbirinden bağımsız olduğu varsayımıyla bacak
olasılıklarının çarpımıdır. Hiçbir kupon garanti değildir.
"""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

COUPONS = Path("data/coupons.csv")
KEY = ["league", "date", "home", "away"]
LEG_COLUMNS = ["coupon_id", "type", "created_at"] + KEY + ["kickoff", "market", "selection", "prob", "odds"]
WINDOW_DAYS = 3                       # yalnızca önümüzdeki 3 günün maçları
TYPES = {
    "dusuk": "En düşük risk",
    "orta": "Orta riskli",
    "riskli": "Riskli",
}
BACKFILL_SUFFIX = "-gd"              # geriye dönük kurulan kuponların kimlik eki
RUN_TIMES = ["05:00", "12:00", "15:30"]   # geriye dönük kurulumda günlük çalıştırma saatleri (UTC)
LABELS = {"H": "1", "D": "X", "A": "2", "1X": "1X", "X2": "X2", "12": "12", "O": "Üst 2.5", "U": "Alt 2.5"}


def candidates(preds: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Her maç için oynanabilir seçimler: modelin 1-X-2 seçimi, çifte şans ve Üst/Alt 2.5."""
    stamp = now.strftime("%Y-%m-%dT%H:%MZ")
    until = (now + timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%dT%H:%MZ")
    up = preds[(preds["kickoff"] > stamp) & (preds["kickoff"] <= until)]
    up = up[up[["odds_h", "odds_d", "odds_a"]].notna().all(axis=1)]
    rows = []
    for r in up.itertuples():
        base = {k: getattr(r, k) for k in KEY + ["kickoff"]}
        p = {"H": r.p_h, "D": r.p_d, "A": r.p_a}
        o = {"H": r.odds_h, "D": r.odds_d, "A": r.odds_a}
        for k in ("H", "D", "A"):
            rows.append({**base, "market": "1X2", "selection": k, "prob": p[k], "odds": o[k], "is_pick": k == r.pick})
        # Çifte şans oranı bültende yok; 1X2 oranlarından yaklaşık hesaplanır.
        dc = {"1X": ("H", "D"), "X2": ("D", "A"), "12": ("H", "A")}
        best = max(dc, key=lambda k: p[dc[k][0]] + p[dc[k][1]])
        a, b = dc[best]
        rows.append({**base, "market": "CS", "selection": best, "prob": p[a] + p[b],
                     "odds": round(1 / (1 / o[a] + 1 / o[b]), 2), "is_pick": True})
        if pd.notna(r.p_o25) and pd.notna(r.odds_o25) and pd.notna(r.odds_u25):
            over = r.p_o25 >= 0.5
            rows.append({**base, "market": "OU", "selection": "O" if over else "U",
                         "prob": r.p_o25 if over else 1 - r.p_o25, "odds": r.odds_o25 if over else r.odds_u25,
                         "is_pick": True})
    return pd.DataFrame(rows, columns=KEY + ["kickoff", "market", "selection", "prob", "odds", "is_pick"])


def _one_per_match(c: pd.DataFrame) -> pd.DataFrame:
    return c.drop_duplicates(KEY, keep="first")


def build(kind: str, cands: pd.DataFrame) -> pd.DataFrame:
    """Kupon kuralları. Seçimlerin sırası olasılığa ya da değere göredir; her maçtan en fazla bir bacak."""
    cols = KEY + ["kickoff", "market", "selection", "prob", "odds"]
    picks = cands[cands["is_pick"].astype(bool)]
    if kind == "dusuk":
        # En olası seçimler (çifte şans dahil); kupon olasılığı %55'in altına düşmeyecek kadar, en fazla 3 bacak
        legs, prob = [], 1.0
        for r in _one_per_match(picks[cols].sort_values("prob", ascending=False)).itertuples(index=False):
            if len(legs) >= 3 or (len(legs) >= 2 and prob * r.prob < 0.55):
                break
            legs.append(r._asdict())
            prob *= r.prob
        return pd.DataFrame(legs, columns=cols)
    if kind == "orta":
        # Modelin en az %60 emin olduğu en olası 3 seçim (çifte şans yok)
        pool = picks[(picks["market"] != "CS") & (picks["prob"] >= 0.6)].sort_values("prob", ascending=False)
        return _one_per_match(pool)[cols].head(3)
    if kind == "riskli":
        # Oranı en az 1.80 olan seçimlerden, model olasılığı x oranı en yüksek 4 tanesi
        pool = cands[(cands["market"] != "CS") & (cands["prob"] >= 0.35) & (cands["odds"] >= 1.8)]
        pool = pool.assign(value=pool["prob"] * pool["odds"]).sort_values("value", ascending=False)
        return _one_per_match(pool)[cols].head(4)
    raise ValueError(kind)


def update(old: pd.DataFrame, preds: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Açık kuponları yeniden kurar, başlamış olanlara dokunmaz."""
    stamp = now.strftime("%Y-%m-%dT%H:%MZ")
    first_kick = old.groupby("coupon_id")["kickoff"].transform("min")
    kept = old[first_kick <= stamp]                     # donmuş kuponlar
    cands = candidates(preds, now)
    new = []
    for kind in TYPES:
        legs = build(kind, cands)
        if len(legs) < 2:
            continue
        new.append(legs.assign(coupon_id=f"{kind}-{now:%Y%m%d%H%M}", type=kind, created_at=stamp))
    out = pd.concat([kept, *new], ignore_index=True) if new else kept
    return out.reindex(columns=LEG_COLUMNS).sort_values(["created_at", "type", "kickoff"]).reset_index(drop=True)


def backfill(preds: pd.DataFrame, until: str) -> pd.DataFrame:
    """Kupon özelliği başlamadan önceki günler için kuponları, günlük çalıştırmalar o günlerde
    yapılmış gibi aynı kurallarla kurar. Yalnızca `until` anından önce donmuş kuponlar döner.

    Bu dönemin tahminleri geriye dönüktür (Elo modeli, kapanış oranları); bu yüzden kuponlar
    ayrı işaretlenir ve canlı kupon istatistiğine katılmaz.
    """
    legs = pd.DataFrame(columns=LEG_COLUMNS)
    start = pd.Timestamp(preds["kickoff"].min()).normalize() - pd.Timedelta(days=WINDOW_DAYS)
    end = pd.Timestamp(until)
    for day in pd.date_range(start, end, freq="D"):
        for t in RUN_TIMES:
            now = pd.Timestamp(f"{day:%Y-%m-%d}T{t}Z").to_pydatetime()
            if now >= end:
                break
            legs = update(legs, preds, now)
    first_kick = legs.groupby("coupon_id")["kickoff"].transform("min")
    legs = legs[first_kick <= until].copy()
    legs["coupon_id"] = legs["coupon_id"] + BACKFILL_SUFFIX
    return legs.reset_index(drop=True)


def leg_result(legs: pd.DataFrame, scored: pd.DataFrame) -> pd.Series:
    m = legs.merge(scored[KEY + ["ftr", "fthg", "ftag", "status"]], on=KEY, how="left")
    sel, ftr = m["selection"], m["ftr"]
    goals = m["fthg"] + m["ftag"]
    won = np.select(
        [m["market"] == "1X2", m["market"] == "CS", sel == "O", sel == "U"],
        [sel == ftr,
         sel.str.contains("1") & (ftr == "H") | sel.str.contains("X") & (ftr == "D") | sel.str.contains("2") & (ftr == "A"),
         goals > 2.5, goals < 2.5],
        default=False)
    res = np.select([m["status"] == "bitti", m["status"] == "ertelendi"],
                    [np.where(won, "kazandi", "kaybetti"), "iptal"], default="bekliyor")
    return pd.Series(res, index=legs.index)


def to_records(legs: pd.DataFrame, scored: pd.DataFrame) -> list[dict]:
    if legs.empty:
        return []
    legs = legs.assign(result=leg_result(legs, scored))
    out = []
    for cid, g in legs.groupby("coupon_id", sort=False):
        live = g[g["result"] != "iptal"]       # ertelenen bacak kupondan düşer (oranı 1 sayılır)
        if (g["result"] == "kaybetti").any():
            status = "kaybetti"
        elif live.empty:
            status = "iptal"
        elif (live["result"] == "kazandi").all():
            status = "kazandi"
        else:
            status = "bekliyor"
        out.append({
            "id": cid, "type": g["type"].iloc[0], "created_at": g["created_at"].iloc[0], "status": status,
            "backfill": cid.endswith(BACKFILL_SUFFIX),
            "odds": round(float(live["odds"].prod()), 2) if len(live) else None,
            "prob": round(float(live["prob"].prod()), 4) if len(live) else None,
            "legs": [{"league": r.league, "home": r.home, "away": r.away, "kickoff": r.kickoff,
                      "selection": LABELS[r.selection], "market": r.market, "prob": round(float(r.prob), 4),
                      "odds": round(float(r.odds), 2), "result": r.result}
                     for r in g.sort_values("kickoff").itertuples()],
        })
    out.sort(key=lambda c: c["created_at"], reverse=True)
    return out


def summary(records: list[dict], backfill: bool = False) -> dict:
    """Tür başına sonuçlanan kupon sayısı, tutan kupon ve 1 birim yatırılsa net kazanç.

    Canlı ve geriye dönük kuponlar ayrı sayılır.
    """
    out = {}
    for kind in TYPES:
        done = [c for c in records if c["type"] == kind and c["status"] in ("kazandi", "kaybetti")
                and c.get("backfill", False) == backfill]
        won = [c for c in done if c["status"] == "kazandi"]
        out[kind] = {"n": len(done), "won": len(won),
                     "net": round(sum(c["odds"] - 1 for c in won) - (len(done) - len(won)), 2)}
    return out


def recent(records: list[dict], per_type: int = 60) -> list[dict]:
    return [c for kind in TYPES for c in [r for r in records if r["type"] == kind][:per_type]]


def with_backfill(old: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """Geriye dönük kuponlar henüz yoksa, ilk canlı kupondan önceki dönem için bir kez kurar."""
    if old.empty or old["coupon_id"].str.endswith(BACKFILL_SUFFIX).any():
        return old
    past = backfill(preds, old["created_at"].min())
    print(f"Kuponlar: {past['coupon_id'].nunique()} geriye dönük kupon eklendi")
    out = pd.concat([past, old], ignore_index=True).reindex(columns=LEG_COLUMNS)
    return out.sort_values(["created_at", "type", "kickoff"]).reset_index(drop=True)
