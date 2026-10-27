from datetime import datetime, timezone

import pandas as pd

from iddia import coupons

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)


def row(home, kickoff, p, odds, p_o25=0.6, o25=(1.8, 2.0)):
    return dict(league="T1", date=kickoff[:10], home=home, away="X", kickoff=kickoff,
                p_h=p[0], p_d=p[1], p_a=p[2], pick="HDA"[max(range(3), key=lambda i: p[i])],
                odds_h=odds[0], odds_d=odds[1], odds_a=odds[2], p_o25=p_o25, odds_o25=o25[0], odds_u25=o25[1])


PREDS = pd.DataFrame([
    row("Guclu", "2026-10-26T17:00Z", (0.75, 0.15, 0.10), (1.30, 5.0, 9.0)),
    row("Iyi", "2026-10-26T19:00Z", (0.62, 0.22, 0.16), (1.60, 3.9, 5.5)),
    row("Denk", "2026-10-27T17:00Z", (0.40, 0.30, 0.30), (2.40, 3.2, 3.0), p_o25=0.45),
    row("Surpriz", "2026-10-27T19:00Z", (0.30, 0.25, 0.45), (3.10, 3.4, 2.10)),
    row("Uzak", "2026-11-05T17:00Z", (0.90, 0.05, 0.05), (1.05, 15, 30)),       # 3 günden sonra
    row("Basladi", "2026-10-25T11:00Z", (0.90, 0.05, 0.05), (1.05, 15, 30)),    # başlamış
])


def test_rules_pick_expected_legs():
    c = coupons.candidates(PREDS, NOW)
    assert set(c["home"]) == {"Guclu", "Iyi", "Denk", "Surpriz"}
    low = coupons.build("dusuk", c)
    assert 2 <= len(low) <= 3
    assert low["home"].is_unique
    mid = coupons.build("orta", c)
    assert (mid["prob"] >= 0.6).all() and "CS" not in set(mid["market"])
    risky = coupons.build("riskli", c)
    assert (risky["odds"] >= 1.8).all() and risky["home"].is_unique


def test_open_coupon_rebuilt_started_coupon_frozen():
    first = coupons.update(pd.DataFrame(columns=coupons.LEG_COLUMNS), PREDS, NOW)
    assert set(first["type"]) == set(coupons.TYPES)
    # Bir gün sonra ilk maçlar başladı: o kuponlar donar, yenileri kalan maçlardan kurulur
    later = datetime(2026, 10, 26, 18, 0, tzinfo=timezone.utc)
    second = coupons.update(first, PREDS, later)
    old_ids = set(first["coupon_id"])
    frozen = second[second["coupon_id"].isin(old_ids)]
    pd.testing.assert_frame_equal(frozen.reset_index(drop=True),
                                  first[first["coupon_id"].isin(set(frozen["coupon_id"]))].reset_index(drop=True))
    new = second[~second["coupon_id"].isin(old_ids)]
    assert (new["kickoff"] > "2026-10-26T18:00Z").all()


def test_results_and_summary():
    legs = pd.DataFrame([
        dict(coupon_id="a", type="orta", created_at="t", league="T1", date="2026-10-20", home="A", away="X",
             kickoff="2026-10-20T17:00Z", market="1X2", selection="H", prob=0.7, odds=1.5),
        dict(coupon_id="a", type="orta", created_at="t", league="T1", date="2026-10-20", home="B", away="X",
             kickoff="2026-10-20T17:00Z", market="OU", selection="O", prob=0.6, odds=1.8),
        dict(coupon_id="b", type="dusuk", created_at="t", league="T1", date="2026-10-20", home="C", away="X",
             kickoff="2026-10-20T17:00Z", market="CS", selection="X2", prob=0.8, odds=1.2),
        dict(coupon_id="b", type="dusuk", created_at="t", league="T1", date="2026-10-20", home="A", away="X",
             kickoff="2026-10-20T17:00Z", market="1X2", selection="H", prob=0.7, odds=1.5),
    ])
    scored = pd.DataFrame(dict(league="T1", date="2026-10-20", home=["A", "B", "C"], away="X",
                               ftr=["H", "A", "H"], fthg=[2, 1, 1], ftag=[0, 2, 0], status="bitti"))
    rec = {c["id"]: c for c in coupons.to_records(legs, scored)}
    assert rec["a"]["status"] == "kazandi" and rec["a"]["odds"] == 2.7      # 1 ve Üst (1-2: 3 gol)
    assert rec["b"]["status"] == "kaybetti"                                 # X2 dedik, ev sahibi kazandı
    s = coupons.summary(list(rec.values()))
    assert s["orta"] == {"n": 1, "won": 1, "net": 1.7}
    assert s["dusuk"] == {"n": 1, "won": 0, "net": -1}


def test_backfill_builds_frozen_coupons_once_before_live_ones():
    live = coupons.update(pd.DataFrame(columns=coupons.LEG_COLUMNS), PREDS, NOW)
    out = coupons.with_backfill(live, PREDS)
    past = out[out["coupon_id"].str.endswith(coupons.BACKFILL_SUFFIX)]
    assert len(past) and (past["created_at"] < live["created_at"].min()).all()
    # her geriye dönük kupon, canlı kuponlar başlamadan donmuş olmalı
    assert (past.groupby("coupon_id")["kickoff"].min() <= live["created_at"].min()).all()
    assert len(coupons.with_backfill(out, PREDS)) == len(out)        # ikinci kez eklenmez
    recs = coupons.to_records(out, PREDS.assign(ftr=None, fthg=None, ftag=None, status="yaklasan"))
    assert any(r["backfill"] for r in recs) and not all(r["backfill"] for r in recs)
