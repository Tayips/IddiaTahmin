"""Benzer durumlar tablosu: model geçmişte bir sonuca bu kadar olasılık verdiğinde ne oldu?

Günlük modelin aynısı (form + piyasa, lojistik regresyon) son 5 tam sezonu
sezon sezon ileri yürüyerek tahmin eder. Maçlar modelin seçtiği sonuca (1, X, 2)
ve verdiği olasılığa göre 5 puanlık kovalara ayrılır; her kovada kaç maçta
seçimin tuttuğu sayılır. Analiz ekranı bu tabloyu okur.

Yeniden üretmek için: python -m iddia.benzer
"""
import json

import numpy as np

from iddia import metrics
from iddia.compare import prepare
from iddia.daily import SIMILAR
from iddia.features import FEATURES_NO_MARKET, MARKET_FEATURES
from iddia.models.ml import Model

BIN = 0.05
TEST_SEASONS = 5


def main() -> None:
    df, X = prepare()
    seasons = sorted(df["season"].unique())[:-1][-TEST_SEASONS:]
    has_mk = X["mk_h"].notna()
    probs = np.full((len(df), 3), np.nan)
    for season in seasons:
        train = (df["season"] < season) & has_mk
        test = ((df["season"] == season) & has_mk).to_numpy()
        model = Model("lr", FEATURES_NO_MARKET + MARKET_FEATURES).fit(X[train], df.loc[train, "ftr"])
        probs[test] = model.predict(X[test])
    ok = np.isfinite(probs).all(axis=1)
    p, y = probs[ok], df.loc[ok, "ftr"].to_numpy()
    pick = np.array(metrics.OUTCOMES)[p.argmax(axis=1)]
    top = p.max(axis=1)
    lo = np.floor(top / BIN) * BIN
    table = []
    for outcome in metrics.OUTCOMES:
        for b in sorted(set(lo[pick == outcome])):
            sel = (pick == outcome) & (lo == b)
            table.append({"pick": outcome, "lo": round(float(b), 2), "hi": round(float(b + BIN), 2),
                          "n": int(sel.sum()), "hit": int((y[sel] == outcome).sum()),
                          "avg_p": round(float(top[sel].mean()), 4)})
    out = {"seasons": [str(s) for s in seasons], "n": int(ok.sum()), "bins": table}
    SIMILAR.parent.mkdir(exist_ok=True)
    SIMILAR.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for r in table:
        print(f"{r['pick']} %{r['lo']*100:.0f}-{r['hi']*100:.0f}: {r['hit']}/{r['n']}  (ort. tahmin %{r['avg_p']*100:.1f})")


if __name__ == "__main__":
    main()
